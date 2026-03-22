from __future__ import annotations

import argparse
import concurrent.futures
import logging
import os
import re
import subprocess
import sys
import threading
from collections import deque
from pathlib import Path
from typing import List, Optional, Tuple

from omegaconf import OmegaConf

from syn_ver2_breezy import PipelineConfig


def _script_dir() -> Path:
    return Path(__file__).resolve().parent


def _load_pipeline_cfg(config_path: Path):
    raw_cfg = OmegaConf.load(str(config_path))
    if "hydra" in raw_cfg:
        del raw_cfg["hydra"]
    return OmegaConf.merge(OmegaConf.structured(PipelineConfig), raw_cfg)


def _slugify_topic(topic: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(topic).strip())
    slug = slug.strip("._-")
    return slug or "unknown"


def _infer_topics_from_run_root(run_root: Path, config_path: Path) -> List[str]:
    cfg = _load_pipeline_cfg(config_path)
    data_root = str(cfg.get("data_root") or "TEST_syn_data").strip() or "TEST_syn_data"
    mode_name = str(cfg.get("mode_name") or "normal").strip() or "normal"

    base = run_root / data_root / mode_name
    if not base.exists():
        # Fallback to common default path used by this repo.
        base = run_root / "TEST_syn_data" / mode_name

    if not base.exists():
        raise FileNotFoundError(f"Cannot find topics directory under: {base}")

    topics = sorted([p.name for p in base.iterdir() if p.is_dir()])
    if not topics:
        raise ValueError(f"No topics found under: {base}")
    return topics


def _detect_gpus() -> List[str]:
    # Prefer explicit CUDA_VISIBLE_DEVICES if set (treat as the pool).
    cvd = os.getenv("CUDA_VISIBLE_DEVICES")
    if cvd:
        parts = [p.strip() for p in cvd.split(",") if p.strip()]
        # Preserve the user's order.
        if parts:
            return parts

    # Otherwise try nvidia-smi.
    try:
        proc = subprocess.run(
            ["nvidia-smi", "-L"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
        lines = [ln for ln in (proc.stdout or "").splitlines() if ln.strip().lower().startswith("gpu ")]
        if lines:
            return [str(i) for i in range(len(lines))]
    except Exception:
        pass

    return ["0"]


def _run_one_topic(
    python_bin: str,
    config_path: Path,
    run_root: Path,
    topic: str,
    gpu_id: str,
    logs_dir: Path,
) -> Tuple[str, int, str]:
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{_slugify_topic(topic)}.log"

    cmd = [
        str(python_bin),
        str(_script_dir() / "run_topic_tts.py"),
        "--config",
        str(config_path),
        "--topic",
        str(topic),
        "--run-root",
        str(run_root),
    ]

    env = os.environ.copy()
    env.setdefault("PYTHONNOUSERSITE", "1")
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write(f"GPU: {gpu_id}\n")
        log_file.write("COMMAND: " + " ".join(cmd) + "\n\n")
        log_file.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(_script_dir()),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )

    return topic, int(proc.returncode), str(log_path)


def parse_args() -> argparse.Namespace:
    default_config = _script_dir() / "conf" / "base_v2_breezy.yaml"

    p = argparse.ArgumentParser(description="Run TTS stage for multiple topics with one topic per GPU.")
    p.add_argument("--config", type=str, default=str(default_config), help="Path to pipeline config YAML")
    p.add_argument("--run-root", type=str, required=True, help="Existing session/run root")
    p.add_argument(
        "--topics",
        type=str,
        default=None,
        help="Optional comma-separated topic list. If omitted, inferred from run_root.",
    )
    p.add_argument(
        "--python-bin",
        type=str,
        default=sys.executable,
        help="Python interpreter used to launch run_topic_tts.py",
    )
    p.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Optional comma-separated GPU id list. If omitted, auto-detect.",
    )
    p.add_argument(
        "--topics-per-gpu",
        type=int,
        default=1,
        help=(
            "How many topics to assign to each GPU (sequentially). "
            "Example: 2 means each GPU will process 2 topics one-by-one (last group may be smaller)."
        ),
    )
    p.add_argument(
        "--concurrent-per-gpu",
        type=int,
        default=1,
        help=(
            "How many topic TTS jobs to run concurrently on the same GPU. "
            "Default 1 (safe). Example: 4 means one GPU may run 4 topics at the same time."
        ),
    )
    p.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Do not stop the overall launcher when a topic job fails",
    )
    return p.parse_args()


def _split_csv(value: Optional[str]) -> List[str]:
    if value is None:
        return []
    return [x.strip() for x in str(value).split(",") if x.strip()]


def _chunked(items: List[str], chunk_size: int) -> List[List[str]]:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def _run_topics_on_gpu(
    *,
    topics: List[str],
    gpu_id: str,
    python_bin: str,
    config_path: Path,
    run_root: Path,
    logs_dir: Path,
    concurrent_per_gpu: int,
    continue_on_error: bool,
) -> List[Tuple[str, int, str]]:
    """Run a list of topics on a single GPU.

    When concurrent_per_gpu > 1, multiple topics are executed concurrently on the same GPU.
    If continue_on_error is False, we will not start *new* jobs after the first failure,
    but we do not force-terminate already running subprocesses.
    """
    if concurrent_per_gpu <= 1 or len(topics) <= 1:
        out: List[Tuple[str, int, str]] = []
        for topic in topics:
            topic_name, returncode, log_path = _run_one_topic(
                python_bin=python_bin,
                config_path=config_path,
                run_root=run_root,
                topic=topic,
                gpu_id=gpu_id,
                logs_dir=logs_dir,
            )
            out.append((topic_name, returncode, log_path))
            status = "OK" if returncode == 0 else f"FAILED({returncode})"
            print(f"[{status}] topic={topic_name} | gpu={gpu_id} | log={log_path}", flush=True)
            if returncode != 0 and not continue_on_error:
                break
        return out

    max_workers = min(int(concurrent_per_gpu), len(topics))
    stop_starting_new = False
    results: List[Tuple[str, int, str]] = []
    futures: List[concurrent.futures.Future] = []

    def submit_one(executor: concurrent.futures.Executor, t: str) -> None:
        futures.append(
            executor.submit(
                _run_one_topic,
                python_bin,
                config_path,
                run_root,
                t,
                gpu_id,
                logs_dir,
            )
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        it = iter(topics)
        # Prime the pool
        for _ in range(max_workers):
            try:
                submit_one(executor, next(it))
            except StopIteration:
                break

        while futures:
            done, pending = concurrent.futures.wait(
                futures,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )

            for fut in done:
                futures.remove(fut)
                topic_name, returncode, log_path = fut.result()
                results.append((topic_name, returncode, log_path))
                status = "OK" if returncode == 0 else f"FAILED({returncode})"
                print(f"[{status}] topic={topic_name} | gpu={gpu_id} | log={log_path}", flush=True)
                if returncode != 0 and not continue_on_error:
                    stop_starting_new = True

            if stop_starting_new:
                # Do not schedule more tasks; just wait for current ones to finish.
                continue

            # Backfill with next topics
            while len(futures) < max_workers:
                try:
                    submit_one(executor, next(it))
                except StopIteration:
                    break

    return results


def main() -> None:
    args = parse_args()

    config_path = Path(args.config).expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    run_root = Path(args.run_root).expanduser()
    if not run_root.exists():
        raise FileNotFoundError(f"run_root does not exist: {run_root}")

    topics = _split_csv(args.topics)
    if not topics:
        topics = _infer_topics_from_run_root(run_root, config_path)

    topics_per_gpu = int(args.topics_per_gpu)
    if topics_per_gpu <= 0:
        raise ValueError(f"topics-per-gpu must be > 0, got {topics_per_gpu}")

    concurrent_per_gpu = int(args.concurrent_per_gpu)
    if concurrent_per_gpu <= 0:
        raise ValueError(f"concurrent-per-gpu must be > 0, got {concurrent_per_gpu}")

    gpus = _split_csv(args.gpus) or _detect_gpus()
    if not gpus:
        raise ValueError("No GPUs detected")

    logs_dir = run_root / "logs_tts"
    logs_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("=" * 80, flush=True)
    print(f"Run root: {run_root}", flush=True)
    print(f"Topics: {len(topics)}", flush=True)
    print(f"GPU pool: {','.join(gpus)}", flush=True)
    print(f"topics_per_gpu={topics_per_gpu} (sequential per GPU)", flush=True)
    print(f"concurrent_per_gpu={concurrent_per_gpu} (parallel topics per GPU)", flush=True)
    print(f"Logs: {logs_dir}", flush=True)
    print("=" * 80, flush=True)

    results: List[Tuple[str, int, str]] = []

    topic_groups = _chunked(topics, topics_per_gpu)

    # If we have enough GPUs, assign exactly one group to each GPU to match the
    # 'N topics per GPU' intent as closely as possible.
    if len(gpus) >= len(topic_groups):
        for group_idx, group in enumerate(topic_groups):
            gpu_id = gpus[group_idx]
            group_results = _run_topics_on_gpu(
                topics=list(group),
                gpu_id=gpu_id,
                python_bin=str(args.python_bin),
                config_path=config_path,
                run_root=run_root,
                logs_dir=logs_dir,
                concurrent_per_gpu=concurrent_per_gpu,
                continue_on_error=bool(args.continue_on_error),
            )
            results.extend(group_results)
            if (not args.continue_on_error) and any(rc != 0 for _, rc, _ in group_results):
                failed_topic = next((t for t, rc, _ in group_results if rc != 0), "unknown")
                raise RuntimeError(f"Topic TTS failed on gpu={gpu_id}: {failed_topic}. See logs under: {logs_dir}")
    else:
        # Not enough GPUs to keep each group on a distinct GPU. We still guarantee:
        # - At most one active job per GPU at any time.
        # - Topics are processed in groups of size N, but a GPU may process multiple groups sequentially.
        group_queue = deque(topic_groups)
        queue_lock = threading.Lock()
        results_lock = threading.Lock()

        def gpu_worker(gpu_id: str) -> None:
            while True:
                with queue_lock:
                    if not group_queue:
                        return
                    group = group_queue.popleft()

                group_results = _run_topics_on_gpu(
                    topics=list(group),
                    gpu_id=gpu_id,
                    python_bin=str(args.python_bin),
                    config_path=config_path,
                    run_root=run_root,
                    logs_dir=logs_dir,
                    concurrent_per_gpu=concurrent_per_gpu,
                    continue_on_error=bool(args.continue_on_error),
                )

                with results_lock:
                    results.extend(group_results)

                if (not args.continue_on_error) and any(rc != 0 for _, rc, _ in group_results):
                    failed_topic = next((t for t, rc, _ in group_results if rc != 0), "unknown")
                    raise RuntimeError(f"Topic TTS failed on gpu={gpu_id}: {failed_topic}. See logs under: {logs_dir}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as executor:
            futures = [executor.submit(gpu_worker, gpu_id) for gpu_id in gpus]
            for future in concurrent.futures.as_completed(futures):
                future.result()

    failed = [r for r in results if r[1] != 0]
    print("=" * 80, flush=True)
    print(f"Completed {len(results)} topic TTS jobs. Failed={len(failed)}", flush=True)
    print(f"Run root: {run_root}", flush=True)
    print("=" * 80, flush=True)

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
