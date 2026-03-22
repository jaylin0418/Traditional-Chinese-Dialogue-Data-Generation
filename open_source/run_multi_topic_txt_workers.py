from __future__ import annotations

import argparse
import concurrent.futures
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from omegaconf import OmegaConf

from syn_ver2_breezy import PipelineConfig


def _script_dir() -> Path:
    return Path(__file__).resolve().parent


def _load_pipeline_cfg(config_path: Path):
    raw_cfg = OmegaConf.load(str(config_path))
    if "hydra" in raw_cfg:
        del raw_cfg["hydra"]
    return OmegaConf.merge(OmegaConf.structured(PipelineConfig), raw_cfg)


def _split_topics(value, default_topics: List[str]) -> List[str]:
    if value is None:
        return list(default_topics)
    if isinstance(value, (list, tuple)):
        topics = [str(x).strip() for x in value if str(x).strip()]
    else:
        topics = [x.strip() for x in str(value).split(",") if x.strip()]
    seen = set()
    deduped: List[str] = []
    for topic in topics:
        if topic not in seen:
            deduped.append(topic)
            seen.add(topic)
    return deduped


def _slugify_topic(topic: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(topic).strip())
    slug = slug.strip("._-")
    return slug or "unknown"


def _defaults(config_path: Path) -> Dict[str, object]:
    cfg = _load_pipeline_cfg(config_path)
    mt = cfg.get("multi_topic_run", {}) or {}
    return {
        "output_root_base": str(mt.get("output_root_base") or "/work/jaylin0418"),
        "workers": int(mt.get("workers") or 1),
        "per_topic_count": int(mt.get("per_topic_count") or 130),
        "batch_size": int(mt.get("batch_size") or cfg.get("batch_run", {}).get("batch_size") or 10),
        "topics": list(mt.get("topics") or []),
        "final_huggingface_only": bool(mt.get("final_huggingface_only", True)),
    }


def parse_args() -> argparse.Namespace:
    default_config = _script_dir() / "conf" / "base_v2_breezy.yaml"
    defaults = _defaults(default_config)

    p = argparse.ArgumentParser(
        description="Launch multiple topic batch-generation jobs with a worker pool."
    )
    p.add_argument("--config", type=str, default=str(default_config), help="Path to pipeline config YAML")
    p.add_argument(
        "--topics",
        type=str,
        default=None,
        help="Comma-separated topic list. Defaults to multi_topic_run.topics from config.",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=int(defaults["workers"]),
        help="How many topic jobs to run in parallel",
    )
    p.add_argument(
        "--per-topic-count",
        type=int,
        default=int(defaults["per_topic_count"]),
        help="How many scripts/scenarios to generate for each topic",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=int(defaults["batch_size"]),
        help="How many scenarios to generate per batch for each topic",
    )
    p.add_argument(
        "--output-root-base",
        type=str,
        default=str(defaults["output_root_base"]),
        help="All topic run roots will be created under this directory",
    )
    p.add_argument(
        "--python-bin",
        type=str,
        default=sys.executable,
        help="Python interpreter used to launch run_topic_txt.py",
    )
    p.add_argument(
        "--export-every-batch",
        action="store_true",
        help="Run the huggingface stage in every batch instead of only the last one",
    )
    p.add_argument(
        "--stages",
        type=str,
        default=None,
        help="Optional comma-separated stage override passed through to run_topic_txt.py",
    )
    p.add_argument(
        "--with-emotion",
        action="store_true",
        help="Pass --with-emotion through to each topic worker (emotion-tagged dialogue prompts)",
    )
    p.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Do not stop the overall launcher when a topic job fails",
    )
    return p.parse_args()


def _validate_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")


def _build_topic_command(
    python_bin: str,
    config_path: Path,
    topic: str,
    batch_size: int,
    per_topic_count: int,
    run_root: Path,
    export_every_batch: bool,
    stages: str | None,
    with_emotion: bool = False,
) -> List[str]:
    cmd = [
        str(python_bin),
        str(_script_dir() / "run_topic_txt.py"),
        "--config",
        str(config_path),
        "--topic",
        str(topic),
        "--batch-size",
        str(batch_size),
        "--total-count",
        str(per_topic_count),
        "--run-root",
        str(run_root),
    ]
    if export_every_batch:
        cmd.append("--export-every-batch")
    if stages:
        cmd.extend(["--stages", str(stages)])
    if with_emotion:
        cmd.append("--with-emotion")
    return cmd


def _run_one_topic(
    python_bin: str,
    config_path: Path,
    topic: str,
    batch_size: int,
    per_topic_count: int,
    run_root: Path,
    logs_dir: Path,
    export_every_batch: bool,
    stages: str | None,
    with_emotion: bool = False,
) -> Tuple[str, int, str, str]:
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{_slugify_topic(topic)}.log"
    cmd = _build_topic_command(
        python_bin=python_bin,
        config_path=config_path,
        topic=topic,
        batch_size=batch_size,
        per_topic_count=per_topic_count,
        run_root=run_root,
        export_every_batch=export_every_batch,
        stages=stages,
        with_emotion=with_emotion,
    )

    env = os.environ.copy()
    env.setdefault("PYTHONNOUSERSITE", "1")

    with open(log_path, "w", encoding="utf-8") as log_file:
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
    return topic, int(proc.returncode), str(run_root), str(log_path)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    _validate_positive("workers", int(args.workers))
    _validate_positive("per_topic_count", int(args.per_topic_count))
    _validate_positive("batch_size", int(args.batch_size))

    defaults = _defaults(config_path)
    topics = _split_topics(args.topics, list(defaults["topics"]))
    if not topics:
        raise ValueError("No topics provided")

    output_root_base = Path(args.output_root_base).expanduser()
    output_root_base.mkdir(parents=True, exist_ok=True)

    session_root = output_root_base
    logs_dir = session_root / "logs"
    session_root.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("=" * 80, flush=True)
    print(f"Session root: {session_root}", flush=True)
    print(f"Topics: {len(topics)} | workers={args.workers} | per_topic_count={args.per_topic_count} | batch_size={args.batch_size}", flush=True)
    print("=" * 80, flush=True)

    futures = []
    results: List[Tuple[str, int, str, str]] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=int(args.workers)) as executor:
        for topic in topics:
            topic_slug = _slugify_topic(topic)
            run_root = session_root
            future = executor.submit(
                _run_one_topic,
                python_bin=str(args.python_bin),
                config_path=config_path,
                topic=topic,
                batch_size=int(args.batch_size),
                per_topic_count=int(args.per_topic_count),
                run_root=run_root,
                logs_dir=logs_dir,
                export_every_batch=bool(args.export_every_batch),
                stages=args.stages,
                with_emotion=bool(args.with_emotion),
            )
            futures.append(future)

        for future in concurrent.futures.as_completed(futures):
            topic, returncode, run_root, log_path = future.result()
            results.append((topic, returncode, run_root, log_path))
            status = "OK" if returncode == 0 else f"FAILED({returncode})"
            print(f"[{status}] topic={topic} | session_root={run_root} | log={log_path}", flush=True)
            if returncode != 0 and not args.continue_on_error:
                raise RuntimeError(f"Topic job failed: {topic}. See log: {log_path}")

    failed = [r for r in results if r[1] != 0]
    print("=" * 80, flush=True)
    print(f"Completed {len(results)} topic jobs. Failed={len(failed)}", flush=True)
    print(f"Session root: {session_root}", flush=True)
    print("=" * 80, flush=True)

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
