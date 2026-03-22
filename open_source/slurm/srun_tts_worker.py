from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

from omegaconf import OmegaConf


def _script_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def _slugify_topic(topic: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(topic).strip())
    slug = slug.strip("._-")
    return slug or "unknown"


def _split_csv(value: Optional[str]) -> List[str]:
    if value is None:
        return []
    return [x.strip() for x in str(value).split(",") if x.strip()]


def _load_pipeline_cfg(config_path: Path):
    raw_cfg = OmegaConf.load(str(config_path))
    if "hydra" in raw_cfg:
        del raw_cfg["hydra"]

    # Keep it lightweight: we only need data_root/mode_name for topic discovery.
    return raw_cfg


def _infer_topics_from_run_root(run_root: Path, config_path: Path) -> List[str]:
    cfg = _load_pipeline_cfg(config_path)
    data_root = str(cfg.get("data_root") or "TEST_syn_data").strip() or "TEST_syn_data"
    mode_name = str(cfg.get("mode_name") or "normal").strip() or "normal"

    base = run_root / data_root / mode_name
    if not base.exists():
        base = run_root / "TEST_syn_data" / mode_name

    if not base.exists():
        raise FileNotFoundError(f"Cannot find topics directory under: {base}")

    topics = sorted([p.name for p in base.iterdir() if p.is_dir() and not p.name.startswith(".")])
    if not topics:
        raise ValueError(f"No topics found under: {base}")
    return topics


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, ""))
    except Exception:
        return int(default)


def _append_progress_line(path: Path, line: str) -> None:
    """Append a progress line with a simple cross-process file lock."""
    try:
        import fcntl  # Linux only (OK on SLURM clusters)

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            except Exception:
                pass
            f.write(line.rstrip("\n") + "\n")
            f.flush()
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
    except Exception:
        # Progress is best-effort; never fail the worker because of it.
        return


def parse_args() -> argparse.Namespace:
    default_config = _script_dir() / "conf" / "base_v2_breezy.yaml"

    p = argparse.ArgumentParser(
        description=(
            "SLURM srun worker for TTS-only runs. "
            "Each SLURM task handles a disjoint subset of topics (round-robin by rank)."
        )
    )
    p.add_argument("--config", type=str, default=str(default_config), help="Path to pipeline config YAML")
    p.add_argument("--run-root", type=str, required=True, help="Dataset/run root (contains {data_root}/{mode_name}/...")
    p.add_argument(
        "--topics",
        type=str,
        default="auto",
        help="Comma-separated topic list, or 'auto' to scan run_root",
    )
    p.add_argument(
        "--rank",
        type=int,
        default=None,
        help="Optional rank override (default: SLURM_PROCID)",
    )
    p.add_argument(
        "--world-size",
        type=int,
        default=None,
        help="Optional world size override (default: SLURM_NTASKS)",
    )
    p.add_argument(
        "--logs-dir",
        type=str,
        default=None,
        help="Optional logs dir (default: <run_root>/logs_tts)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    config_path = Path(args.config).expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    run_root = Path(str(args.run_root)).expanduser().resolve()
    if not run_root.exists():
        raise FileNotFoundError(f"run_root does not exist: {run_root}")

    rank = int(args.rank) if args.rank is not None else _env_int("SLURM_PROCID", 0)
    world_size = int(args.world_size) if args.world_size is not None else _env_int("SLURM_NTASKS", 1)
    if world_size <= 0:
        world_size = 1
    if rank < 0:
        rank = 0

    if args.logs_dir:
        logs_dir = Path(args.logs_dir).expanduser()
    else:
        logs_dir = run_root / "logs_tts"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Propagate progress directory to the per-topic TTS subprocesses.
    # syn_ver2_breezy.py will append per-txt progress events under this folder.
    os.environ["SYN_TTS_PROGRESS_DIR"] = str(logs_dir)

    progress_all = logs_dir / "progress_all.tsv"
    progress_rank = logs_dir / f"progress_rank{rank:03d}.tsv"
    progress_txt = logs_dir / "progress_tts_txt.tsv"
    header_line = "ts\trank\tworld\tgpu\ttopic\tstatus\treturncode\tidx\ttotal\tlog\n"
    if not progress_rank.exists():
        _append_progress_line(progress_rank, header_line.strip("\n"))
    if not progress_all.exists():
        _append_progress_line(progress_all, header_line.strip("\n"))

    topics: List[str]
    raw_topics = str(args.topics).strip()
    if raw_topics.lower() in {"auto", "all"}:
        topics = _infer_topics_from_run_root(run_root, config_path)
    else:
        topics = _split_csv(raw_topics)
        if not topics:
            raise ValueError("No topics provided")

    assigned = [t for i, t in enumerate(topics) if (i % world_size) == rank]

    header = (
        f"rank={rank}/{world_size} | assigned_topics={len(assigned)} | "
        f"CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES', '')}"
    )
    print(header, flush=True)
    print(f"Per-txt progress: {progress_txt}", flush=True)

    # Record assignment.
    _append_progress_line(
        progress_rank,
        f"{int(time.time())}\t{rank}\t{world_size}\t{os.getenv('CUDA_VISIBLE_DEVICES', '')}\t<ASSIGNED>\tINFO\t0\t0\t{len(assigned)}\t{','.join(assigned)}",
    )
    _append_progress_line(
        progress_all,
        f"{int(time.time())}\t{rank}\t{world_size}\t{os.getenv('CUDA_VISIBLE_DEVICES', '')}\t<ASSIGNED>\tINFO\t0\t0\t{len(assigned)}\t{','.join(assigned)}",
    )

    repo_root = _script_dir()
    run_topic = repo_root / "run_topic_tts.py"

    failures = 0
    total = len(assigned)
    for idx, topic in enumerate(assigned, start=1):
        log_path = logs_dir / f"rank{rank:03d}_{_slugify_topic(topic)}.log"
        cmd = [
            sys.executable,
            str(run_topic),
            "--config",
            str(config_path),
            "--topic",
            str(topic),
            "--run-root",
            str(run_root),
        ]

        with open(log_path, "w", encoding="utf-8") as f:
            f.write(header + "\n")
            f.write("COMMAND: " + " ".join(cmd) + "\n\n")
            f.flush()
            proc = subprocess.run(cmd, cwd=str(repo_root), stdout=f, stderr=subprocess.STDOUT, text=True)

        if proc.returncode != 0:
            failures += 1
            print(f"[FAILED({proc.returncode})] topic={topic} log={log_path}", flush=True)
            status = "FAILED"
        else:
            print(f"[OK] topic={topic} log={log_path}", flush=True)
            status = "OK"

        ts = int(time.time())
        line = f"{ts}\t{rank}\t{world_size}\t{os.getenv('CUDA_VISIBLE_DEVICES', '')}\t{topic}\t{status}\t{proc.returncode}\t{idx}\t{total}\t{log_path}"
        _append_progress_line(progress_rank, line)
        _append_progress_line(progress_all, line)

    if failures:
        raise SystemExit(failures)


if __name__ == "__main__":
    main()
