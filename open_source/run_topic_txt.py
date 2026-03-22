from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import List, Optional

from omegaconf import OmegaConf

from syn_ver2_breezy import Pipeline, PipelineConfig, prepare_run_output_root


def _script_dir() -> Path:
    return Path(__file__).resolve().parent


def _load_pipeline_cfg(config_path: Path):
    raw_cfg = OmegaConf.load(str(config_path))
    if "hydra" in raw_cfg:
        del raw_cfg["hydra"]
    return OmegaConf.merge(OmegaConf.structured(PipelineConfig), raw_cfg)


def _parse_stages(stages_value) -> List[str]:
    if isinstance(stages_value, (list, tuple)):
        return [str(x).strip() for x in stages_value if str(x).strip()]
    if isinstance(stages_value, str):
        return [s.strip() for s in stages_value.split(",") if s.strip()]
    return []


def _batch_defaults(config_path: Path) -> dict:
    cfg = _load_pipeline_cfg(config_path)
    batch_cfg = cfg.get("batch_run", {}) or {}
    return {
        "topic": str(batch_cfg.get("topic") or cfg.scenario.get("topic") or "Travel"),
        "batch_size": int(batch_cfg.get("batch_size") or cfg.scenario.get("n") or 1),
        "total_count": int(batch_cfg.get("total_count") or cfg.scenario.get("n") or 1),
        "final_huggingface_only": bool(batch_cfg.get("final_huggingface_only", True)),
        "run_root": str(cfg.get("run_root") or "").strip() or None,
    }


def parse_args() -> argparse.Namespace:
    default_config = _script_dir() / "conf" / "base_v2_breezy.yaml"
    defaults = _batch_defaults(default_config)

    p = argparse.ArgumentParser(
        description="Run the synthetic pipeline for one topic in repeated batches. Counts are scenario counts."
    )
    p.add_argument("--config", type=str, default=str(default_config), help="Path to pipeline config YAML")
    p.add_argument("--topic", type=str, default=defaults["topic"], help="Topic/domain to generate, e.g. Travel")
    p.add_argument(
        "--batch-size",
        type=int,
        default=defaults["batch_size"],
        help="How many scenarios to generate per batch",
    )
    p.add_argument(
        "--total-count",
        type=int,
        default=defaults["total_count"],
        help="Total number of scenarios to generate for the topic",
    )
    p.add_argument(
        "--run-root",
        type=str,
        default=defaults["run_root"],
        help="Existing or target run root. Leave empty to auto-create once and reuse it for all batches.",
    )
    p.add_argument(
        "--export-every-batch",
        action="store_true",
        help="If set, keep the `huggingface` stage in every batch instead of only the final batch.",
    )
    p.add_argument(
        "--stages",
        type=str,
        default=None,
        help="Optional comma-separated stage override, e.g. scenario,system_prompt,dialogue,overlap,filler,tts",
    )
    p.add_argument(
        "--with-emotion",
        action="store_true",
        help="Use emotion-tagged prompts (adds (emotion:xxx) prefix to each dialogue turn)",
    )
    return p.parse_args()


def _validate_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")


def _stages_for_batch(base_stages: List[str], is_final_batch: bool, export_every_batch: bool) -> List[str]:
    if is_final_batch or export_every_batch:
        return list(base_stages)
    return [stage for stage in base_stages if stage != "huggingface"]


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    _validate_positive("batch_size", int(args.batch_size))
    _validate_positive("total_count", int(args.total_count))

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    initial_cfg = _load_pipeline_cfg(config_path)
    base_stages = _parse_stages(args.stages) or _parse_stages(initial_cfg.get("stages", []))
    if not base_stages:
        raise ValueError("No stages configured for batch generation")

    export_every_batch = bool(args.export_every_batch)
    if not export_every_batch:
        try:
            export_every_batch = not bool(initial_cfg.get("batch_run", {}).get("final_huggingface_only", True))
        except Exception:
            export_every_batch = False

    topic = str(args.topic).strip()
    if not topic:
        raise ValueError("topic must be non-empty")

    run_root: Optional[str] = str(args.run_root).strip() if args.run_root else None
    total_batches = int(math.ceil(int(args.total_count) / int(args.batch_size)))
    remaining = int(args.total_count)

    print("=" * 80, flush=True)
    print(f"Topic batch generation start: topic={topic}", flush=True)
    print(
        f"Total scenarios={args.total_count}, batch_size={args.batch_size}, batches={total_batches}",
        flush=True,
    )
    print("=" * 80, flush=True)

    for batch_idx in range(1, total_batches + 1):
        current_batch_size = min(int(args.batch_size), remaining)
        is_final_batch = batch_idx == total_batches

        cfg = _load_pipeline_cfg(config_path)
        cfg["run_root"] = run_root
        cfg.dialogue["with_emotion"] = bool(args.with_emotion)
        cfg["topic_folder_suffix"] = "_emo" if args.with_emotion else ""
        cfg.scenario["topic"] = topic
        cfg.batch_run["topic"] = topic
        cfg.batch_run["batch_size"] = int(args.batch_size)
        cfg.batch_run["total_count"] = int(args.total_count)
        cfg.scenario["n"] = current_batch_size
        cfg["stages"] = _stages_for_batch(base_stages, is_final_batch, export_every_batch)

        prepare_run_output_root(cfg)
        run_root = str(cfg.get("run_root"))

        print(
            f"[Batch {batch_idx}/{total_batches}] run_root={run_root} | scenario.n={current_batch_size} | stages={list(cfg.stages)}",
            flush=True,
        )

        Pipeline(cfg).run()
        remaining -= current_batch_size

    print("=" * 80, flush=True)
    print(f"Completed topic={topic} under run_root={run_root}", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
