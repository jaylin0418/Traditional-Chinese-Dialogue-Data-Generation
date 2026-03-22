from __future__ import annotations

import argparse
import logging
from pathlib import Path

from omegaconf import OmegaConf

from syn_ver2_breezy import Pipeline, PipelineConfig, prepare_run_output_root


def _script_dir() -> Path:
    return Path(__file__).resolve().parent


def _load_pipeline_cfg(config_path: Path):
    raw_cfg = OmegaConf.load(str(config_path))
    if "hydra" in raw_cfg:
        del raw_cfg["hydra"]
    return OmegaConf.merge(OmegaConf.structured(PipelineConfig), raw_cfg)


def parse_args() -> argparse.Namespace:
    default_config = _script_dir() / "conf" / "base_v2_breezy.yaml"

    p = argparse.ArgumentParser(description="Run TTS stage for one topic under an existing run_root.")
    p.add_argument("--config", type=str, default=str(default_config), help="Path to pipeline config YAML")
    p.add_argument("--topic", type=str, required=True, help="Topic/domain to run TTS for")
    p.add_argument(
        "--run-root",
        type=str,
        required=True,
        help="Existing session/run root (e.g. /work/.../syn_multi_topic_YYYYMMDD_HHMMSS)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    topic = str(args.topic).strip()
    if not topic:
        raise ValueError("topic must be non-empty")

    run_root = str(args.run_root).strip()
    if not run_root:
        raise ValueError("run_root must be non-empty")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    cfg = _load_pipeline_cfg(config_path)
    cfg["run_root"] = run_root
    cfg.scenario["topic"] = topic

    # Only run TTS stage.
    cfg["stages"] = ["tts"]

    prepare_run_output_root(cfg)
    Pipeline(cfg).run()


if __name__ == "__main__":
    main()
