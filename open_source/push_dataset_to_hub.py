from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from datasets import load_from_disk
from omegaconf import OmegaConf


def _script_dir() -> Path:
    return Path(__file__).resolve().parent


def _resolve_under(base: Optional[Path], value: Optional[str]) -> Optional[Path]:
    if value is None:
        return None
    p = Path(str(value)).expanduser()
    if p.is_absolute() or base is None:
        return p
    return base / p


def _load_config(config_path: Path) -> Dict[str, Any]:
    cfg = OmegaConf.load(str(config_path))
    resolved = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(resolved, dict):
        raise ValueError(f"Invalid config structure: {config_path}")
    return resolved


def _config_defaults(config_path: Path, run_root: Optional[str]) -> Dict[str, Any]:
    cfg = _load_config(config_path)
    huggingface_cfg = dict(cfg.get("huggingface", {}) or {})

    run_root_path = Path(run_root).expanduser() if run_root else None
    data_root = _resolve_under(run_root_path, cfg.get("data_root", "TEST_syn_data"))
    output_dir = _resolve_under(run_root_path, huggingface_cfg.get("output_dir"))
    if output_dir is None:
        output_dir = (data_root or Path("TEST_syn_data")) / "huggingface_dataset"

    return {
        "output_dir": output_dir,
        "dataset_path": output_dir / "dataset",
        "repo_id": huggingface_cfg.get("hub_repo_id") or "Jaylin0418/synthetic_dialogue_zh",
        "private": bool(huggingface_cfg.get("hub_private", False)),
        "revision": huggingface_cfg.get("hub_revision") or None,
    }


def _parse_topics_arg(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    if raw.lower() in {"auto", "all"}:
        return None
    items = [t.strip() for t in raw.split(",")]
    return [t for t in items if t]


def _iter_topics_with_wavs(data_root: Path, mode_name: str) -> List[str]:
    mode_dir = data_root / mode_name
    if not mode_dir.exists():
        return []
    topics: List[str] = []
    for p in sorted(mode_dir.iterdir()):
        if not p.is_dir() or p.name.startswith("."):
            continue
        wav_dir = p / "wav"
        if not wav_dir.exists() or not wav_dir.is_dir():
            continue
        # Only include topics that have at least one dialogue folder with metadata.
        has_any = False
        try:
            for d in wav_dir.iterdir():
                if not d.is_dir() or d.name.startswith("."):
                    continue
                if (d / "individual" / "turn_metadata.json").exists():
                    has_any = True
                    break
        except Exception:
            has_any = False
        if has_any:
            topics.append(p.name)
    return topics


def _load_cfg_for_topic(config_path: Path, *, run_root: str, topic: str) -> Any:
    """Load config, set run_root/topic, resolve interpolations, and normalize paths under run_root."""
    raw = OmegaConf.load(str(config_path))
    OmegaConf.update(raw, "run_root", str(run_root), merge=False)
    OmegaConf.update(raw, "scenario.topic", str(topic), merge=False)
    # Force-enable HF export so the exporter writes per-turn metadata/audio paths.
    OmegaConf.update(raw, "huggingface.enabled", True, merge=False)

    resolved = OmegaConf.to_container(raw, resolve=True)
    cfg = OmegaConf.create(resolved)

    try:
        from syn_ver2_breezy import PipelineConfig  # type: ignore

        cfg = OmegaConf.merge(OmegaConf.structured(PipelineConfig), cfg)
    except Exception:
        pass

    try:
        from syn_ver2_breezy import prepare_run_output_root  # type: ignore

        prepare_run_output_root(cfg, create_dirs=True)
    except Exception as e:
        raise RuntimeError(f"Failed to prepare run_root paths: {e}")

    return cfg


def _export_multi_topic_dataset(
    *,
    config_path: Path,
    run_root: str,
    topics: Optional[Sequence[str]],
    strict: bool,
    combined_output_dir: Optional[Path],
    keep_by_topic: bool,
) -> Path:
    """Export a single combined dataset for all topics under run_root.

    Uses the existing V2 exporter to ensure the dataset schema/columns remain unchanged.
    """
    try:
        from datasets import concatenate_datasets  # type: ignore
    except Exception as e:
        raise RuntimeError(f"datasets is required for export/concat: {e}")

    # Build a base cfg (topic doesn't matter) just to derive resolved data_root/mode_name.
    base_cfg = _load_cfg_for_topic(config_path, run_root=run_root, topic="_dummy_")
    data_root = Path(str(base_cfg.get("data_root")))
    mode_name = str(base_cfg.get("mode_name", "normal")).strip() or "normal"
    default_output_dir = Path(str(base_cfg.get("huggingface", {}).get("output_dir")))
    output_dir = combined_output_dir if combined_output_dir is not None else default_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_topics: List[str]
    if topics is None:
        selected_topics = _iter_topics_with_wavs(data_root, mode_name)
        if not selected_topics:
            raise FileNotFoundError(f"No topics with wav outputs found under: {data_root / mode_name}")
    else:
        selected_topics = list(topics)

    available_topics = set(_iter_topics_with_wavs(data_root, mode_name))
    missing = [t for t in selected_topics if t not in available_topics]
    if missing and strict:
        raise FileNotFoundError(
            "Missing/empty wav outputs for topics: " + ", ".join(missing) + f" (under {data_root / mode_name})"
        )

    topics_to_export = [t for t in selected_topics if t in available_topics]
    if not topics_to_export:
        raise FileNotFoundError("No exportable topics found after filtering")

    by_topic_root = output_dir / "by_topic"
    by_topic_root.mkdir(parents=True, exist_ok=True)

    per_topic_ds_paths: List[Tuple[str, Path]] = []

    for topic in topics_to_export:
        cfg_topic = _load_cfg_for_topic(config_path, run_root=run_root, topic=topic)
        # Prevent per-topic exporter from pushing; we will push the combined dataset.
        try:
            cfg_topic.huggingface["push_to_hub"] = False
        except Exception:
            pass

        topic_out_dir = by_topic_root / topic
        topic_out_dir.mkdir(parents=True, exist_ok=True)
        try:
            cfg_topic.huggingface["output_dir"] = str(topic_out_dir)
        except Exception:
            pass

        try:
            from syn_ver2_breezy import export_to_huggingface  # type: ignore

            export_to_huggingface(cfg_topic)
        except Exception as e:
            if strict:
                raise RuntimeError(f"Export failed for topic={topic}: {e}")
            print(f"[WARN] Export failed for topic={topic}: {e}", flush=True)
            continue

        ds_path = topic_out_dir / "dataset"
        if not ds_path.exists():
            if strict:
                raise FileNotFoundError(f"Expected dataset not found after export: {ds_path}")
            print(f"[WARN] Dataset missing after export for topic={topic}: {ds_path}", flush=True)
            continue
        per_topic_ds_paths.append((topic, ds_path))

    if not per_topic_ds_paths:
        raise RuntimeError("No per-topic datasets were produced; cannot create combined dataset")

    datasets_list = []
    for topic, ds_path in per_topic_ds_paths:
        try:
            datasets_list.append(load_from_disk(str(ds_path)))
        except Exception as e:
            if strict:
                raise RuntimeError(f"Failed to load per-topic dataset for topic={topic}: {e}")
            print(f"[WARN] Failed to load per-topic dataset for topic={topic}: {e}", flush=True)

    if not datasets_list:
        raise RuntimeError("No per-topic datasets could be loaded")

    combined = concatenate_datasets(datasets_list)

    combined_ds_path = output_dir / "dataset"
    if combined_ds_path.exists():
        shutil.rmtree(combined_ds_path)
    combined.save_to_disk(str(combined_ds_path))

    if not keep_by_topic:
        try:
            shutil.rmtree(by_topic_root)
        except Exception:
            pass

    return combined_ds_path


def parse_args() -> argparse.Namespace:
    default_config = _script_dir() / "conf" / "base_v2_breezy.yaml"
    p = argparse.ArgumentParser(description="Push an on-disk HuggingFace Dataset to the Hub")
    p.add_argument(
        "--config",
        type=str,
        default=str(default_config),
        help="Path to pipeline config YAML used to derive defaults",
    )
    p.add_argument(
        "--run-root",
        type=str,
        default=None,
        help="Optional run root used when the dataset was produced under a specific run_root",
    )
    p.add_argument(
        "--export",
        action="store_true",
        help="Export dataset from a run_root before pushing (recommended after TTS)",
    )
    p.add_argument(
        "--topics",
        type=str,
        default="auto",
        help="Comma-separated topics to export, or 'auto' to scan all topics under {data_root}/{mode_name}",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any requested topic is missing/has no wav outputs",
    )
    p.add_argument(
        "--keep-by-topic",
        action="store_true",
        help="Keep intermediate per-topic exports under output_dir/by_topic (default: delete)",
    )
    p.add_argument(
        "--no-push",
        action="store_true",
        help="Only export/save locally; do not push to hub",
    )
    p.add_argument(
        "--path",
        type=str,
        default=None,
        help="Path to dataset folder produced by load_from_disk(); defaults to config-derived output_dir/dataset",
    )
    p.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Target Hub repo id (e.g. user/name or org/name)",
    )
    p.add_argument(
        "--private",
        action="store_true",
        help="Create/push as a private repo (overrides config)",
    )
    p.add_argument(
        "--public",
        action="store_true",
        help="Force push as a public repo (overrides config)",
    )
    p.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Optional target branch name",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser()
    defaults = _config_defaults(config_path, args.run_root)

    output_dir = Path(defaults["output_dir"]) if defaults.get("output_dir") else None
    path = Path(args.path).expanduser() if args.path else Path(defaults["dataset_path"])
    repo_id = str(args.repo_id or defaults["repo_id"])

    if args.private and args.public:
        raise ValueError("--private and --public cannot be used together")

    if args.private:
        private = True
    elif args.public:
        private = False
    else:
        private = bool(defaults["private"])

    revision = args.revision if args.revision is not None else defaults["revision"]

    if args.export:
        if not args.run_root:
            raise ValueError("--export requires --run-root")
        combined_output_dir = output_dir
        topics = _parse_topics_arg(args.topics)
        keep_by_topic = bool(args.keep_by_topic)

        print("Exporting HuggingFace dataset from run_root...", flush=True)
        path = _export_multi_topic_dataset(
            config_path=config_path,
            run_root=str(args.run_root),
            topics=topics,
            strict=bool(args.strict),
            combined_output_dir=combined_output_dir,
            keep_by_topic=keep_by_topic,
        )

    if not path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {path}")

    print("Loading dataset from:", str(path), flush=True)
    ds = load_from_disk(str(path))
    # Avoid decoding any Audio samples here (which may require extra deps like torchcodec).
    print("Dataset loaded.", flush=True)
    try:
        print("rows:", ds.num_rows, flush=True)
        print("columns:", ds.column_names, flush=True)
    except Exception:
        print(ds, flush=True)

    if args.no_push:
        print("--no-push set; skipping push_to_hub.", flush=True)
        return

    print("Pushing to hub:", repo_id, flush=True)
    push_kwargs = {
        "private": private,
    }
    if revision:
        push_kwargs["revision"] = str(revision)
    url = ds.push_to_hub(repo_id, **push_kwargs)
    print("Pushed:", url, flush=True)


if __name__ == "__main__":
    main()
