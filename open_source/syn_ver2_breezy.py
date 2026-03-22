from __future__ import annotations

# ── standard library ─────────────────────────────
import os
import re
import json
import time
import fcntl
from datetime import datetime
import random
import logging
import subprocess
import tempfile
import csv
import platform
from functools import lru_cache
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any

# ── third-party ──────────────────────────────────
import torch
import torchaudio
from tqdm import tqdm
from omegaconf import OmegaConf
import hydra
from dotenv import load_dotenv, dotenv_values
from openai import OpenAI

# NOTE: TTS backends are imported lazily inside the TTS stage.

_DOTENV_PATH = Path(__file__).resolve().parent / ".env"
# Load env vars from this project folder by default (more reliable than relying on cwd).
load_dotenv(dotenv_path=_DOTENV_PATH, override=False)


def _append_tsv_line_locked(path: Path, line: str) -> None:
    """Append a TSV line with a cross-process file lock (best effort)."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        return

    try:
        if platform.system().lower() == "linux":
            import fcntl  # type: ignore

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
            return
    except Exception:
        pass

    # Fallback (no lock)
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line.rstrip("\n") + "\n")
    except Exception:
        return


def _get_openrouter_api_key() -> str:
    """Return OpenRouter API key from environment.

    - Supports both OPENROUTER_API_KEY (preferred) and OPENAI_API_KEY (fallback).
    - Strips whitespace and surrounding quotes to avoid sending an empty/invalid header.
    """
    raw = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
    key = str(raw).strip()
    if len(key) >= 2 and key[0] == key[-1] and key[0] in ("\"", "'"):
        key = key[1:-1].strip()

    # If the environment has an explicitly empty value (common on clusters),
    # fall back to reading the project .env file directly.
    if not key:
        try:
            values = dotenv_values(_DOTENV_PATH)
            raw2 = values.get("OPENROUTER_API_KEY") or values.get("OPENAI_API_KEY") or ""
            key2 = str(raw2).strip()
            if len(key2) >= 2 and key2[0] == key2[-1] and key2[0] in ("\"", "'"):
                key2 = key2[1:-1].strip()
            key = key2
        except Exception:
            pass
    return key


def _sanitize_json_newlines_in_strings(s: str) -> str:
    """Make a best-effort attempt to turn LLM 'almost JSON' into valid JSON.

    OpenRouter/LLM responses sometimes include literal newlines inside JSON string
    values (e.g., within "description": "...\n...") without escaping them.
    That is invalid JSON and breaks json.loads.

    This sanitizer replaces '\n'/'\r' characters that appear *inside* JSON
    strings with spaces, preserving structure.
    """
    if not s:
        return s

    out: List[str] = []
    in_string = False
    escaped = False

    for ch in s:
        if in_string:
            if escaped:
                out.append(ch)
                escaped = False
                continue
            if ch == "\\":
                out.append(ch)
                escaped = True
                continue
            if ch == '"':
                out.append(ch)
                in_string = False
                continue
            if ch in ("\n", "\r"):
                out.append(" ")
                continue
            out.append(ch)
        else:
            if ch == '"':
                out.append(ch)
                in_string = True
                escaped = False
                continue
            out.append(ch)

    return "".join(out)


def _parse_llm_json_object(text: str) -> Dict[str, Any]:
    """Parse a JSON object from LLM output with best-effort cleanup."""
    if text is None:
        raise ValueError("Empty LLM response")

    raw = str(text).strip()
    # Remove common code fences
    raw = re.sub(r"```(?:json)?|```", "", raw, flags=re.I).strip()

    # Extract the outermost JSON object region if the model wrapped extra text
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        raw_obj = raw[start : end + 1]
    else:
        raw_obj = raw

    try:
        return json.loads(raw_obj)
    except json.JSONDecodeError:
        cleaned = _sanitize_json_newlines_in_strings(raw_obj)
        return json.loads(cleaned)


class ScenarioSchemaError(ValueError):
    pass


def _scenario_index(sid: str) -> Optional[int]:
    m = re.match(r"^scenario(\d+)$", str(sid).strip())
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _normalize_scenarios_json(parsed: Any) -> Dict[str, Any]:
    """Normalize common LLM scenario JSON shape issues.

    Expected canonical shape:
      {
        "scenarios": [
          {"scenario1": {"description": "..."}},
          ...
        ]
      }

    Handles common mistakes such as:
      - Returning {"scenario1": {...}, "scenario2": {...}} without wrapping list.
      - Nested repetition: {"scenario2": {"scenario2": {"description": ...}}}
      - Returning scenario values as strings directly.
    """
    if not isinstance(parsed, dict):
        raise ScenarioSchemaError(f"Scenario payload must be a JSON object, got: {type(parsed).__name__}")

    scenarios_raw: Any = None
    if isinstance(parsed.get("scenarios"), list):
        scenarios_raw = parsed["scenarios"]
    else:
        # Sometimes the model returns a top-level dict of scenario keys.
        scenario_keys = [k for k in parsed.keys() if str(k).strip().lower().startswith("scenario")]
        if scenario_keys:
            scenarios_raw = [{k: parsed[k]} for k in scenario_keys]

    if not isinstance(scenarios_raw, list):
        raise ScenarioSchemaError("Missing `scenarios` list")

    normalized: List[Dict[str, Dict[str, str]]] = []
    for entry in scenarios_raw:
        if isinstance(entry, str):
            # No scenario id; skip (we prefer retry over guessing ids).
            continue
        if not isinstance(entry, dict) or not entry:
            continue

        scenario_key: Optional[str] = None
        scenario_val: Any = None

        if len(entry) == 1:
            scenario_key, scenario_val = next(iter(entry.items()))
        else:
            # Try to find a scenario-like key among multiple keys.
            for k in entry.keys():
                if str(k).strip().lower().startswith("scenario"):
                    scenario_key = k
                    scenario_val = entry.get(k)
                    break
            if scenario_key is None:
                continue

        sid = _canonicalize_scenario_id(scenario_key)

        # Fix nested repetition: {"scenario2": {"scenario2": {...}}}
        if isinstance(scenario_val, dict) and sid in scenario_val and isinstance(scenario_val.get(sid), dict):
            scenario_val = scenario_val[sid]
        elif isinstance(scenario_val, dict) and str(scenario_key) in scenario_val and isinstance(scenario_val.get(str(scenario_key)), dict):
            scenario_val = scenario_val[str(scenario_key)]

        desc: Optional[str] = None
        if isinstance(scenario_val, dict):
            d = scenario_val.get("description")
            if isinstance(d, str) and d.strip():
                desc = d.strip()
        elif isinstance(scenario_val, str) and scenario_val.strip():
            desc = scenario_val.strip()

        if not desc:
            continue

        normalized.append({sid: {"description": desc}})

    # Sort by scenario index if possible for determinism.
    normalized.sort(key=lambda x: (_scenario_index(next(iter(x.keys()))) is None, _scenario_index(next(iter(x.keys()))) or 10**9))
    return {"scenarios": normalized}


def _validate_scenarios_json(data: Any, want_n: int) -> None:
    if not isinstance(data, dict) or not isinstance(data.get("scenarios"), list):
        raise ScenarioSchemaError("Scenario JSON must be an object with a `scenarios` list")
    scenarios = data["scenarios"]
    if want_n > 0 and len(scenarios) < want_n:
        raise ScenarioSchemaError(f"Need at least {want_n} scenarios, got {len(scenarios)}")

    seen_ids: set[str] = set()
    for entry in scenarios:
        if not isinstance(entry, dict) or len(entry) != 1:
            raise ScenarioSchemaError("Each scenarios[] item must be a single-key object")
        sid, inner = next(iter(entry.items()))
        sid_norm = _canonicalize_scenario_id(sid)
        if sid_norm != sid:
            raise ScenarioSchemaError(f"Scenario id not canonical: {sid} (expected {sid_norm})")
        if sid_norm in seen_ids:
            raise ScenarioSchemaError(f"Duplicate scenario id: {sid_norm}")
        seen_ids.add(sid_norm)
        if not isinstance(inner, dict):
            raise ScenarioSchemaError(f"Scenario {sid_norm} value must be an object")
        desc = inner.get("description")
        if not isinstance(desc, str) or not desc.strip():
            raise ScenarioSchemaError(f"Scenario {sid_norm} missing non-empty description")


def _extract_topic_label(topic: str) -> str:
    """Extract a compact topic label.

    Para pipeline typically uses topics like "規劃（Planning）" but writes folders/datasets
    using the label inside parentheses.
    """
    if not topic:
        return "unknown"
    s = str(topic).strip()
    m = re.search(r"（([^）]+)）", s)
    return (m.group(1).strip() if m else s)


def _canonicalize_scenario_id(value: Any) -> str:
    """Normalize scenario ids to compact form like `scenario3`."""
    s = str(value).strip()
    m = re.match(r"^(?:(.+?)_)?scenario(\d+)$", s)
    if m:
        return f"scenario{int(m.group(2))}"
    m = re.match(r"^scenario(\d+)$", s)
    if m:
        return f"scenario{int(m.group(1))}"
    if re.fullmatch(r"\d+", s):
        return f"scenario{int(s)}"
    return s


def _topic_scenario_id(topic_label: str, scenario_id: str) -> str:
    return _scenario_file_prefix(topic_label, scenario_id)


def _scenario_file_prefix(topic_label: str, scenario_id: str) -> str:
    """Return a filename prefix like `Travel_scenario3`."""
    topic = _extract_topic_label(str(topic_label or "unknown"))
    sid = _canonicalize_scenario_id(scenario_id)
    return f"{topic}_{sid}"


def _stage_txt_root(cfg) -> Path:
    """Root folder to archive per-stage text outputs.

    Layout:
      {data_root}/{mode_name}/{topic}{topic_folder_suffix}/txt/<stage>/*.txt
    """
    data_root = Path(cfg.get("data_root", "data_v2"))
    mode_name = str(cfg.get("mode_name", "v2")).strip() or "v2"
    topic = _extract_topic_label(str(cfg.scenario.get("topic", "unknown")))
    suffix = str(cfg.get("topic_folder_suffix", "") or "")
    return data_root / mode_name / (topic + suffix) / "txt"

def _topic_root(cfg, topic_label: Optional[str] = None) -> Path:
    data_root = Path(cfg.get("data_root", "data_v2"))
    mode_name = str(cfg.get("mode_name", "v2")).strip() or "v2"
    topic = _extract_topic_label(str(topic_label or cfg.scenario.get("topic", "unknown")))
    suffix = str(cfg.get("topic_folder_suffix", "") or "")
    return data_root / mode_name / (topic + suffix)

def _full_progress_path(cfg) -> Path:
    run_root = str(cfg.get("run_root", "") or "").strip()
    if run_root:
        return Path(run_root) / "full_progress.json"
    return Path(cfg.get("data_root", "data_v2")) / "full_progress.json"

def _topic_progress_path(cfg, topic_label: Optional[str] = None) -> Path:
    return _topic_root(cfg, topic_label) / "topic_progress.json"

def _load_json_dict(path: Path) -> Dict[str, Any]:
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}

def _write_json_dict(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def _with_file_lock(lock_path: Path, fn):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            return fn()
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

def _extract_scenario_index_from_name(name: str) -> Optional[int]:
    # Accept both bare ids and filenames with suffixes/extensions, e.g.
    #   Travel_scenario12
    #   Travel_scenario12.txt
    #   scenario12_3
    m = re.search(r"(?:^|_)scenario(\d+)(?:\D|$)", str(name))
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None

def _existing_topic_scenario_max(cfg, topic_label: Optional[str] = None) -> int:
    topic_dir = _topic_root(cfg, topic_label)
    max_idx = 0
    try:
        progress = _load_json_dict(_topic_progress_path(cfg, topic_label))
        for item in progress.get("items", []) or []:
            if not isinstance(item, dict):
                continue
            idx = _extract_scenario_index_from_name(item.get("scenario_id") or item.get("id") or "")
            if idx:
                max_idx = max(max_idx, idx)
    except Exception:
        pass

    try:
        if topic_dir.exists():
            for p in topic_dir.rglob("*"):
                idx = _extract_scenario_index_from_name(p.name)
                if idx:
                    max_idx = max(max_idx, idx)
    except Exception:
        pass
    return max_idx

def _next_topic_scenario_index(cfg, topic_label: Optional[str] = None) -> int:
    return _existing_topic_scenario_max(cfg, topic_label) + 1


def _normalize_progress_item(topic: str, item: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize legacy progress schema.

    Ensures `scenario_id` is topic-prefixed (e.g. `Travel_scenario1`) while
    keeping/deriving `base_scenario_id` (e.g. `scenario1`).
    """
    if not isinstance(item, dict):
        return item

    topic_label = _extract_topic_label(str(topic or "unknown"))
    out = dict(item)

    raw_scenario = str(out.get("scenario_id") or out.get("id") or "").strip()
    base = _canonicalize_scenario_id(raw_scenario)

    idx = _extract_scenario_index_from_name(raw_scenario) or _extract_scenario_index_from_name(base)
    if idx is None:
        return out

    base_scenario_id = f"scenario{idx}"
    topic_scenario_id = _topic_scenario_id(topic_label, base_scenario_id)

    out["scenario_id"] = topic_scenario_id

    raw_id = str(out.get("id") or "").strip()
    # If `id` is missing or unprefixed/ambiguous, align it with `scenario_id`.
    if not raw_id or _canonicalize_scenario_id(raw_id) == base_scenario_id:
        out["id"] = topic_scenario_id

    if not out.get("base_scenario_id"):
        out["base_scenario_id"] = base_scenario_id

    if out.get("topic") is None:
        out["topic"] = topic_label

    return out


def _normalize_progress_items(topic: str, items: List[Any]) -> List[Any]:
    normalized: List[Any] = []
    for item in items or []:
        if isinstance(item, dict):
            normalized.append(_normalize_progress_item(topic, item))
        else:
            normalized.append(item)
    return normalized

def _append_progress_records(cfg, topic_label: str, records: List[Dict[str, Any]]) -> None:
    if not records:
        return

    now = datetime.now().isoformat(timespec="seconds")
    topic = _extract_topic_label(str(topic_label or "unknown"))
    topic_path = _topic_progress_path(cfg, topic)
    full_path = _full_progress_path(cfg)

    topic_progress = _load_json_dict(topic_path)
    topic_items = topic_progress.get("items")
    if not isinstance(topic_items, list):
        topic_items = []

    # Migrate/normalize any legacy items so we don't keep writing unprefixed ids.
    topic_items = _normalize_progress_items(topic, topic_items)

    topic_item_by_id: Dict[str, Dict[str, Any]] = {
        str(item.get("scenario_id")): item for item in topic_items if isinstance(item, dict)
    }
    for rec in records:
        rec_norm = _normalize_progress_item(topic, rec)
        sid = str(rec_norm.get("scenario_id"))
        if sid not in topic_item_by_id:
            topic_items.append(rec_norm)
            topic_item_by_id[sid] = rec_norm
            continue

        # Repair: if existing item has empty/missing description, fill it from new record.
        existing = topic_item_by_id[sid]
        existing_desc = str(existing.get("description") or "").strip()
        new_desc = str(rec_norm.get("description") or "").strip()
        if not existing_desc and new_desc:
            existing["description"] = new_desc

    topic_progress = {
        "topic": topic,
        "updated_at": now,
        "count": len(topic_items),
        "next_scenario_index": _next_topic_scenario_index(cfg, topic),
        "items": topic_items,
    }
    _write_json_dict(topic_path, topic_progress)

    def _update_full_progress() -> None:
        full_progress = _load_json_dict(full_path)
        full_items = full_progress.get("items")
        if not isinstance(full_items, list):
            full_items = []

        # Normalize legacy items in the shared full progress.
        normalized_full_items: List[Any] = []
        for item in full_items:
            if isinstance(item, dict):
                t = _extract_topic_label(str(item.get("topic") or "unknown"))
                normalized_full_items.append(_normalize_progress_item(t, item))
            else:
                normalized_full_items.append(item)
        full_items = normalized_full_items

        full_item_by_key: Dict[Tuple[str, str], Dict[str, Any]] = {
            (str(item.get("topic")), str(item.get("scenario_id"))): item
            for item in full_items
            if isinstance(item, dict)
        }
        for rec in records:
            rec_norm = _normalize_progress_item(topic, rec)
            key = (str(rec_norm.get("topic")), str(rec_norm.get("scenario_id")))
            if key not in full_item_by_key:
                full_items.append(rec_norm)
                full_item_by_key[key] = rec_norm
                continue

            existing = full_item_by_key[key]
            existing_desc = str(existing.get("description") or "").strip()
            new_desc = str(rec_norm.get("description") or "").strip()
            if not existing_desc and new_desc:
                existing["description"] = new_desc

        topics_meta = full_progress.get("topics")
        if not isinstance(topics_meta, dict):
            topics_meta = {}
        topics_meta[topic] = {
            "count": len(topic_items),
            "next_scenario_index": topic_progress["next_scenario_index"],
            "topic_progress_path": str(topic_path),
            "updated_at": now,
        }

        updated = {
            "updated_at": now,
            "count": len(full_items),
            "topics": topics_meta,
            "items": full_items,
        }
        _write_json_dict(full_path, updated)

    _with_file_lock(full_path.with_suffix(full_path.suffix + ".lock"), _update_full_progress)


def _archive_stage_txt(cfg, stage: str, src_path: Path) -> Optional[Path]:
    """Copy a stage txt into the archive tree under data_root.

    Returns the archived path when successful.
    """
    try:
        src = Path(src_path)
        if not src.exists():
            return None
        dst_dir = _stage_txt_root(cfg) / stage
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / src.name
        try:
            if src.resolve() == dst.resolve():
                return dst
        except Exception:
            pass
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        return dst
    except Exception as e:
        logging.warning(f"Failed to archive {stage} txt {src_path}: {e}")
        return None


def _current_scenario_ids(cfg) -> List[str]:
    """Return scenario ids (e.g. ["scenario1", ...]) for the current run.

    Prefer reading the scenario JSONL produced by the scenario stage so we don't
    accidentally reprocess old txt files.
    """
    try:
        scenario_jsonl = Path(cfg.scenario["out_file"]).with_suffix(".jsonl")
        if scenario_jsonl.exists():
            ids: List[str] = []
            for line in scenario_jsonl.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                obj = json.loads(line)
                sid = obj.get("id")
                if isinstance(sid, str) and sid:
                    ids.append(_canonicalize_scenario_id(sid))
            if ids:
                return ids
    except Exception:
        pass

    # Fallback: infer from cfg.scenario.n
    try:
        n = int(cfg.scenario.get("n", 1))
    except Exception:
        n = 1
    return [f"scenario{i}" for i in range(1, max(n, 1) + 1)]


def _filter_files_by_current_scenarios(cfg, files: List[Path]) -> List[Path]:
    allowed = set(_current_scenario_ids(cfg))
    out: List[Path] = []
    for p in files:
        try:
            parsed = _parse_dialogue_txt_stem(p.stem)
            if parsed:
                scenario_id = str(parsed["scenario_id"])
            else:
                stem = p.stem
                scenario_id = _canonicalize_scenario_id(stem.split("_", 1)[-1])
            if scenario_id in allowed:
                out.append(p)
        except Exception:
            continue
    return out


def _dialogue_file_stem(
    scenario_id: str,
    dialogue_number: int,
    dialogues_per_scenario: int,
    *,
    topic_label: str,
) -> str:
    """Build the txt stem for a generated dialogue.

    When only one dialogue is generated per scenario, keep filenames compact as:
      Travel_scenario1.txt
    Otherwise preserve the indexed form:
      Travel_scenario1_1.txt
      Travel_scenario1_2.txt
    """
    base = _scenario_file_prefix(topic_label, scenario_id)
    try:
        per_scenario = int(dialogues_per_scenario or 1)
    except Exception:
        per_scenario = 1
    if per_scenario <= 1:
        return base
    return f"{base}_{int(dialogue_number)}"


def _parse_dialogue_txt_stem(stem: str) -> Optional[Dict[str, Optional[str]]]:
    """Parse dialogue txt stem supporting both bare and indexed names.

    Accepted forms:
      Travel_scenario1
      Travel_scenario1_1
      scenario1
      scenario1_1
      scenario000001
      scenario000001_1
    """
    m = re.match(r"^(?:(.+?)_)?(scenario(\d+))(?:_(\d+))?$", str(stem).strip())
    if not m:
        return None
    return {
        "topic_prefix": m.group(1),
        "scenario_id": _canonicalize_scenario_id(m.group(2)),
        "scenario_idx": str(int(m.group(3))),
        "dialogue_idx": m.group(4),
    }


def _auto_detect_breezy_python(breezy_python: str) -> str:
    """Resolve a workable Python interpreter for BreezyVoice.

    BreezyVoice commonly requires Python 3.10 due to wheels like `ttsfrd`.
    This project typically installs it into a conda env named `breezyvoice_py310`.

    If user explicitly provides a path, keep it.
    If left as 'python'/'python3', try common conda env locations.
    """

    if breezy_python and breezy_python not in ("python", "python3"):
        return breezy_python

    home = Path.home()
    env_names = ["breezyvoice_py310", "breezyvoice"]
    prefixes = [
        home / "miniconda3" / "envs",
        home / "anaconda3" / "envs",
        home / "mambaforge" / "envs",
        home / "micromamba" / "envs",
    ]

    for prefix in prefixes:
        for env in env_names:
            cand = prefix / env / "bin" / "python"
            if cand.exists():
                return str(cand)

    return breezy_python or "python"


# ──────────────────────────────  Global / Helper  ────────────────────────────────
def set_global_seed(seed: Optional[int]):
    """Make runs more reproducible (LLM is still stochastic, but local randomness becomes stable)."""
    if seed is None:
        return
    random.seed(seed)
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def validate_openrouter_model_name(name: str, where: str):
    """Fail-fast if model name isn't OpenRouter format provider/model."""
    if not name or "/" not in name:
        raise ValueError(
            f"[{where}] model 名稱必須是 OpenRouter 格式 'provider/model'，你給的是：{name!r}"
        )


def _try_get_reference_sentence(
    speaker_meta_by_path: Optional[Dict[str, Any]], audio_path: str
) -> Optional[str]:
    if not speaker_meta_by_path:
        return None
    try:
        meta = speaker_meta_by_path.get(str(audio_path), None)
        if not isinstance(meta, dict):
            return None
        s = meta.get("sentence")
        if isinstance(s, str) and s.strip():
            return s.strip()
    except Exception:
        return None
    return None


def get_gen_kwargs(cfg_section) -> Dict:
    """
    Read stage generation params from yaml.
    Expect:
      gen:
        max_tokens: ...
        temperature: ...
        top_p: ...
        ...
    """
    if cfg_section is None:
        return {}
    try:
        g = cfg_section.get("gen", None)
        return dict(g) if g else {}
    except Exception:
        return {}


# ──────────────────────────────  Normal UTILS  ────────────────────────────────
def convert_nested_json_to_jsonl(
    input_path,
    output_path,
    scenario_txt_dir: Optional[Path] = None,
    topic_label: Optional[str] = None,
    start_index: int = 1,
    cfg=None,
):
    # 1. Read the entire JSON object
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    scenarios = data.get("scenarios", [])
    if not scenarios:
        raise ValueError("No 'scenarios' key or empty list in the input JSON.")

    scenario_txt_dir_path = None
    if scenario_txt_dir is not None:
        scenario_txt_dir_path = Path(scenario_txt_dir)
        scenario_txt_dir_path.mkdir(parents=True, exist_ok=True)

    progress_records: List[Dict[str, Any]] = []

    with open(output_path, "w", encoding="utf-8") as f_out:
        for offset, entry in enumerate(scenarios):
            # Each entry looks like {"scenario1": {...}}; get the value, but
            # re-number the output id from the persisted topic progress.
            _scenario_key, inner = next(iter(entry.items()))
            scenario_id = _canonicalize_scenario_id(f"scenario{int(start_index) + offset}")
            scenario_record_id = (
                _topic_scenario_id(str(topic_label), scenario_id) if topic_label else scenario_id
            )

            # Some LLM outputs incorrectly nest scenario keys, e.g.
            #   {"scenario2": {"scenario2": {"description": "..."}}}
            # Extract the real description defensively.
            description: str = ""
            if isinstance(inner, dict):
                d = inner.get("description")
                if isinstance(d, str) and d.strip():
                    description = d.strip()
                else:
                    sid = _canonicalize_scenario_id(_scenario_key)
                    nested = None
                    if sid in inner and isinstance(inner.get(sid), dict):
                        nested = inner.get(sid)
                    elif str(_scenario_key) in inner and isinstance(inner.get(str(_scenario_key)), dict):
                        nested = inner.get(str(_scenario_key))
                    if isinstance(nested, dict):
                        d2 = nested.get("description")
                        if isinstance(d2, str) and d2.strip():
                            description = d2.strip()

            if not description:
                raise ValueError(
                    f"Scenario description is empty/missing while converting to JSONL: {output_path}"
                )
            # Write a flat dict
            out_obj = {"id": scenario_record_id, "description": description}
            f_out.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

            if scenario_txt_dir_path is not None:
                file_stem = (
                    _scenario_file_prefix(str(topic_label), scenario_id)
                    if topic_label
                    else scenario_id
                )
                (scenario_txt_dir_path / f"{file_stem}.txt").write_text(description, encoding="utf-8")

            if cfg is not None and topic_label:
                progress_records.append(
                    {
                        "topic": str(topic_label),
                        "id": scenario_record_id,
                        "scenario_id": scenario_record_id,
                        "base_scenario_id": scenario_id,
                        "description": description,
                    }
                )

    if cfg is not None and topic_label:
        _append_progress_records(cfg, str(topic_label), progress_records)

    print(f"Converted {len(scenarios)} scenarios to JSONL → {output_path}")


def format_headers_in_lines(text: str) -> str:
    """Repair LLM outputs where a header token is on its own line.

    We only join when a line is *exactly* a bracketed header like:
        [overlap]\nUser: ...
    to become:
        [overlap] User: ...

    Important: do NOT remove newlines after bracket tokens that are part of
    a longer line (e.g. the trailing '[pause]' in '[pause] User: [pause]').
    """

    return re.sub(r"(?m)^(\[[^\]]+\])\s*\n\s*", r"\1 ", text)


def merge_overlapping_user_lines(lines):
    merged_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # Check if this is a User line and the next line is an [overlap] User line
        if line.startswith("User:") and i + 1 < len(lines) and lines[i + 1].startswith(
            "[overlap] User:"
        ):
            # Remove the label from the overlap line and merge
            merged_text = (
                line.rstrip()
                + " "
                + lines[i + 1].replace("[overlap] User:", "", 1).lstrip()
            )
            merged_lines.append("User:" + merged_text[len("User:") :])
            i += 2  # Skip the next line as it's merged
        else:
            merged_lines.append(line)
            i += 1
    return merged_lines


PAUSE_MARKER_CANONICAL = "[pause] User: [pause]"
PAT_PAUSE_MARKER = re.compile(r"\[pause\]\s*User:\s*\[pause\]", flags=re.IGNORECASE)


PAT_EMOTION_PREFIX = re.compile(
    r"^\s*\(\s*emotion\s*:\s*[^)]+\)\s*", flags=re.IGNORECASE
)


def strip_leading_emotion_tag(text: str) -> str:
    """Remove a leading '(emotion:xxx)' tag if present."""
    if not text:
        return text
    return PAT_EMOTION_PREFIX.sub("", text).strip()


def _extract_leading_emotion_tag(text: str) -> Optional[str]:
    """Return the emotion name from a leading '(emotion:xxx)' tag, or None."""
    if not text:
        return None
    m = re.match(r"^\s*\(\s*emotion\s*:\s*([^)]+)\)\s*", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip().lower()
    return None


def strip_emotion_tags_from_dialogue_text(dialogue_text: str) -> str:
    """Strip leading emotion tags from every turn line.

    Handles lines like:
      - User: (emotion:xxx) ...
      - Agent: (emotion:xxx) ...
      - [overlap] User: (emotion:xxx) ...
      - [backchannel] Agent: (emotion:xxx) ...
    """
    if not dialogue_text:
        return dialogue_text
    out_lines: List[str] = []
    for raw in dialogue_text.splitlines():
        line = raw.rstrip("\n")
        if not line.strip():
            continue
        if ":" not in line:
            out_lines.append(line.strip())
            continue
        role, content = line.split(":", 1)
        cleaned = strip_leading_emotion_tag(content.strip())
        out_lines.append(f"{role.strip()}: {cleaned}" if cleaned else f"{role.strip()}:")
    return "\n".join(out_lines)


_PAT_SPACED_LETTERS = re.compile(r"\b(?:[A-Za-z]\s+){2,}[A-Za-z]\b")


def sanitize_tts_text(text: str) -> str:
    """Light sanitization for TTS stability.

    - Collapse spaced-letter abbreviations like 'A P P' -> 'APP'
    - Keep BreezyVoice pronunciation annotations like '[:ㄧㄠ4]' intact (we do not strip them).
    """
    if not text:
        return text

    s = str(text)

    def _collapse(m: re.Match) -> str:
        return "".join(m.group(0).split())

    s = _PAT_SPACED_LETTERS.sub(_collapse, s)
    # Strip trailing bare/full emotion tags like "(curiosity)" or "(emotion:happy)"
    s = re.sub(r"\s*\(\s*(?:emotion\s*[:=]\s*)?[A-Za-z_\-]+\s*\)\s*$", "", s)
    return s.strip()


def normalize_emotion_tag_prefix(text: str, emotion_pool: List[str]) -> str:
    """Normalize various emotion-tag variants into canonical '(emotion:xxx) ...'.

    Observed bad variants from LLM outputs:
      - **(frustration) 句子...          ← bare name, leading
      - (frustration) 句子...            ← bare name, leading
      - **(emotion:frustration) 句子...  ← correct but with markdown noise
      - (emotion: frustration) 句子...   ← space after colon
      - /emotion:happy/ 句子...          ← wrong bracket type
      - 句子... (anxiety)                ← tag at end, bare name
      - 句子... (emotion:anxiety)        ← tag at end, full form

    If we can confidently map the tag to the provided emotion pool, we rewrite
    it into canonical '(emotion:xxx) rest' format. Otherwise we return the
    original text unchanged.
    """

    if not text:
        return text

    pool = {str(e).strip() for e in (emotion_pool or []) if str(e).strip()}
    s = text.strip()

    # Strip leading markdown emphasis markers that frequently leak into outputs.
    s = re.sub(r"^\*+\s*", "", s)

    def _canonical(emo: str, rest: str) -> str:
        rest = rest.strip()
        return f"(emotion:{emo})" + (f" {rest}" if rest else "")

    def _match_inner(inner: str):
        """Return emotion name if inner matches a known variant, else None."""
        inner = inner.strip()
        # Full form: emotion:xxx or emotion=xxx
        m = re.match(r"^emotion\s*[:=]\s*([A-Za-z_\-]+)\s*$", inner, flags=re.IGNORECASE)
        if m:
            emo = m.group(1)
            if not pool or emo in pool:
                return emo
        # Bare emotion name
        if inner in pool:
            return inner
        return None

    # ── 1. Leading /emotion:xxx/ or /xxx/ (wrong bracket type) ───────────────
    m_slash = re.match(r"^/([^/]+)/\s*(.*)", s, flags=re.DOTALL)
    if m_slash:
        emo = _match_inner(m_slash.group(1))
        if emo:
            return _canonical(emo, m_slash.group(2))

    # ── 2. Leading (…) ───────────────────────────────────────────────────────
    if s.startswith("("):
        close = s.find(")")
        if close != -1:
            inner = s[1:close]
            rest = s[close + 1:].lstrip()
            rest = re.sub(r"^\*+\s*", "", rest)
            emo = _match_inner(inner)
            if emo:
                return _canonical(emo, rest)

    # ── 3. Leading emotion:xxx (no brackets) ─────────────────────────────────
    m_bare = re.match(r"^emotion\s*[:=]\s*([A-Za-z_\-]+)\s*(.*)", s, flags=re.IGNORECASE | re.DOTALL)
    if m_bare:
        emo = m_bare.group(1)
        if not pool or emo in pool:
            return _canonical(emo, m_bare.group(2))

    # ── 4. Trailing tag: 句子 (emotion:xxx) or 句子 (xxx) ────────────────────
    # Match a tag-like token at the very end of the string.
    m_trail = re.search(r"\s*[\(\[/]([^\)\]/]+)[\)\]/]\s*$", s)
    if m_trail:
        emo = _match_inner(m_trail.group(1))
        if emo:
            body = s[: m_trail.start()].strip()
            return _canonical(emo, body)

    return text


def split_pause_marker_to_own_line(lines: List[str]) -> List[str]:
    """Ensure the pause marker is always a standalone line.

    Some LLM outputs may incorrectly produce e.g.:
      [pause] User: [pause] User: (emotion:xxx) ...
    This will be normalized to:
      [pause] User: [pause]
      User: (emotion:xxx) ...

    If the pause marker appears mid-line, we split into: before / marker / after.
    """

    fixed: List[str] = []
    for line in lines:
        raw = line.strip("\n")
        if not raw.strip():
            continue

        s = raw.strip()
        m_full = PAT_PAUSE_MARKER.fullmatch(s)
        if m_full:
            fixed.append(PAUSE_MARKER_CANONICAL)
            continue

        if not PAT_PAUSE_MARKER.search(s):
            fixed.append(raw.rstrip())
            continue

        idx = 0
        for m in PAT_PAUSE_MARKER.finditer(s):
            before = s[idx : m.start()]
            if before.strip():
                fixed.append(before.strip())
            fixed.append(PAUSE_MARKER_CANONICAL)
            idx = m.end()

        after = s[idx:]
        if after.strip():
            fixed.append(after.strip())

    return fixed


# ═══════════════════════════════════════════════════════════════════════════
# SPEAKER REFERENCE RESOLUTION - File/Directory/Fallback
# ═══════════════════════════════════════════════════════════════════════════
def resolve_speaker_reference(
    ref: Optional[str],
    *,
    fallback_pool: List[Path],
    role_label: str,
) -> str:
    """
    Resolve a speaker reference input into a concrete audio file path.
    """
    # Normalize falsy values (None, "", etc.) to trigger fallback.
    if not ref:
        if not fallback_pool:
            raise ValueError(
                f"{role_label}: fallback speaker pool is empty; cannot select a reference."
            )
        return str(random.choice(fallback_pool))

    p = Path(ref)

    if p.is_file():
        return str(p)

    if p.is_dir():
        # Support common audio formats for reference prompts.
        candidates = (
            list(p.glob("*.wav"))
            + list(p.glob("*.mp3"))
            + list(p.glob("*.flac"))
            + list(p.glob("*.m4a"))
        )
        if not candidates:
            raise ValueError(
                f"{role_label}: reference directory contains no supported audio files: {p}"
            )
        return str(random.choice(candidates))

    raise ValueError(f"{role_label}: reference path does not exist: {p}")


# ═══════════════════════════════════════════════════════════════════════════
# COMMON VOICE REFERENCE POOL (validated.tsv)
# ═══════════════════════════════════════════════════════════════════════════
def _normalize_cv_gender(gender_raw: Optional[str]) -> Optional[str]:
    if gender_raw is None:
        return None
    s = str(gender_raw).strip().lower()
    if not s:
        return None
    if s.startswith("male") or s in {"m", "man", "masculine"}:
        return "male"
    if s.startswith("female") or s in {"f", "woman", "feminine"}:
        return "female"
    return "other"


@dataclass(frozen=True)
class CVReferenceCandidate:
    audio_path: str
    reference_id: str
    gender_norm: Optional[str]
    metadata: Dict[str, Any]


@lru_cache(maxsize=8)
def _load_common_voice_validated_candidates(
    *, tsv_path: str, clips_dir: str
) -> Tuple[CVReferenceCandidate, ...]:
    """Load Common Voice zh-TW validated.tsv candidates.

    We keep absolute audio_path for local generation, but store a stable
    reference_id (typically the `path` field, e.g. common_voice_...mp3) for
    dataset metadata.
    """

    tsv = Path(tsv_path).expanduser().resolve()
    clips = Path(clips_dir).expanduser().resolve()

    if not tsv.exists():
        raise FileNotFoundError(f"Common Voice validated.tsv not found: {tsv}")
    if not clips.exists():
        raise FileNotFoundError(f"Common Voice clips dir not found: {clips}")

    wanted_keys = [
        "client_id",
        "path",
        "sentence_id",
        "sentence",
        "sentence_domain",
        "up_votes",
        "down_votes",
        "age",
        "gender",
        "accents",
        "variant",
        "locale",
        "segment",
    ]

    candidates: List[CVReferenceCandidate] = []
    with tsv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if not reader.fieldnames:
            raise ValueError(f"validated.tsv appears to have no header: {tsv}")

        for row in reader:
            rel = (row.get("path") or "").strip()
            if not rel:
                continue

            audio_file = clips / rel
            if not audio_file.exists():
                continue

            meta: Dict[str, Any] = {k: (row.get(k) or None) for k in wanted_keys}
            gender_norm = _normalize_cv_gender(meta.get("gender"))
            meta["gender_norm"] = gender_norm
            meta["source"] = "common_voice_validated"

            reference_id = str(meta.get("path") or rel)
            candidates.append(
                CVReferenceCandidate(
                    audio_path=str(audio_file.resolve()),
                    reference_id=reference_id,
                    gender_norm=gender_norm,
                    metadata=meta,
                )
            )

    if not candidates:
        raise ValueError(
            f"No usable Common Voice clips found from: {tsv} (clips={clips})"
        )

    return tuple(candidates)


def _pick_cv_reference(
    candidates: List[CVReferenceCandidate], *, gender_filter: Optional[str] = None
) -> CVReferenceCandidate:
    """Pick one reference candidate, optionally filtering by gender.

    gender_filter supports: None/'any'/'male'/'female'.
    """
    gf = (str(gender_filter).strip().lower() if gender_filter is not None else "")
    if gf and gf not in {"any", "none", "null"}:
        filtered = [c for c in candidates if c.gender_norm == gf]
        if filtered:
            return random.choice(filtered)
    return random.choice(candidates)


# ═══════════════════════════════════════════════════════════════════════════
# ELEVEN LAB EMOTION REFERENCE POOL
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class EmotionRefPool:
    """Lookup tables built from {emotion}_{speaker}_{num}.wav files."""
    # (emotion_name, speaker_name) -> list of wav paths
    emotion_speaker: Dict[Tuple[str, str], List[Path]]
    # speaker_name -> list of wav paths (no-emotion fallback)
    speaker: Dict[str, List[Path]]
    # all unique speaker names
    speakers: List[str]


def _load_emotion_reference_pool(
    emotion_dir: str | Path, known_emotions: List[str]
) -> EmotionRefPool:
    """Scan emotion_dir for wav files and build lookup tables.

    Expected filenames:
      - With emotion : {emotion_name}_{speaker_name}_{num}.wav
      - Without emotion: {speaker_name}_{num}.wav
    Disambiguation: if the first segment of the stem matches a known emotion, treat as emotion file.
    """
    emotion_set = {e.lower() for e in known_emotions}
    emotion_speaker: Dict[Tuple[str, str], List[Path]] = {}
    speaker: Dict[str, List[Path]] = {}

    for wav in Path(emotion_dir).rglob("*.wav"):
        parts = wav.stem.split("_")
        if len(parts) >= 3 and parts[0].lower() in emotion_set:
            # {emotion}_{speaker...}_{num}  – join middle parts as speaker name
            emo = parts[0].lower()
            spk = "_".join(parts[1:-1])
            emotion_speaker.setdefault((emo, spk), []).append(wav)
            # Also register speaker for no-emotion fallback
            speaker.setdefault(spk, []).append(wav)
        elif len(parts) >= 2:
            # {speaker...}_{num}
            spk = "_".join(parts[:-1])
            speaker.setdefault(spk, []).append(wav)

    speakers = sorted(speaker.keys())
    return EmotionRefPool(
        emotion_speaker=emotion_speaker,
        speaker=speaker,
        speakers=speakers,
    )


_EXCLUDED_SPEAKERS = {"bella"}  # speakers with undesired accent/style

def _pick_emotion_speakers(pool: EmotionRefPool) -> Tuple[str, str]:
    """Pick two different speaker names for User (A) and Agent (B)."""
    available = [s for s in pool.speakers if s not in _EXCLUDED_SPEAKERS]
    if len(available) < 2:
        raise ValueError(
            f"eleven_lab_emotion reference pool needs >=2 distinct speakers after exclusions, found: {available}"
        )
    spk_a = random.choice(available)
    remaining = [s for s in available if s != spk_a]
    spk_b = random.choice(remaining)
    return spk_a, spk_b


def _pick_emotion_ref_wav(
    pool: EmotionRefPool, emotion: Optional[str], speaker_name: str
) -> Path:
    """Pick a reference wav for (emotion, speaker). Falls back gracefully."""
    if emotion:
        key = (emotion.lower(), speaker_name)
        if key in pool.emotion_speaker:
            return random.choice(pool.emotion_speaker[key])
    # Fallback: any wav for this speaker
    if speaker_name in pool.speaker:
        return random.choice(pool.speaker[speaker_name])
    # Ultimate fallback: any wav in the pool
    all_wavs = [w for wavs in pool.speaker.values() for w in wavs]
    if all_wavs:
        return random.choice(all_wavs)
    raise ValueError(f"No reference wav found for speaker={speaker_name!r}, emotion={emotion!r}")


def mix_segments_to_stereo_and_save(
    *,
    audio_segments: List[Tuple[str, torch.Tensor]],
    output: str | Path,
    sample_rate: int,
    inter_turn_silence_sec: float = 0.25,
    overlap_shift_sec_min: float = 0.6,
    overlap_shift_sec_max: float = 1.0,
    overlap_pause_sec: Optional[float] = None,
    return_timeline: bool = False,
):
    """Mix a sequence of mono segments into a 2-channel dialogue wav.

    Convention:
      - User turns go to left channel
      - Agent turns go to right channel
      - [overlap] turns overlap with previous audio
    """

    l_ch = None
    r_ch = None

    timeline: List[Dict[str, Any]] = []

    for idx, (role, wav) in enumerate(audio_segments):
        wav_len = int(wav.shape[-1])
        is_user = role in ("User", "[overlap] User", "[pause] User") or role.strip().lower().startswith(
            "[pause] user"
        )
        is_overlap = role.strip().lower().startswith("[overlap]")

        # Both channels are kept aligned after each iteration; use left length as current time.
        current_len = int(l_ch.shape[-1]) if l_ch is not None else 0
        seg_start = 0
        seg_end = wav_len

        if idx == 0:
            if is_user:
                l_ch = wav
                r_ch = torch.zeros_like(wav)
            else:
                r_ch = wav
                l_ch = torch.zeros_like(wav)
        else:
            if is_overlap:
                # How much earlier this overlap turn starts (by trimming the overlapping speaker channel).
                # Effective audible overlap is roughly: overlap_frame - overlap_pause_frames.
                s_min = float(overlap_shift_sec_min)
                s_max = float(overlap_shift_sec_max)
                if s_min <= 0:
                    s_min = 0.01
                if s_max < s_min:
                    s_max = s_min

                overlap_frame = int(random.uniform(s_min, s_max) * sample_rate)
                overlap_frame = max(0, overlap_frame)

                channel_len = l_ch.shape[-1] if is_user else r_ch.shape[-1]
                overlap_frame = min(overlap_frame, wav.shape[-1], channel_len)
                pad_len = max(0, wav.shape[-1] - overlap_frame)
                padded = torch.zeros(1, pad_len)
                pause_sec = inter_turn_silence_sec if overlap_pause_sec is None else float(overlap_pause_sec)
                pause_frames = max(0, int(round(pause_sec * sample_rate)))
                pause = torch.zeros(1, pause_frames)

                seg_start = max(0, current_len - overlap_frame) + pause_frames
                seg_end = seg_start + wav_len

                if is_user:
                    l_ch = l_ch[:, :-overlap_frame] if overlap_frame > 0 else l_ch
                    l_ch = torch.cat([l_ch, pause, wav], -1)
                    r_ch = torch.cat([r_ch, pause, padded], -1)
                else:
                    r_ch = r_ch[:, :-overlap_frame] if overlap_frame > 0 else r_ch
                    r_ch = torch.cat([r_ch, pause, wav], -1)
                    l_ch = torch.cat([l_ch, pause, padded], -1)
            else:
                padded = torch.zeros_like(wav)
                pause_frames = max(0, int(round(float(inter_turn_silence_sec) * sample_rate)))
                pause = torch.zeros(1, pause_frames)

                seg_start = current_len + pause_frames
                seg_end = seg_start + wav_len

                if is_user:
                    l_ch = torch.cat([l_ch, pause, wav], -1)
                    r_ch = torch.cat([r_ch, pause, padded], -1)
                else:
                    r_ch = torch.cat([r_ch, pause, wav], -1)
                    l_ch = torch.cat([l_ch, pause, padded], -1)

        timeline.append(
            {
                "role": role,
                "start_frame": int(seg_start),
                "end_frame": int(seg_end),
                "duration_frames": int(max(0, seg_end - seg_start)),
            }
        )

    full_dialog = torch.cat([l_ch, r_ch], dim=0)
    torchaudio.save(str(output), full_dialog, sample_rate)
    logging.info(f"Saved TTS audio to: {output}")

    if return_timeline:
        return timeline

    return None


def _role_is_user(role: str) -> bool:
    r = (role or "").strip().lower()
    # Matches: "User", "[overlap] User", "[pause] User", etc.
    return r.endswith("user") or (" user" in r)


def _role_is_pause(role: str) -> bool:
    return (role or "").strip().lower().startswith("[pause]")


def _role_is_overlap(role: str) -> bool:
    return (role or "").strip().lower().startswith("[overlap]")


def mix_segments_to_stereo_and_save_clean(
    *, audio_segments: List[Tuple[str, torch.Tensor]], output: str | Path, sample_rate: int
):
    """Mix a sequence of mono segments into a 2-channel dialogue wav (clean).

    Clean means:
      - NO extra inter-turn silence
      - NO [pause] insertion (pause turns are skipped)
      - NO overlap behavior ([overlap] treated as a normal sequential turn)
    """

    l_ch = torch.zeros(1, 0)
    r_ch = torch.zeros(1, 0)

    for role, wav in audio_segments:
        if _role_is_pause(role):
            continue

        is_user = _role_is_user(role)
        if wav.ndim == 1:
            wav = wav.view(1, -1)
        if wav.shape[0] > 1:
            wav = wav[0:1, :]

        padded = torch.zeros_like(wav)
        if is_user:
            l_ch = torch.cat([l_ch, wav], -1)
            r_ch = torch.cat([r_ch, padded], -1)
        else:
            r_ch = torch.cat([r_ch, wav], -1)
            l_ch = torch.cat([l_ch, padded], -1)

    full_dialog = torch.cat([l_ch, r_ch], dim=0)
    torchaudio.save(str(output), full_dialog, sample_rate)
    logging.info(f"Saved clean TTS audio to: {output}")


def _ensure_wav_mono_16k(src_audio: str, dst_wav: str):
    """Ensure a wav file exists at 16kHz mono for BreezyVoice prompt audio."""
    src = Path(src_audio)
    if src.suffix.lower() == ".wav" and Path(dst_wav).resolve() == src.resolve():
        return

    wav, sr = torchaudio.load(str(src))
    if wav.ndim == 2 and wav.shape[0] > 1:
        wav = wav[0:1, :]
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    Path(dst_wav).parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(dst_wav, wav, 16000)


def _breezyvoice_transcribe(
    *, breezy_python: str, breezy_repo_dir: str, audio_path: str
) -> str:
    """Transcribe prompt audio ONCE via BreezyVoice's helper (Whisper-base + OpenCC)."""
    repo_dir = str(Path(breezy_repo_dir).resolve())
    audio_path = str(Path(audio_path).resolve())
    cmd = [
        breezy_python,
        "-c",
        (
            "import sys; "
            "sys.path.insert(0, r'" + repo_dir + "'); "
            "import single_inference as s; "
            "print(s.transcribe_audio(r'" + audio_path + "'))"
        ),
    ]
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    proc = subprocess.run(
        cmd,
        cwd=repo_dir,
        env=env,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    lines = [ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def BreezyVoice_gen(
    script,
    output,
    *,
    breezy_repo_dir: str,
    breezy_python: str = "python",
    model_path: str = "MediaTek-Research/BreezyVoice-300M",
    spk_audio_dir: str = "XTTS_reference",
    user_ref_audio: Optional[str] = None,
    agent_ref_audio: Optional[str] = None,
    emotion_ref_pool: Optional["EmotionRefPool"] = None,
    user_speaker_name: Optional[str] = None,
    agent_speaker_name: Optional[str] = None,
    user_prompt_text_transcription: Optional[str] = None,
    agent_prompt_text_transcription: Optional[str] = None,
    sample_rate: int = 24000,
    save_individual_turns: bool = False,
    turn_output_dir: Optional[Path] = None,
    speaker_meta_by_path: Optional[Dict[str, Dict[str, Any]]] = None,
    cuda_visible_devices: Optional[str] = None,
    # Mixing knobs (effects wav only)
    inter_turn_silence_sec: float = 0.25,
    overlap_shift_sec_min: float = 0.6,
    overlap_shift_sec_max: float = 1.0,
    overlap_pause_sec: Optional[float] = None,
    write_pause_overlap_mix: Optional[bool] = None,
):
    """Generate TTS audio via BreezyVoice (batch_inference) and mix into stereo.

    Design goals:
      - Avoid importing BreezyVoice into this Python env (Py3.10 constraint).
      - Load BreezyVoice model once per dialogue by using batch_inference.py.
      - Keep existing stereo mixing + overlap/pause behavior.

        Notes:
            - BreezyVoice prompt audio must be WAV; we convert/copy to temp WAV(16k mono).
            - Emotion tags are not supported by BreezyVoice and are stripped.
    """

    repo_dir = Path(breezy_repo_dir)
    if not repo_dir.exists():
        raise FileNotFoundError(
            f"BreezyVoice repo not found: {repo_dir}. Set tts.breezyvoice_repo_dir correctly."
        )

    # When emotion_ref_pool is provided, speaker refs come from the emotion dir per turn.
    # Otherwise fall back to the legacy spk_audio_dir pool.
    if emotion_ref_pool is not None:
        spk_A_name = user_speaker_name or ""
        spk_B_name = agent_speaker_name or ""
        spk_A_audio = ""  # resolved per turn
        spk_B_audio = ""  # resolved per turn
        logging.info(f"Speaker A (User) [emotion mode]: {spk_A_name}")
        logging.info(f"Speaker B (Agent) [emotion mode]: {spk_B_name}")
    else:
        # Build fallback speaker pool
        all_dir = Path(spk_audio_dir) / "all"
        spk_audio_files = list(all_dir.glob("*.wav"))  # BreezyVoice batch assumes .wav
        if len(spk_audio_files) < 2:
            raise ValueError(
                f"BreezyVoice backend needs >=2 .wav files in {all_dir}. Found {len(spk_audio_files)}"
            )

        spk_A_audio = resolve_speaker_reference(
            user_ref_audio, fallback_pool=spk_audio_files, role_label="User"
        )
        spk_B_audio = resolve_speaker_reference(
            agent_ref_audio, fallback_pool=spk_audio_files, role_label="Agent"
        )

        if (not user_ref_audio) and (not agent_ref_audio):
            if Path(spk_A_audio).resolve() == Path(spk_B_audio).resolve():
                remaining = [
                    p for p in spk_audio_files if p.resolve() != Path(spk_A_audio).resolve()
                ]
                spk_B_audio = str(random.choice(remaining))

        logging.info(f"Speaker A (User): {Path(spk_A_audio).name}")
        logging.info(f"Speaker B (Agent): {Path(spk_B_audio).name}")

    # If reference metadata provides a transcript (e.g. Common Voice validated.tsv `sentence`),
    # prefer it over Whisper transcription for better stability.
    if speaker_meta_by_path is not None:
        if not user_prompt_text_transcription:
            user_prompt_text_transcription = _try_get_reference_sentence(
                speaker_meta_by_path, str(spk_A_audio)
            )
        if not agent_prompt_text_transcription:
            agent_prompt_text_transcription = _try_get_reference_sentence(
                speaker_meta_by_path, str(spk_B_audio)
            )

    # Prepare temp workspace for this dialogue
    with tempfile.TemporaryDirectory(prefix=f"breezyvoice_{os.getpid()}_") as td:
        td_path = Path(td)
        prompt_dir = td_path / "prompts"
        out_dir = td_path / "out"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)

        if emotion_ref_pool is None:
            # ── Standard mode: one reference per speaker, copied once ──────
            user_prompt_wav = prompt_dir / "user.wav"
            agent_prompt_wav = prompt_dir / "agent.wav"
            _ensure_wav_mono_16k(spk_A_audio, str(user_prompt_wav))
            _ensure_wav_mono_16k(spk_B_audio, str(agent_prompt_wav))

            # Prompt transcriptions (highly recommended). Compute once if not provided.
            if not user_prompt_text_transcription:
                logging.info("BreezyVoice: transcribing user prompt audio (once)...")
                user_prompt_text_transcription = _breezyvoice_transcribe(
                    breezy_python=breezy_python,
                    breezy_repo_dir=str(repo_dir),
                    audio_path=str(user_prompt_wav),
                )
            if not agent_prompt_text_transcription:
                logging.info("BreezyVoice: transcribing agent prompt audio (once)...")
                agent_prompt_text_transcription = _breezyvoice_transcribe(
                    breezy_python=breezy_python,
                    breezy_repo_dir=str(repo_dir),
                    audio_path=str(agent_prompt_wav),
                )

        # Build CSV rows for non-pause turns
        csv_rows: List[Dict[str, str]] = []
        # (role, clean_text, out_wav, speaker_reference, emotion_tag)
        turn_meta: List[Tuple[str, str, Optional[Path], str, Optional[str]]] = []

        for idx, (role, text) in enumerate(script, 1):
            is_pause = role.strip().lower().startswith("[pause]")
            is_user = role in ("User", "[overlap] User", "[pause] User") or role.strip().lower().startswith(
                "[pause] user"
            )

            emotion_tag = _extract_leading_emotion_tag(text)  # e.g. "happy" or None
            clean_text = sanitize_tts_text(strip_leading_emotion_tag(text))

            if emotion_ref_pool is not None:
                # ── Emotion mode: per-turn reference ─────────────────────
                spk_name_for_role = spk_A_name if is_user else spk_B_name
                ref_wav = _pick_emotion_ref_wav(emotion_ref_pool, emotion_tag, spk_name_for_role)
                turn_ref_name = f"turn_{idx:03d}_ref"
                ref_wav_in_prompt = prompt_dir / f"{turn_ref_name}.wav"
                _ensure_wav_mono_16k(str(ref_wav), str(ref_wav_in_prompt))
                spk_ref = str(ref_wav)

                if is_pause:
                    turn_meta.append((role, clean_text, None, spk_ref, emotion_tag))
                    continue

                out_stem = f"turn_{idx:03d}"
                out_wav = out_dir / f"{out_stem}.wav"
                turn_meta.append((role, clean_text, out_wav, spk_ref, emotion_tag))
                _ref_sentence = ""
                if speaker_meta_by_path is not None:
                    _ref_sentence = (speaker_meta_by_path.get(str(ref_wav), {}) or {}).get("sentence", "") or ""
                csv_rows.append(
                    {
                        "speaker_prompt_audio_filename": turn_ref_name,
                        "speaker_prompt_text_transcription": _ref_sentence,
                        "content_to_synthesize": clean_text,
                        "output_audio_filename": out_stem,
                    }
                )
            else:
                # ── Standard mode: dialogue-level reference ───────────────
                spk_ref = spk_A_audio if is_user else spk_B_audio
                if is_pause:
                    turn_meta.append((role, clean_text, None, str(spk_ref), emotion_tag))
                    continue

                out_stem = f"turn_{idx:03d}"
                out_wav = out_dir / f"{out_stem}.wav"
                turn_meta.append((role, clean_text, out_wav, str(spk_ref), emotion_tag))

                if is_user:
                    spk_name = "user"
                    spk_txt = user_prompt_text_transcription
                else:
                    spk_name = "agent"
                    spk_txt = agent_prompt_text_transcription

                csv_rows.append(
                    {
                        "speaker_prompt_audio_filename": spk_name,
                        "speaker_prompt_text_transcription": spk_txt,
                        "content_to_synthesize": clean_text,
                        "output_audio_filename": out_stem,
                    }
                )

        # Run BreezyVoice batch inference once for all turns
        csv_path = td_path / "batch.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "speaker_prompt_audio_filename",
                    "speaker_prompt_text_transcription",
                    "content_to_synthesize",
                    "output_audio_filename",
                ],
            )
            writer.writeheader()
            for r in csv_rows:
                writer.writerow(r)

        cmd = [
            breezy_python,
            "batch_inference.py",
            "--csv_file",
            str(csv_path),
            "--speaker_prompt_audio_folder",
            str(prompt_dir),
            "--output_audio_folder",
            str(out_dir),
            "--model_path",
            str(model_path),
        ]
        env = dict(os.environ)
        env["PYTHONUTF8"] = "1"
        if cuda_visible_devices is not None and str(cuda_visible_devices).strip() != "":
            env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices).strip()
        logging.info("BreezyVoice: running batch_inference.py for this dialogue...")
        subprocess.run(cmd, cwd=str(repo_dir), env=env, check=True)

        # Load generated wavs and mix.
        # We always write the clean full-dialogue mix:
        #   - full.wav
        # The pause/overlap effects mix is optional and can be disabled:
        #   - full_with_overlap_and_pause.wav
        audio_segments_with_effects: List[Tuple[str, torch.Tensor]] = []
        audio_segments_clean: List[Tuple[str, torch.Tensor]] = []
        turn_metadata_list: List[Dict[str, Any]] = []
        # Mapping to update pause/overlap timing + audio later.
        effect_segment_to_metadata: List[Optional[Dict[str, Any]]] = []
        current_time = 0.0
        saved_turn_idx = 0
        has_pause_or_overlap = False

        for role, clean_text, out_wav, spk_ref, emotion_tag in turn_meta:
            is_pause = role.strip().lower().startswith("[pause]")
            is_overlap = role.strip().lower().startswith("[overlap]")
            is_user = role in ("User", "[overlap] User", "[pause] User") or role.strip().lower().startswith(
                "[pause] user"
            )
            if is_pause:
                has_pause_or_overlap = True
                pause_duration = random.uniform(0.6, 1.0)
                pause_frames = int(pause_duration * sample_rate)
                wav = torch.zeros(1, pause_frames)
                audio_segments_with_effects.append((role, wav))
                effect_segment_to_metadata.append(None)
                continue
            if is_overlap:
                has_pause_or_overlap = True

            try:
                if out_wav is None or not out_wav.exists():
                    raise FileNotFoundError(f"Missing BreezyVoice output wav: {out_wav}")

                wav, sr = torchaudio.load(str(out_wav))
                if sr != sample_rate:
                    wav = torchaudio.functional.resample(wav, sr, sample_rate)
                if wav.shape[0] > 1:
                    wav = wav[0:1, :]
            except Exception as e:
                logging.error(f"Error loading BreezyVoice audio for role={role}: {e}")

                wav = torch.zeros(1, sample_rate)

            if save_individual_turns:
                if turn_output_dir is None:
                    raise ValueError("save_individual_turns=True requires turn_output_dir")

                turn_output_dir.mkdir(parents=True, exist_ok=True)
                turn_duration = wav.shape[1] / sample_rate
                audio_start = current_time
                audio_end = current_time + turn_duration
                current_time = audio_end

                turn_audio_path = turn_output_dir / f"turn{saved_turn_idx:02d}.wav"
                torchaudio.save(str(turn_audio_path), wav, sample_rate)

                # V2 pipeline has no para tags; keep schema compatible with para exporter.
                para_tags_structured = {
                    "gender": None,
                    "pitch": None,
                    "speed": None,
                    "volume": None,
                    "emotion": None,
                }

                spk_ref_meta = None
                if speaker_meta_by_path is not None:
                    spk_ref_meta = speaker_meta_by_path.get(str(spk_ref), None)

                spk_ref_id = None
                if isinstance(spk_ref_meta, dict):
                    spk_ref_id = spk_ref_meta.get("reference_id") or spk_ref_meta.get("path")
                if not spk_ref_id:
                    spk_ref_id = Path(str(spk_ref)).name

                meta_entry: Dict[str, Any] = {
                    "turn_idx": saved_turn_idx,
                    "role": role,
                    "speaker": "user" if is_user else "agent",
                    "text": clean_text,
                    "para_tags": para_tags_structured,
                    "audio_path": str(turn_audio_path),
                    "audio_start": audio_start,
                    "audio_end": audio_end,
                    "audio_duration": turn_duration,
                    # New: pause+overlap turn-level audio + timing (filled after effects mix)
                    "audio_path_pause_overlap": None,
                    "audio_start_pause_overlap": None,
                    "audio_end_pause_overlap": None,
                    "audio_duration_pause_overlap": None,
                    "speaker_reference": str(spk_ref),
                    "speaker_reference_id": str(spk_ref_id),
                    "speaker_reference_metadata": spk_ref_meta,
                    "emotion": emotion_tag,
                    "emotion_reference": [],
                }

                turn_metadata_list.append(meta_entry)
                effect_segment_to_metadata.append(meta_entry)
                saved_turn_idx += 1
            else:
                effect_segment_to_metadata.append(None)

            audio_segments_with_effects.append((role, wav))
            audio_segments_clean.append((role, wav))

        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)

        output_clean = output
        output_with_effects = output.with_name("full_with_overlap_and_pause.wav")

        mix_segments_to_stereo_and_save_clean(
            audio_segments=audio_segments_clean, output=output_clean, sample_rate=sample_rate
        )

        do_fx = bool(write_pause_overlap_mix) if write_pause_overlap_mix is not None else bool(has_pause_or_overlap)
        effects_timeline = None
        if do_fx:
            effects_timeline = mix_segments_to_stereo_and_save(
                audio_segments=audio_segments_with_effects,
                output=output_with_effects,
                sample_rate=sample_rate,
                inter_turn_silence_sec=inter_turn_silence_sec,
                overlap_shift_sec_min=overlap_shift_sec_min,
                overlap_shift_sec_max=overlap_shift_sec_max,
                overlap_pause_sec=overlap_pause_sec,
                return_timeline=bool(save_individual_turns),
            )

            # Save pause+overlap per-turn audio slices and timing.
            if save_individual_turns and effects_timeline is not None and turn_output_dir is not None:
                try:
                    full_fx, fx_sr = torchaudio.load(str(output_with_effects))
                    if fx_sr != sample_rate:
                        full_fx = torchaudio.functional.resample(full_fx, fx_sr, sample_rate)
                        fx_sr = sample_rate

                    # Ensure (2, T)
                    if full_fx.ndim == 1:
                        full_fx = full_fx.unsqueeze(0)
                    if full_fx.shape[0] == 1:
                        full_fx = torch.cat([full_fx, torch.zeros_like(full_fx)], dim=0)

                    include_next_overlap_sec = 0.35

                    for seg_idx, seg_info in enumerate(effects_timeline):
                        meta_entry = (
                            effect_segment_to_metadata[seg_idx]
                            if seg_idx < len(effect_segment_to_metadata)
                            else None
                        )
                        if not isinstance(meta_entry, dict):
                            continue

                        start_f = int(seg_info.get("start_frame", 0) or 0)
                        end_f = int(seg_info.get("end_frame", 0) or 0)

                        # Include an explicit [pause] segment immediately before this turn.
                        if seg_idx > 0:
                            prev = effects_timeline[seg_idx - 1]
                            prev_role = str(prev.get("role", "") or "")
                            if _role_is_pause(prev_role):
                                prev_start = int(prev.get("start_frame", 0) or 0)
                                prev_end = int(prev.get("end_frame", 0) or 0)
                                if prev_end > prev_start and prev_end <= start_f:
                                    start_f = prev_start

                        # If the next segment is an [overlap] turn, extend the current turn slightly
                        # into the overlap so BOTH adjacent per-turn clips contain audible overlap.
                        if seg_idx + 1 < len(effects_timeline):
                            nxt = effects_timeline[seg_idx + 1]
                            nxt_role = str(nxt.get("role", "") or "")
                            if _role_is_overlap(nxt_role):
                                nxt_start = int(nxt.get("start_frame", 0) or 0)
                                bump = int(round(float(include_next_overlap_sec) * float(fx_sr)))
                                end_f = max(end_f, nxt_start + bump)
                        if end_f <= start_f:
                            continue
                        start_f = max(0, min(start_f, int(full_fx.shape[-1])))
                        end_f = max(0, min(end_f, int(full_fx.shape[-1])))
                        if end_f <= start_f:
                            continue

                        turn_idx = int(meta_entry.get("turn_idx", 0) or 0)
                        turn_fx_path = turn_output_dir / f"turn{turn_idx:02d}_pause_overlap.wav"
                        fx_slice = full_fx[:, start_f:end_f]
                        torchaudio.save(str(turn_fx_path), fx_slice, fx_sr)

                        meta_entry["audio_path_pause_overlap"] = str(turn_fx_path)
                        meta_entry["audio_start_pause_overlap"] = float(start_f) / float(fx_sr)
                        meta_entry["audio_end_pause_overlap"] = float(end_f) / float(fx_sr)
                        meta_entry["audio_duration_pause_overlap"] = float(end_f - start_f) / float(fx_sr)
                except Exception as e:
                    logging.warning("Failed to create pause/overlap per-turn audio slices: %s", e)

        if save_individual_turns:
            for m in turn_metadata_list:
                m["full_dialogue_audio"] = str(output_clean)
                m["full_dialogue_audio_with_overlap_and_pause"] = str(output_with_effects) if do_fx else None
            return turn_metadata_list

    return None


def _find_final_filler_txt_path(
    *,
    data_root: Path,
    mode_name: str,
    topic: str,
    topic_dir: Optional[str] = None,
    scenario_idx: str,
    dialogue_idx: Optional[str],
) -> Optional[Path]:
    """Locate the final script txt used for TTS (prefer archived filler output).

        Priority:
            1) final filler txt: {data_root}/{mode}/{topic_dir}/txt/filler/{topic}_scenario{idx}.txt
            2) indexed txt: {data_root}/{mode}/{topic_dir}/txt/filler/{topic}_scenario{idx}_{didx}.txt
            3) legacy final filler txt: {data_root}/{mode}/{topic_dir}/txt/filler/scenario{idx}.txt
            4) legacy indexed txt: {data_root}/{mode}/{topic_dir}/txt/filler/scenario{idx}_{didx}.txt
    """
    _tdir = topic_dir or topic
    candidates: List[Path] = [
        data_root / mode_name / _tdir / "txt" / "filler" / f"{topic}_scenario{scenario_idx}.txt",
        data_root / mode_name / _tdir / "txt" / "filler" / f"scenario{scenario_idx}.txt"
    ]
    if dialogue_idx:
        candidates.append(
            data_root / mode_name / _tdir / "txt" / "filler" / f"{topic}_scenario{scenario_idx}_{dialogue_idx}.txt"
        )
        candidates.append(
            data_root / mode_name / _tdir / "txt" / "filler" / f"scenario{scenario_idx}_{dialogue_idx}.txt"
        )
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def _find_final_txt_path(
    *,
    data_root: Path,
    mode_name: str,
    topic: str,
    topic_dir: Optional[str] = None,
    scenario_idx: str,
    dialogue_idx: Optional[str],
    preferred_stage: Optional[str] = None,
) -> Optional[Path]:
    """Locate the final script txt used for TTS.

    When the user skips overlap/filler, the "final" script is usually the dialogue-stage txt.
    This helper searches multiple stage folders.

    preferred_stage:
      - 'filler' | 'overlap' | 'dialogue' to force a specific stage first
      - None/'auto' to fall back in order: filler -> overlap -> dialogue
    """
    _tdir = topic_dir or topic
    stage_order = ["filler", "overlap", "dialogue"]
    if preferred_stage:
        ps = str(preferred_stage).strip().lower()
        if ps in stage_order:
            stage_order = [ps] + [s for s in stage_order if s != ps]

    def _cands(stage: str) -> List[Path]:
        base = data_root / mode_name / _tdir / "txt" / stage
        out: List[Path] = [
            base / f"{topic}_scenario{scenario_idx}.txt",
            base / f"scenario{scenario_idx}.txt",
        ]
        if dialogue_idx:
            out.append(base / f"{topic}_scenario{scenario_idx}_{dialogue_idx}.txt")
            out.append(base / f"scenario{scenario_idx}_{dialogue_idx}.txt")
        return out

    for stage in stage_order:
        for cand in _cands(stage):
            if cand.exists():
                return cand
    return None


def _ensure_pause_overlap_turn_slices(
    *,
    dialogue_folder: Path,
    data_root: Path,
    sample_rate: int,
    search_window_sec: float = 2.0,
) -> None:
    """Backfill per-turn pause/overlap slices for existing runs.

    If `turn_metadata.json` is missing *_pause_overlap fields or files, we align each clean
    per-turn wav against the full pause/overlap mix (per-speaker channel) and slice the
    stereo segment from `full_with_overlap_and_pause.wav`.
    """

    try:
        metadata_file = dialogue_folder / "individual" / "turn_metadata.json"
        if not metadata_file.exists():
            return

        turn_list = json.loads(metadata_file.read_text(encoding="utf-8"))
        if not isinstance(turn_list, list) or not turn_list:
            return

        fx_wav = dialogue_folder / "full_with_overlap_and_pause.wav"
        if not fx_wav.exists():
            return

        needs = False
        for m in turn_list:
            if not isinstance(m, dict):
                continue
            p = m.get("audio_path_pause_overlap")
            if not p:
                needs = True
                break
            if not Path(str(p)).exists():
                needs = True
                break

        if not needs:
            return

        full_fx, fx_sr = torchaudio.load(str(fx_wav))
        if fx_sr != sample_rate:
            full_fx = torchaudio.functional.resample(full_fx, fx_sr, sample_rate)
            fx_sr = sample_rate

        if full_fx.ndim == 1:
            full_fx = full_fx.unsqueeze(0)
        if full_fx.shape[0] == 1:
            full_fx = torch.cat([full_fx, torch.zeros_like(full_fx)], dim=0)

        out_dir = dialogue_folder / "individual"
        out_dir.mkdir(parents=True, exist_ok=True)

        win = int(round(float(search_window_sec) * fx_sr))

        for m in turn_list:
            if not isinstance(m, dict):
                continue

            # Already present and file exists.
            existing = m.get("audio_path_pause_overlap")
            if existing and Path(str(existing)).exists():
                continue

            clean_path = m.get("audio_path")
            if not clean_path or not Path(str(clean_path)).exists():
                continue

            clean_wav, clean_sr = torchaudio.load(str(clean_path))
            if clean_sr != fx_sr:
                clean_wav = torchaudio.functional.resample(clean_wav, clean_sr, fx_sr)
                clean_sr = fx_sr

            if clean_wav.ndim == 1:
                clean_wav = clean_wav.unsqueeze(0)
            if clean_wav.shape[0] > 1:
                clean_wav = clean_wav[0:1, :]

            speaker = str(m.get("speaker", "")).strip().lower()
            ch = 0 if speaker == "user" else 1
            x = full_fx[ch : ch + 1, :]

            turn_len = int(clean_wav.shape[-1])
            if turn_len <= 0:
                continue

            expected_start = int(round(float(m.get("audio_start", 0.0) or 0.0) * fx_sr))
            s0 = max(0, expected_start - win)
            s1 = min(int(x.shape[-1]), expected_start + win + turn_len)
            if s1 - s0 < turn_len:
                s0 = max(0, min(s0, int(x.shape[-1]) - turn_len))
                s1 = min(int(x.shape[-1]), s0 + max(turn_len, 1))
            if s1 - s0 < turn_len:
                continue

            # Correlate within search window.
            seg = x[:, s0:s1]
            # Normalize to reduce amplitude sensitivity.
            seg_z = seg - seg.mean(dim=-1, keepdim=True)
            t = clean_wav - clean_wav.mean(dim=-1, keepdim=True)
            denom = (t.pow(2).sum().sqrt() + 1e-6)
            t = t / denom

            # conv1d correlation: (1,1,L) * (1,1,K)
            seg_in = seg_z.unsqueeze(0)
            w = t.flip(-1).unsqueeze(0)
            corr = torch.nn.functional.conv1d(seg_in, w)
            best = int(torch.argmax(torch.abs(corr)).item())
            start_f = int(s0 + best)
            end_f = int(start_f + turn_len)
            start_f = max(0, min(start_f, int(full_fx.shape[-1])))
            end_f = max(0, min(end_f, int(full_fx.shape[-1])))
            if end_f <= start_f:
                continue

            turn_idx = int(m.get("turn_idx", 0) or 0)
            fx_path = out_dir / f"turn{turn_idx:02d}_pause_overlap.wav"
            fx_slice = full_fx[:, start_f:end_f]
            torchaudio.save(str(fx_path), fx_slice, fx_sr)

            m["audio_path_pause_overlap"] = str(fx_path)
            m["audio_start_pause_overlap"] = float(start_f) / float(fx_sr)
            m["audio_end_pause_overlap"] = float(end_f) / float(fx_sr)
            m["audio_duration_pause_overlap"] = float(end_f - start_f) / float(fx_sr)

        metadata_file.write_text(json.dumps(turn_list, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        logging.warning("Backfill pause_overlap slices failed for %s: %s", dialogue_folder, e)


def _default_run_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _default_run_root_base() -> Path:
    """Preferred base folder for new runs.

    If /work/jaylin0418 exists (common workspace mount), write all new outputs there.
    Fallback to current working directory otherwise.
    """

    cand = Path("/work/jaylin0418")
    if cand.exists() and cand.is_dir():
        return cand
    return Path.cwd()


def _system_prompt_dir(*, data_root: Path, mode_name: str, topic_label: str) -> Path:
    return data_root / mode_name / topic_label / "txt" / "system_prompt_txt"


def _legacy_system_prompt_dir(*, data_root: Path, mode_name: str, topic_label: str) -> Path:
    return data_root / mode_name / topic_label / "system_prompt_txt"


def _system_prompt_path(
    *, data_root: Path, mode_name: str, topic_label: str, scenario_id: str, topic_dir: Optional[str] = None
) -> Path:
    return _system_prompt_dir(data_root=data_root, mode_name=mode_name, topic_label=topic_dir or topic_label) / (
        f"{_scenario_file_prefix(topic_label, scenario_id)}_system_prompt.txt"
    )


def generate_system_prompts(cfg, max_retries: int = 10) -> None:
    """Stage 1.5: generate a scenario-level system prompt per scenario.

    Reads scenarios from: <cfg.scenario.out_file>.jsonl
    Writes to: {data_root}/{mode_name}/{topic}/txt/system_prompt_txt/{scenario_id}_system_prompt.txt
    """

    if cfg.get("system_prompt", None) is None:
        logging.info("system_prompt stage skipped: cfg.system_prompt is missing")
        return

    scen_path = Path(cfg.scenario["out_file"]).with_suffix(".jsonl")
    if not scen_path.exists():
        raise FileNotFoundError(f"Scenario JSONL not found for system_prompt stage: {scen_path}")

    data_root = Path(cfg.get("data_root", "TEST_syn_data"))
    mode_name = str(cfg.get("mode_name", "v2")).strip() or "v2"
    topic_label = _extract_topic_label(str(cfg.scenario.get("topic", "unknown")))
    suffix = str(cfg.get("topic_folder_suffix", "") or "")
    topic_dir = topic_label + suffix

    out_dir = _system_prompt_dir(data_root=data_root, mode_name=mode_name, topic_label=topic_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sys_tmpl = str(cfg.system_prompt.get("prompt", "") or "").strip()
    model = str(cfg.system_prompt.get("model", "") or "").strip()
    if not sys_tmpl or not model:
        logging.info("system_prompt stage skipped: missing system_prompt.prompt or system_prompt.model")
        return

    gen_kwargs = {}
    try:
        gen_kwargs = get_gen_kwargs(cfg.system_prompt)
    except Exception:
        gen_kwargs = {}

    scenarios = [json.loads(l) for l in scen_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    for s in tqdm(scenarios, desc="system prompts"):
        scenario_id = _canonicalize_scenario_id(s.get("id", "unknown") or "unknown")
        out_path = _system_prompt_path(
            data_root=data_root, mode_name=mode_name, topic_label=topic_label, scenario_id=scenario_id,
            topic_dir=topic_dir,
        )
        if out_path.exists():
            logging.info("Skipping %s, system prompt already exists.", scenario_id)
            continue

        for attempt in range(1, int(max_retries) + 1):
            try:
                messages = [
                    {"role": "system", "content": sys_tmpl},
                    {"role": "user", "content": json.dumps(s, ensure_ascii=False)},
                ]
                script = chat_completion(
                    model,
                    messages,
                    **({"max_tokens": 1024, "temperature": 0.9, "top_p": 0.9} | gen_kwargs),
                )
                script = str(script or "").strip()
                if script:
                    out_path.write_text(script, encoding="utf-8")
                    logging.info("Saved system prompt: %s", out_path)
                    break
                logging.warning("Attempt %d/%d failed for %s (empty)", attempt, max_retries, scenario_id)
            except Exception as e:
                logging.warning("Attempt %d/%d failed for %s: %s", attempt, max_retries, scenario_id, e)


def _read_scenario_system_prompt(
    *,
    topic_label: str,
    scenario_id: str,
    cfg=None,
    data_root: Optional[Path] = None,
    mode_name: Optional[str] = None,
) -> Optional[str]:
    """Best-effort read of scenario-level system prompt text."""

    try:
        suffix = ""
        if cfg is not None:
            if data_root is None:
                data_root = Path(cfg.get("data_root", "TEST_syn_data"))
            if mode_name is None:
                mode_name = str(cfg.get("mode_name", "v2")).strip() or "v2"
            suffix = str(cfg.get("topic_folder_suffix", "") or "")

        if data_root is None:
            data_root = Path("TEST_syn_data")
        else:
            data_root = Path(data_root)

        if mode_name is None:
            mode_name = "v2"

        topic_dir = topic_label + suffix
        p = _system_prompt_path(
            data_root=data_root,
            mode_name=mode_name,
            topic_label=topic_label,
            scenario_id=str(scenario_id),
            topic_dir=topic_dir,
        )
        if p.exists():
            return p.read_text(encoding="utf-8").strip() or None
        legacy_dir = _legacy_system_prompt_dir(
            data_root=data_root,
            mode_name=mode_name,
            topic_label=topic_dir,
        )
        legacy_candidates = [
            legacy_dir / f"{_scenario_file_prefix(topic_label, scenario_id)}_system_prompt.txt",
            legacy_dir / f"{_canonicalize_scenario_id(str(scenario_id))}_system_prompt.txt",
        ]
        m = re.match(r"^scenario(\d+)$", _canonicalize_scenario_id(str(scenario_id)))
        if m:
            legacy_candidates.append(legacy_dir / f"scenario{int(m.group(1)):06d}_system_prompt.txt")
        for legacy_p in legacy_candidates:
            if legacy_p.exists():
                return legacy_p.read_text(encoding="utf-8").strip() or None
    except Exception:
        return None
    return None


def _resolve_under(root: Path, p: Any) -> Any:
    if not isinstance(p, str):
        return p
    raw = p.strip()
    if not raw:
        return p
    # Preserve OmegaConf interpolations like ${...}
    if raw.startswith("${"):
        return p
    expanded = Path(raw).expanduser()
    try:
        root_abs = root.expanduser().resolve()
    except Exception:
        root_abs = root

    if expanded.is_absolute():
        return str(expanded)

    # If this relative path already resolves under the run_root (e.g. because it
    # already contains run_root as its first path component), keep it as-is.
    try:
        abs_candidate = (Path.cwd() / expanded).resolve()
        if abs_candidate == root_abs or root_abs in abs_candidate.parents:
            return str(abs_candidate)
    except Exception:
        pass

    return str(root_abs / expanded)


def prepare_run_output_root(cfg, *, create_dirs: bool = True) -> None:
    """Create a per-run root folder and rewrite relative output paths under it.

        Notes:
        - Some configs still default intermediate paths to `data_v2/...`.
        - In batch generation, paths may be rewritten per-iteration under
            `{data_root}/{mode_name}/{topic}/...`, so pre-creating unused folders is just
            noisy. Use `create_dirs=False` to only rewrite paths.
        """

    run_root = str(cfg.get("run_root", "") or "").strip()
    if not run_root:
        base = _default_run_root_base()
        run_root = str(base / f"syn_data_zh_{_default_run_timestamp()}")
        cfg["run_root"] = run_root

    run_root_path = Path(run_root).expanduser()
    run_root_path.mkdir(parents=True, exist_ok=True)

    # Rewrite top-level data_root if it is relative.
    cfg["data_root"] = _resolve_under(run_root_path, cfg.get("data_root", "TEST_syn_data"))

    # Rewrite stage IO paths if they are relative.
    cfg.scenario["out_file"] = _resolve_under(run_root_path, cfg.scenario.get("out_file"))
    cfg.dialogue["out_dir"] = _resolve_under(run_root_path, cfg.dialogue.get("out_dir"))
    cfg.overlap["out_dir"] = _resolve_under(run_root_path, cfg.overlap.get("out_dir"))
    cfg.filler["out_dir"] = _resolve_under(run_root_path, cfg.filler.get("out_dir"))
    cfg.judge["out_dir"] = _resolve_under(run_root_path, cfg.judge.get("out_dir"))
    cfg.tts["wav_dir"] = _resolve_under(run_root_path, cfg.tts.get("wav_dir"))
    cfg.tts["load_dir"] = _resolve_under(run_root_path, cfg.tts.get("load_dir"))
    cfg.huggingface["output_dir"] = _resolve_under(run_root_path, cfg.huggingface.get("output_dir"))

    if not create_dirs:
        return

    # Create only the directories that will actually be used.
    try:
        Path(str(cfg.get("data_root"))).mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    try:
        out_file = cfg.scenario.get("out_file")
        if isinstance(out_file, str) and out_file:
            Path(out_file).expanduser().parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    for k in ("dialogue", "overlap", "filler", "judge"):
        try:
            out_dir = cfg.get(k, {}).get("out_dir")
            if isinstance(out_dir, str) and out_dir:
                Path(out_dir).expanduser().mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
    for k in ("wav_dir", "load_dir"):
        try:
            p = cfg.tts.get(k)
            if isinstance(p, str) and p:
                Path(p).expanduser().mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
    try:
        out_dir = cfg.get("huggingface", {}).get("output_dir")
        if isinstance(out_dir, str) and out_dir:
            Path(out_dir).expanduser().mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def export_to_huggingface(cfg):
    """Export V2 generated dialogues and audio to HuggingFace dataset format.

    Mirrors the para pipeline dataset schema for compatibility.
    Requires TTS to have been run with huggingface.enabled=true so that
    per-turn wavs and `turn_metadata.json` exist.
    """
    try:
        from datasets import Dataset, Features, Value, Sequence
    except ImportError:
        logging.error("HuggingFace datasets library not installed. Install with: pip install datasets")
        return

    if not cfg.get("huggingface", {}).get("enabled", False):
        logging.info("HuggingFace dataset export disabled in config")
        return

    logging.info("=" * 80)
    logging.info("EXPORTING TO HUGGINGFACE DATASET FORMAT (V2)")
    logging.info("=" * 80)

    data_root = Path(cfg.get("data_root", "data_v2"))
    mode_name = str(cfg.get("mode_name", "v2")).strip() or "v2"
    export_mode = str(cfg.get("huggingface", {}).get("mode_override", "normal")).strip() or "normal"
    topic = _extract_topic_label(str(cfg.scenario.get("topic", "unknown")))
    _suffix = str(cfg.get("topic_folder_suffix", "") or "")
    topic_dir = topic + _suffix
    wav_dir = data_root / mode_name / topic_dir / "wav"

    output_dir_cfg = cfg.get("huggingface", {}).get("output_dir", None)
    if isinstance(output_dir_cfg, str) and output_dir_cfg.strip():
        output_dir = Path(output_dir_cfg).expanduser()
    else:
        output_dir = data_root / "huggingface_dataset"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Scenarios (JSONL created during scenario stage)
    scenario_jsonl = Path(cfg.scenario["out_file"]).with_suffix(".jsonl")
    scenarios: Dict[str, Dict[str, Any]] = {}
    if scenario_jsonl.exists():
        for line in scenario_jsonl.read_text(encoding="utf-8").splitlines():
            s = json.loads(line)
            sid = _canonicalize_scenario_id(s.get("id", ""))
            scenarios[sid] = s

    llm_user = cfg.dialogue.get("user_model", "unknown")
    llm_agent = cfg.dialogue.get("agent_model", "unknown")

    # Enumerate dialogue folders
    if not wav_dir.exists():
        logging.error(f"No wav directory found: {wav_dir}")
        logging.error("Please run the TTS stage first with huggingface.enabled=true")
        return

    dialogue_folders = [p for p in sorted(wav_dir.iterdir()) if p.is_dir() and not p.name.startswith(".")]
    if not dialogue_folders:
        logging.error(f"No dialogue folders found under {wav_dir}")
        logging.error("Please run the TTS stage first with huggingface.enabled=true")
        return

    dataset_rows: List[Dict[str, Any]] = []

    include_pause_overlap_cfg = cfg.get("huggingface", {}).get("include_pause_overlap", None)
    include_pause_overlap: Optional[bool]
    if include_pause_overlap_cfg in (None, "", "null", "None"):
        include_pause_overlap = None
    else:
        include_pause_overlap = bool(include_pause_overlap_cfg)

    final_txt_stage = cfg.get("huggingface", {}).get("final_txt_stage", None)
    if isinstance(final_txt_stage, str) and final_txt_stage.strip().lower() in {"", "auto", "none", "null"}:
        final_txt_stage = None

    for dialogue_folder in tqdm(dialogue_folders, desc="Building HuggingFace dataset"):
        dialogue_id = dialogue_folder.name
        metadata_file = dialogue_folder / "individual" / "turn_metadata.json"
        if not metadata_file.exists():
            logging.warning(f"No metadata file found for {dialogue_id}, skipping")
            continue

        # Auto-detect pause/overlap availability when include_pause_overlap is not explicitly set.
        # If the effects mix does not exist, we skip pause_overlap columns and do not try to backfill.
        if include_pause_overlap is None:
            include_pause_overlap = (dialogue_folder / "full_with_overlap_and_pause.wav").exists()

        if include_pause_overlap:
            _ensure_pause_overlap_turn_slices(
                dialogue_folder=dialogue_folder,
                data_root=data_root,
                sample_rate=int(cfg.tts.get("sample_rate", 24000)),
            )

        with open(metadata_file, "r", encoding="utf-8") as f:
            turn_list = json.load(f)

        # Parse dialogue folder name:
        #   {topic}_{mode}_{scenario_idx}
        #   {topic}_{mode}_{scenario_idx}_{dialogue_idx}
        parts = dialogue_id.split("_")
        scenario_idx = None
        dialogue_idx = None
        if len(parts) >= 4:
            scenario_idx = parts[-2]
            dialogue_idx = parts[-1]
            scenario_id = _canonicalize_scenario_id(scenario_idx)
            scenario_description = scenarios.get(scenario_id, {}).get("description", None)
            conversation_id = f"{topic}_{mode_name}_{scenario_idx}"
        elif len(parts) >= 3:
            scenario_idx = parts[-1]
            scenario_id = _canonicalize_scenario_id(scenario_idx)
            scenario_description = scenarios.get(scenario_id, {}).get("description", None)
            conversation_id = f"{topic}_{mode_name}_{scenario_idx}"
        else:
            scenario_description = None
            conversation_id = dialogue_id

        full_dialogue_path = dialogue_folder / "full.wav"
        full_dialogue_pause_overlap_path = dialogue_folder / "full_with_overlap_and_pause.wav"

        final_txt_path = None
        if scenario_idx is not None:
            final_txt_path = _find_final_txt_path(
                data_root=data_root,
                mode_name=mode_name,
                topic=topic,
                topic_dir=topic_dir,
                scenario_idx=str(scenario_idx),
                dialogue_idx=str(dialogue_idx) if dialogue_idx is not None else None,
                preferred_stage=str(final_txt_stage) if final_txt_stage is not None else None,
            )

        system_prompt = None
        try:
            # Store a per-dialogue system prompt for downstream SFT formatting.
            # Prefer Stage 1.5 per-scenario system prompt text when available.
            if scenario_idx is not None:
                scenario_system_prompt = _read_scenario_system_prompt(
                    data_root=data_root,
                    mode_name=mode_name,
                    topic_label=topic,
                    scenario_id=scenario_id,
                    cfg=cfg,
                )
                system_prompt = scenario_system_prompt.strip() or None

            # Legacy fallback: agent system prompt + topic/scenario context.
            if not system_prompt:
                agent_system_prompt = str(cfg.dialogue.get("agent_prompt", "") or "")
                if scenario_idx is not None:
                    context_info = (
                        f"Topic（分類名稱，不要直接講出英文）：{topic}\n"
                        f"Scenario ID：{scenario_id}\n"
                        f"Scenario description：{scenario_description or ''}\n"
                    )
                    system_prompt = (agent_system_prompt + "\n\n" + context_info).strip() or None
                else:
                    system_prompt = agent_system_prompt.strip() or None
        except Exception:
            system_prompt = None

        if system_prompt:
            try:
                (dialogue_folder / "system_prompt.txt").write_text(system_prompt, encoding="utf-8")
            except Exception:
                pass

        for turn_metadata in turn_list:
            para_tags = turn_metadata.get("para_tags", {}) or {}
            emotion_ref_raw = turn_metadata.get("emotion_reference", None)
            if emotion_ref_raw is None:
                emotion_ref_list: List[str] = []
            elif isinstance(emotion_ref_raw, list):
                emotion_ref_list = emotion_ref_raw
            elif isinstance(emotion_ref_raw, str):
                emotion_ref_list = [emotion_ref_raw] if emotion_ref_raw else []
            else:
                emotion_ref_list = []

            turn_idx = int(turn_metadata.get("turn_idx", 0) or 0)
            clean_turn_path = dialogue_folder / "individual" / f"turn{turn_idx:02d}.wav"
            fx_turn_path = dialogue_folder / "individual" / f"turn{turn_idx:02d}_pause_overlap.wav"

            try:
                rel_audio_path = str(clean_turn_path.relative_to(data_root))
            except Exception:
                rel_audio_path = str(turn_metadata.get("audio_path", ""))

            row = {
                "conversation_id": conversation_id,
                "mode": export_mode,
                "topic": topic,
                "scenario": scenario_description,
                "final_txt_path": (
                    str(final_txt_path.relative_to(data_root))
                    if isinstance(final_txt_path, Path) and final_txt_path.exists()
                    else None
                ),
                "system_prompt": system_prompt,
                "turn_index": int(turn_metadata.get("turn_idx", 0)),
                "LLM1": llm_user,
                "LLM2": llm_agent,
                "speaker": turn_metadata.get("speaker", "unknown"),
                "text": turn_metadata.get("text", ""),
                "paralinguistic_info": {
                    "gender": para_tags.get("gender", None),
                    "pitch": para_tags.get("pitch", None),
                    "speed": para_tags.get("speed", None),
                    "volume": para_tags.get("volume", None),
                    "emotion": para_tags.get("emotion", None),
                },
                # Clean per-turn audio
                "audio_path": rel_audio_path,
                "audio": (
                    str(clean_turn_path) if clean_turn_path.exists() else str(turn_metadata.get("audio_path", ""))
                ),
                "audio_start": float(turn_metadata.get("audio_start", 0.0)),
                "audio_end": float(turn_metadata.get("audio_end", 0.0)),
                "audio_duration": float(turn_metadata.get("audio_duration", 0.0)),
                "full_dialogue_audio": str(full_dialogue_path),
                "emotion": turn_metadata.get("emotion", None),
                "reference": turn_metadata.get("speaker_reference", None),
                "reference_id": turn_metadata.get("speaker_reference_id", None),
                "reference_metadata": (
                    json.dumps(
                        turn_metadata.get("speaker_reference_metadata", None),
                        ensure_ascii=False,
                    )
                    if isinstance(turn_metadata.get("speaker_reference_metadata", None), dict)
                    else None
                ),
                "reference_age": (
                    turn_metadata.get("speaker_reference_metadata", {}).get("age")
                    if isinstance(turn_metadata.get("speaker_reference_metadata", None), dict)
                    else None
                ),
                "reference_gender": (
                    turn_metadata.get("speaker_reference_metadata", {}).get("gender_norm")
                    if isinstance(turn_metadata.get("speaker_reference_metadata", None), dict)
                    else None
                ),
                "reference_accent": (
                    turn_metadata.get("speaker_reference_metadata", {}).get("accents")
                    if isinstance(turn_metadata.get("speaker_reference_metadata", None), dict)
                    else None
                ),
                "emotion_reference": emotion_ref_list,
            }

            if include_pause_overlap:
                try:
                    rel_audio_path_fx = str(fx_turn_path.relative_to(data_root))
                except Exception:
                    rel_audio_path_fx = str(turn_metadata.get("audio_path_pause_overlap", ""))

                row.update(
                    {
                        "audio_path_pause_overlap": rel_audio_path_fx,
                        "audio_pause_overlap": (
                            str(fx_turn_path)
                            if fx_turn_path.exists()
                            else str(turn_metadata.get("audio_path_pause_overlap", ""))
                        ),
                        "audio_start_pause_overlap": (
                            float(turn_metadata.get("audio_start_pause_overlap"))
                            if turn_metadata.get("audio_start_pause_overlap", None) is not None
                            else None
                        ),
                        "audio_end_pause_overlap": (
                            float(turn_metadata.get("audio_end_pause_overlap"))
                            if turn_metadata.get("audio_end_pause_overlap", None) is not None
                            else None
                        ),
                        "audio_duration_pause_overlap": (
                            float(turn_metadata.get("audio_duration_pause_overlap"))
                            if turn_metadata.get("audio_duration_pause_overlap", None) is not None
                            else None
                        ),
                        "full_dialogue_audio_pause_overlap": str(full_dialogue_pause_overlap_path),
                    }
                )

            dataset_rows.append(row)

    logging.info(f"Creating HuggingFace dataset with {len(dataset_rows)} rows...")

    if not dataset_rows:
        logging.error("No dataset rows were collected; skipping save/push step")
        return

    features_dict: Dict[str, Any] = {
            "conversation_id": Value("string"),
            "mode": Value("string"),
            "topic": Value("string"),
            "scenario": Value("string"),
            "final_txt_path": Value("string"),
            "system_prompt": Value("string"),
            "turn_index": Value("int32"),
            "LLM1": Value("string"),
            "LLM2": Value("string"),
            "speaker": Value("string"),
            "text": Value("string"),
            "paralinguistic_info": {
                "gender": Value("string"),
                "pitch": Value("string"),
                "speed": Value("string"),
                "volume": Value("string"),
                "emotion": Value("string"),
            },
            # Clean block
            "audio_path": Value("string"),
            "audio": Value("string"),
            "audio_start": Value("float32"),
            "audio_end": Value("float32"),
            "audio_duration": Value("float32"),
            "full_dialogue_audio": Value("string"),
            "emotion": Value("string"),
            "reference": Value("string"),
            "reference_id": Value("string"),
            "reference_metadata": Value("string"),
            "reference_age": Value("string"),
            "reference_gender": Value("string"),
            "reference_accent": Value("string"),
            "emotion_reference": Sequence(Value("string")),
        }

    if include_pause_overlap:
        features_dict.update(
            {
                "audio_path_pause_overlap": Value("string"),
                "audio_pause_overlap": Value("string"),
                "audio_start_pause_overlap": Value("float32"),
                "audio_end_pause_overlap": Value("float32"),
                "audio_duration_pause_overlap": Value("float32"),
                "full_dialogue_audio_pause_overlap": Value("string"),
            }
        )

    features = Features(features_dict)

    dataset = Dataset.from_list(dataset_rows, features=features)

    # Optional cast to Audio
    try:
        from datasets import Audio as AudioFeature

        dataset = dataset.cast_column("audio", AudioFeature(sampling_rate=int(cfg.tts.get("sample_rate", 24000))))
        if include_pause_overlap and "audio_pause_overlap" in dataset.column_names:
            dataset = dataset.cast_column(
                "audio_pause_overlap", AudioFeature(sampling_rate=int(cfg.tts.get("sample_rate", 24000)))
            )
        logging.info("Successfully cast audio column to Audio feature type")
    except ImportError:
        logging.info(
            "Audio stored as file paths. To load audio, install torchcodec and use: dataset.cast_column('audio', Audio())"
        )
    except Exception as e:
        logging.warning(f"Could not cast audio column to Audio feature: {e}")

    dataset_save_path = output_dir / "dataset"
    dataset.save_to_disk(str(dataset_save_path))
    logging.info(f"Dataset saved to: {dataset_save_path}")

    json_path = output_dir / "dataset_metadata.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(dataset_rows, f, indent=2, ensure_ascii=False)
    logging.info(f"Metadata saved to: {json_path}")

    if cfg.huggingface.get("push_to_hub", False):
        hub_repo_id = cfg.huggingface.get("hub_repo_id")
        if hub_repo_id:
            push_kwargs = {
                "private": cfg.huggingface.get("hub_private", False),
            }
            hub_revision = cfg.huggingface.get("hub_revision", None)
            if hub_revision:
                push_kwargs["revision"] = str(hub_revision)
            try:
                logging.info(f"Pushing dataset to HuggingFace Hub: {hub_repo_id}")
                dataset.push_to_hub(hub_repo_id, **push_kwargs)
                logging.info("Dataset successfully pushed to HuggingFace Hub!")
            except Exception as e:
                logging.exception(f"Failed to push dataset to HuggingFace Hub: {e}")
                logging.info(f"Dataset remains saved locally at: {dataset_save_path}")
        else:
            logging.warning("push_to_hub is True but hub_repo_id not specified")

    logging.info("=" * 80)
    logging.info("HUGGINGFACE DATASET EXPORT COMPLETE (V2)")
    logging.info(f"Total turns exported: {len(dataset_rows)}")
    logging.info(f"Dataset location: {dataset_save_path}")
    logging.info("=" * 80)


def chat_completion(model_name: str, messages: List[Dict], **gen_kwargs) -> str:
    """
    Unified chat completion function that works with OpenRouter API.
    OpenRouter model names format: provider/model-name
    """
    api_key = _get_openrouter_api_key()
    if not api_key:
        raise EnvironmentError(
            "Missing OpenRouter API key. Set OPENROUTER_API_KEY (preferred) or OPENAI_API_KEY. "
            f"Tried loading .env at: {_DOTENV_PATH}"
        )

    validate_openrouter_model_name(model_name, where="chat_completion")

    base_url = "https://openrouter.ai/api/v1"
    timeout_env = (
        os.getenv("OPENROUTER_TIMEOUT_SECONDS")
        or os.getenv("OPENROUTER_TIMEOUT")
        or os.getenv("OPENAI_TIMEOUT")
        or "120"
    )
    try:
        timeout_seconds = float(timeout_env)
    except ValueError:
        timeout_seconds = 120.0

    # OpenAI SDK internal retries are disabled here because we implement our own
    # retry policy across transient errors (timeouts, rate limits, connection resets).
    #
    # NOTE: We also set Authorization explicitly via default_headers. On some
    # environments, users accidentally provide an empty key or headers get
    # stripped; this makes intent explicit and improves debuggability.
    default_headers = {"Authorization": f"Bearer {api_key}"}
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout_seconds,
        max_retries=0,
        default_headers=default_headers,
    )

    # Retry policy (env-configurable)
    # - OPENROUTER_REQUEST_MAX_RETRIES: total attempts (default 20)
    # - OPENROUTER_RETRY_BACKOFF_BASE: exponential base seconds (default 2)
    # - OPENROUTER_RETRY_BACKOFF_CAP: max sleep seconds (default 300)
    try:
        max_attempts = int(
            os.getenv("OPENROUTER_REQUEST_MAX_RETRIES")
            or os.getenv("OPENROUTER_MAX_RETRIES")
            or "20"
        )
    except Exception:
        max_attempts = 20
    max_attempts = max(1, max_attempts)

    try:
        backoff_base = float(os.getenv("OPENROUTER_RETRY_BACKOFF_BASE") or "2")
    except Exception:
        backoff_base = 2.0
    try:
        backoff_cap = float(os.getenv("OPENROUTER_RETRY_BACKOFF_CAP") or "300")
    except Exception:
        backoff_cap = 300.0

    # Import exception types lazily (keeps this file runnable even if SDK changes).
    try:
        from openai import (  # type: ignore
            APITimeoutError,
            APIConnectionError,
            RateLimitError,
            InternalServerError,
            APIStatusError,
            AuthenticationError,
        )
        transient_openai_errors = (APITimeoutError, APIConnectionError, RateLimitError, InternalServerError)
        status_error_cls = APIStatusError
        auth_error_cls = AuthenticationError
    except Exception:  # pragma: no cover
        transient_openai_errors = tuple()
        status_error_cls = None
        auth_error_cls = None

    def _is_transient(e: Exception) -> bool:
        # Known SDK exception types
        if transient_openai_errors and isinstance(e, transient_openai_errors):
            return True

        # Some errors are wrapped as APIStatusError; retry on common transient statuses.
        if status_error_cls is not None and isinstance(e, status_error_cls):
            try:
                code = int(getattr(e, "status_code", 0) or 0)
            except Exception:
                code = 0
            if code in (408, 409, 429, 500, 502, 503, 504):
                return True

        # Fallback: message/classname heuristics
        msg = (str(e) or "").lower()
        name = e.__class__.__name__.lower()
        if "timeout" in msg or "timed out" in msg or "readtimeout" in msg:
            return True
        if "connection" in msg and ("reset" in msg or "aborted" in msg or "refused" in msg):
            return True
        if "temporarily" in msg and "unavailable" in msg:
            return True
        if name in ("readtimeout", "connecttimeout"):
            return True
        return False

    last_err: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[{"role": m["role"], "content": m["content"]} for m in messages],
                **gen_kwargs,
            )
            content = getattr(completion.choices[0].message, "content", None)
            return (content or "").strip()
        except Exception as e:
            last_err = e

            # Fail-fast with a clearer message for auth issues.
            if auth_error_cls is not None and isinstance(e, auth_error_cls):
                raise EnvironmentError(
                    "OpenRouter authentication failed (401/403). "
                    "Check OPENROUTER_API_KEY (must be an OpenRouter key, not an OpenAI key), "
                    f"and confirm it is non-empty (len={len(api_key)})."
                ) from e
            if status_error_cls is not None and isinstance(e, status_error_cls):
                try:
                    code = int(getattr(e, "status_code", 0) or 0)
                except Exception:
                    code = 0
                if code in (401, 403):
                    raise EnvironmentError(
                        f"OpenRouter authentication failed (HTTP {code}). "
                        "Check OPENROUTER_API_KEY (OpenRouter key) and proxy settings; "
                        f"current key len={len(api_key)}."
                    ) from e

            if not _is_transient(e) or attempt >= max_attempts:
                raise
            sleep_s = min(backoff_cap, (backoff_base ** (attempt - 1)))
            sleep_s = float(sleep_s) + random.random() * 0.35
            logging.warning(
                "Transient LLM request error (%s/%s) for model=%s; sleeping %.1fs then retry. err=%s",
                attempt,
                max_attempts,
                model_name,
                sleep_s,
                repr(e),
            )
            time.sleep(sleep_s)

    # Should not reach here, but keep a clear failure mode.
    raise RuntimeError(f"LLM request failed after {max_attempts} attempts: {last_err}")


# ─────────────────────────────  SCENARIO GEN  ──────────────────────────────
def generate_scenarios(cfg):
    out_path = Path(cfg.scenario["out_file"])

    # Fail-fast: out_file must be .json (jsonl will be generated after)
    if out_path.suffix.lower() != ".json":
        raise ValueError(
            f"scenario.out_file 必須是 .json（例如 data_v2/scenarios.json），你現在是：{out_path}"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)

    topic = cfg.scenario["topic"]
    topic_desc = dict(cfg.get("domain_descriptions", {})).get(topic, "")
    system_prompt = cfg.scenario["prompt"].format(n=cfg.scenario["n"], topic=topic, topic_desc=topic_desc)
    base_msgs = [{"role": "system", "content": system_prompt}]

    gen_kwargs = get_gen_kwargs(cfg.scenario)

    max_retries = int(cfg.scenario.get("max_retries", 3) or 3)
    want_n = int(cfg.scenario.get("n", 1) or 1)
    data = None
    last_err: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        msgs = list(base_msgs)
        if attempt > 1:
            msgs.append(
                {
                    "role": "system",
                    "content": (
                        "上一次輸出可能格式不合格或內容太相似。請重新產生，並嚴格遵守以下要求：\n"
                        "1) 輸出必須是『純 JSON』，不要加任何說明文字、不要用 Markdown code fence。\n"
                        "2) 格式必須是：{\"scenarios\":[{\"scenario1\":{\"description\":\"...\"}}, ...]}\n"
                        "3) 每個 scenarios[i] 必須只有一個 key（scenario1/scenario2/...），且 value 內必須只有 description。\n"
                        "4) 不要出現巢狀重複 key，例如 scenario2 裡面再包一層 scenario2。\n"
                        "5) 每個 description 必須彼此不同。"
                    ),
                }
            )

        # LLM sometimes violates the "no newlines" constraint; parse defensively.
        text = chat_completion(cfg.scenario["model"], msgs, **gen_kwargs)
        logging.debug("Scenario raw LLM output (attempt %s/%s):\n%s", attempt, max_retries, text)

        # Archive raw scenario output as txt for debugging/repro.
        try:
            raw_dir = _stage_txt_root(cfg) / "scenario"
            raw_dir.mkdir(parents=True, exist_ok=True)
            suffix = f"_{attempt}" if max_retries > 1 else ""
            # Include out_file stem to avoid overwriting when batch scripts vary out_file.
            raw_name = f"{out_path.stem}_raw{suffix}.txt"
            (raw_dir / raw_name).write_text(text, encoding="utf-8")
        except Exception as e:
            logging.warning(f"Failed to archive scenario raw txt: {e}")

        try:
            parsed = _parse_llm_json_object(text)
        except Exception as e:
            last_err = e
            logging.warning(
                "Scenario JSON parse failed (attempt %s/%s): %s; retrying...",
                attempt,
                max_retries,
                repr(e),
            )
            continue

        try:
            parsed = _normalize_scenarios_json(parsed)
            _validate_scenarios_json(parsed, want_n=want_n)
        except ScenarioSchemaError as e:
            last_err = e
            logging.warning(
                "Scenario JSON schema invalid (attempt %s/%s): %s; retrying...",
                attempt,
                max_retries,
                str(e),
            )
            continue

        # If requesting multiple scenarios, avoid exact duplicate descriptions.
        if want_n > 1:
            try:
                entries = parsed.get("scenarios", []) if isinstance(parsed, dict) else []
                descs: List[str] = []
                for entry in entries:
                    if not isinstance(entry, dict) or not entry:
                        continue
                    _, inner = next(iter(entry.items()))
                    if isinstance(inner, dict):
                        d = inner.get("description")
                        if isinstance(d, str) and d.strip():
                            descs.append(d.strip())
                if descs and len(set(descs)) != len(descs):
                    logging.warning(
                        "Scenario descriptions are duplicated (attempt %s/%s); retrying...",
                        attempt,
                        max_retries,
                    )
                    continue
            except Exception:
                pass

        data = parsed
        break

    if data is None:
        msg = "Failed to generate valid scenarios JSON after retries"
        if last_err is not None:
            msg = f"{msg}: {last_err}"
        raise ValueError(msg)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logging.info("Saved %s → %s", type(data).__name__, out_path)


# ─────────────────────────────  DIALOGUE GEN V2 (DUAL LLM)  ──────────────────────────────
def generate_dialogues_v2(cfg):
    """
    VERSION 2: Uses TWO separate LLMs to create dialogue through interaction.
    """
    scen_path = Path(cfg.scenario["out_file"]).with_suffix(".jsonl")
    if not scen_path.exists():
        raise FileNotFoundError(
            f"Scenario JSONL not found: {scen_path}\n"
            f"Please run: python syn_ver2_zh.py (with stages including scenario)\n"
        )

    out_dir = Path(cfg.dialogue["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    scenarios = [json.loads(l) for l in scen_path.read_text(encoding="utf-8").splitlines()]

    user_model = cfg.dialogue["user_model"]
    agent_model = cfg.dialogue["agent_model"]
    with_emotion = bool(cfg.dialogue.get("with_emotion", False))
    emotion_rate = float(cfg.dialogue.get("emotion_rate", 0.6))
    emotion_list: List[str] = []
    user_system_prompt = cfg.dialogue["user_prompt"]
    agent_system_prompt = cfg.dialogue["agent_prompt"]
    if with_emotion:
        # The YAML prompts use {EMOTION_LIST} as a plain placeholder (not OmegaConf interpolation).
        # We replace it here with the actual comma-separated emotion names.
        emotion_list = list(cfg.get("emotion", []))
        emotion_names_str = "、".join(emotion_list)
        raw_user_emo = str(cfg.dialogue.get("user_prompt_with_emotion") or "")
        raw_agent_emo = str(cfg.dialogue.get("agent_prompt_with_emotion") or "")
        user_system_prompt_emo = raw_user_emo.replace("{EMOTION_LIST}", emotion_names_str) or user_system_prompt
        agent_system_prompt_emo = raw_agent_emo.replace("{EMOTION_LIST}", emotion_names_str) or agent_system_prompt
        logging.info("Emotion mode ON (rate=%.0f%%) — emotion list: %s", emotion_rate * 100, emotion_names_str)
    else:
        user_system_prompt_emo = user_system_prompt
        agent_system_prompt_emo = agent_system_prompt
    max_turns = cfg.dialogue.get("max_turns", 10)
    min_turns = cfg.dialogue.get("min_turns", 4)
    dialogues_per_scenario = cfg.dialogue.get("per_scenario", 3)

    # YAML-controlled generation params
    user_gen_kwargs = dict(cfg.dialogue.get("user_gen", {})) if cfg.dialogue.get("user_gen", None) else {}
    agent_gen_kwargs = dict(cfg.dialogue.get("agent_gen", {})) if cfg.dialogue.get("agent_gen", None) else {}

    logging.info("Generating dialogues with dual LLM system:")
    logging.info(f"  User Model: {user_model}")
    logging.info(f"  Agent Model: {agent_model}")
    logging.info(f"  Turn Range: {min_turns}-{max_turns} (randomized per dialogue)")
    logging.info(f"  Dialogues per Scenario: {dialogues_per_scenario}")

    topic_label = _extract_topic_label(str(cfg.scenario.get("topic", "unknown")))

    for scenario in tqdm(scenarios, desc="dialogues"):
        scenario_id = _canonicalize_scenario_id(scenario.get("id", ""))
        scenario_description = str(scenario.get("description", "")).strip()

        scenario_system_prompt = _read_scenario_system_prompt(
            cfg=cfg, topic_label=topic_label, scenario_id=scenario_id
        )

        for dialogue_idx in range(dialogues_per_scenario):
            dialogue_turns = []
            conversation_history = []

            # Provide topic + scenario to the LLMs as explicit context.
            # NOTE: topic is a category label; the dialogue should not explicitly mention the English domain label.
            context_info = (
                f"Topic（分類名稱，不要直接講出英文）：{topic_label}\n"
                f"Scenario ID：{scenario_id}\n"
                f"Scenario description：{scenario_description}\n\n"
            )

            num_turns = random.randint(min_turns, max_turns)
            dialogue_stem = _dialogue_file_stem(
                scenario_id,
                dialogue_idx + 1,
                dialogues_per_scenario,
                topic_label=topic_label,
            )
            logging.info(f"  Generating dialogue {dialogue_stem} with {num_turns} turns")

            for turn in range(num_turns):
                turn_use_emotion = with_emotion and (random.random() < emotion_rate)
                if with_emotion:
                    logging.debug("turn=%d turn_use_emotion=%s", turn, turn_use_emotion)
                if turn % 2 == 0:
                    backchannel_rate = float(cfg.dialogue.get("backchannel_rate", 0.2))
                    force_backchannel = (turn > 0) and (random.random() < backchannel_rate)
                    if turn == 0:
                        user_instruction = (
                            f"{context_info}"
                            f"現在是對話開始。請根據情境，產生使用者的第一句話。"
                            f"只輸出使用者要說的內容，不要加任何前綴或解釋文字。"
                        )
                    elif force_backchannel:
                        history_text = "\n".join(conversation_history)
                        user_instruction = (
                            f"{context_info}"
                            f"目前對話：\n{history_text}\n\n"
                            f"這一輪請模擬使用者剛聽完對方說話的自然短回應，十個字以內。"
                            f"依當下情境自由發揮，例如：表示聽到了（嗯、喔好、哦這樣）、需要時間處理（等我一下、我看看、讓我想想）、"
                            f"接受對方說法（好好、知道了、我懂）、有點意外（哦真的假的、欸這樣喔）、答不出來（我不確定欸、這個我不太清楚）……不限於此，配合情境自創。"
                            f"不要展開說明，不要加任何前綴或解釋文字。"
                        )
                    else:
                        history_text = "\n".join(conversation_history)
                        user_instruction = (
                            f"{context_info}"
                            f"目前對話：\n{history_text}\n\n"
                            f"請根據代理人上一句回覆，產生使用者的下一句回應。"
                            f"只輸出使用者要說的內容，不要加任何前綴或解釋文字。"
                        )

                    _usr_prompt = user_system_prompt_emo if turn_use_emotion else user_system_prompt
                    user_messages = [
                        {
                            "role": "system",
                            "content": (
                                (str(_usr_prompt or "") + "\n\n" + str(scenario_system_prompt or "")).strip()
                                if scenario_system_prompt
                                else str(_usr_prompt or "")
                            ),
                        },
                        {"role": "user", "content": user_instruction},
                    ]

                    max_retries = cfg.dialogue.get("max_retries", 3)
                    user_utterance = None

                    for retry in range(max_retries):
                        try:
                            user_utterance = chat_completion(
                                user_model,
                                user_messages,
                                **user_gen_kwargs,
                            ).strip()

                            user_utterance = re.sub(r"^User:\s*", "", user_utterance, flags=re.IGNORECASE)
                            if turn_use_emotion:
                                user_utterance = normalize_emotion_tag_prefix(user_utterance, emotion_list)
                            else:
                                user_utterance = strip_leading_emotion_tag(user_utterance)

                            if user_utterance:
                                break
                            else:
                                logging.warning(
                                    f"Empty response from user LLM at turn {turn}, retry {retry + 1}/{max_retries}"
                                )

                        except Exception as e:
                            logging.error(
                                f"Error in user LLM turn {turn} (retry {retry + 1}/{max_retries}): {e}"
                            )
                            if retry < max_retries - 1:
                                wait_time = 2 ** retry
                                logging.info(f"Waiting {wait_time}s before retry...")
                                time.sleep(wait_time)
                            else:
                                logging.error(f"Failed after {max_retries} retries, breaking dialogue...")

                    if user_utterance:
                        dialogue_turns.append(f"User: {user_utterance}")
                        conversation_history.append(f"User: {user_utterance}")
                    else:
                        logging.error(
                            f"Could not generate user response after {max_retries} retries, ending dialogue early"
                        )
                        break

                else:
                    history_text = "\n".join(conversation_history)
                    agent_instruction = (
                        f"{context_info}"
                        f"目前對話：\n{history_text}\n\n"
                        f"請回覆使用者上一句話：務實、可執行、自然口語。"
                        f"只輸出代理人要說的內容，不要加任何前綴或解釋文字。"
                    )

                    _agt_prompt = agent_system_prompt_emo if turn_use_emotion else agent_system_prompt
                    agent_messages = [
                        {
                            "role": "system",
                            "content": (
                                (str(_agt_prompt or "") + "\n\n" + str(scenario_system_prompt or "")).strip()
                                if scenario_system_prompt
                                else str(_agt_prompt or "")
                            ),
                        },
                        {"role": "user", "content": agent_instruction},
                    ]

                    max_retries = cfg.dialogue.get("max_retries", 3)
                    agent_utterance = None

                    for retry in range(max_retries):
                        try:
                            agent_utterance = chat_completion(
                                agent_model,
                                agent_messages,
                                **agent_gen_kwargs,
                            ).strip()

                            agent_utterance = re.sub(r"^Agent:\s*", "", agent_utterance, flags=re.IGNORECASE)
                            if turn_use_emotion:
                                agent_utterance = normalize_emotion_tag_prefix(agent_utterance, emotion_list)
                            else:
                                agent_utterance = strip_leading_emotion_tag(agent_utterance)

                            if agent_utterance:
                                break
                            else:
                                logging.warning(
                                    f"Empty response from agent LLM at turn {turn}, retry {retry + 1}/{max_retries}"
                                )

                        except Exception as e:
                            logging.error(
                                f"Error in agent LLM turn {turn} (retry {retry + 1}/{max_retries}): {e}"
                            )
                            if retry < max_retries - 1:
                                wait_time = 2 ** retry
                                logging.info(f"Waiting {wait_time}s before retry...")
                                time.sleep(wait_time)
                            else:
                                logging.error(f"Failed after {max_retries} retries, breaking dialogue...")

                    if agent_utterance:
                        dialogue_turns.append(f"Agent: {agent_utterance}")
                        conversation_history.append(f"Agent: {agent_utterance}")
                    else:
                        logging.error(
                            f"Could not generate agent response after {max_retries} retries, ending dialogue early"
                        )
                        break

            if dialogue_turns:
                raw_dialogue_text = "\n".join(dialogue_turns)
                dialogue_text = raw_dialogue_text if with_emotion else strip_emotion_tags_from_dialogue_text(raw_dialogue_text)
                output_file = out_dir / f"{dialogue_stem}.txt"
                output_file.write_text(dialogue_text, encoding="utf-8")
                logging.info(f"Saved dialogue: {output_file}")
                _archive_stage_txt(cfg, "dialogue", output_file)


# ────────────────────────────  OVERLAP INSERT  ─────────────────────────────
def insert_overlap(cfg):
    src = Path(cfg.dialogue["out_dir"])
    dst = Path(cfg.overlap["out_dir"])
    dst.mkdir(parents=True, exist_ok=True)
    files = _filter_files_by_current_scenarios(cfg, list(src.glob("*.txt")))

    rule_prompt = cfg.overlap.prompt
    gen_kwargs = get_gen_kwargs(cfg.overlap)

    stage_retries = int(cfg.overlap.get("max_retries", 15) or 15)

    for f in tqdm(files, desc="overlap"):
        txt = f.read_text("utf-8")
        messages = [
            {"role": "system", "content": rule_prompt},
            {"role": "user", "content": txt},
        ]
        new_txt = ""
        for attempt in range(1, stage_retries + 1):
            try:
                new_txt = chat_completion(cfg.overlap.model, messages, **gen_kwargs)
                if not str(new_txt).strip():
                    raise ValueError("Empty overlap output")
                break
            except Exception as e:
                if attempt >= stage_retries:
                    raise
                wait_time = min(30.0, float(2 ** (attempt - 1)) + random.random() * 0.35)
                logging.warning(
                    "overlap failed for %s (attempt %s/%s): %s; retrying in %.1fs",
                    f.name,
                    attempt,
                    stage_retries,
                    repr(e),
                    wait_time,
                )
                time.sleep(wait_time)
        # LLM occasionally concatenates pause marker with other content on the same line.
        # Normalize before saving to keep downstream parsing stable.
        new_lines = split_pause_marker_to_own_line(new_txt.splitlines())
        new_txt = "\n".join(new_lines)
        new_txt = strip_emotion_tags_from_dialogue_text(new_txt)
        out_file = dst / f.name
        out_file.write_text(new_txt, "utf-8")
        _archive_stage_txt(cfg, "overlap", out_file)


# ────────────────────────────────  Filler  ────────────────────────────────────
def insert_filler(cfg):
    src = Path(cfg.overlap["out_dir"])
    dst = Path(cfg.filler["out_dir"])
    dst.mkdir(parents=True, exist_ok=True)
    files = _filter_files_by_current_scenarios(cfg, list(src.glob("*.txt")))

    gen_kwargs = get_gen_kwargs(cfg.filler)

    stage_retries = int(cfg.filler.get("max_retries", 15) or 15)

    for f in tqdm(files, desc="filler"):
        txt = f.read_text("utf-8")
        messages = [
            {"role": "system", "content": cfg.filler.prompt},
            {"role": "user", "content": txt},
        ]
        new_txt = ""
        for attempt in range(1, stage_retries + 1):
            try:
                new_txt = chat_completion(cfg.filler.model, messages, **gen_kwargs)
                if not str(new_txt).strip():
                    raise ValueError("Empty filler output")
                break
            except Exception as e:
                if attempt >= stage_retries:
                    raise
                wait_time = min(30.0, float(2 ** (attempt - 1)) + random.random() * 0.35)
                logging.warning(
                    "filler failed for %s (attempt %s/%s): %s; retrying in %.1fs",
                    f.name,
                    attempt,
                    stage_retries,
                    repr(e),
                    wait_time,
                )
                time.sleep(wait_time)
        new_txt = format_headers_in_lines(new_txt)
        lines = split_pause_marker_to_own_line(new_txt.splitlines())
        new_txt = merge_overlapping_user_lines(lines)
        new_txt = "\n".join(new_txt)
        new_txt = strip_emotion_tags_from_dialogue_text(new_txt)
        out_file = dst / f.name
        out_file.write_text(new_txt, "utf-8")
        _archive_stage_txt(cfg, "filler", out_file)


# ────────────────────────────────  Judge  ────────────────────────────────────
def llm_judge(cfg):
    src = Path(cfg.filler["out_dir"])
    dst = Path(cfg.judge["out_dir"])
    dst.mkdir(parents=True, exist_ok=True)

    gen_kwargs = get_gen_kwargs(cfg.judge)

    all_files = sorted(src.glob("*.txt"))
    for scenario_id in _current_scenario_ids(cfg):
        group_files = [
            f
            for f in all_files
            if (_parse_dialogue_txt_stem(f.stem) or {}).get("scenario_id") == scenario_id
        ]
        results = []
        for f in tqdm(group_files, desc=f"Processing {scenario_id}"):
            txt = f.read_text(encoding="utf-8")
            messages = [
                {"role": "system", "content": cfg.judge.prompt},
                {"role": "user", "content": txt},
            ]
            response = chat_completion(cfg.judge.model, messages, **gen_kwargs)
            m = re.search(r"Score:\s*(\d+)", response)
            if not m:
                logging.warning("Judge output missing Score for %s", f.name)
                continue
            score = int(m.group(1))
            results.append({"file": f, "score": score})

        top_x = cfg.judge.top_x
        results_sorted = sorted(results, key=lambda x: x["score"], reverse=True)
        top_results = results_sorted[:top_x]
        new_folder = dst / f"top_{top_x}_{scenario_id}"
        new_folder.mkdir(parents=True, exist_ok=True)
        for item in top_results:
            src_file = item["file"]
            dst_file = new_folder / src_file.name
            dst_file.write_text(src_file.read_text(encoding="utf-8"), encoding="utf-8")


# ────────────────────────────────  TTS  ────────────────────────────────────
def tts_batch(cfg):
    """
        Batch process dialogue text files to generate TTS audio.
        This pipeline is BreezyVoice-only.
    """
    src = Path(str(cfg.tts.get("load_dir", "") or "")).expanduser()

    data_root = Path(cfg.get("data_root", Path(cfg.tts.wav_dir).parent))
    mode_name = str(cfg.get("mode_name", "v2")).strip() or "v2"
    topic = _extract_topic_label(str(cfg.scenario.get("topic", "unknown")))
    _suffix = str(cfg.get("topic_folder_suffix", "") or "")
    topic_dir = topic + _suffix

    # When HF export is enabled, enforce para-compatible wav folder layout.
    save_individual_turns = cfg.get("huggingface", {}).get("enabled", False)
    if save_individual_turns:
        wav_dir = data_root / mode_name / topic_dir / "wav"
    else:
        wav_dir = Path(cfg.tts.wav_dir)

    wav_dir.mkdir(parents=True, exist_ok=True)

    project_root = Path(__file__).resolve().parent

    spk_audio_dir = cfg.tts.get("spk_audio_dir", "XTTS_reference")
    if spk_audio_dir is not None:
        spk_audio_dir_path = Path(str(spk_audio_dir)).expanduser()
        if not spk_audio_dir_path.is_absolute():
            spk_audio_dir = str(project_root / spk_audio_dir_path)
    sample_rate = int(cfg.tts.get("sample_rate", 24000))

    with_emotion_tts = bool(cfg.dialogue.get("with_emotion", False))
    reference_source = (
        "eleven_lab_emotion" if with_emotion_tts
        else str(cfg.tts.get("reference_source", "common_voice_validated")).strip().lower()
    )
    user_ref_audio_cfg = cfg.tts.get("user_ref_audio", None)
    agent_ref_audio_cfg = cfg.tts.get("agent_ref_audio", None)

    cv_candidates: Optional[List[CVReferenceCandidate]] = None
    cv_gender_filter = cfg.tts.get("common_voice_gender_filter", None)
    if reference_source in {"common_voice_validated", "commonvoice_validated", "cv_validated"}:
        cv_tsv = cfg.tts.get(
            "common_voice_validated_tsv",
            "/home/jaylin0418/SpeechLab/ref_audio/cv-corpus-24.0-2025-12-05/zh-TW/validated.tsv",
        )
        cv_clips = cfg.tts.get(
            "common_voice_clips_dir",
            "/home/jaylin0418/SpeechLab/ref_audio/cv-corpus-24.0-2025-12-05/zh-TW/clips",
        )

        cv_tsv_path = Path(str(cv_tsv)).expanduser()
        if not cv_tsv_path.is_absolute():
            cv_tsv_path = project_root / cv_tsv_path

        cv_clips_path = Path(str(cv_clips)).expanduser()
        if not cv_clips_path.is_absolute():
            cv_clips_path = project_root / cv_clips_path

        cv_candidates = list(
            _load_common_voice_validated_candidates(
                tsv_path=str(cv_tsv_path),
                clips_dir=str(cv_clips_path),
            )
        )
        logging.info(
            "Using Common Voice validated.tsv reference pool: %s candidates (gender_filter=%s)",
            len(cv_candidates),
            cv_gender_filter,
        )

    # ── Eleven Lab emotion reference pool ─────────────────────────────────
    emotion_ref_pool: Optional[EmotionRefPool] = None
    if reference_source in {"eleven_lab_emotion", "elevenlabs_emotion"}:
        eleven_lab_dir = cfg.tts.get(
            "eleven_lab_emotion_dir",
            "/home/jaylin0418/SpeechLab/ref_audio/eleven_lab_emotion",
        )
        eleven_lab_path = Path(str(eleven_lab_dir)).expanduser()
        known_emotions = list(cfg.get("emotion", []))
        emotion_ref_pool = _load_emotion_reference_pool(eleven_lab_path, known_emotions)
        logging.info(
            "Using eleven_lab_emotion reference pool: %d speakers, %d emotion-speaker combos",
            len(emotion_ref_pool.speakers),
            len(emotion_ref_pool.emotion_speaker),
        )
        # Build wav-path → speaker metadata for the emotion pool
        _eleven_lab_speakers_cfg = cfg.tts.get("eleven_lab_speakers", {}) or {}
        # Load transcriptions.json (relative path → sentence text)
        _trans_path = eleven_lab_path / "transcriptions.json"
        _transcriptions: Dict[str, str] = {}
        if _trans_path.exists():
            try:
                import json as _json
                _transcriptions = _json.loads(_trans_path.read_text(encoding="utf-8"))
            except Exception as _e:
                logging.warning("Could not load transcriptions.json: %s", _e)
        emotion_speaker_meta_by_path: Dict[str, Dict[str, Any]] = {}
        for spk_name, wav_paths in emotion_ref_pool.speaker.items():
            spk_info = _eleven_lab_speakers_cfg.get(spk_name, {}) or {}
            for wav in wav_paths:
                # Infer gender from parent directory name (male / female)
                gender = wav.parent.name if wav.parent.name in {"male", "female"} else None
                # Build relative key matching transcriptions.json format
                try:
                    rel_key = wav.relative_to(eleven_lab_path).as_posix()
                except ValueError:
                    rel_key = wav.name
                sentence = _transcriptions.get(rel_key, "")
                emotion_speaker_meta_by_path[str(wav)] = {
                    "reference_id": wav.name,
                    "speaker_name": spk_name,
                    "sentence": sentence,
                    "gender": gender,
                    "voice_id": str(spk_info.get("voice_id", "")),
                    "display_name": str(spk_info.get("display_name", spk_name)),
                    "description": str(spk_info.get("description", "")),
                }

    error_log_path = wav_dir / "tts_errors.txt"

    breezy_repo_dir = cfg.tts.get("breezyvoice_repo_dir", "BreezyVoice")
    breezy_repo_dir_path = Path(str(breezy_repo_dir)).expanduser()
    if not breezy_repo_dir_path.is_absolute():
        breezy_repo_dir = str(project_root / breezy_repo_dir_path)
    breezy_python = _auto_detect_breezy_python(cfg.tts.get("breezy_python", "python"))
    breezy_model_path = cfg.tts.get("model_path", "MediaTek-Research/BreezyVoice-300M")
    user_prompt_text_transcription = cfg.tts.get("user_prompt_text_transcription", None)
    agent_prompt_text_transcription = cfg.tts.get("agent_prompt_text_transcription", None)
    wav_suffix = cfg.tts.get("wav_suffix", "BreezyVoice")

    if save_individual_turns:
        logging.info("HuggingFace export enabled - saving individual turn audio files + metadata")

    logging.info(
        "Using BreezyVoice backend (repo=%s, python=%s, model=%s)",
        breezy_repo_dir,
        breezy_python,
        breezy_model_path,
    )

    # Pick a workable txt input folder.
    # Default config points to txt/filler, but users may intentionally skip overlap/filler
    # and want to run TTS directly on txt/dialogue.
    stage_txt_root = data_root / mode_name / topic_dir / "txt"

    def _has_txt_files(p: Path) -> bool:
        try:
            return p.exists() and p.is_dir() and any(p.glob("*.txt"))
        except Exception:
            return False

    # If user specifies an explicit stage, honor it.
    input_stage = str(cfg.tts.get("input_stage", "") or "").strip().lower()
    stage_map = {
        "dialogue": stage_txt_root / "dialogue",
        "overlap": stage_txt_root / "overlap",
        "filler": stage_txt_root / "filler",
    }

    candidates: List[Tuple[str, Path]] = []
    if input_stage in stage_map:
        candidates.append((input_stage, stage_map[input_stage]))
    else:
        # Priority: configured load_dir first, then common stage folders.
        candidates.append(("load_dir", src))
        candidates.append(("filler", stage_txt_root / "filler"))
        candidates.append(("overlap", stage_txt_root / "overlap"))
        candidates.append(("dialogue", stage_txt_root / "dialogue"))

    txt_folder = None
    picked_label = ""
    for label, p in candidates:
        if _has_txt_files(p):
            txt_folder = p
            picked_label = label
            break

    if txt_folder is None:
        raise FileNotFoundError(
            "No input .txt files found for TTS. Checked: "
            + ", ".join([f"{lbl}={path}" for (lbl, path) in candidates])
        )

    if picked_label != "load_dir":
        logging.info(
            "TTS input txt folder fallback: load_dir=%s -> %s (%s)",
            src,
            txt_folder,
            picked_label,
        )
    else:
        logging.info("TTS input txt folder: %s", txt_folder)
    folder_wav_dir = wav_dir
    folder_wav_dir.mkdir(parents=True, exist_ok=True)

    # Default behavior: filter by scenario JSONL (if present) to avoid reprocessing
    # unrelated historical scripts. When running TTS-only on an existing dataset where
    # scenario JSONL is missing/unavailable, fall back to processing all txt files.
    filter_enabled = cfg.tts.get("filter_by_current_scenarios", None)
    if filter_enabled is None:
        try:
            scenario_jsonl = Path(cfg.scenario["out_file"]).with_suffix(".jsonl")
            filter_enabled = scenario_jsonl.exists()
        except Exception:
            filter_enabled = False

    def _num_sort_key(p):
        idx = _extract_scenario_index_from_name(p.name)
        return (idx is None, idx or 10**9)

    all_txt_files = sorted(txt_folder.glob("*.txt"), key=_num_sort_key)
    txt_files = (
        sorted(_filter_files_by_current_scenarios(cfg, all_txt_files), key=_num_sort_key)
        if filter_enabled
        else all_txt_files
    )
    if not txt_files:
        logging.warning("No txt files to TTS under %s (filter=%s)", txt_folder, filter_enabled)
        return

    # Per-txt progress logging (for SLURM monitoring).
    # Written as TSV so it can be tailed and grepped easily.
    progress_dir_env = os.getenv("SYN_TTS_PROGRESS_DIR")
    if progress_dir_env:
        progress_dir = Path(progress_dir_env).expanduser()
    else:
        run_root_val = str(cfg.get("run_root", "") or "").strip()
        progress_dir = Path(run_root_val).expanduser() / "logs_tts" if run_root_val else (project_root / "logs_tts")

    progress_path = progress_dir / "progress_tts_txt.tsv"
    if not progress_path.exists():
        _append_tsv_line_locked(
            progress_path,
            "ts\trank\tworld\tgpu\ttopic\ttxt\tdialogue_id\tstatus\treturncode\tdone\ttotal\tgenerated\tskipped\tfailed\telapsed_sec\tlog_or_err",
        )

    slurm_rank = os.getenv("SLURM_PROCID", "")
    slurm_world = os.getenv("SLURM_NTASKS", "")
    gpu_visible = os.getenv("CUDA_VISIBLE_DEVICES", "")
    total_txt = len(txt_files)
    start_time = time.time()
    done = 0
    generated = 0
    skipped = 0
    failed = 0

    for txt_file in tqdm(txt_files, desc="tts"):
        stem = txt_file.stem
        parsed_stem = _parse_dialogue_txt_stem(stem)
        if parsed_stem:
            scenario_idx = str(parsed_stem["scenario_idx"])
            dialogue_idx = parsed_stem.get("dialogue_idx")
            dialogue_id = (
                f"{topic}_{mode_name}_{scenario_idx}_{dialogue_idx}"
                if dialogue_idx
                else f"{topic}_{mode_name}_{scenario_idx}"
            )
        else:
            dialogue_id = stem

        if save_individual_turns:
            # Para-compatible output layout:
            #   wav_dir/<dialogue_id>/full.wav
            #   wav_dir/<dialogue_id>/individual/turnXX.wav
            #   wav_dir/<dialogue_id>/individual/turn_metadata.json
            dialogue_folder = folder_wav_dir / dialogue_id
            dialogue_folder.mkdir(parents=True, exist_ok=True)
            wav_file_path = dialogue_folder / "full.wav"
            wav_file_with_effects_path = dialogue_folder / "full_with_overlap_and_pause.wav"
            dialogue_turn_dir = dialogue_folder / "individual"
            metadata_file_path = dialogue_turn_dir / "turn_metadata.json"

            # If pause/overlap mix is disabled (or auto-disabled), full_with_overlap_and_pause.wav may not exist.
            fx_policy = cfg.tts.get("write_pause_overlap_mix", None)
            needs_fx_file = bool(fx_policy) if fx_policy is not None else False
            if wav_file_path.exists() and metadata_file_path.exists() and (not needs_fx_file or wav_file_with_effects_path.exists()):
                logging.info(f"Skipping {txt_file.name} - output already exists")
                done += 1
                skipped += 1
                elapsed = time.time() - start_time
                _append_tsv_line_locked(
                    progress_path,
                    f"{int(time.time())}\t{slurm_rank}\t{slurm_world}\t{gpu_visible}\t{topic}\t{txt_file.name}\t{dialogue_id}\tSKIP\t0\t{done}\t{total_txt}\t{generated}\t{skipped}\t{failed}\t{elapsed:.1f}\toutput_exists",
                )
                continue
        else:
            wav_file_path = folder_wav_dir / f"{dialogue_id}_{wav_suffix}.wav"
            dialogue_turn_dir = None
            metadata_file_path = None

            if wav_file_path.exists():
                print(f"Skipping {txt_file.name} - wav file already exists")
                done += 1
                skipped += 1
                elapsed = time.time() - start_time
                _append_tsv_line_locked(
                    progress_path,
                    f"{int(time.time())}\t{slurm_rank}\t{slurm_world}\t{gpu_visible}\t{topic}\t{txt_file.name}\t{dialogue_id}\tSKIP\t0\t{done}\t{total_txt}\t{generated}\t{skipped}\t{failed}\t{elapsed:.1f}\toutput_exists",
                )
                continue

        try:
            text = txt_file.read_text("utf-8")
            script = []
            for _, line in enumerate(text.strip().splitlines()):
                line = line.strip()
                if not line:
                    continue
                if ":" in line:
                    role, content = line.split(":", 1)
                    script.append((role.strip(), content.strip()))

            logging.info(f"Generating TTS for {txt_file.name}...")

            speaker_meta_by_path = None
            user_ref_audio = user_ref_audio_cfg
            agent_ref_audio = agent_ref_audio_cfg

            mix_cfg = None
            try:
                mix_cfg = cfg.tts.get("mix", None)
            except Exception:
                mix_cfg = None

            inter_turn_silence_sec = 0.25
            overlap_shift_sec_min = 0.6
            overlap_shift_sec_max = 1.0
            overlap_pause_sec = None
            if isinstance(mix_cfg, dict) or OmegaConf.is_config(mix_cfg):
                try:
                    inter_turn_silence_sec = float(mix_cfg.get("inter_turn_silence_sec", inter_turn_silence_sec))
                except Exception:
                    pass
                try:
                    overlap_shift_sec_min = float(mix_cfg.get("overlap_shift_sec_min", overlap_shift_sec_min))
                except Exception:
                    pass
                try:
                    overlap_shift_sec_max = float(mix_cfg.get("overlap_shift_sec_max", overlap_shift_sec_max))
                except Exception:
                    pass

                raw_overlap_pause = mix_cfg.get("overlap_pause_sec", None)
                if raw_overlap_pause not in (None, "", "null", "None"):
                    try:
                        overlap_pause_sec = float(raw_overlap_pause)
                    except Exception:
                        overlap_pause_sec = None

            user_speaker_name: Optional[str] = None
            agent_speaker_name: Optional[str] = None

            if emotion_ref_pool is not None:
                user_speaker_name, agent_speaker_name = _pick_emotion_speakers(emotion_ref_pool)
                speaker_meta_by_path = emotion_speaker_meta_by_path
            elif cv_candidates is not None:
                a = _pick_cv_reference(cv_candidates, gender_filter=cv_gender_filter)
                b = _pick_cv_reference(cv_candidates, gender_filter=cv_gender_filter)
                if len(cv_candidates) > 1 and a.audio_path == b.audio_path:
                    for _ in range(3):
                        b = _pick_cv_reference(cv_candidates, gender_filter=cv_gender_filter)
                        if b.audio_path != a.audio_path:
                            break

                user_ref_audio = a.audio_path
                agent_ref_audio = b.audio_path
                speaker_meta_by_path = {
                    a.audio_path: {**(a.metadata or {}), "reference_id": a.reference_id, "speaker_name": a.reference_id},
                    b.audio_path: {**(b.metadata or {}), "reference_id": b.reference_id, "speaker_name": b.reference_id},
                }

            turn_metadata = BreezyVoice_gen(
                script,
                wav_file_path,
                breezy_repo_dir=breezy_repo_dir,
                breezy_python=breezy_python,
                model_path=breezy_model_path,
                spk_audio_dir=spk_audio_dir,
                user_ref_audio=user_ref_audio,
                agent_ref_audio=agent_ref_audio,
                emotion_ref_pool=emotion_ref_pool,
                user_speaker_name=user_speaker_name,
                agent_speaker_name=agent_speaker_name,
                user_prompt_text_transcription=user_prompt_text_transcription,
                agent_prompt_text_transcription=agent_prompt_text_transcription,
                sample_rate=sample_rate,
                save_individual_turns=save_individual_turns,
                turn_output_dir=dialogue_turn_dir,
                speaker_meta_by_path=speaker_meta_by_path,
                cuda_visible_devices=cfg.tts.get("cuda_visible_devices", None),
                inter_turn_silence_sec=inter_turn_silence_sec,
                overlap_shift_sec_min=overlap_shift_sec_min,
                overlap_shift_sec_max=overlap_shift_sec_max,
                overlap_pause_sec=overlap_pause_sec,
                write_pause_overlap_mix=cfg.tts.get("write_pause_overlap_mix", None),
            )

            if save_individual_turns and metadata_file_path is not None:
                dialogue_turn_dir.mkdir(parents=True, exist_ok=True)
                with open(metadata_file_path, "w", encoding="utf-8") as f:
                    json.dump(turn_metadata or [], f, indent=2, ensure_ascii=False)
                logging.info(f"Saved turn metadata to: {metadata_file_path}")

            done += 1
            generated += 1
            elapsed = time.time() - start_time
            _append_tsv_line_locked(
                progress_path,
                f"{int(time.time())}\t{slurm_rank}\t{slurm_world}\t{gpu_visible}\t{topic}\t{txt_file.name}\t{dialogue_id}\tOK\t0\t{done}\t{total_txt}\t{generated}\t{skipped}\t{failed}\t{elapsed:.1f}\t{wav_file_path}",
            )

        except Exception as e:
            print(f"Error processing {txt_file.name}: {str(e)}")
            with open(error_log_path, "a", encoding="utf-8") as error_file:
                error_file.write(f"{txt_file.name}: {str(e)}\n")

            done += 1
            failed += 1
            elapsed = time.time() - start_time
            _append_tsv_line_locked(
                progress_path,
                f"{int(time.time())}\t{slurm_rank}\t{slurm_world}\t{gpu_visible}\t{topic}\t{txt_file.name}\t{dialogue_id}\tFAIL\t1\t{done}\t{total_txt}\t{generated}\t{skipped}\t{failed}\t{elapsed:.1f}\t{repr(e).replace(chr(9),' ')}",
            )
            continue


# ─────────────────────────────  ORCHESTRATOR  ──────────────────────────────
@dataclass
class PipelineConfig:
    device: str = "cuda"
    seed: Optional[int] = 3407

    # Per-run output root. If not set, a folder named syn_data_zh_{timestamp} will be created.
    run_root: Optional[str] = None

    # Match para-style root layout: {data_root}/{mode_name}/{topic}/...
    data_root: str = "data_v2"
    mode_name: str = "v2"
    topic_folder_suffix: str = ""

    batch_run: OmegaConf = OmegaConf.create(
        {
            "topic": "Planning",
            "batch_size": 1,
            "total_count": 1,
            "final_huggingface_only": True,
        }
    )

    multi_topic_run: OmegaConf = OmegaConf.create(
        {
            "output_root_base": "/work/jaylin0418",
            "workers": 2,
            "per_topic_count": 130,
            "batch_size": 10,
            "final_huggingface_only": True,
            "topics": [
                "Art",
                "Books",
                "Cars",
                "Celebrities",
                "Coding",
                "Cooking",
                "Education",
                "Events",
                "Fashion",
                "Fitness",
                "Finance",
                "Food",
                "Gaming",
                "Gardening",
                "Health",
                "History",
                "Hobbies",
                "Holidays",
                "Home",
                "Languages",
                "Makeup",
                "Movies",
                "Music",
                "Nature",
                "News",
                "Pets",
                "Philosophy",
                "Photography",
                "Podcasts",
                "Politics",
                "Relationships",
                "Science",
                "Shopping",
                "Social Media",
                "Spirituality",
                "Sports",
                "Technology",
                "Traditions",
                "Travel",
                "Weather",
                "Work",
            ],
        }
    )

    stages: List[str] = field(
        default_factory=lambda: ["scenario", "system_prompt", "dialogue", "overlap", "filler", "judge", "tts"]
    )

    emotion: List[str] = field(default_factory=lambda: [])

    domain_descriptions: OmegaConf = OmegaConf.create({})

    scenario: OmegaConf = OmegaConf.create(
        {
            "model": "openai/gpt-4o-mini",
            "n": 200,
            "out_file": "data_v2/scenarios.json",
            "topic": "Planning",
            "prompt": "",
            "gen": {"max_tokens": 1800, "temperature": 0.9, "top_p": 0.9},
        }
    )

    # Stage 1.5: scenario-level system prompt generation.
    system_prompt: OmegaConf = OmegaConf.create(
        {
            "model": "openai/gpt-4o-mini",
            "n": 5,
            "prompt": "",
            "gen": {"max_tokens": 1024, "temperature": 0.9, "top_p": 0.9},
        }
    )

    dialogue: OmegaConf = OmegaConf.create(
        {
            "user_model": "meta-llama/llama-3.3-70b-instruct",
            "agent_model": "openai/gpt-4o-mini",
            "user_prompt": "",
            "agent_prompt": "",
            "out_dir": "data_v2/dialogue_multi_txt",
            "min_turns": 6,
            "max_turns": 10,
            "per_scenario": 1,
            "max_retries": 3,
            "user_gen": {"max_tokens": 320, "temperature": 0.9, "top_p": 0.9},
            "agent_gen": {"max_tokens": 420, "temperature": 0.8, "top_p": 0.9},
        }
    )

    overlap: OmegaConf = OmegaConf.create(
        {
            "model": "openai/gpt-4o-mini",
            "out_dir": "data_v2/overlap_multi_txt",
            "prompt": "",
            "gen": {"max_tokens": 1200, "temperature": 0.8},
        }
    )

    filler: OmegaConf = OmegaConf.create(
        {
            "model": "openai/gpt-4o-mini",
            "out_dir": "data_v2/filler_multi_txt",
            "prompt": "",
            "gen": {"max_tokens": 1400, "temperature": 0.7},
        }
    )

    judge: OmegaConf = OmegaConf.create(
        {
            "model": "openai/gpt-4o-mini",
            "out_dir": "data_v2/judge_multi_txt",
            "top_x": 3,
            "prompt": "",
            "gen": {"max_tokens": 1200, "temperature": 0.2},
        }
    )

    tts: OmegaConf = OmegaConf.create(
        {
            "backend": "breezyvoice",
            "breezyvoice_repo_dir": "BreezyVoice",
            "breezy_python": "python",
            "model_path": "MediaTek-Research/BreezyVoice-300M",
            "user_prompt_text_transcription": None,
            "agent_prompt_text_transcription": None,
            "sample_rate": 24000,
            "wav_suffix": "BreezyVoice",
            "spk_audio_dir": "XTTS_reference",
            "wav_dir": "data_v2/wav",
            "load_dir": "data_v2/filler_multi_txt",
            "user_ref_audio": None,
            "agent_ref_audio": None,
        }
    )

    huggingface: OmegaConf = OmegaConf.create(
        {
            "enabled": False,
            "mode_override": "normal",
            "dataset_name": "synthetic_dialogues_v2",
            "output_dir": "data_v2/huggingface_dataset",
            "push_to_hub": False,
            "hub_repo_id": None,
            "hub_private": False,
        }
    )


class Pipeline:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg

    def run(self):
        # YAML-first reproducibility for local randomness
        set_global_seed(self.cfg.get("seed", None))

        st = set(self.cfg.stages)

        if "scenario" in st:
            print("Scenario Creating...")
            generate_scenarios(self.cfg)
            original_path = Path(self.cfg.scenario["out_file"])
            scenario_txt_dir = original_path.parent.parent / "scenario_txt"
            topic_label = _extract_topic_label(str(self.cfg.scenario.get("topic", "unknown")))
            start_index = _next_topic_scenario_index(self.cfg, topic_label)
            convert_nested_json_to_jsonl(
                original_path,
                original_path.with_suffix(".jsonl"),
                scenario_txt_dir=scenario_txt_dir,
                topic_label=topic_label,
                start_index=start_index,
                cfg=self.cfg,
            )

        if "system_prompt" in st:
            print("System prompt Generating (Stage 1.5)...")
            generate_system_prompts(self.cfg)

        if "dialogue" in st:
            print("Dialogue Generating (V2 - Dual LLM)...")
            generate_dialogues_v2(self.cfg)

        if "overlap" in st:
            print("Inserting overlap dialogue...")
            insert_overlap(self.cfg)

        if "filler" in st:
            print("Inserting filler dialogue...")
            insert_filler(self.cfg)

        if "judge" in st:
            print("Judge dialogue...")
            llm_judge(self.cfg)

        if "tts" in st:
            print("Speech dialogue Generating...")
            tts_batch(self.cfg)

        if "huggingface" in st:
            print("Exporting to HuggingFace Dataset...")
            export_to_huggingface(self.cfg)


# ─────────────────────────────  CLI ENTRY  ────────────────────────────────
def main():  # pragma: no cover
    @hydra.main(config_path="conf", config_name="base_v2_breezy", version_base=None)
    def _run(cfg):
        if not isinstance(cfg, PipelineConfig):
            cfg = OmegaConf.merge(OmegaConf.structured(PipelineConfig), cfg)

        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

        prepare_run_output_root(cfg)
        logging.info("Run output root: %s", cfg.get("run_root"))
        Pipeline(cfg).run()

    print("Finish Config Matching")
    _run()


if __name__ == "__main__":
    main()
