#!/usr/bin/env python3
"""Build a global YAML index of models created by experiment manifests."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "runs"
EXPORTED_PARAMS_DIR = REPO_ROOT / "exported_params"
DEFAULT_EXPORTED_MANIFEST = EXPORTED_PARAMS_DIR / "manifest.csv"
DEFAULT_OUTPUT = REPO_ROOT / "registry" / "model_index.yaml"

PHASE_DIRS = {
    "f02": "f02_events",
    "f03": "f03_windows",
    "f04": "f04_targets",
    "f05": "f05_modeling",
    "f06": "f06_quant",
}

F05_INDEX_METRIC_KEYS = [
    "val_precision",
    "val_recall",
    "val_f1",
    "val_pr_auc",
    "test_precision",
    "test_recall",
    "test_f1",
    "test_pr_auc",
    "test_tp",
    "test_tn",
    "test_fp",
    "test_fn",
    "tp",
    "tn",
    "fp",
    "fn",
]


def normalize_path(value: str | Path) -> str:
    return str(value).replace("\\", "/")


def relative_path(path: Path) -> str:
    try:
        return normalize_path(path.relative_to(REPO_ROOT))
    except ValueError:
        return normalize_path(path)


def empty_index(source_manifests: list[Path] | None = None) -> dict[str, Any]:
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_manifests": [relative_path(path) for path in (source_manifests or [])],
        "summary": {
            "n_models": 0,
            "n_measures": 0,
            "n_target_groups": 0,
            "pipelines": {},
        },
        "models_by_measure": {},
    }


def find_manifest_paths(runs_dir: Path = RUNS_DIR) -> list[Path]:
    if not runs_dir.exists():
        print("No runs found.")
        return []
    manifests = sorted(runs_dir.glob("*/manifest.csv"))
    if not manifests:
        print("No manifests found.")
    return manifests


def read_manifest_rows(manifest_paths: list[Path], include_failed: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for manifest_path in manifest_paths:
        with manifest_path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                status = str(row.get("status", ""))
                if status == "COMPLETED" or (include_failed and status == "FAILED"):
                    item = dict(row)
                    item["_source_manifest"] = manifest_path
                    rows.append(item)
    return rows


def phase_variant_lookup(exported: dict[str, Any], phase: str) -> dict[str, dict[str, Any]]:
    variants = exported.get("variants") or {}
    phase_items = variants.get(phase) if isinstance(variants, dict) else []
    if not isinstance(phase_items, list):
        return {}
    return {
        str(item.get("variant")): item
        for item in phase_items
        if isinstance(item, dict) and item.get("variant")
    }


def exported_yaml_path(row: dict[str, Any]) -> Path | None:
    yaml_text = str(row.get("yaml_path", "")).strip()
    if yaml_text:
        yaml_path = Path(yaml_text)
        if yaml_path.exists():
            return yaml_path
    pipeline = str(row.get("pipeline", "")).strip()
    fallback = EXPORTED_PARAMS_DIR / f"{pipeline}.yaml"
    return fallback if fallback.exists() else None


def repo_path_for_exported_row(exported: dict[str, Any]) -> str | None:
    repo_path = str(exported.get("repo_path", "")).strip()
    return normalize_path(repo_path) if repo_path else None


def temporal_id_from_row(row: dict[str, Any]) -> dict[str, Any]:
    parts = str(row.get("temporal_id", "")).lower().split("_")
    parsed: dict[str, Any] = {}
    for part in parts:
        if part.startswith("ow"):
            parsed["OW"] = part[2:]
        elif part.startswith("pw"):
            parsed["PW"] = part[2:]
        elif part.startswith("lt"):
            parsed["LT"] = part[2:]
    return parsed


def exported_row_to_index_row(row: dict[str, Any], exported: dict[str, Any], manifest_path: Path) -> dict[str, Any]:
    f02 = phase_variant_lookup(exported, "f02").get(str(row.get("f02_variant", "")), {})
    f03 = phase_variant_lookup(exported, "f03").get(str(row.get("f03_variant", "")), {})
    f05 = phase_variant_lookup(exported, "f05").get(str(row.get("f05_variant", "")), {})
    f02_params = f02.get("params") if isinstance(f02.get("params"), dict) else {}
    f03_params = f03.get("params") if isinstance(f03.get("params"), dict) else {}
    f05_params = f05.get("params") if isinstance(f05.get("params"), dict) else {}
    temporal = temporal_id_from_row(row)

    bands = f02_params.get("bands") if isinstance(f02_params.get("bands"), list) else []
    item = dict(row)
    item.update({
        "_source_manifest": manifest_path,
        "status": "COMPLETED",
        "run_id": "exported_params",
        "repo_path": repo_path_for_exported_row(exported),
        "bands": json.dumps(bands),
        "OW": f03_params.get("OW", temporal.get("OW")),
        "PW": f03_params.get("PW", temporal.get("PW")),
        "LT": f03_params.get("LT", temporal.get("LT")),
        "window_strategy": f03_params.get("window_strategy"),
        "nan_mode": f03_params.get("nan_mode") or f02_params.get("nan_mode"),
        "model_family": f05_params.get("model_family"),
        "seed": f05_params.get("seed"),
        "imbalance_strategy": f05_params.get("imbalance_strategy"),
        "imbalance_max_maj": f05_params.get("imbalance_max_majority_samples"),
        "deduplication_mode": f05_params.get("deduplication_mode") or "none",
        "output_path": "",
    })
    return item


def read_exported_manifest_rows(manifest_path: Path = DEFAULT_EXPORTED_MANIFEST) -> list[dict[str, Any]]:
    if not manifest_path.exists():
        return []

    rows: list[dict[str, Any]] = []
    yaml_cache: dict[Path, dict[str, Any]] = {}
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            yaml_path = exported_yaml_path(row)
            if not yaml_path:
                continue
            exported = yaml_cache.setdefault(yaml_path, load_yaml(yaml_path))
            rows.append(exported_row_to_index_row(dict(row), exported, manifest_path))
    return rows


def parse_scalar(value: Any) -> Any:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        number = float(text)
    except ValueError:
        return text
    if number.is_integer():
        return int(number)
    return number


def parse_bands(value: Any) -> list[Any]:
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        try:
            parsed = yaml.safe_load(text)
        except yaml.YAMLError:
            return [item.strip() for item in text.split(",") if item.strip()]
    if isinstance(parsed, list):
        return [parse_scalar(item) for item in parsed]
    return []


def compact_value(value: Any) -> str:
    parsed = parse_scalar(value)
    if parsed is None:
        return "unknown"
    if isinstance(parsed, float) and parsed.is_integer():
        return str(int(parsed))
    return str(parsed).replace(".", "p").replace(" ", "_")


def clean_key_part(value: Any, default: str = "unknown") -> str:
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    return text.replace(" ", "_").replace("/", "_").replace("\\", "_").replace(":", "_")


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def walk_values(data: Any, key: str) -> Any:
    if isinstance(data, dict):
        if key in data:
            return data[key]
        for value in data.values():
            found = walk_values(value, key)
            if found is not None:
                return found
    elif isinstance(data, list):
        for item in data:
            found = walk_values(item, key)
            if found is not None:
                return found
    return None


def first_available(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value not in (None, ""):
            return value
    return None


def build_target_group(row: dict[str, Any]) -> str:
    return (
        f"{clean_key_part(row.get('measure'))}_"
        f"{clean_key_part(row.get('direction'))}_"
        f"{compact_value(row.get('effective_threshold'))}"
    )


def build_temporal_config_id(row: dict[str, Any]) -> str:
    return (
        f"OW{compact_value(row.get('OW'))}_"
        f"PW{compact_value(row.get('PW'))}_"
        f"LT{compact_value(row.get('LT'))}_"
        f"{clean_key_part(row.get('window_strategy'))}"
    )


def build_bands_key(row: dict[str, Any]) -> str:
    if str(row.get("pipeline", "")) == "ts":
        return "no_bands"
    return f"bands_{clean_key_part(row.get('bands_id'))}"


def build_model_key(row: dict[str, Any]) -> str:
    pipeline = clean_key_part(row.get("pipeline"))
    measure = clean_key_part(row.get("measure"))
    direction = clean_key_part(row.get("direction"))
    threshold = f"thr{compact_value(row.get('effective_threshold'))}"
    temporal = build_temporal_config_id(row)
    model_family = clean_key_part(row.get("model_family"), "")
    deduplication_mode = clean_key_part(row.get("deduplication_mode") or "none")
    seed = f"seed{compact_value(row.get('seed'))}"
    parts = [pipeline, measure, direction, threshold]
    if pipeline != "ts":
        parts.append(build_bands_key(row))
    parts.extend([temporal, model_family, f"dedup_{deduplication_mode}", seed])
    return "|".join(parts)


def build_variant_paths(row: dict[str, Any]) -> dict[str, str | None]:
    repo_path = str(row.get("repo_path", "")).strip()
    paths: dict[str, str | None] = {}
    for phase, phase_dir in PHASE_DIRS.items():
        variant = str(row.get(f"{phase}_variant", "")).strip()
        if repo_path and variant:
            paths[phase] = normalize_path(Path(repo_path) / "executions" / phase_dir / variant)
        else:
            paths[phase] = None
    return paths


def build_local_output_path(row: dict[str, Any]) -> str | None:
    existing = str(row.get("output_path", "")).strip()
    if existing:
        return normalize_path(existing)
    run_id = str(row.get("run_id", "")).strip()
    pipeline = str(row.get("pipeline", "")).strip()
    job_key = str(row.get("job_key", "")).strip()
    if run_id and pipeline and job_key:
        return normalize_path(Path("runs") / run_id / "outputs" / pipeline / job_key)
    return None


def resolve_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def read_f05_metric_snapshot(row: dict[str, Any]) -> dict[str, Any]:
    candidates: list[Path] = []
    output_path = resolve_path(build_local_output_path(row))
    if output_path:
        candidates.append(output_path / "f05" / "outputs.yaml")

    variant_paths = build_variant_paths(row)
    f05_path = resolve_path(variant_paths.get("f05"))
    if f05_path:
        candidates.append(f05_path / "outputs.yaml")

    outputs: dict[str, Any] = {}
    for candidate in candidates:
        outputs = load_yaml(candidate)
        if outputs:
            break
    if not outputs:
        return {}

    metrics = {key: walk_values(outputs, key) for key in F05_INDEX_METRIC_KEYS}
    metrics.update({
        "test_fn_effective": first_available(metrics, "test_fn", "fn"),
        "test_pr_auc_effective": first_available(metrics, "test_pr_auc"),
    })
    return {key: value for key, value in metrics.items() if value not in (None, "")}


def parse_timestamp(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def choose_latest_duplicate(current: dict[str, Any] | None, candidate: dict[str, Any]) -> dict[str, Any]:
    if current is None:
        return candidate

    current_completed = current.get("status") == "COMPLETED"
    candidate_completed = candidate.get("status") == "COMPLETED"
    if current_completed != candidate_completed:
        return candidate if candidate_completed else current

    current_finished = parse_timestamp(current.get("finished_at"))
    candidate_finished = parse_timestamp(candidate.get("finished_at"))
    if current_finished and candidate_finished and current_finished != candidate_finished:
        return candidate if candidate_finished > current_finished else current
    if candidate_finished and not current_finished:
        return candidate
    if current_finished and not candidate_finished:
        return current

    return candidate if str(candidate.get("run_id", "")) > str(current.get("run_id", "")) else current


def row_to_model_entry(row: dict[str, Any], model_key: str) -> dict[str, Any]:
    bands = parse_bands(row.get("bands"))
    entry: dict[str, Any] = {}
    if str(row.get("pipeline", "")) != "ts":
        entry["bands_id"] = row.get("bands_id") or None
        entry["bands"] = bands
    entry.update({
        "model_key": model_key,
        "run_id": row.get("run_id") or None,
        "job_key": row.get("job_key") or None,
        "status": row.get("status") or None,
        "repo_path": normalize_path(row.get("repo_path", "")) if row.get("repo_path") else None,
        "local_output_path": build_local_output_path(row),
        "variants": {
            "f02": row.get("f02_variant") or None,
            "f03": row.get("f03_variant") or None,
            "f04": row.get("f04_variant") or None,
            "f05": row.get("f05_variant") or None,
            "f06": row.get("f06_variant") or None,
        },
        "variant_paths": build_variant_paths(row),
        "model": {
            "model_family": row.get("model_family") or None,
            "seed": parse_scalar(row.get("seed")),
            "imbalance_strategy": row.get("imbalance_strategy") or None,
            "imbalance_max_maj": parse_scalar(row.get("imbalance_max_maj")),
            "deduplication_mode": row.get("deduplication_mode") or "none",
        },
        "phase_keys": {
            "f02": row.get("f02_key") or None,
            "f03": row.get("f03_key") or None,
            "f04": row.get("f04_key") or None,
            "f05": row.get("f05_key") or None,
            "f06": row.get("f06_key") or None,
        },
        "phase_reused": {
            "f02": row.get("f02_reused") or None,
            "f03": row.get("f03_reused") or None,
            "f04": row.get("f04_reused") or None,
            "f05": row.get("f05_reused") or None,
            "f06": row.get("f06_reused") or None,
        },
        "metrics": read_f05_metric_snapshot(row),
    })
    return entry


def build_index(rows: list[dict[str, Any]], source_manifests: list[Path]) -> dict[str, Any]:
    latest_by_model_key: dict[str, dict[str, Any]] = {}
    for row in rows:
        model_key = build_model_key(row)
        row["_model_key"] = model_key
        latest_by_model_key[model_key] = choose_latest_duplicate(latest_by_model_key.get(model_key), row)

    index = empty_index(source_manifests)
    models_by_measure: dict[str, Any] = {}
    pipeline_counts: dict[str, int] = {}
    target_groups: set[str] = set()

    for model_key, row in sorted(latest_by_model_key.items()):
        measure = clean_key_part(row.get("measure"))
        direction = clean_key_part(row.get("direction"))
        threshold_key = f"threshold_{compact_value(row.get('effective_threshold'))}"
        temporal_config_id = build_temporal_config_id(row)
        bands_key = build_bands_key(row)
        pipeline = clean_key_part(row.get("pipeline"))
        target_group = build_target_group(row)

        pipeline_counts[pipeline] = pipeline_counts.get(pipeline, 0) + 1
        target_groups.add(target_group)

        threshold_node = (
            models_by_measure
            .setdefault(measure, {})
            .setdefault(direction, {})
            .setdefault(threshold_key, {
                "target_group": target_group,
                "temporal_configs": {},
            })
        )
        temporal_node = threshold_node["temporal_configs"].setdefault(temporal_config_id, {
            "OW": parse_scalar(row.get("OW")),
            "PW": parse_scalar(row.get("PW")),
            "LT": parse_scalar(row.get("LT")),
            "window_strategy": row.get("window_strategy") or None,
            "nan_mode": row.get("nan_mode") or None,
            "pipelines": {},
        })
        pipeline_node = temporal_node["pipelines"].setdefault(pipeline, {})
        bands_node = pipeline_node.setdefault(bands_key, {
            "bands_id": row.get("bands_id") or ("no_bands" if pipeline == "ts" else None),
            "bands": parse_bands(row.get("bands")) if pipeline != "ts" else [],
            "models": [],
        })
        bands_node.setdefault("models", []).append(row_to_model_entry(row, model_key))

    index["summary"] = {
        "n_models": len(latest_by_model_key),
        "n_measures": len(models_by_measure),
        "n_target_groups": len(target_groups),
        "pipelines": dict(sorted(pipeline_counts.items())),
    }
    index["models_by_measure"] = models_by_measure
    return index


def write_yaml(index: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(index, f, sort_keys=False, allow_unicode=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build registry/model_index.yaml from run manifests.")
    parser.add_argument("--include-failed", action="store_true", help="Include FAILED rows when no COMPLETED duplicate exists.")
    parser.add_argument(
        "--source",
        choices=["exported", "runs", "all"],
        default="all",
        help="Rows to index. Default: all.",
    )
    parser.add_argument(
        "--exported-manifest",
        default=str(DEFAULT_EXPORTED_MANIFEST),
        help="Exported params manifest to include. Default: exported_params/manifest.csv",
    )
    parser.add_argument(
        "--no-exported-params",
        action="store_true",
        help="Do not include exported_params/manifest.csv rows.",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output YAML path. Default: registry/model_index.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path

    include_runs = args.source in ("runs", "all")
    include_exported = args.source in ("exported", "all") and not args.no_exported_params

    manifest_paths = find_manifest_paths() if include_runs else []
    source_manifests = list(manifest_paths)
    rows = read_manifest_rows(manifest_paths, include_failed=args.include_failed) if manifest_paths else []

    if include_exported:
        exported_manifest = Path(args.exported_manifest)
        if not exported_manifest.is_absolute():
            exported_manifest = REPO_ROOT / exported_manifest
        exported_rows = read_exported_manifest_rows(exported_manifest)
        if exported_rows:
            rows.extend(exported_rows)
            source_manifests.append(exported_manifest)
        elif exported_manifest.exists():
            print(f"No valid exported params rows found in {relative_path(exported_manifest)}.")

    if not rows:
        print("No valid manifest rows found.")

    index = build_index(rows, source_manifests)
    write_yaml(index, output_path)
    print(f"Wrote {relative_path(output_path)} with {index['summary']['n_models']} models.")


if __name__ == "__main__":
    main()