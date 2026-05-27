#!/usr/bin/env python3
"""Simple incremental orchestrator for comparing TS and event-based MLOps pipelines."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "experiment.yaml"
RUNS_DIR = REPO_ROOT / "runs"
F56_GPU_FOR_SCRIPT5 = True

PHASES = {
    "f02": ("f02_events", 2),
    "f03": ("f03_windows", 3),
    "f04": ("f04_targets", 4),
    "f05": ("f05_modeling", 5),
}

ALLOWED_PIPELINES = {"ts", "events", "uniEvents"}

MANIFEST_COLUMNS = [
    "run_id",
    "job_key",
    "pipeline",
    "measure",
    "direction",
    "temporal_id",
    "OW",
    "LT",
    "PW",
    "window_strategy",
    "nan_mode",
    "event_strategy",
    "model_family",
    "imbalance_strategy",
    "imbalance_max_maj",
    "deduplication_mode",
    "seed",
    "bands_id",
    "bands",
    "effective_threshold",
    "status",
    "message",
    "f02_variant",
    "f03_variant",
    "f04_variant",
    "f05_variant",
    "f02_key",
    "f03_key",
    "f04_key",
    "f05_key",
    "f02_reused",
    "f03_reused",
    "f04_reused",
    "f05_reused",
    "started_at",
    "finished_at",
    "repo_path",
    "output_path",
]


DEFAULT_OUTPUTS_CONFIG = {
    "enabled": True,
    "collect_after_job": True,
    "phases": {
        "f02": {"phase_name": "f02_events", "variant_column": "f02_variant"},
        "f03": {"phase_name": "f03_windows", "variant_column": "f03_variant"},
        "f04": {"phase_name": "f04_targets", "variant_column": "f04_variant"},
        "f05": {"phase_name": "f05_modeling", "variant_column": "f05_variant"},
    },
    "destination_template": "runs/{run_id}/outputs/{pipeline}/{job_key}/{phase}",
    "always_copy": ["params.yaml", "outputs.yaml", "metadata.yaml"],
    "include_extensions": [".yaml", ".yml", ".json", ".csv", ".txt", ".html", ".png", ".svg"],
    "exclude_extensions": [".parquet", ".h5", ".tflite", ".pkl", ".npy", ".npz"],
    "max_file_mb": 10,
    "write_manifest_row": False,
    "overwrite_existing": True,
}


class CommandError(RuntimeError):
    pass


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing experiment config: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("experiment.yaml must contain a YAML mapping")
    return data


def validate_config(cfg: dict[str, Any]) -> None:
    for section in ["experiment", "repos", "parents", "base", "grid", "pipelines"]:
        if section not in cfg:
            raise ValueError(f"Missing required section in experiment.yaml: {section}")

    pipelines = cfg.get("pipelines") or []
    unknown = sorted(set(pipelines) - ALLOWED_PIPELINES)
    if unknown:
        raise ValueError(f"Unknown pipeline(s): {unknown}. Allowed: {sorted(ALLOWED_PIPELINES)}")

    grid = cfg["grid"]
    if not grid.get("measures"):
        raise ValueError("grid.measures must not be empty")
    if not grid.get("directions"):
        raise ValueError("grid.directions must not be empty")
    if not grid.get("event_band_sets"):
        raise ValueError("grid.event_band_sets must not be empty")

    for direction in grid["directions"]:
        if direction not in {"high", "low"}:
            raise ValueError(f"Unsupported direction: {direction}. Use high or low")

    for band_set in grid["event_band_sets"]:
        if "id" not in band_set or "values" not in band_set:
            raise ValueError("Each event_band_sets entry must have id and values")
        values = band_set["values"]
        if not isinstance(values, list) or not values:
            raise ValueError(f"Band set {band_set.get('id')} must contain a non-empty values list")
        validate_bands(values, str(band_set["id"]))

    temporal_ids: set[str] = set()
    for temporal in get_temporal_configs(cfg):
        temporal_id = str(temporal["id"])
        if temporal_id in temporal_ids:
            raise ValueError(f"Duplicate temporal config id: {temporal_id}")
        temporal_ids.add(temporal_id)
        for key_name in ["OW", "LT", "PW"]:
            if int(temporal[key_name]) <= 0:
                raise ValueError(f"Temporal config {temporal_id} has invalid {key_name}: {temporal[key_name]}")
        if temporal["window_strategy"] not in {"synchro", "asynOW"}:
            raise ValueError(f"Temporal config {temporal_id} has unsupported window_strategy: {temporal['window_strategy']}")
        if temporal["nan_mode"] not in {"keep", "discard"}:
            raise ValueError(f"Temporal config {temporal_id} has unsupported nan_mode: {temporal['nan_mode']}")

    for key in ALLOWED_PIPELINES:
        if key in pipelines:
            repo = Path(str(cfg["repos"].get(key, "")))
            if not repo.exists():
                raise ValueError(f"Repo path for {key} does not exist: {repo}")


def validate_bands(values: list[Any], bands_id: str) -> None:
    try:
        nums = [float(v) for v in values]
    except Exception as exc:
        raise ValueError(f"Band set {bands_id} contains non-numeric values: {values}") from exc
    if nums != sorted(nums):
        raise ValueError(f"Band set {bands_id} must be sorted ascending: {values}")
    if len(set(nums)) != len(nums):
        raise ValueError(f"Band set {bands_id} contains duplicate values: {values}")


def derive_effective_threshold(direction: str, bands: list[Any]) -> int | float:
    validate_bands(bands, "<inline>")
    value = bands[-1] if direction == "high" else bands[0]
    value = float(value)
    return int(value) if value.is_integer() else value


def compact_number(value: Any) -> str:
    try:
        f = float(value)
        if f.is_integer():
            return str(int(f))
        return str(f).replace(".", "p")
    except Exception:
        return str(value).replace(".", "p")


def safe_name(value: str) -> str:
    name = re.sub(r"[^a-z0-9_]+", "_", value.lower())
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "unnamed"


def json_dumps_compact(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def build_job_key(job_identity: dict[str, Any]) -> str:
    payload = json.dumps(job_identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def build_identity_key(identity: dict[str, Any]) -> str:
    return build_job_key(identity)


def field_value(source: dict[str, Any], key: str, default: Any = "") -> Any:
    if key in source and source.get(key) not in (None, ""):
        return source.get(key)
    identity = source.get("identity") if isinstance(source.get("identity"), dict) else {}
    if key in identity and identity.get(key) not in (None, ""):
        return identity.get(key)
    base = source.get("base") if isinstance(source.get("base"), dict) else {}
    if key in base and base.get(key) not in (None, ""):
        return base.get(key)
    temporal = source.get("temporal") if isinstance(source.get("temporal"), dict) else {}
    if key in temporal and temporal.get(key) not in (None, ""):
        return temporal.get(key)
    return default


def normalized_repo_path(value: Any) -> str:
    return str(Path(str(value)).resolve()) if value not in (None, "") else ""


def parse_manifest_bands(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    text = str(value or "").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        try:
            parsed = yaml.safe_load(text)
        except yaml.YAMLError:
            return [item.strip() for item in text.split(",") if item.strip()]
    return parsed if isinstance(parsed, list) else []


def job_bands(source: dict[str, Any]) -> list[Any]:
    return parse_manifest_bands(source.get("bands"))


def job_repo_path(source: dict[str, Any]) -> str:
    return normalized_repo_path(source.get("repo_path", ""))


def job_parent_f01(source: dict[str, Any]) -> Any:
    return source.get("parent_f01") or field_value(source, "parent_f01", "")


def job_deduplication_mode(source: dict[str, Any]) -> str:
    return str(field_value(source, "deduplication_mode", "none") or "none")


def build_f02_key(job: dict[str, Any]) -> str:
    pipeline = str(field_value(job, "pipeline"))
    identity: dict[str, Any] = {
        "pipeline": pipeline,
        "repo_path": job_repo_path(job),
        "parent_f01": job_parent_f01(job),
    }
    if pipeline == "ts":
        identity["measure"] = field_value(job, "measure")
    elif pipeline == "events":
        identity.update({
            "event_strategy": field_value(job, "event_strategy"),
            "bands_id": field_value(job, "bands_id"),
            "bands": job_bands(job),
            "nan_mode": field_value(job, "nan_mode"),
        })
    elif pipeline == "uniEvents":
        identity.update({
            "measure": field_value(job, "measure"),
            "event_strategy": field_value(job, "event_strategy"),
            "bands_id": field_value(job, "bands_id"),
            "bands": job_bands(job),
            "nan_mode": field_value(job, "nan_mode"),
        })
    return build_identity_key(identity)


def build_f03_key(job: dict[str, Any], f02_key: str) -> str:
    identity = {
        "pipeline": field_value(job, "pipeline"),
        "repo_path": job_repo_path(job),
        "f02_key": f02_key,
        "OW": int(field_value(job, "OW", 0)),
        "PW": int(field_value(job, "PW", 0)),
        "LT": int(field_value(job, "LT", 0)),
        "window_strategy": field_value(job, "window_strategy"),
        "nan_mode": field_value(job, "nan_mode"),
    }
    return build_identity_key(identity)


def build_f04_key(job: dict[str, Any], f03_key: str) -> str:
    pipeline = str(field_value(job, "pipeline"))
    identity = {
        "pipeline": pipeline,
        "repo_path": job_repo_path(job),
        "f03_key": f03_key,
        "measure": field_value(job, "measure"),
        "direction": field_value(job, "direction"),
        "effective_threshold": field_value(job, "effective_threshold"),
        "bands_id": "" if pipeline == "ts" else field_value(job, "bands_id"),
        "bands": [] if pipeline == "ts" else job_bands(job),
    }
    target_events = field_value(job, "target_events", "")
    if target_events not in (None, ""):
        identity["target_events"] = target_events
    return build_identity_key(identity)


def build_f05_key(job: dict[str, Any], f04_key: str) -> str:
    identity = {
        "pipeline": field_value(job, "pipeline"),
        "repo_path": job_repo_path(job),
        "f04_key": f04_key,
        "model_family": field_value(job, "model_family"),
        "seed": field_value(job, "seed"),
        "imbalance_strategy": field_value(job, "imbalance_strategy"),
        "imbalance_max_maj": field_value(job, "imbalance_max_maj"),
        "deduplication_mode": job_deduplication_mode(job),
    }
    training = field_value(job, "training", {}) or {}
    if training:
        identity["training"] = training
    return build_identity_key(identity)


def attach_phase_keys(job: dict[str, Any]) -> None:
    f02_key = build_f02_key(job)
    f03_key = build_f03_key(job, f02_key)
    f04_key = build_f04_key(job, f03_key)
    f05_key = build_f05_key(job, f04_key)
    job.update({"f02_key": f02_key, "f03_key": f03_key, "f04_key": f04_key, "f05_key": f05_key})


def get_temporal_configs(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    grid = cfg.get("grid") or {}
    base = cfg.get("base") or {}
    raw_temporals = grid.get("temporal_configs")

    if raw_temporals is None:
        required = ["OW", "LT", "PW", "window_strategy", "nan_mode"]
        missing = [key for key in required if key not in base]
        if missing:
            raise ValueError(
                "Define grid.temporal_configs or keep temporal keys in base. "
                f"Missing: {missing}"
            )
        raw_temporals = [{
            "id": f"ow{base['OW']}_lt{base['LT']}_pw{base['PW']}_{base['window_strategy']}_{base['nan_mode']}",
            "OW": base["OW"],
            "LT": base["LT"],
            "PW": base["PW"],
            "window_strategy": base["window_strategy"],
            "nan_mode": base["nan_mode"],
        }]

    if not isinstance(raw_temporals, list) or not raw_temporals:
        raise ValueError("grid.temporal_configs must be a non-empty list")

    temporals: list[dict[str, Any]] = []
    for item in raw_temporals:
        if not isinstance(item, dict):
            raise ValueError("Each temporal config must be a mapping")
        for key in ["id", "OW", "LT", "PW", "window_strategy", "nan_mode"]:
            if key not in item:
                raise ValueError(f"Temporal config missing required key: {key}")
        temporals.append({
            "id": str(item["id"]),
            "OW": int(item["OW"]),
            "LT": int(item["LT"]),
            "PW": int(item["PW"]),
            "window_strategy": str(item["window_strategy"]),
            "nan_mode": str(item["nan_mode"]),
        })
    return temporals


def expand_jobs(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    base = cfg["base"]
    grid = cfg["grid"]
    pipelines = set(cfg.get("pipelines") or [])
    temporal_configs = get_temporal_configs(cfg)
    model_families = list(grid.get("model_families") or [base.get("model_family")])
    deduplication_modes = list(grid.get("deduplication_modes") or [base.get("deduplication_mode", "none")])
    jobs: list[dict[str, Any]] = []

    for temporal in temporal_configs:
        for measure in grid["measures"]:
            for direction in grid["directions"]:
                for band_set in grid["event_band_sets"]:
                    bands_id = str(band_set["id"])
                    bands = list(band_set["values"])
                    effective_threshold = derive_effective_threshold(direction, bands)

                    for model_family in model_families:
                        for deduplication_mode in deduplication_modes:
                            for pipeline in ["events", "uniEvents"]:
                                if pipeline not in pipelines:
                                    continue
                                identity = {
                                    "pipeline": pipeline,
                                    "measure": measure,
                                    "direction": direction,
                                    "temporal_id": temporal["id"],
                                    "bands_id": bands_id,
                                    "bands": bands,
                                    "effective_threshold": effective_threshold,
                                    "OW": temporal["OW"],
                                    "PW": temporal["PW"],
                                    "LT": temporal["LT"],
                                    "window_strategy": temporal["window_strategy"],
                                    "nan_mode": temporal["nan_mode"],
                                    "event_strategy": base["event_strategy"],
                                    "model_family": model_family,
                                    "imbalance_strategy": base["imbalance_strategy"],
                                    "imbalance_max_maj": base["imbalance_max_maj"],
                                    "deduplication_mode": deduplication_mode,
                                    "seed": base["seed"],
                                }
                                jobs.append(make_job(identity, cfg, bands_id, bands, temporal))

    # TS jobs are deduplicated by effective threshold within each temporal config.
    seen_ts: set[str] = set()
    if "ts" in pipelines:
        for temporal in temporal_configs:
            for measure in grid["measures"]:
                for direction in grid["directions"]:
                    for band_set in grid["event_band_sets"]:
                        bands = list(band_set["values"])
                        effective_threshold = derive_effective_threshold(direction, bands)
                        for model_family in model_families:
                            for deduplication_mode in deduplication_modes:
                                identity = {
                                    "pipeline": "ts",
                                    "measure": measure,
                                    "direction": direction,
                                    "temporal_id": temporal["id"],
                                    "effective_threshold": effective_threshold,
                                    "OW": temporal["OW"],
                                    "PW": temporal["PW"],
                                    "LT": temporal["LT"],
                                    "window_strategy": temporal["window_strategy"],
                                    "nan_mode": temporal["nan_mode"],
                                    "model_family": model_family,
                                    "imbalance_strategy": base["imbalance_strategy"],
                                    "imbalance_max_maj": base["imbalance_max_maj"],
                                    "deduplication_mode": deduplication_mode,
                                    "seed": base["seed"],
                                }
                                job_key = build_job_key(identity)
                                if job_key in seen_ts:
                                    continue
                                seen_ts.add(job_key)
                                jobs.append(make_job(identity, cfg, "", [], temporal))

    return jobs


def make_job(identity: dict[str, Any], cfg: dict[str, Any], bands_id: str, bands: list[Any], temporal: dict[str, Any]) -> dict[str, Any]:
    pipeline = identity["pipeline"]
    parent_key = f"{pipeline}_f01"
    parent_f01 = cfg["parents"].get(parent_key)
    if parent_f01 is None and pipeline == "uniEvents":
        parent_f01 = cfg["parents"].get("events_f01")
    if parent_f01 is None:
        raise ValueError(f"Missing parent F01 config: parents.{parent_key}")
    job = {
        "job_key": build_job_key(identity),
        "identity": identity,
        "pipeline": pipeline,
        "repo_path": Path(str(cfg["repos"][pipeline])).resolve(),
        "parent_f01": parent_f01,
        "measure": identity["measure"],
        "direction": identity["direction"],
        "temporal_id": temporal["id"],
        "temporal": temporal,
        "bands_id": bands_id,
        "bands": bands,
        "effective_threshold": identity["effective_threshold"],
        "base": {
            **cfg["base"],
            "model_family": identity.get("model_family"),
            "deduplication_mode": identity.get("deduplication_mode", "none"),
        },
        "deduplication_mode": identity.get("deduplication_mode", "none"),
    }
    attach_phase_keys(job)
    return job


def parse_datetime(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def is_newer_attempt(candidate: dict[str, Any], current: dict[str, Any] | None) -> bool:
    if current is None:
        return True
    candidate_finished = parse_datetime(candidate.get("finished_at"))
    current_finished = parse_datetime(current.get("finished_at"))
    if candidate_finished and current_finished and candidate_finished != current_finished:
        return candidate_finished > current_finished
    if candidate_finished and not current_finished:
        return True
    if current_finished and not candidate_finished:
        return False
    candidate_run_id = str(candidate.get("run_id", ""))
    current_run_id = str(current.get("run_id", ""))
    if candidate_run_id != current_run_id:
        return candidate_run_id > current_run_id
    return True


def load_job_attempts(runs_dir: Path = RUNS_DIR) -> tuple[set[str], dict[str, dict[str, Any]]]:
    completed: set[str] = set()
    failed_attempts: dict[str, dict[str, Any]] = {}
    for manifest in sorted(runs_dir.glob("*/manifest.csv")):
        try:
            with manifest.open("r", encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    job_key = row.get("job_key")
                    if not job_key:
                        continue
                    if row.get("status") == "COMPLETED":
                        completed.add(job_key)
                    elif row.get("status") == "FAILED" and is_newer_attempt(row, failed_attempts.get(job_key)):
                        failed_attempts[job_key] = row
        except Exception as exc:
            print(f"Warning: could not read manifest {manifest}: {exc}")
    return completed, failed_attempts


def load_completed_jobs(runs_dir: Path = RUNS_DIR) -> set[str]:
    completed, _ = load_job_attempts(runs_dir)
    return completed


def variant_dir_exists(repo_path: Path, phase: str, variant: str) -> bool:
    if not variant:
        return False
    phase_name = PHASES[phase][0]
    return (repo_path / "executions" / phase_name / variant).exists()


def build_phase_cache(runs_dir: Path = RUNS_DIR) -> dict[str, dict[str, dict[str, Any]]]:
    phase_cache: dict[str, dict[str, dict[str, Any]]] = {phase: {} for phase in PHASES}
    for manifest in sorted(runs_dir.glob("*/manifest.csv")):
        try:
            with manifest.open("r", encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    if row.get("status") != "COMPLETED":
                        continue
                    repo_text = str(row.get("repo_path", "")).strip()
                    if not repo_text:
                        continue
                    repo_path = Path(repo_text)
                    if not repo_path.exists():
                        continue

                    f02_key = row.get("f02_key") or build_f02_key(row)
                    f03_key = row.get("f03_key") or build_f03_key(row, f02_key)
                    f04_key = row.get("f04_key") or build_f04_key(row, f03_key)
                    f05_key = row.get("f05_key") or build_f05_key(row, f04_key)
                    keys = {"f02": f02_key, "f03": f03_key, "f04": f04_key, "f05": f05_key}

                    for phase, key in keys.items():
                        variant = str(row.get(f"{phase}_variant", "")).strip()
                        if not key or not variant or not variant_dir_exists(repo_path, phase, variant):
                            continue
                        phase_cache[phase][key] = {
                            "variant": variant,
                            "repo_path": str(repo_path),
                            "pipeline": row.get("pipeline", ""),
                            "run_id": row.get("run_id", ""),
                            "job_key": row.get("job_key", ""),
                        }
        except Exception as exc:
            print(f"Warning: could not read manifest for phase cache {manifest}: {exc}")
    return phase_cache


def cache_completed_phase(
    phase_cache: dict[str, dict[str, dict[str, Any]]],
    phase: str,
    key: str,
    variant: str,
    row: dict[str, Any],
) -> None:
    phase_cache.setdefault(phase, {})[key] = {
        "variant": variant,
        "repo_path": row.get("repo_path", ""),
        "pipeline": row.get("pipeline", ""),
        "run_id": row.get("run_id", ""),
        "job_key": row.get("job_key", ""),
    }


def reusable_cached_variant(
    job: dict[str, Any],
    phase_cache: dict[str, dict[str, dict[str, Any]]],
    phase: str,
    log_path: Path,
) -> str:
    key = job.get(f"{phase}_key") or ""
    cached = (phase_cache.get(phase) or {}).get(key)
    if not cached:
        return ""
    variant = str(cached.get("variant", "")).strip()
    repo_path = Path(str(cached.get("repo_path") or job["repo_path"]))
    if variant and variant_dir_exists(repo_path, phase, variant):
        with log_path.open("a", encoding="utf-8") as log:
            log.write(
                f"\n[REUSE] Reusing {phase.upper()} variant {variant} "
                f"from run {cached.get('run_id')} job {cached.get('job_key')}\n"
            )
        return variant
    return ""


def create_run_dir(runs_dir: Path = RUNS_DIR) -> tuple[str, Path]:
    today = datetime.now().strftime("%Y-%m-%d")
    runs_dir.mkdir(parents=True, exist_ok=True)
    idx = 1
    while True:
        run_id = f"{today}_{idx:03d}"
        run_dir = runs_dir / run_id
        if not run_dir.exists():
            (run_dir / "logs").mkdir(parents=True)
            (run_dir / "outputs").mkdir(parents=True)
            return run_id, run_dir
        idx += 1


def next_free_variant(repo_path: Path, phase_name: str, phase_digit: int) -> str:
    phase_dir = repo_path / "executions" / phase_name
    used: set[int] = set()
    if phase_dir.exists():
        prefix = f"v{phase_digit}_"
        for child in phase_dir.iterdir():
            if not child.is_dir() or not child.name.startswith(prefix):
                continue
            suffix = child.name[len(prefix):]
            if suffix.isdigit():
                used.add(int(suffix))
    idx = 0
    while idx in used:
        idx += 1
    return f"v{phase_digit}_{idx:04d}"


def format_command(command: str | list[str]) -> str:
    if isinstance(command, str):
        return command
    return subprocess.list2cmdline(command)


def run_command(command: str | list[str], cwd: Path, log_path: Path) -> None:
    timestamp = datetime.now().isoformat(timespec="seconds")
    display_command = format_command(command)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{timestamp}] COMMAND\n{display_command}\n")
        proc = subprocess.run(
            command,
            cwd=str(cwd),
            shell=isinstance(command, str),
            text=True,
            capture_output=True,
        )
        log.write(f"\n[{timestamp}] STDOUT\n{proc.stdout}\n")
        log.write(f"\n[{timestamp}] STDERR\n{proc.stderr}\n")
        log.write(f"\n[{timestamp}] RETURN_CODE {proc.returncode}\n")
    if proc.returncode != 0:
        raise CommandError(f"Command failed with return code {proc.returncode}: {display_command}")


def run_make(cwd: Path, log_path: Path, target: str, **kwargs: Any) -> None:
    command = ["make", target]
    command.extend(f"{key}={value}" for key, value in kwargs.items())
    run_command(command, cwd, log_path)


def run_script5(cwd: Path, log_path: Path, variant: str) -> None:
    kwargs: dict[str, Any] = {"VARIANT": variant}
    if F56_GPU_FOR_SCRIPT5:
        kwargs["F56_GPU"] = "true"
    run_make(cwd, log_path, "script5", **kwargs)


def log_retry(log_path: Path, message: str) -> None:
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[RETRY] {message}\n")


def reusable_previous_variant(job: dict[str, Any], phase: str, log_path: Path) -> str:
    previous = job.get("previous_attempt") or {}
    variant = str(previous.get(f"{phase}_variant", "")).strip()
    if not variant:
        return ""

    phase_name = PHASES[phase][0]
    variant_dir = job["repo_path"] / "executions" / phase_name / variant
    if variant_dir.exists():
        log_retry(log_path, f"Reusing existing {phase} variant {variant} from previous FAILED attempt.")
        log_retry(log_path, f"Running script{PHASES[phase][1]} directly.")
        return variant

    log_warning(
        log_path,
        f"Previous FAILED attempt has {phase}_variant={variant}, but variant directory does not exist: {variant_dir}. "
        "Creating a new variant for this phase.",
    )
    return ""


def write_manifest(manifest_path: Path, rows: list[dict[str, Any]]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in MANIFEST_COLUMNS})


def append_or_update_manifest_row(manifest_path: Path, rows: list[dict[str, Any]], row: dict[str, Any]) -> None:
    for idx, existing in enumerate(rows):
        if existing.get("job_key") == row.get("job_key"):
            rows[idx] = row
            break
    else:
        rows.append(row)
    write_manifest(manifest_path, rows)


def initial_manifest_row(run_id: str, run_dir: Path, job: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "job_key": job["job_key"],
        "pipeline": job["pipeline"],
        "measure": job["measure"],
        "direction": job["direction"],
        "temporal_id": job["temporal_id"],
        "OW": job["temporal"]["OW"],
        "LT": job["temporal"]["LT"],
        "PW": job["temporal"]["PW"],
        "window_strategy": job["temporal"]["window_strategy"],
        "nan_mode": job["temporal"]["nan_mode"],
        "event_strategy": job["base"].get("event_strategy", ""),
        "model_family": job["base"].get("model_family", ""),
        "imbalance_strategy": job["base"].get("imbalance_strategy", ""),
        "imbalance_max_maj": job["base"].get("imbalance_max_maj", ""),
        "deduplication_mode": job.get("deduplication_mode", "none"),
        "seed": job["base"].get("seed", ""),
        "bands_id": job["bands_id"],
        "bands": json_dumps_compact(job["bands"]) if job["bands"] else "",
        "effective_threshold": job["effective_threshold"],
        "status": "RUNNING",
        "message": "",
        "f02_variant": "",
        "f03_variant": "",
        "f04_variant": "",
        "f05_variant": "",
        "f02_key": job.get("f02_key", ""),
        "f03_key": job.get("f03_key", ""),
        "f04_key": job.get("f04_key", ""),
        "f05_key": job.get("f05_key", ""),
        "f02_reused": False,
        "f03_reused": False,
        "f04_reused": False,
        "f05_reused": False,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "finished_at": "",
        "repo_path": str(job["repo_path"]),
        "output_path": str((run_dir / "outputs" / job["pipeline"] / job["job_key"]).resolve()),
    }


def log_warning(log_path: Path | None, message: str) -> None:
    if log_path is None:
        print(f"Warning: {message}")
        return
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[WARNING] {message}\n")


def load_outputs_config(config: dict[str, Any]) -> dict[str, Any]:
    user_cfg = config.get("outputs") or {}
    if not isinstance(user_cfg, dict):
        raise ValueError("outputs must be a mapping when present")

    merged = dict(DEFAULT_OUTPUTS_CONFIG)
    merged["phases"] = dict(DEFAULT_OUTPUTS_CONFIG["phases"])
    for key, value in user_cfg.items():
        if key == "phases" and isinstance(value, dict):
            phases = dict(merged["phases"])
            for phase, phase_cfg in value.items():
                current = dict(phases.get(phase, {}))
                current.update(phase_cfg or {})
                phases[phase] = current
            merged["phases"] = phases
        else:
            merged[key] = value

    for key in ["always_copy", "include_extensions", "exclude_extensions"]:
        merged[key] = list(merged.get(key) or [])
    merged["include_extensions"] = {str(ext).lower() for ext in merged["include_extensions"]}
    merged["exclude_extensions"] = {str(ext).lower() for ext in merged["exclude_extensions"]}
    merged["always_copy"] = {str(name) for name in merged["always_copy"]}
    merged["max_file_bytes"] = int(float(merged.get("max_file_mb", 10)) * 1024 * 1024)
    return merged


def should_copy_file(path: Path, outputs_config: dict[str, Any]) -> bool:
    if not path.is_file():
        return False
    if path.name in outputs_config["always_copy"]:
        return True
    suffix = path.suffix.lower()
    if suffix in outputs_config["exclude_extensions"]:
        return False
    if suffix not in outputs_config["include_extensions"]:
        return False
    try:
        if path.stat().st_size > int(outputs_config["max_file_bytes"]):
            return False
    except OSError:
        return False
    return True


def output_destination(run_dir: Path, row: dict[str, Any], outputs_config: dict[str, Any], phase: str) -> Path:
    rendered = str(outputs_config["destination_template"]).format(
        run_id=row.get("run_id", ""),
        pipeline=row.get("pipeline", ""),
        job_key=row.get("job_key", ""),
        phase=phase,
    )
    dst = Path(rendered)
    if dst.is_absolute():
        return dst
    if dst.parts and dst.parts[0] == "runs":
        return REPO_ROOT / dst
    return run_dir / dst


def safe_copy_file(src: Path, dst: Path, log_path: Path | None) -> None:
    try:
        shutil.copy2(src, dst)
    except Exception as exc:
        log_warning(log_path, f"Could not copy {src} -> {dst}: {exc}")


def write_manifest_row_yaml(row_or_job: dict[str, Any], output_dir: Path) -> None:
    variants = {
        "f02": row_or_job.get("f02_variant", ""),
        "f03": row_or_job.get("f03_variant", ""),
        "f04": row_or_job.get("f04_variant", ""),
        "f05": row_or_job.get("f05_variant", ""),
    }
    payload = dict(row_or_job)
    payload["variants"] = variants
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest_row.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def copy_configured_outputs(row_or_job: dict[str, Any], outputs_config: dict[str, Any], run_dir: Path, log_path: Path | None = None) -> None:
    if not outputs_config.get("enabled", True):
        return

    repo_path = Path(str(row_or_job.get("repo_path", "")))
    if not repo_path.exists():
        log_warning(log_path, f"Repo path not found while collecting outputs: {repo_path}")
        return

    job_root = run_dir / "outputs" / str(row_or_job.get("pipeline", "")) / str(row_or_job.get("job_key", ""))
    if outputs_config.get("write_manifest_row", False):
        write_manifest_row_yaml(row_or_job, job_root)

    overwrite = bool(outputs_config.get("overwrite_existing", True))
    for phase, phase_cfg in (outputs_config.get("phases") or {}).items():
        phase_name = phase_cfg.get("phase_name")
        variant_column = phase_cfg.get("variant_column")
        variant = row_or_job.get(variant_column, "") if variant_column else ""
        if not phase_name or not variant:
            log_warning(log_path, f"Skipping output collection for {phase}: missing phase_name or variant")
            continue

        src_dir = repo_path / "executions" / str(phase_name) / str(variant)
        if not src_dir.exists():
            log_warning(log_path, f"Variant output directory not found: {src_dir}")
            continue

        dst_dir = output_destination(run_dir, row_or_job, outputs_config, phase)
        dst_dir.mkdir(parents=True, exist_ok=True)

        copied: set[str] = set()
        for name in outputs_config["always_copy"]:
            src = src_dir / name
            if src.exists() and src.is_file():
                dst = dst_dir / src.name
                if overwrite or not dst.exists():
                    safe_copy_file(src, dst, log_path)
                copied.add(src.name)

        for src in src_dir.iterdir():
            if src.name in copied:
                continue
            if should_copy_file(src, outputs_config):
                dst = dst_dir / src.name
                if overwrite or not dst.exists():
                    safe_copy_file(src, dst, log_path)


def mark_phase(row: dict[str, Any], phase: str, variant: str, reused: bool, manifest_path: Path, rows: list[dict[str, Any]]) -> None:
    row[f"{phase}_variant"] = variant
    row[f"{phase}_reused"] = bool(reused)
    append_or_update_manifest_row(manifest_path, rows, row)


def execute_ts_job(
    job: dict[str, Any],
    run_dir: Path,
    log_path: Path,
    row: dict[str, Any],
    manifest_path: Path,
    rows: list[dict[str, Any]],
    phase_cache: dict[str, dict[str, dict[str, Any]]],
) -> None:
    repo = job["repo_path"]
    base = job["base"]
    temporal = job["temporal"]
    measure = job["measure"]
    direction = job["direction"]
    threshold = job["effective_threshold"]
    name = safe_name(f"{measure}_{direction}_{compact_number(threshold)}")

    v2 = reusable_previous_variant(job, "f02", log_path)
    if v2:
        mark_phase(row, "f02", v2, False, manifest_path, rows)
        run_make(repo, log_path, "script2", VARIANT=v2)
        cache_completed_phase(phase_cache, "f02", job["f02_key"], v2, row)
    else:
        v2 = reusable_cached_variant(job, phase_cache, "f02", log_path)
    if v2 and row.get("f02_variant") != v2:
        mark_phase(row, "f02", v2, True, manifest_path, rows)
    elif not v2:
        v2 = next_free_variant(repo, "f02_events", 2)
        mark_phase(row, "f02", v2, False, manifest_path, rows)
        run_make(repo, log_path, "variant2", VARIANT=v2, PARENT=job["parent_f01"], MEASURE=measure)
        run_make(repo, log_path, "script2", VARIANT=v2)
        cache_completed_phase(phase_cache, "f02", job["f02_key"], v2, row)

    v3 = reusable_previous_variant(job, "f03", log_path)
    if v3:
        mark_phase(row, "f03", v3, False, manifest_path, rows)
        run_make(repo, log_path, "script3", VARIANT=v3)
        cache_completed_phase(phase_cache, "f03", job["f03_key"], v3, row)
    else:
        v3 = reusable_cached_variant(job, phase_cache, "f03", log_path)
    if v3 and row.get("f03_variant") != v3:
        mark_phase(row, "f03", v3, True, manifest_path, rows)
    elif not v3:
        v3 = next_free_variant(repo, "f03_windows", 3)
        mark_phase(row, "f03", v3, False, manifest_path, rows)
        run_make(
            repo,
            log_path,
            "variant3",
            VARIANT=v3,
            PARENT=v2,
            OW=temporal["OW"],
            LT=temporal["LT"],
            PW=temporal["PW"],
            STRATEGY=temporal["window_strategy"],
            NAN_MODE=temporal["nan_mode"],
        )
        run_make(repo, log_path, "script3", VARIANT=v3)
        cache_completed_phase(phase_cache, "f03", job["f03_key"], v3, row)

    v4 = reusable_previous_variant(job, "f04", log_path)
    if v4:
        mark_phase(row, "f04", v4, False, manifest_path, rows)
        run_make(repo, log_path, "script4", VARIANT=v4)
        cache_completed_phase(phase_cache, "f04", job["f04_key"], v4, row)
    else:
        v4 = reusable_cached_variant(job, phase_cache, "f04", log_path)
    if v4 and row.get("f04_variant") != v4:
        mark_phase(row, "f04", v4, True, manifest_path, rows)
    elif not v4:
        v4 = next_free_variant(repo, "f04_targets", 4)
        mark_phase(row, "f04", v4, False, manifest_path, rows)
        run_make(
            repo,
            log_path,
            "variant4",
            VARIANT=v4,
            PARENT=v3,
            THRESHOLD=threshold,
            DIRECTION=direction,
            NAME=name,
        )
        run_make(repo, log_path, "script4", VARIANT=v4)
        cache_completed_phase(phase_cache, "f04", job["f04_key"], v4, row)

    v5 = reusable_previous_variant(job, "f05", log_path)
    if v5:
        mark_phase(row, "f05", v5, False, manifest_path, rows)
        run_script5(repo, log_path, v5)
        cache_completed_phase(phase_cache, "f05", job["f05_key"], v5, row)
    else:
        v5 = reusable_cached_variant(job, phase_cache, "f05", log_path)
    if v5 and row.get("f05_variant") != v5:
        mark_phase(row, "f05", v5, True, manifest_path, rows)
    elif not v5:
        v5 = next_free_variant(repo, "f05_modeling", 5)
        mark_phase(row, "f05", v5, False, manifest_path, rows)
        run_make(
            repo,
            log_path,
            "variant5",
            VARIANT=v5,
            PARENT=v4,
            MODEL_FAMILY=base["model_family"],
            IMBALANCE_STRATEGY=base["imbalance_strategy"],
            IMBALANCE_MAX_MAJ=base["imbalance_max_maj"],
            DEDUP_MODE=job.get("deduplication_mode", "none"),
            SEED=base["seed"],
        )
        run_script5(repo, log_path, v5)
        cache_completed_phase(phase_cache, "f05", job["f05_key"], v5, row)


def execute_event_based_job(
    job: dict[str, Any],
    run_dir: Path,
    log_path: Path,
    row: dict[str, Any],
    manifest_path: Path,
    rows: list[dict[str, Any]],
    requires_measure_in_f02: bool,
    phase_cache: dict[str, dict[str, dict[str, Any]]],
) -> None:
    repo = job["repo_path"]
    base = job["base"]
    temporal = job["temporal"]
    measure = job["measure"]
    direction = job["direction"]
    threshold = job["effective_threshold"]
    bands = job["bands"]
    bands_id = job["bands_id"]
    bands_json = json_dumps_compact(bands)

    v2 = reusable_previous_variant(job, "f02", log_path)
    if v2:
        mark_phase(row, "f02", v2, False, manifest_path, rows)
        run_make(repo, log_path, "script2", VARIANT=v2)
        cache_completed_phase(phase_cache, "f02", job["f02_key"], v2, row)
    else:
        v2 = reusable_cached_variant(job, phase_cache, "f02", log_path)
    if v2 and row.get("f02_variant") != v2:
        mark_phase(row, "f02", v2, True, manifest_path, rows)
    elif not v2:
        v2 = next_free_variant(repo, "f02_events", 2)
        mark_phase(row, "f02", v2, False, manifest_path, rows)
        variant2_args = {
            "VARIANT": v2,
            "PARENT": job["parent_f01"],
        }
        if requires_measure_in_f02:
            variant2_args["MEASURE"] = measure
        variant2_args.update({
            "STRATEGY": base["event_strategy"],
            "BANDS": bands_json,
            "NAN_MODE": temporal["nan_mode"],
        })
        run_make(repo, log_path, "variant2", **variant2_args)
        run_make(repo, log_path, "script2", VARIANT=v2)
        cache_completed_phase(phase_cache, "f02", job["f02_key"], v2, row)

    v3 = reusable_previous_variant(job, "f03", log_path)
    if v3:
        mark_phase(row, "f03", v3, False, manifest_path, rows)
        run_make(repo, log_path, "script3", VARIANT=v3)
        cache_completed_phase(phase_cache, "f03", job["f03_key"], v3, row)
    else:
        v3 = reusable_cached_variant(job, phase_cache, "f03", log_path)
    if v3 and row.get("f03_variant") != v3:
        mark_phase(row, "f03", v3, True, manifest_path, rows)
    elif not v3:
        v3 = next_free_variant(repo, "f03_windows", 3)
        mark_phase(row, "f03", v3, False, manifest_path, rows)
        run_make(
            repo,
            log_path,
            "variant3",
            VARIANT=v3,
            PARENT=v2,
            OW=temporal["OW"],
            LT=temporal["LT"],
            PW=temporal["PW"],
            STRATEGY=temporal["window_strategy"],
            NAN_MODE=temporal["nan_mode"],
        )
        run_make(repo, log_path, "script3", VARIANT=v3)
        cache_completed_phase(phase_cache, "f03", job["f03_key"], v3, row)

    v4 = reusable_previous_variant(job, "f04", log_path)
    if v4:
        mark_phase(row, "f04", v4, False, manifest_path, rows)
        run_make(repo, log_path, "script4", VARIANT=v4)
        cache_completed_phase(phase_cache, "f04", job["f04_key"], v4, row)
    else:
        v4 = reusable_cached_variant(job, phase_cache, "f04", log_path)
    if v4 and row.get("f04_variant") != v4:
        mark_phase(row, "f04", v4, True, manifest_path, rows)
    elif not v4:
        target_events = resolve_target_events(repo, v2, v3, measure, direction, threshold, bands, bands_id)
        events_json = json_dumps_compact(target_events)
        v4 = next_free_variant(repo, "f04_targets", 4)
        mark_phase(row, "f04", v4, False, manifest_path, rows)
        name = safe_name(f"{measure}_{direction}_{compact_number(threshold)}_{bands_id}")
        run_make(
            repo,
            log_path,
            "variant4",
            VARIANT=v4,
            PARENT=v3,
            NAME=name,
            OPERATOR="OR",
            EVENTS=events_json,
        )
        run_make(repo, log_path, "script4", VARIANT=v4)
        cache_completed_phase(phase_cache, "f04", job["f04_key"], v4, row)

    v5 = reusable_previous_variant(job, "f05", log_path)
    if v5:
        mark_phase(row, "f05", v5, False, manifest_path, rows)
        run_script5(repo, log_path, v5)
        cache_completed_phase(phase_cache, "f05", job["f05_key"], v5, row)
    else:
        v5 = reusable_cached_variant(job, phase_cache, "f05", log_path)
    if v5 and row.get("f05_variant") != v5:
        mark_phase(row, "f05", v5, True, manifest_path, rows)
    elif not v5:
        v5 = next_free_variant(repo, "f05_modeling", 5)
        mark_phase(row, "f05", v5, False, manifest_path, rows)
        run_make(
            repo,
            log_path,
            "variant5",
            VARIANT=v5,
            PARENT=v4,
            MODEL_FAMILY=base["model_family"],
            IMBALANCE_STRATEGY=base["imbalance_strategy"],
            IMBALANCE_MAX_MAJ=base["imbalance_max_maj"],
            DEDUP_MODE=job.get("deduplication_mode", "none"),
            SEED=base["seed"],
        )
        run_script5(repo, log_path, v5)
        cache_completed_phase(phase_cache, "f05", job["f05_key"], v5, row)


def execute_events_job(
    job: dict[str, Any],
    run_dir: Path,
    log_path: Path,
    row: dict[str, Any],
    manifest_path: Path,
    rows: list[dict[str, Any]],
    phase_cache: dict[str, dict[str, dict[str, Any]]],
) -> None:
    execute_event_based_job(job, run_dir, log_path, row, manifest_path, rows, requires_measure_in_f02=False, phase_cache=phase_cache)


def execute_uni_events_job(
    job: dict[str, Any],
    run_dir: Path,
    log_path: Path,
    row: dict[str, Any],
    manifest_path: Path,
    rows: list[dict[str, Any]],
    phase_cache: dict[str, dict[str, dict[str, Any]]],
) -> None:
    execute_event_based_job(job, run_dir, log_path, row, manifest_path, rows, requires_measure_in_f02=True, phase_cache=phase_cache)


def load_structured_file(path: Path) -> Any:
    try:
        if path.suffix.lower() in {".yaml", ".yml"}:
            with path.open("r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        if path.suffix.lower() == ".json":
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        return None
    return None


def walk_strings(value: Any) -> list[str]:
    found: list[str] = []
    if isinstance(value, str):
        found.append(value)
    elif isinstance(value, dict):
        for key, item in value.items():
            found.extend(walk_strings(key))
            found.extend(walk_strings(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(walk_strings(item))
    return found


def resolve_target_events(
    repo_path: Path,
    f02_variant: str,
    f03_variant: str,
    measure: str,
    direction: str,
    effective_threshold: int | float,
    bands: list[Any],
    bands_id: str,
) -> list[str]:
    """Best-effort resolver for the legacy events pipeline F04 target EVENTS list.

    The events pipeline needs event names, while the comparison config only knows a
    measure, direction and effective threshold. We inspect generated YAML/JSON files
    from F02/F03 and only return names that are actually present. If the local
    format is different, this function is the intended small adjustment point.
    """
    threshold_token = compact_number(effective_threshold).replace("p", ".")
    if direction == "high":
        target_label = f"{threshold_token}_100"
        preferred = f"-to-{target_label}"
    else:
        target_label = f"0_{threshold_token}"
        preferred = f"-to-{target_label}"

    search_dirs = [
        repo_path / "executions" / "f02_events" / f02_variant,
        repo_path / "executions" / "f03_windows" / f03_variant,
    ]
    candidate_strings: list[str] = []
    for directory in search_dirs:
        if not directory.exists():
            continue
        for path in directory.iterdir():
            if path.suffix.lower() not in {".yaml", ".yml", ".json"}:
                continue
            data = load_structured_file(path)
            if data is not None:
                candidate_strings.extend(walk_strings(data))

    # Catalog JSONs often store event_name -> id, so dictionary keys are included.
    unique = sorted(set(candidate_strings))
    measure_matches = [s for s in unique if measure in s]
    preferred_matches = [s for s in measure_matches if preferred in s]
    if preferred_matches:
        return preferred_matches

    label_matches = [s for s in measure_matches if target_label in s]
    if label_matches:
        return label_matches

    raise RuntimeError(
        "Could not resolve target events for "
        f"measure={measure}, direction={direction}, effective_threshold={effective_threshold}, "
        f"bands_id={bands_id}, bands={bands}. "
        "Adjust resolve_target_events() for the actual events outputs format."
    )


def execute_job(
    job: dict[str, Any],
    run_id: str,
    run_dir: Path,
    manifest_path: Path,
    rows: list[dict[str, Any]],
    outputs_config: dict[str, Any],
    phase_cache: dict[str, dict[str, dict[str, Any]]],
) -> None:
    log_path = run_dir / "logs" / f"{job['job_key']}_{job['pipeline']}.log"
    row = initial_manifest_row(run_id, run_dir, job)
    append_or_update_manifest_row(manifest_path, rows, row)
    try:
        if job["pipeline"] == "ts":
            execute_ts_job(job, run_dir, log_path, row, manifest_path, rows, phase_cache)
        elif job["pipeline"] == "events":
            execute_events_job(job, run_dir, log_path, row, manifest_path, rows, phase_cache)
        elif job["pipeline"] == "uniEvents":
            execute_uni_events_job(job, run_dir, log_path, row, manifest_path, rows, phase_cache)
        else:
            raise RuntimeError(f"Unsupported pipeline: {job['pipeline']}")
        row["status"] = "COMPLETED"
        row["message"] = "OK"
        if outputs_config.get("collect_after_job", True):
            copy_configured_outputs(row, outputs_config, run_dir, log_path)
    except Exception as exc:
        row["status"] = "FAILED"
        row["message"] = str(exc)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n[FAILED] {exc}\n")
    finally:
        row["finished_at"] = datetime.now().isoformat(timespec="seconds")
        append_or_update_manifest_row(manifest_path, rows, row)


def main() -> None:
    cfg = load_config()
    validate_config(cfg)
    jobs = expand_jobs(cfg)
    outputs_config = load_outputs_config(cfg)
    completed, failed_attempts = load_job_attempts()
    phase_cache = build_phase_cache()
    pending = [job for job in jobs if job["job_key"] not in completed]
    for job in pending:
        previous_attempt = failed_attempts.get(job["job_key"])
        if previous_attempt:
            job["retry_from_failed"] = True
            job["previous_attempt"] = previous_attempt

    print(f"Desired jobs: {len(jobs)}")
    print(f"Already completed: {len(completed.intersection({j['job_key'] for j in jobs}))}")
    print(f"Pending jobs: {len(pending)}")
    print(f"Pending retries from FAILED: {sum(1 for job in pending if job.get('retry_from_failed'))}")

    if not pending:
        print("No new jobs to run. All desired jobs are already COMPLETED.")
        return

    run_id, run_dir = create_run_dir()
    manifest_path = run_dir / "manifest.csv"
    rows: list[dict[str, Any]] = []
    write_manifest(manifest_path, rows)

    print(f"Created run: {run_id}")
    for idx, job in enumerate(pending, start=1):
        print(f"[{idx}/{len(pending)}] Running {job['pipeline']} job {job['job_key']} ({job['measure']} {job['direction']} temporal={job['temporal_id']} threshold={job['effective_threshold']})")
        execute_job(job, run_id, run_dir, manifest_path, rows, outputs_config, phase_cache)

    print(f"Done. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
