#!/usr/bin/env python3
"""Backfill PR-AUC metrics for already trained F05 models."""

from __future__ import annotations

import argparse
import inspect
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_INDEX_PATH = REPO_ROOT / "registry" / "model_index.yaml"
WORKER_PATH = REPO_ROOT / "scripts" / "pr_auc_worker.py"


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def normalize_path(value: Any) -> str:
    return str(value).replace("\\", "/")


def local_path(value: Any) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    return path if path.is_absolute() else REPO_ROOT / path


def iter_models(index: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    models_by_measure = index.get("models_by_measure") or {}
    for directions in models_by_measure.values():
        for thresholds in (directions or {}).values():
            for threshold_node in (thresholds or {}).values():
                temporal_configs = (threshold_node or {}).get("temporal_configs") or {}
                for temporal_node in temporal_configs.values():
                    pipelines = (temporal_node or {}).get("pipelines") or {}
                    for bands_map in pipelines.values():
                        for bands_node in (bands_map or {}).values():
                            models = bands_node.get("models") if isinstance(bands_node, dict) else None
                            if isinstance(models, list):
                                rows.extend(model for model in models if isinstance(model, dict))
    return rows


def repo_python(repo_path: Path) -> Path:
    candidate = repo_path / ".venv" / "Scripts" / "python.exe"
    return candidate if candidate.exists() else Path(sys.executable)


def compute_pr_auc(model: dict[str, Any]) -> dict[str, Any]:
    repo_path = local_path(model.get("repo_path"))
    f05_path = local_path((model.get("variant_paths") or {}).get("f05"))
    if repo_path is None or f05_path is None:
        raise RuntimeError("Missing repo_path or f05 variant path")

    command = [
        str(repo_python(repo_path)),
        str(WORKER_PATH),
        "--repo-path",
        str(repo_path),
        "--f05-dir",
        str(f05_path),
    ]
    proc = subprocess.run(command, cwd=repo_path, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or "worker failed").strip())
    return json.loads(proc.stdout)


def update_outputs(path: Path, metrics: dict[str, Any]) -> bool:
    outputs = load_yaml(path)
    if not outputs:
        return False

    changed = False
    for section_name in ("exports", "metrics"):
        section = outputs.setdefault(section_name, {})
        if isinstance(section, dict):
            for key in ("val_pr_auc", "test_pr_auc"):
                if section.get(key) != metrics[key]:
                    section[key] = metrics[key]
                    changed = True

    mlflow = outputs.setdefault("mlflow_registration", {})
    if isinstance(mlflow, dict):
        mlflow_metrics = mlflow.setdefault("metrics", {})
        if isinstance(mlflow_metrics, dict):
            for key in ("val_pr_auc", "test_pr_auc"):
                if mlflow_metrics.get(key) != metrics[key]:
                    mlflow_metrics[key] = metrics[key]
                    changed = True

    if changed:
        write_yaml(path, outputs)
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-index", default=str(MODEL_INDEX_PATH))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    index = load_yaml(Path(args.model_index))
    models = iter_models(index)
    if args.limit is not None:
        models = models[: args.limit]

    updated = 0
    skipped = 0
    failed = 0
    for pos, model in enumerate(models, start=1):
        f05_path = local_path((model.get("variant_paths") or {}).get("f05"))
        local_output = local_path(model.get("local_output_path"))
        label = f"{model.get('run_id')}/{model.get('job_key')}/{(model.get('variants') or {}).get('f05')}"
        if not f05_path:
            skipped += 1
            print(f"[SKIP] {label}: missing f05 path")
            continue

        try:
            metrics = compute_pr_auc(model)
        except Exception as exc:
            failed += 1
            print(f"[FAIL] {label}: {exc}")
            continue

        targets = [f05_path / "outputs.yaml"]
        if local_output:
            targets.append(local_output / "f05" / "outputs.yaml")

        changed_any = False
        if not args.dry_run:
            for target in targets:
                changed_any = update_outputs(target, metrics) or changed_any
        if changed_any:
            updated += 1
            status = "UPDATED"
        else:
            skipped += 1
            status = "OK"
        print(
            f"[{status}] {pos}/{len(models)} {label}: "
            f"val_pr_auc={metrics['val_pr_auc']:.6f}, test_pr_auc={metrics['test_pr_auc']:.6f}"
        )

    print(f"Done. updated={updated}, skipped={skipped}, failed={failed}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
