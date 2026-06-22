#!/usr/bin/env python3
"""Compute PR-AUC for one F05 variant inside its source repo environment."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-path", required=True)
    parser.add_argument("--f05-dir", required=True)
    args = parser.parse_args()

    repo_path = Path(args.repo_path)
    f05_dir = Path(args.f05_dir)
    sys.path.insert(0, str(repo_path))

    from scripts.phases import f05_model as f05  # type: ignore
    import tensorflow as tf

    outputs = load_yaml(f05_dir / "outputs.yaml")
    params_data = load_yaml(f05_dir / "params.yaml")
    params = params_data.get("parameters") or {}
    exports = outputs.get("exports") or {}
    metrics = outputs.get("metrics") or {}
    artifacts = outputs.get("artifacts") or {}

    dataset_name = ((artifacts.get("labeled_dataset") or {}).get("path")) or "05_modeling_training_dataset.parquet"
    model_name = ((artifacts.get("model") or {}).get("path")) or "05_modeling_model.h5"
    dataset_path = f05_dir / dataset_name
    model_path = f05_dir / model_name

    n_train = int(metrics["n_train"])
    n_val = int(metrics["n_val"])
    n_test = int(metrics["n_test"])
    n_needed = n_train + n_val + n_test

    df = pd.read_parquet(dataset_path)
    if len(df) < n_needed:
        raise RuntimeError(f"Dataset has {len(df)} rows, expected at least {n_needed}")

    label_col = "label" if "label" in df.columns else "target"
    if label_col not in df.columns:
        raise RuntimeError("Could not infer label column: expected 'label' or 'target'")

    model_family = str(exports.get("model_family") or params.get("model_family"))
    vectorize = f05.vectorize_for_family
    signature = inspect.signature(vectorize)
    if len(signature.parameters) == 4:
        event_type_count = int(exports.get("event_type_count") or params.get("event_type_count") or 0)
        X, y, _info = vectorize(df, label_col, model_family, event_type_count)
    else:
        X, y, _info = vectorize(df, label_col, model_family)

    val_start = n_train
    test_start = n_train + n_val
    X_val = X[val_start:test_start]
    y_val = y[val_start:test_start]
    X_test = X[test_start:test_start + n_test]
    y_test = y[test_start:test_start + n_test]

    model = tf.keras.models.load_model(model_path, compile=False)
    batch_size = 32 if model_family == "cnn1d" else 64
    y_val_prob = model.predict(X_val, batch_size=batch_size, verbose=0).ravel()
    y_test_prob = model.predict(X_test, batch_size=batch_size, verbose=0).ravel()
    if hasattr(f05, "sanitize_probabilities"):
        y_val_prob = f05.sanitize_probabilities(y_val_prob, "val_pr_auc_backfill")
        y_test_prob = f05.sanitize_probabilities(y_test_prob, "test_pr_auc_backfill")
    else:
        y_val_prob = np.clip(np.nan_to_num(y_val_prob, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        y_test_prob = np.clip(np.nan_to_num(y_test_prob, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

    result = {
        "val_pr_auc": float(average_precision_score(y_val, y_val_prob)),
        "test_pr_auc": float(average_precision_score(y_test, y_test_prob)),
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
