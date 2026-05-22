#!/usr/bin/env python3
"""Collect configured outputs for already completed manifest rows."""

from __future__ import annotations

import csv
from pathlib import Path

from run_experiment import RUNS_DIR, copy_configured_outputs, load_config, load_outputs_config


def collect_outputs() -> None:
    cfg = load_config()
    outputs_config = load_outputs_config(cfg)
    if not outputs_config.get("enabled", True):
        print("Output collection is disabled in experiment.yaml")
        return

    manifests = sorted(RUNS_DIR.glob("*/manifest.csv"))
    if not manifests:
        print("No manifests found under runs/*/manifest.csv")
        return

    total = 0
    for manifest in manifests:
        run_dir = manifest.parent
        log_path = run_dir / "logs" / "collect_outputs.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row.get("status") != "COMPLETED":
                    continue
                copy_configured_outputs(row, outputs_config, run_dir, log_path)
                total += 1

    print(f"Collected outputs for {total} COMPLETED manifest rows.")


if __name__ == "__main__":
    collect_outputs()
