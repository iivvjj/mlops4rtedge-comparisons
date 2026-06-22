#!/usr/bin/env python3
"""Rerun only F06 for all configured events jobs and refresh copied outputs."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from run_experiment import (
    build_phase_cache,
    copy_configured_outputs,
    create_run_dir,
    expand_jobs,
    initial_manifest_row,
    load_config,
    load_outputs_config,
    mark_phase,
    next_free_variant,
    reusable_cached_variant,
    run_make,
    run_script6,
    validate_config,
    variant_parent_matches,
    write_manifest,
)


PHASES_BEFORE_F06 = ("f02", "f03", "f04", "f05")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rerun F06 for every events job expanded from experiment.yaml. "
            "F02-F05 are reused from completed manifests."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be rerun; do not create variants, runs or outputs.",
    )
    parser.add_argument(
        "--force-new-variant",
        action="store_true",
        help="Always create a fresh F06 variant instead of rerunning the cached one.",
    )
    return parser.parse_args()


def cached_variant(job: dict, phase_cache: dict, phase: str, log_path: Path) -> str:
    return reusable_cached_variant(job, phase_cache, phase, log_path)


def prepare_row(
    job: dict,
    phase_cache: dict,
    run_id: str,
    run_dir: Path,
    manifest_path: Path,
    rows: list[dict],
    log_path: Path,
) -> tuple[dict, str]:
    row = initial_manifest_row(run_id, run_dir, job)
    for phase in PHASES_BEFORE_F06:
        variant = cached_variant(job, phase_cache, phase, log_path)
        if not variant:
            raise RuntimeError(
                f"Missing reusable {phase.upper()} variant for events job {job['job_key']}. "
                "Run scripts/run_experiment.py first so F05 exists."
            )
        mark_phase(row, phase, variant, True, manifest_path, rows)
    return row, row["f05_variant"]


def find_existing_f06(job: dict, phase_cache: dict, log_path: Path, parent_f05: str) -> str:
    variant = cached_variant(job, phase_cache, "f06", log_path)
    if not variant:
        return ""
    if not variant_parent_matches(job["repo_path"], "f06", variant, parent_f05):
        return ""
    return variant


def rerun_f06_job(
    job: dict,
    phase_cache: dict,
    run_id: str,
    run_dir: Path,
    manifest_path: Path,
    rows: list[dict],
    outputs_config: dict,
    force_new_variant: bool,
) -> None:
    log_path = run_dir / "logs" / f"{job['job_key']}_{job['pipeline']}_f06.log"
    row: dict | None = None
    try:
        row, parent_f05 = prepare_row(job, phase_cache, run_id, run_dir, manifest_path, rows, log_path)
        repo = job["repo_path"]
        f06_variant = "" if force_new_variant else find_existing_f06(job, phase_cache, log_path, parent_f05)

        if f06_variant:
            mark_phase(row, "f06", f06_variant, False, manifest_path, rows)
            run_script6(repo, log_path, f06_variant)
            row["message"] = f"Reran existing F06 variant {f06_variant}"
        else:
            f06_variant = next_free_variant(repo, "f06_quant", 6)
            mark_phase(row, "f06", f06_variant, False, manifest_path, rows)
            run_make(repo, log_path, "variant6", VARIANT=f06_variant, PARENT=parent_f05)
            run_script6(repo, log_path, f06_variant)
            row["message"] = f"Created and ran F06 variant {f06_variant}"

        row["status"] = "COMPLETED"
        copy_configured_outputs(row, outputs_config, run_dir, log_path)
    except Exception as exc:
        if row is None:
            row = initial_manifest_row(run_id, run_dir, job)
        row["status"] = "FAILED"
        row["message"] = str(exc)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n[FAILED] {exc}\n")
    finally:
        row["finished_at"] = datetime.now().isoformat(timespec="seconds")
        mark_phase(row, "f06", row.get("f06_variant", ""), bool(row.get("f06_reused", False)), manifest_path, rows)


def main() -> None:
    args = parse_args()
    cfg = load_config()
    validate_config(cfg)
    jobs = [job for job in expand_jobs(cfg) if job["pipeline"] == "events"]
    phase_cache = build_phase_cache()

    print(f"Events jobs from experiment.yaml: {len(jobs)}")
    if args.dry_run:
        for job in jobs:
            print(
                f"Would rerun F06 for {job['job_key']} "
                f"({job['measure']} {job['direction']} temporal={job['temporal_id']} "
                f"bands={job['bands_id']} threshold={job['effective_threshold']})"
            )
        return

    outputs_config = load_outputs_config(cfg)
    run_id, run_dir = create_run_dir()
    manifest_path = run_dir / "manifest.csv"
    rows: list[dict] = []
    write_manifest(manifest_path, rows)

    print(f"Created run: {run_id}")
    for idx, job in enumerate(jobs, start=1):
        print(f"[{idx}/{len(jobs)}] Rerunning F06 for events job {job['job_key']}")
        rerun_f06_job(
            job,
            phase_cache,
            run_id,
            run_dir,
            manifest_path,
            rows,
            outputs_config,
            args.force_new_variant,
        )

    print(f"Done. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
