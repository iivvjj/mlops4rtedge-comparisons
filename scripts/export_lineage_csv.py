#!/usr/bin/env python3
"""Export a clean, shareable CSV of experiments from 3 external repos
(events/ts/uniEvents): one row per experiment (leaf variant of the lineage
chain), with only the parameters and results a non-technical reader needs.
No filesystem paths or internal variant/lineage codes in the output.

Each repo tracks its own execution history under `executions/`:
  - If `executions/pipeline_lineage.html` exists, it embeds a JS constant
    `variantConfigs` with a JSON registry of every (phase, variant) node:
    its parameters, its parent variant, and its lifecycle metadata. This is
    the richest source and survives pruning of on-disk params/metadata files.
  - Otherwise we fall back to reading `executions/f0N_xxx/vN_YYYY/params.yaml`
    and `metadata.yaml` directly from disk.

Metrics (precision/recall/f1/...) live in a separate `outputs.yaml` written
next to params.yaml/metadata.yaml at each phase variant folder. This file is
never embedded in pipeline_lineage.html, so it is always read from disk
(regardless of which lineage source was used) for every node in a chain.

A "leaf" is a (phase, variant) that no other node references as its parent.
Leaves capture both fully completed experiments (e.g. reaching f06/f07/f08)
and experiments that stalled partway (e.g. a f04 variant nobody ever built a
f05 on top of, or a phase that failed) -- surfaced via the `estado` column.
`fase_alcanzada` is the last phase that actually produced a real, compatible
outputs.yaml: some phases (f04/f05/f06/f07) auto-advance to the next one even
after flagging themselves incompatible/not-trainable/not-edge-capable, only
writing an inert stub downstream, so the leaf's own phase is not always the
right answer -- see `phase_is_compatible`.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_REPOS: dict[str, Path] = {
    # Sibling clones next to this repo; override with --repo NAME=PATH
    "events": REPO_ROOT.parent / "mlops4rtedge",
    "ts": REPO_ROOT.parent / "mlops4rtedgeTS",
    "uniEvents": REPO_ROOT.parent / "mlops4rtedgeUni",
}

DEFAULT_OUTPUT = REPO_ROOT / "exported_params" / "lineage_export.csv"

PHASE_NUM_RE = re.compile(r"^f(\d+)_")
LINEAGE_JS_RE = re.compile(r"const variantConfigs\s*=\s*(\{.*?\});\s*\n", re.DOTALL)
VARIANT_KEY_RE = re.compile(r"^(f\d+_\w+)_(v\d+_\d+)$")


def phase_num(phase: str) -> int:
    m = PHASE_NUM_RE.match(phase)
    return int(m.group(1)) if m else -1


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def load_nodes_from_lineage_html(html_path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    content = html_path.read_text(encoding="utf-8", errors="replace")
    match = LINEAGE_JS_RE.search(content)
    if not match:
        return {}
    try:
        raw = json.loads(match.group(1))
    except json.JSONDecodeError:
        return {}

    nodes: dict[tuple[str, str], dict[str, Any]] = {}
    for key, entry in raw.items():
        params = entry.get("params") or {}
        phase = params.get("phase")
        variant = params.get("variant")
        if not phase or not variant:
            key_match = VARIANT_KEY_RE.match(key)
            if not key_match:
                continue
            phase, variant = key_match.group(1), key_match.group(2)
        nodes[(phase, variant)] = {
            "phase": phase,
            "variant": variant,
            "parent": params.get("parent"),
            "parameters": params.get("parameters") or {},
            "metadata": entry.get("metadata") or {},
        }
    return nodes


def load_nodes_from_disk(executions_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    nodes: dict[tuple[str, str], dict[str, Any]] = {}
    for phase_dir in sorted(p for p in executions_dir.glob("f*_*") if p.is_dir()):
        phase = phase_dir.name
        for variant_dir in sorted(v for v in phase_dir.glob("v*_*") if v.is_dir()):
            params = load_yaml(variant_dir / "params.yaml")
            metadata = load_yaml(variant_dir / "metadata.yaml")
            variant = params.get("variant") or variant_dir.name
            nodes[(phase, variant)] = {
                "phase": phase,
                "variant": variant,
                "parent": params.get("parent"),
                "parameters": params.get("parameters") or {},
                "metadata": metadata,
            }
    return nodes


def load_repo_nodes(repo_path: Path) -> tuple[dict[tuple[str, str], dict[str, Any]], str]:
    """The pipeline_lineage.html registry can miss variants that were run
    without going through the normal registration step (e.g. manual local
    runs) even though their folder, params.yaml, metadata.yaml and
    outputs.yaml all exist on disk. So: start from the registry (it has the
    richest data and survives file pruning), then add any variant folder
    found on disk that the registry doesn't already know about."""
    executions_dir = repo_path / "executions"
    lineage_html = executions_dir / "pipeline_lineage.html"
    nodes: dict[tuple[str, str], dict[str, Any]] = {}
    source = "none"
    if lineage_html.exists():
        nodes = load_nodes_from_lineage_html(lineage_html)
        if nodes:
            source = "pipeline_lineage.html"

    if executions_dir.exists():
        disk_nodes = load_nodes_from_disk(executions_dir)
        added = 0
        for key, node in disk_nodes.items():
            if key not in nodes:
                nodes[key] = node
                added += 1
        if not nodes:
            source = "disk_scan"
        elif added:
            source = f"{source}+disk_scan({added} extra)"

    return nodes, source


def is_phantom_leaf(chain: list[dict[str, Any]]) -> bool:
    """A leaf that is just a reserved variant slot: never configured, never
    linked to a parent, and not the dataset-exploration root (f01)."""
    if len(chain) != 1:
        return False
    node = chain[0]
    return not node.get("parameters") and not node.get("parent") and phase_num(node["phase"]) != 1


def find_leaves(nodes: dict[tuple[str, str], dict[str, Any]]) -> list[tuple[str, str]]:
    phases_present = sorted({phase for phase, _ in nodes}, key=phase_num)
    phase_by_num = {phase_num(p): p for p in phases_present}

    referenced = set()
    for (phase, _variant), node in nodes.items():
        parent = node.get("parent")
        if not parent:
            continue
        parent_phase = phase_by_num.get(phase_num(phase) - 1)
        if parent_phase is not None:
            referenced.add((parent_phase, parent))

    return [key for key in nodes if key not in referenced]


def build_chain(
    leaf_key: tuple[str, str], nodes: dict[tuple[str, str], dict[str, Any]]
) -> tuple[list[dict[str, Any]], bool, str]:
    phases_present = sorted({phase for phase, _ in nodes}, key=phase_num)
    phase_by_num = {phase_num(p): p for p in phases_present}

    chain: list[dict[str, Any]] = []
    current_key: tuple[str, str] | None = leaf_key
    complete = True
    gap = ""
    seen = set()
    while current_key is not None:
        if current_key in seen:
            gap = f"cycle detected at {current_key}"
            complete = False
            break
        seen.add(current_key)
        node = nodes.get(current_key)
        if node is None:
            gap = f"missing node {current_key}"
            complete = False
            break
        chain.append(node)
        parent = node.get("parent")
        phase, _variant = current_key
        if not parent:
            current_key = None
            continue
        parent_phase = phase_by_num.get(phase_num(phase) - 1)
        if parent_phase is None:
            current_key = None
            continue
        parent_key = (parent_phase, parent)
        if parent_key not in nodes:
            gap = f"missing parent {parent} in {parent_phase}"
            complete = False
            current_key = None
            continue
        current_key = parent_key
    chain.reverse()
    return chain, complete, gap


def read_phase_outputs(repo_path: Path, phase: str, variant: str) -> dict[str, Any]:
    outputs_path = repo_path / "executions" / phase / variant / "outputs.yaml"
    return load_yaml(outputs_path)


# Once a phase marks its own outputs as incompatible/not-trainable/not-edge-capable,
# downstream phases keep auto-advancing but only write an inert stub outputs.yaml
# (no real artifacts, no real work) that just carries the same failure forward.
# These are the per-phase keys (checked in `exports`) that signal a *real* pass.
PHASE_COMPATIBILITY_KEYS: dict[int, tuple[str, ...]] = {
    4: ("compatible", "target_compatible"),
    5: ("trainable",),
    6: ("edge_capable",),
    7: ("edge_capable",),
}


def phase_is_compatible(phase_outputs: dict[str, Any], phase_n: int) -> bool:
    keys = PHASE_COMPATIBILITY_KEYS.get(phase_n)
    if not keys:
        return True
    exports = phase_outputs.get("exports") or {}
    for key in keys:
        value = exports.get(key)
        if value is not None:
            return bool(value)
    return True


def node_status(metadata: dict[str, Any]) -> str:
    if "lifecycle_state" in metadata and metadata["lifecycle_state"]:
        return str(metadata["lifecycle_state"])
    if str(metadata.get("registred")) == "none":
        return "NOT_EXECUTED"
    return "UNKNOWN"

STATUS_LABELS = {
    "EXECUTION_COMPLETED": "Completado",
    "EXECUTION_FAILED": "Fallido",
    "VARIANT_CREATED": "Creado (no ejecutado)",
    "NOT_EXECUTED": "No ejecutado",
    "UNKNOWN": "Desconocido",
}

def dig(data: Any, *keys: str) -> Any:
    current = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def chain_index(chain: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {phase_num(node["phase"]): node for node in chain}


def find_param(idx: dict[int, dict[str, Any]], phases: tuple[int, ...], key: str) -> Any:
    for n in phases:
        node = idx.get(n)
        if node is None:
            continue
        value = (node.get("parameters") or {}).get(key)
        if value not in (None, ""):
            return value
    return None


def find_output(
    outs: dict[int, dict[str, Any]], phases: tuple[int, ...], *keys: str
) -> Any:
    for n in phases:
        value = dig(outs.get(n) or {}, "exports", *keys)
        if value not in (None, ""):
            return value
    return None


def load_known_measures() -> list[str]:
    data = load_yaml(REPO_ROOT / "experiment.yaml")
    return list((data.get("grid") or {}).get("measures") or [])


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


PREDICTION_NAME_RE = re.compile(r"^(high|low)_(\d+)(?:_(.+))?$")


def parse_prediction_name(name: str, measure_slugs: dict[str, str]) -> dict[str, Any]:
    if not name:
        return {}
    for slug in sorted(measure_slugs, key=len, reverse=True):
        prefix = f"{slug}_"
        if not name.startswith(prefix):
            continue
        match = PREDICTION_NAME_RE.match(name[len(prefix):])
        if match:
            return {
                "measure": measure_slugs[slug],
                "direction": match.group(1),
                "threshold": int(match.group(2)),
                "bands_id": match.group(3),
            }
    return {}


def build_row(
    pipeline: str, repo_path: Path, leaf_key: tuple[str, str], chain: list[dict[str, Any]],
    measure_slugs: dict[str, str],
) -> dict[str, Any]:
    leaf_phase, leaf_variant = leaf_key
    leaf_node = chain[-1] if chain else {}
    leaf_status = node_status(leaf_node.get("metadata", {})) if chain else "UNKNOWN"

    idx = chain_index(chain)
    outs = {n: read_phase_outputs(repo_path, node["phase"], node["variant"]) for n, node in idx.items()}

    # The lineage registry's lifecycle_state can go stale (e.g. a variant run
    # manually outside the normal flow never gets its VARIANT_CREATED record
    # updated). A real outputs.yaml on disk for the leaf phase is stronger
    # evidence that it actually ran than a "not executed" registry status.
    leaf_outputs = outs.get(phase_num(leaf_phase)) or {}
    if leaf_outputs and leaf_status in ("NOT_EXECUTED", "VARIANT_CREATED", "UNKNOWN"):
        leaf_status = "EXECUTION_COMPLETED"

    # fase_alcanzada reports the last phase that actually produced usable
    # outputs, not just the leaf variant's phase: a leaf can exist (e.g. f05
    # FAILED) without ever writing outputs for that phase, and a phase can
    # write outputs.yaml while flagging itself incompatible (see
    # PHASE_COMPATIBILITY_KEYS) -- downstream phases then just carry that
    # failure forward as inert stubs. Walk the chain from the root and stop
    # at the first phase that has no outputs or fails its own gate.
    last_output_phase = 0
    for n in sorted(idx.keys()):
        phase_outputs = outs.get(n)
        if not phase_outputs or not phase_is_compatible(phase_outputs, n):
            break
        last_output_phase = n
    if last_output_phase == 0:
        last_output_phase = phase_num(leaf_phase)

    bands = find_param(idx, (2,), "bands")
    bands_text = "; ".join(str(b) for b in bands) if isinstance(bands, list) else ""

    # Older variants only recorded `prediction_name` (e.g. "ge_active_power_high_90_coarse_10"),
    # not the explicit measure_name/direction/threshold fields added later. Parse it as a fallback.
    prediction_name = find_param(idx, (4, 5, 6, 7), "prediction_name")
    parsed = parse_prediction_name(prediction_name, measure_slugs)

    measure = find_param(idx, (2, 3), "measure_name") or find_output(outs, (2, 3, 4, 5, 6, 7), "measure_name") or parsed.get("measure")
    direction = find_param(idx, (4,), "direction") or parsed.get("direction")
    threshold = find_param(idx, (4,), "threshold") or parsed.get("threshold")

    variantes_padre = json.dumps([node["variant"] for node in chain], ensure_ascii=False)

    row: dict[str, Any] = {
        "pipeline": pipeline,
        "id_experimento": f"{pipeline}_{leaf_variant}",
        "estado": STATUS_LABELS.get(leaf_status, leaf_status),
        "fase_alcanzada": f"f{last_output_phase:02d}",
        "variantes_padre": variantes_padre,
        "measure": measure,
        "direction": direction,
        "threshold": threshold,
        "bands": bands_text,
        "OW": find_param(idx, (3, 4, 5, 6, 7), "OW"),
        "LT": find_param(idx, (3, 4, 5, 6, 7), "LT"),
        "PW": find_param(idx, (3, 4, 5, 6, 7), "PW"),
        "window_strategy": find_param(idx, (3,), "window_strategy") or find_output(outs, (5,), "window_strategy"),
        "nan_mode": find_param(idx, (2, 3), "nan_mode"),
        "event_strategy": find_param(idx, (4,), "event_strategy"),
        "model_family": find_param(idx, (5,), "model_family") or find_output(outs, (5, 6), "model_family"),
        "seed": find_param(idx, (5,), "seed"),
        "imbalance_strategy": find_param(idx, (5,), "imbalance_strategy"),
        "imbalance_max_maj": find_param(idx, (5,), "imbalance_max_majority_samples"),
        "dedup_mode": find_param(idx, (5,), "deduplication_mode"),


        "val_precision": find_output(outs, (5,), "val_precision"),
        "val_recall": find_output(outs, (5,), "val_recall"),
        "val_f1": find_output(outs, (5,), "val_f1"),
        "val_pr_auc": find_output(outs, (5,), "val_pr_auc"),
        "test_precision": find_output(outs, (5,), "test_precision"),
        "test_recall": find_output(outs, (5,), "test_recall"),
        "test_f1": find_output(outs, (5,), "test_f1"),
        "test_pr_auc": find_output(outs, (5,), "test_pr_auc"),
        "test_tp": dig(outs.get(5) or {}, "metrics", "tp"),
        "test_tn": dig(outs.get(5) or {}, "metrics", "tn"),
        "test_fp": dig(outs.get(5) or {}, "metrics", "fp"),
        "test_fn": dig(outs.get(5) or {}, "metrics", "fn"),
        "decision_threshold": find_output(outs, (5,), "decision_threshold"),

        "tflite_test_precision": find_output(outs, (6,), "tflite_test_precision"),
        "tflite_test_recall": find_output(outs, (6,), "tflite_test_recall"),
        "tflite_test_f1": find_output(outs, (6,), "tflite_test_f1"),
        "tflite_test_pr_auc": find_output(outs, (6,), "tflite_test_pr_auc"),
        "tflite_test_fp": find_output(outs, (6,), "tflite_test_fp"),
        "tflite_test_fn": find_output(outs, (6,), "tflite_test_fn"),
        "model_size_bytes": find_output(outs, (6,), "model_size_bytes"),
        "arena_estimated_bytes": find_output(outs, (6,), "arena_estimated_bytes"),

        "f07_platform": find_param(idx, (7,), "platform"),
        "f07_virtual": find_param(idx, (7,), "virtual"),
        "f07_time_scale_factor": find_param(idx, (7,), "time_scale_factor"),
        "f07_MTI_MS": find_param(idx, (7,), "MTI_MS"),
        "f07_max_rows": find_param(idx, (7,), "max_rows"),
        "f07_esp_flash_size_mb": find_param(idx, (7,), "esp_flash_size_mb"),

        "edge_capable": find_output(outs, (7,), "edge_capable"),
        "edge_run_completed": find_output(outs, (7,), "edge_run_completed"),
        "edge_phase_status_reason": find_output(outs, (7,), "phase_status_reason"),
        "edge_quality_score": find_output(outs, (7,), "model_stats", "quality_score"),
        "edge_precision": find_output(outs, (7,), "model_stats", "quality", "precision"),
        "edge_recall": find_output(outs, (7,), "model_stats", "quality", "recall"),
        "edge_f1": find_output(outs, (7,), "model_stats", "quality", "f1"),
        "edge_accuracy": find_output(outs, (7,), "model_stats", "quality", "accuracy"),
        "edge_tp": find_output(outs, (7,), "model_stats", "quality", "tp"),
        "edge_tn": find_output(outs, (7,), "model_stats", "quality", "tn"),
        "edge_fp": find_output(outs, (7,), "model_stats", "quality", "fp"),
        "edge_fn": find_output(outs, (7,), "model_stats", "quality", "fn"),
        "edge_n_attempts": find_output(outs, (7,), "model_stats", "n_attempts"),
        "edge_n_ok": find_output(outs, (7,), "model_stats", "n_ok"),
        "edge_n_fail": find_output(outs, (7,), "model_stats", "n_fail"),
        "edge_n_inferences": find_output(outs, (7,), "model_stats", "n_inferences"),
        "edge_ok_rate": find_output(outs, (7,), "model_stats", "ok_rate"),
        "edge_max_latency_ms": find_output(outs, (7,), "model_stats", "timing", "edge_max_latency_ms"),
        "edge_jitter_ms": find_output(outs, (7,), "model_stats", "timing", "edge_jitter_ms"),
        "edge_memory_peak_bytes": find_output(outs, (7,), "model_stats", "memory", "edge_memory_peak_bytes"),
        "edge_model_memory_bytes": find_output(outs, (7,), "model_memory_bytes"),
        "edge_arena_bytes": find_output(outs, (7,), "arena_bytes"),
    }
    return row


COLUMN_ORDER = [
    "pipeline", "id_experimento", "estado", "fase_alcanzada", "variantes_padre",
    "measure", "direction", "threshold", "bands", "OW", "LT", "PW",
    "window_strategy", "nan_mode", "event_strategy",
    "model_family", "seed", "imbalance_strategy", "imbalance_max_maj", "dedup_mode",
    "val_precision", "val_recall", "val_f1", "val_pr_auc",
    "test_precision", "test_recall", "test_f1", "test_pr_auc",
    "test_tp", "test_tn", "test_fp", "test_fn", "decision_threshold",
    "tflite_test_precision", "tflite_test_recall", "tflite_test_f1", "tflite_test_pr_auc",
    "tflite_test_fp", "tflite_test_fn", "model_size_bytes", "arena_estimated_bytes",
    "f07_platform", "f07_virtual", "f07_time_scale_factor", "f07_MTI_MS",
    "f07_max_rows", "f07_esp_flash_size_mb",
    "edge_capable", "edge_run_completed", "edge_phase_status_reason", "edge_quality_score",
    "edge_precision", "edge_recall", "edge_f1", "edge_accuracy",
    "edge_tp", "edge_tn", "edge_fp", "edge_fn",
    "edge_n_attempts", "edge_n_ok", "edge_n_fail", "edge_n_inferences", "edge_ok_rate", "edge_max_latency_ms", "edge_jitter_ms",
    "edge_memory_peak_bytes", "edge_model_memory_bytes", "edge_arena_bytes",
]


def export(repos: dict[str, Path], output_path: Path) -> None:
    all_rows: list[dict[str, Any]] = []
    measure_slugs = {slugify(measure): measure for measure in load_known_measures()}

    for pipeline, repo_path in repos.items():
        nodes, lineage_source = load_repo_nodes(repo_path)
        if not nodes:
            print(f"[{pipeline}] {repo_path}: no execution data found (lineage_source={lineage_source}).")
            continue

        leaves = find_leaves(nodes)
        skipped_phantoms = 0
        kept = 0
        for leaf_key in leaves:
            chain, _complete, _gap = build_chain(leaf_key, nodes)
            if not chain:
                continue
            if is_phantom_leaf(chain):
                skipped_phantoms += 1
                continue
            row = build_row(pipeline, repo_path, leaf_key, chain, measure_slugs)
            all_rows.append(row)
            kept += 1
        print(
            f"[{pipeline}] {repo_path}: {len(nodes)} nodes, {len(leaves)} leaf branches "
            f"(source={lineage_source}); {kept} kept, {skipped_phantoms} never-configured slots skipped."
        )

    if not all_rows:
        print("No rows produced. Nothing written.")
        return

    all_rows.sort(key=lambda r: (r["pipeline"], str(r["measure"]), str(r["id_experimento"])))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMN_ORDER, extrasaction="ignore")
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    print(f"Wrote {len(all_rows)} rows x {len(COLUMN_ORDER)} columns to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--repo",
        action="append",
        default=[],
        metavar="PIPELINE=PATH",
        help="Override/add a repo, e.g. --repo uniEvents=C:/path/to/mlops4rtedgeUni. Repeatable.",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output CSV path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repos = dict(DEFAULT_REPOS)
    for override in args.repo:
        if "=" not in override:
            raise SystemExit(f"--repo must be PIPELINE=PATH, got: {override}")
        pipeline, path = override.split("=", 1)
        repos[pipeline] = Path(path)

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path

    export(repos, output_path)


if __name__ == "__main__":
    main()