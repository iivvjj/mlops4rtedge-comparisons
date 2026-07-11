#!/usr/bin/env python3
"""Build a visual HTML comparison report from model_index.yaml and copied outputs."""

from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

import build_model_index


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_INDEX_PATH = REPO_ROOT / "registry" / "model_index.yaml"
REPORT_PATH = REPO_ROOT / "reports" / "report.html"

PIPELINE_ORDER = ["ts", "events", "uniEvents"]
PIPELINE_LABELS = {"ts": "TS", "events": "Events", "uniEvents": "UniEvents"}

F05_METRIC_KEYS = [
    "selection_metric",
    "selection_score",
    "min_precision",
    "min_precision_satisfied",
    "selected_threshold",
    "threshold_source",
    "threshold_policy",
    "val_precision",
    "val_recall",
    "val_f1",
    "test_tp",
    "test_tn",
    "test_fp",
    "test_fn",
    "val_pr_auc",
    "test_pr_auc",
    "pr_auc",
    "average_precision",
    "test_average_precision",
    "val_precision_at_threshold",
    "val_recall_at_threshold",
    "val_f1_at_threshold",
    "test_precision_at_threshold",
    "test_recall_at_threshold",
    "test_f1_at_threshold",
    "val_positive_ratio",
    "test_positive_ratio",
    "val_cap_used",
    "test_cap_used",
    "val_max_samples",
    "test_max_samples",
    "val_positive_ratio_before",
    "val_positive_ratio_after",
    "test_positive_ratio_before",
    "test_positive_ratio_after",
    "test_precision",
    "test_recall",
    "test_f1",
    "best_val_recall",
    "decision_threshold",
    "n_train",
    "n_val",
    "n_test",
    "positive_ratio_train",
    "positive_ratio_val",
    "positive_ratio_test",
    "tp",
    "tn",
    "fp",
    "fn",
    "execution_time",
]

CRITERION_COLUMNS = [
    "best_by_f1_f1",
    "best_by_f1_recall",
    "best_by_f1_precision",
    "best_by_precision_f1",
    "best_by_precision_recall",
    "best_by_precision_precision",
    "best_by_recall_f1",
    "best_by_recall_recall",
    "best_by_recall_precision",
    "delta_f1_choice_f1_vs_recall_choice",
    "delta_f1_choice_recall_vs_recall_choice",
    "delta_precision_choice_precision_vs_recall_choice",
    "delta_precision_choice_recall_vs_recall_choice",
]

BLOCK_TABLE_COLUMNS = [
    "pipeline",
    "bands_id",
    "deduplication_mode",
    "f05_variant",
    "f05_compatible",
    "f05_incompatibility_reason",
    "f06_variant",
    "tflm_compatible",
    "f06_incompatibility_reason",
    "test_precision_effective",
    "test_recall_effective",
    "test_f1_effective",
    "test_fn_effective",
    "test_pr_auc_effective",
    "val_recall_effective",
    "decision_threshold_effective",
    "threshold_source_effective",
    "threshold_policy_effective",
    "positive_ratio_train",
    "val_positive_ratio_effective",
    "test_positive_ratio_effective",
    "n_train",
    "n_val",
    "n_test",
    *CRITERION_COLUMNS,
    "local_output_path",
]


def load_model_index(path: Path = MODEL_INDEX_PATH) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing model index: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid model index YAML: {path}")
    return data


def rebuild_model_index(path: Path = MODEL_INDEX_PATH) -> None:
    manifest_paths = build_model_index.find_manifest_paths()
    source_manifests = list(manifest_paths)
    rows = build_model_index.read_manifest_rows(manifest_paths, include_failed=False) if manifest_paths else []
    exported_manifest = build_model_index.DEFAULT_EXPORTED_MANIFEST
    exported_rows = build_model_index.read_exported_manifest_rows(exported_manifest)
    if exported_rows:
        rows.extend(exported_rows)
        source_manifests.append(exported_manifest)
    if not rows:
        print("No valid manifest rows found.")
    index = build_model_index.build_index(rows, source_manifests)
    build_model_index.write_yaml(index, path)
    print(f"Wrote {path.relative_to(REPO_ROOT)} with {index['summary']['n_models']} models.")


def load_yaml(path: Path | None) -> dict[str, Any]:
    if not path or not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def normalize_path(value: Any) -> str:
    return str(value or "").replace("\\", "/")


def local_path(value: Any) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text)
    return path if path.is_absolute() else REPO_ROOT / path


def walk_values(data: Any, target_key: str) -> Any:
    if isinstance(data, dict):
        if target_key in data:
            return data[target_key]
        for value in data.values():
            found = walk_values(value, target_key)
            if found not in (None, ""):
                return found
    elif isinstance(data, list):
        for item in data:
            found = walk_values(item, target_key)
            if found not in (None, ""):
                return found
    return None


def as_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt(value: Any) -> str:
    if value in (None, ""):
        return "-"
    if isinstance(value, float):
        return f"{value:.6g}"
    parsed = as_float(value)
    if parsed is not None and "." in str(value):
        return f"{parsed:.6g}"
    return str(value)


def esc(value: Any) -> str:
    return html.escape(fmt(value))


def threshold_sort_value(value: str) -> tuple[int, str]:
    parsed = as_float(str(value).replace("threshold_", "").replace("p", "."))
    if parsed is None:
        return (1, str(value))
    return (0, f"{parsed:020.8f}")


def direction_sort_value(direction: str) -> tuple[int, str]:
    order = {"high": 0, "low": 1}
    return (order.get(direction, 2), direction)


def first_available(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value not in (None, ""):
            return value
    return None


def normalize_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "si", "sí"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def falseish(value: Any) -> bool:
    return normalize_bool(value) is False


def nested_metric(block: Any, metric: str) -> float | None:
    if not isinstance(block, dict):
        return None
    return as_float(block.get(metric))


def delta(left: Any, right: Any) -> float | None:
    left_float = as_float(left)
    right_float = as_float(right)
    if left_float is None or right_float is None:
        return None
    return left_float - right_float


def extract_criterion_metrics(outputs: dict[str, Any]) -> dict[str, Any]:
    metrics_root = outputs.get("metrics") if isinstance(outputs.get("metrics"), dict) else {}
    best_by_f1 = metrics_root.get("mejor_modelo_f1") or walk_values(outputs, "mejor_modelo_f1")
    best_by_precision = metrics_root.get("mejor_modelo_precision") or walk_values(outputs, "mejor_modelo_precision")
    best_by_recall = metrics_root.get("mejor_modelo_recall") or walk_values(outputs, "mejor_modelo_recall")

    extracted = {
        "best_by_f1_f1": nested_metric(best_by_f1, "f1"),
        "best_by_f1_recall": nested_metric(best_by_f1, "recall"),
        "best_by_f1_precision": nested_metric(best_by_f1, "precision"),
        "best_by_precision_f1": nested_metric(best_by_precision, "f1"),
        "best_by_precision_recall": nested_metric(best_by_precision, "recall"),
        "best_by_precision_precision": nested_metric(best_by_precision, "precision"),
        "best_by_recall_f1": nested_metric(best_by_recall, "f1"),
        "best_by_recall_recall": nested_metric(best_by_recall, "recall"),
        "best_by_recall_precision": nested_metric(best_by_recall, "precision"),
    }
    extracted.update({
        "delta_f1_choice_f1_vs_recall_choice": delta(extracted["best_by_f1_f1"], extracted["best_by_recall_f1"]),
        "delta_f1_choice_recall_vs_recall_choice": delta(extracted["best_by_f1_recall"], extracted["best_by_recall_recall"]),
        "delta_precision_choice_precision_vs_recall_choice": delta(extracted["best_by_precision_precision"], extracted["best_by_recall_precision"]),
        "delta_precision_choice_recall_vs_recall_choice": delta(extracted["best_by_precision_recall"], extracted["best_by_recall_recall"]),
    })
    return extracted


def extract_f05_metrics(outputs: dict[str, Any]) -> dict[str, Any]:
    metrics = {key: walk_values(outputs, key) for key in F05_METRIC_KEYS}
    metrics.update(extract_criterion_metrics(outputs))
    trainable = walk_values(outputs, "trainable")
    f05_reason = first_available(
        {
            "incompatibility_reason": walk_values(outputs, "incompatibility_reason"),
            "training_skip_reason": walk_values(outputs, "training_skip_reason"),
            "skip_reason": walk_values(outputs, "skip_reason"),
            "failure_reason": walk_values(outputs, "failure_reason"),
            "error_message": walk_values(outputs, "error_message"),
            "error": walk_values(outputs, "error"),
        },
        "incompatibility_reason",
        "training_skip_reason",
        "skip_reason",
        "failure_reason",
        "error_message",
        "error",
    )
    metrics.update({
        "f05_trainable": trainable,
        "f05_compatible": normalize_bool(trainable) if trainable not in (None, "") else None,
        "f05_incompatibility_reason": f05_reason,
        "test_precision_effective": first_available(metrics, "test_precision", "test_precision_at_threshold"),
        "test_recall_effective": first_available(metrics, "test_recall", "test_recall_at_threshold"),
        "test_f1_effective": first_available(metrics, "test_f1", "test_f1_at_threshold"),
        "test_fn_effective": first_available(metrics, "test_fn", "fn"),
        "test_pr_auc_effective": first_available(metrics, "test_pr_auc", "pr_auc", "test_average_precision", "average_precision"),
        "val_precision_effective": first_available(metrics, "val_precision", "val_precision_at_threshold"),
        "val_recall_effective": first_available(metrics, "val_recall", "val_recall_at_threshold", "best_val_recall"),
        "val_f1_effective": first_available(metrics, "val_f1", "val_f1_at_threshold"),
        "val_pr_auc_effective": first_available(metrics, "val_pr_auc"),
        "decision_threshold_effective": first_available(metrics, "decision_threshold", "selected_threshold"),
        "threshold_source_effective": first_available(metrics, "threshold_source"),
        "threshold_policy_effective": first_available(metrics, "threshold_policy"),
        "val_positive_ratio_effective": first_available(metrics, "val_positive_ratio", "positive_ratio_val"),
        "test_positive_ratio_effective": first_available(metrics, "test_positive_ratio", "positive_ratio_test"),
    })
    if metrics["threshold_source_effective"] in (None, "") and metrics.get("decision_threshold") not in (None, ""):
        metrics["threshold_source_effective"] = "legacy_or_unknown"
    if metrics["threshold_policy_effective"] in (None, "") and metrics.get("decision_threshold") not in (None, ""):
        metrics["threshold_policy_effective"] = "legacy_or_unknown"
    return metrics


def extract_f04_metrics(outputs: dict[str, Any]) -> dict[str, Any]:
    exports = outputs.get("exports") or {}
    dedup = exports.get("deduplication_stats") or {}
    return {
        "f04_positive_ratio": walk_values(outputs, "positive_ratio") or walk_values(outputs, "class_balance_ratio"),
        "f04_n_windows_pos": walk_values(outputs, "n_windows_pos"),
        "f04_n_windows_neg": walk_values(outputs, "n_windows_neg"),
        "f04_ambiguous_ratio": dedup.get("ambiguous_ratio") if isinstance(dedup, dict) else None,
        "f04_duplicate_ratio_after_target": dedup.get("duplicate_ratio") if isinstance(dedup, dict) else None,
    }


def extract_f06_metrics(outputs: dict[str, Any]) -> dict[str, Any]:
    reason = walk_values(outputs, "incompatibility_reason")
    return {
        "tflm_compatible": walk_values(outputs, "tflm_compatible"),
        "f06_incompatibility_reason": reason,
        "incompatibility_reason": reason,
        "tflite_test_precision": walk_values(outputs, "tflite_test_precision"),
        "tflite_test_recall": walk_values(outputs, "tflite_test_recall"),
        "tflite_test_f1": walk_values(outputs, "tflite_test_f1"),
        "tflite_test_pr_auc": walk_values(outputs, "tflite_test_pr_auc"),
    }


def load_metrics_for_model(model: dict[str, Any]) -> dict[str, Any]:
    output_root = local_path(model.get("local_output_path"))
    variant_paths = model.get("variant_paths") or {}
    f05_outputs = load_yaml(output_root / "f05" / "outputs.yaml") if output_root else {}
    f04_outputs = load_yaml(output_root / "f04" / "outputs.yaml") if output_root else {}
    f06_outputs = load_yaml(output_root / "f06" / "outputs.yaml") if output_root else {}
    if not f05_outputs:
        f05_outputs = load_yaml(local_path(variant_paths.get("f05")) / "outputs.yaml" if variant_paths.get("f05") else None)
    if not f04_outputs:
        f04_outputs = load_yaml(local_path(variant_paths.get("f04")) / "outputs.yaml" if variant_paths.get("f04") else None)
    if not f06_outputs:
        f06_outputs = load_yaml(local_path(variant_paths.get("f06")) / "outputs.yaml" if variant_paths.get("f06") else None)
    if not f05_outputs and isinstance(model.get("metrics"), dict):
        f05_outputs = model["metrics"]
    metrics = extract_f05_metrics(f05_outputs)
    metrics.update(extract_f04_metrics(f04_outputs))
    metrics.update(extract_f06_metrics(f06_outputs))
    if not f05_outputs and (variant_paths.get("f05") or (model.get("variants") or {}).get("f05")):
        metrics["f05_compatible"] = "unknown"
        metrics["f05_incompatibility_reason"] = "Phase 5 outputs.yaml not found; compatibility could not be evaluated."
    if not f06_outputs and (variant_paths.get("f06") or (model.get("variants") or {}).get("f06")):
        metrics["tflm_compatible"] = "unknown"
        metrics["f06_incompatibility_reason"] = "Phase 6 outputs.yaml not found; TFLM compatibility could not be evaluated."
        metrics["incompatibility_reason"] = metrics["f06_incompatibility_reason"]
    if falseish(metrics.get("f05_compatible")) and metrics.get("f05_incompatibility_reason") in (None, ""):
        metrics["f05_incompatibility_reason"] = "Phase 5 marked the model as not trainable/compatible, but did not provide a reason."
    if falseish(metrics.get("tflm_compatible")) and metrics.get("f06_incompatibility_reason") in (None, ""):
        metrics["f06_incompatibility_reason"] = "Phase 6 marked the model as not TFLM-compatible, but did not provide a reason."
        metrics["incompatibility_reason"] = metrics["f06_incompatibility_reason"]
    return metrics


def iter_band_models(bands_key: str, bands_node: Any) -> list[dict[str, Any]]:
    if not isinstance(bands_node, dict):
        return []
    models = bands_node.get("models")
    if isinstance(models, list):
        rows = []
        for model in models:
            if not isinstance(model, dict):
                continue
            merged = dict(model)
            merged.setdefault("bands_id", bands_node.get("bands_id"))
            merged.setdefault("bands", bands_node.get("bands"))
            rows.append(merged)
        return rows
    # Backward compatibility with model_index.yaml generated before bands kept lists.
    return [bands_node] if bands_node.get("variants") or bands_node.get("model") else []


def flatten_models_from_index(index: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    models_by_measure = index.get("models_by_measure") or {}
    if not isinstance(models_by_measure, dict):
        return rows

    for measure, directions in models_by_measure.items():
        for direction, thresholds in (directions or {}).items():
            for threshold_key, threshold_node in (thresholds or {}).items():
                temporal_configs = (threshold_node or {}).get("temporal_configs") or {}
                for temporal_config_id, temporal_node in temporal_configs.items():
                    pipelines = (temporal_node or {}).get("pipelines") or {}
                    for pipeline, bands_map in pipelines.items():
                        for bands_key, bands_node in (bands_map or {}).items():
                            for model in iter_band_models(bands_key, bands_node):
                                if not isinstance(model, dict):
                                    continue
                                variants = model.get("variants") or {}
                                variant_paths = model.get("variant_paths") or {}
                                model_info = model.get("model") or {}
                                row = {
                                    "measure": measure,
                                    "direction": direction,
                                    "effective_threshold": str(threshold_key).replace("threshold_", ""),
                                    "threshold_key": threshold_key,
                                    "temporal_config_id": temporal_config_id,
                                    "OW": temporal_node.get("OW"),
                                    "PW": temporal_node.get("PW"),
                                    "LT": temporal_node.get("LT"),
                                    "window_strategy": temporal_node.get("window_strategy"),
                                    "pipeline": pipeline,
                                    "bands_key": bands_key,
                                    "bands_id": model.get("bands_id") or "no_bands",
                                    "bands": model.get("bands") or [],
                                    "model_family": model_info.get("model_family"),
                                    "deduplication_mode": model_info.get("deduplication_mode") or "none",
                                    "seed": model_info.get("seed"),
                                    "job_key": model.get("job_key"),
                                    "run_id": model.get("run_id"),
                                    "status": model.get("status"),
                                    "f05_variant": variants.get("f05"),
                                    "f06_variant": variants.get("f06"),
                                    "local_output_path": normalize_path(model.get("local_output_path")),
                                    "f05_path": normalize_path(variant_paths.get("f05")),
                                    "f06_path": normalize_path(variant_paths.get("f06")),
                                }
                                row.update(load_metrics_for_model(model))
                                rows.append(row)
    return rows


def group_models_by_comparison_block(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str, str], dict[str, list[dict[str, Any]]]]:
    grouped: dict[tuple[str, str, str, str], dict[str, list[dict[str, Any]]]] = {}
    for row in rows:
        key = (row["measure"], row["direction"], row["effective_threshold"], row["temporal_config_id"])
        block = grouped.setdefault(key, {pipeline: [] for pipeline in PIPELINE_ORDER})
        block.setdefault(row["pipeline"], []).append(row)
    return grouped


def build_global_summary(index: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = dict(index.get("summary") or {})
    summary["n_models"] = summary.get("n_models", len(rows))
    summary["n_measures"] = summary.get("n_measures", len({row["measure"] for row in rows}))
    summary["n_target_groups"] = summary.get(
        "n_target_groups",
        len({(row["measure"], row["direction"], row["effective_threshold"]) for row in rows}),
    )
    summary["pipelines"] = summary.get("pipelines") or {
        pipeline: sum(1 for row in rows if row["pipeline"] == pipeline) for pipeline in PIPELINE_ORDER
    }
    return summary


def average(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def build_criterion_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    available = [row for row in rows if as_float(row.get("best_by_recall_recall")) is not None]
    f1_gains = [
        value for value in (delta(row.get("best_by_f1_f1"), row.get("best_by_recall_f1")) for row in available)
        if value is not None
    ]
    precision_gains = [
        value for value in (delta(row.get("best_by_precision_precision"), row.get("best_by_recall_precision")) for row in available)
        if value is not None
    ]
    recall_losses_if_f1 = [
        value for value in (delta(row.get("best_by_recall_recall"), row.get("best_by_f1_recall")) for row in available)
        if value is not None
    ]
    recall_losses_if_precision = [
        value for value in (delta(row.get("best_by_recall_recall"), row.get("best_by_precision_recall")) for row in available)
        if value is not None
    ]
    return {
        "n_blocks": len(available),
        "f1_choice_improves_f1_count": sum(1 for value in f1_gains if value > 0),
        "precision_choice_improves_precision_count": sum(1 for value in precision_gains if value > 0),
        "avg_recall_loss_if_select_f1": average(recall_losses_if_f1),
        "avg_recall_loss_if_select_precision": average(recall_losses_if_precision),
        "avg_f1_gain_if_select_f1": average(f1_gains),
        "avg_precision_gain_if_select_precision": average(precision_gains),
    }


def build_rankings(rows: list[dict[str, Any]], metric: str, limit: int = 10) -> list[dict[str, Any]]:
    ranked = [row for row in rows if as_float(row.get(metric)) is not None]
    ranked.sort(key=lambda row: as_float(row.get(metric)) or float("-inf"), reverse=True)
    return ranked[:limit]


def best_f1_rows(rows: list[dict[str, Any]]) -> set[int]:
    scored = [(idx, as_float(row.get("test_f1_effective"))) for idx, row in enumerate(rows)]
    scored = [(idx, score) for idx, score in scored if score is not None]
    if not scored:
        return set()
    best = max(score for _, score in scored)
    return {idx for idx, score in scored if score == best}


def model_sort_key(row: dict[str, Any]) -> tuple[str, str, float, str]:
    score = as_float(row.get("test_f1_effective"))
    return (
        str(row.get("bands_id")),
        str(row.get("deduplication_mode") or "none"),
        -(score if score is not None else -1.0),
        str(row.get("f05_variant")),
    )


def render_metric(label: str, value: Any) -> str:
    return f'<div class="metric"><span>{esc(label)}</span><strong>{esc(value)}</strong></div>'


def render_kv(label: str, value: Any) -> str:
    return f"<div><span>{esc(label)}</span><strong>{esc(value)}</strong></div>"


def delta_class(value: Any) -> str:
    parsed = as_float(value)
    if parsed is None or parsed == 0:
        return "neutral"
    return "positive" if parsed > 0 else "negative"


def render_delta(label: str, value: Any) -> str:
    if as_float(value) is None:
        return ""
    sign = "+" if (as_float(value) or 0) > 0 else ""
    return f'<div class="criterion-delta {delta_class(value)}"><span>{esc(label)}</span><strong>{sign}{esc(value)}</strong></div>'


def has_criterion_metrics(row: dict[str, Any]) -> bool:
    return any(as_float(row.get(column)) is not None for column in CRITERION_COLUMNS[:9])


def render_criterion_card(title: str, prefix: str, row: dict[str, Any], badges: list[str], deltas: list[tuple[str, str]]) -> str:
    metric_rows = [
        ("F1", row.get(f"{prefix}_f1")),
        ("Recall", row.get(f"{prefix}_recall")),
        ("Precision", row.get(f"{prefix}_precision")),
    ]
    badge_html = "".join(f'<span class="criterion-badge">{esc(badge)}</span>' for badge in badges)
    delta_html = "".join(render_delta(label, row.get(key)) for label, key in deltas)
    return f"""
      <div class="criterion-card">
        <div class="criterion-title"><strong>{esc(title)}</strong>{badge_html}</div>
        <div class="criterion-values">
          {''.join(render_kv(label, value) for label, value in metric_rows)}
        </div>
        {f'<div class="criterion-deltas">{delta_html}</div>' if delta_html else ''}
      </div>
    """


def render_criterion_comparison(row: dict[str, Any]) -> str:
    if not has_criterion_metrics(row):
        return '<p class="muted criterion-note">Criterion comparison not available for this output.</p>'

    f1_badges = []
    precision_badges = []
    recall_badges = ["selected"]
    if (as_float(row.get("delta_f1_choice_f1_vs_recall_choice")) or 0) > 0:
        f1_badges.append("better F1")
    if (as_float(row.get("delta_precision_choice_precision_vs_recall_choice")) or 0) > 0:
        precision_badges.append("better precision")

    return f"""
  <details class="criterion-details">
    <summary>Selection criteria comparison</summary>
    <p class="criterion-note">The final model is still selected by recall. This comparison shows the validation metrics that would result from selecting by F1 or precision, to evaluate whether the decision criterion should change.</p>
    <div class="criterion-grid">
      {render_criterion_card("Best by F1", "best_by_f1", row, f1_badges, [
          ("Delta F1 vs selected", "delta_f1_choice_f1_vs_recall_choice"),
          ("Delta recall vs selected", "delta_f1_choice_recall_vs_recall_choice"),
      ])}
      {render_criterion_card("Best by Precision", "best_by_precision", row, precision_badges, [
          ("Delta precision vs selected", "delta_precision_choice_precision_vs_recall_choice"),
          ("Delta recall vs selected", "delta_precision_choice_recall_vs_recall_choice"),
      ])}
      {render_criterion_card("Best by Recall", "best_by_recall", row, recall_badges, [])}
    </div>
  </details>
"""


def has_f05_compatibility_note(row: dict[str, Any]) -> bool:
    return falseish(row.get("f05_compatible")) or row.get("f05_incompatibility_reason") not in (None, "")


def render_f05_section(row: dict[str, Any]) -> str:
    if not has_f05_compatibility_note(row):
        return ""
    metrics = [
        ("F05 compatible", row.get("f05_compatible")),
        ("Trainable", row.get("f05_trainable")),
        ("Reason", row.get("f05_incompatibility_reason")),
    ]
    return f"""
  <section class="phase-compat-section phase-compat-warning">
    <h6>Phase 5 - Model Training</h6>
    <div class="phase-compat-grid">
      {''.join(render_kv(label, value) for label, value in metrics)}
    </div>
  </section>
"""


def has_f06_metrics(row: dict[str, Any]) -> bool:
    return any(row.get(key) not in (None, "") for key in [
        "tflm_compatible",
        "f06_incompatibility_reason",
        "incompatibility_reason",
        "tflite_test_precision",
        "tflite_test_recall",
        "tflite_test_f1",
        "tflite_test_pr_auc",
    ])


def render_f06_section(row: dict[str, Any]) -> str:
    if not has_f06_metrics(row):
        return ""
    section_class = (
        "phase-compat-section phase-compat-warning"
        if falseish(row.get("tflm_compatible")) or first_available(row, "f06_incompatibility_reason", "incompatibility_reason") not in (None, "")
        else "phase-compat-section"
    )
    metrics = [
        ("TFLM compatible", row.get("tflm_compatible")),
        ("TFLite precision", row.get("tflite_test_precision")),
        ("TFLite recall", row.get("tflite_test_recall")),
        ("TFLite F1", row.get("tflite_test_f1")),
        ("TFLite PR-AUC", row.get("tflite_test_pr_auc")),
        ("Incompatibility reason", first_available(row, "f06_incompatibility_reason", "incompatibility_reason")),
    ]
    return f"""
  <section class="{section_class}">
    <h6>Phase 6 - TFLite/TFLM</h6>
    <div class="phase-compat-grid">
      {''.join(render_kv(label, value) for label, value in metrics)}
    </div>
  </section>
"""


def render_model_card(row: dict[str, Any], is_best: bool) -> str:
    pipeline = row.get("pipeline")
    class_name = f"model-card {pipeline}"
    best_badge = '<span class="badge best">Best F1 in block</span>' if is_best else ""
    bands_line = ""
    if pipeline != "ts":
        bands_line = f'<p class="bands">bands: {esc(row.get("bands"))}</p>'
    secondary = [
        ("val_recall", row.get("val_recall_effective")),
        ("val_precision", row.get("val_precision_effective")),
        ("val_f1", row.get("val_f1_effective")),
        ("val_pr_auc", row.get("val_pr_auc_effective")),
        ("deduplication_mode", row.get("deduplication_mode")),
        ("decision_threshold", row.get("decision_threshold_effective")),
        ("threshold_source", row.get("threshold_source_effective")),
        ("threshold_policy", row.get("threshold_policy_effective")),
        ("positive_ratio_train", row.get("positive_ratio_train")),
        ("val_positive_ratio", row.get("val_positive_ratio_effective")),
        ("test_positive_ratio", row.get("test_positive_ratio_effective")),
        ("val_cap_used", row.get("val_cap_used")),
        ("test_cap_used", row.get("test_cap_used")),
        ("n_train", row.get("n_train")),
        ("n_val", row.get("n_val")),
        ("n_test", row.get("n_test")),
        ("tp", row.get("tp")),
        ("fp", row.get("fp")),
        ("tn", row.get("tn")),
        ("fn", row.get("test_fn_effective")),
        ("execution_time", row.get("execution_time")),
    ]
    trace = [
        ("job_key", row.get("job_key")),
        ("run_id", row.get("run_id")),
        ("local_output_path", row.get("local_output_path")),
        ("f05_path", row.get("f05_path")),
        ("f06_path", row.get("f06_path")),
    ]
    return f"""
<article class="{class_name}">
  <div class="card-head">
    <div>
      <h5>{esc(PIPELINE_LABELS.get(str(pipeline), pipeline))}</h5>
      <p>{esc(row.get("bands_id") or "no_bands")} · {esc(row.get("deduplication_mode") or "none")} · {esc(row.get("f05_variant"))}</p>
    </div>
    <span class="status">{esc(row.get("status"))}</span>
  </div>
  {best_badge}
  {bands_line}
  <div class="main-metrics">
    {render_metric("F1", row.get("test_f1_effective"))}
    {render_metric("Recall", row.get("test_recall_effective"))}
    {render_metric("Precision", row.get("test_precision_effective"))}
    {render_metric("PR-AUC", row.get("test_pr_auc_effective"))}
  </div>
  <div class="kv-grid">
    {''.join(render_kv(label, value) for label, value in secondary)}
  </div>
  {render_f05_section(row)}
  {render_f06_section(row)}
  {render_criterion_comparison(row)}
  <details>
    <summary>Traceability</summary>
    <div class="trace">
      {''.join(render_kv(label, value) for label, value in trace)}
    </div>
  </details>
</article>
"""


def render_pipeline_column(pipeline: str, rows: list[dict[str, Any]], best_indexes: set[int], all_rows: list[dict[str, Any]]) -> str:
    sorted_rows = sorted(rows, key=model_sort_key)
    cards = []
    for row in sorted_rows:
        cards.append(render_model_card(row, all_rows.index(row) in best_indexes))
    if not cards:
        cards.append('<div class="empty">No model available</div>')
    return f"""
<div class="pipeline-column {pipeline}">
  <div class="column-title">{esc(PIPELINE_LABELS[pipeline])}</div>
  {''.join(cards)}
</div>
"""


def render_block_table(rows: list[dict[str, Any]]) -> str:
    sorted_rows = sorted(rows, key=lambda row: (PIPELINE_ORDER.index(row["pipeline"]) if row["pipeline"] in PIPELINE_ORDER else 99, str(row.get("bands_id"))))
    header = "".join(f"<th>{esc(col)}</th>" for col in BLOCK_TABLE_COLUMNS)
    body = []
    for row in sorted_rows:
        cells = []
        for col in BLOCK_TABLE_COLUMNS:
            class_name = "path" if col == "local_output_path" else ""
            cells.append(f'<td class="{class_name}">{esc(row.get(col))}</td>')
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def render_comparison_block(key: tuple[str, str, str, str], block: dict[str, list[dict[str, Any]]]) -> str:
    measure, direction, threshold, temporal_config_id = key
    all_rows = [row for pipeline in PIPELINE_ORDER for row in block.get(pipeline, [])]
    first = all_rows[0] if all_rows else {}
    best_indexes = best_f1_rows(all_rows)
    columns = "".join(render_pipeline_column(pipeline, block.get(pipeline, []), best_indexes, all_rows) for pipeline in PIPELINE_ORDER)
    return f"""
<section class="comparison-block">
  <div class="block-title">
    <div>
      <h2>{esc(measure)} <span>{esc(direction)}</span> <span>threshold {esc(threshold)}</span></h2>
      <p>Temporal config: OW={esc(first.get("OW"))}, PW={esc(first.get("PW"))}, LT={esc(first.get("LT"))}, strategy={esc(first.get("window_strategy"))}</p>
    </div>
  </div>
  <div class="pipeline-grid">
    {columns}
  </div>
  <h3>Compact Comparison Table</h3>
  {render_block_table(all_rows)}
</section>
"""


def render_measure_details(measure: str, items: list[tuple[tuple[str, str, str, str], dict[str, list[dict[str, Any]]]]]) -> str:
    block_count = len(items)
    model_count = sum(
        len(rows)
        for _, block in items
        for rows in block.values()
    )
    blocks_html = "\n".join(render_comparison_block(key, block) for key, block in items)
    return f"""
<details class="measure-details">
  <summary>
    <span class="measure-name">{esc(measure)}</span>
    <span class="measure-meta">{esc(block_count)} comparison blocks &middot; {esc(model_count)} models</span>
  </summary>
  <div class="measure-blocks">
    {blocks_html}
  </div>
</details>
"""


def render_rankings(rows: list[dict[str, Any]]) -> str:
    columns = ["measure", "direction", "effective_threshold", "temporal_config_id", "pipeline", "bands_id", "deduplication_mode", "f05_variant", "test_precision_effective", "test_recall_effective", "test_fn_effective", "test_pr_auc_effective", "test_f1_effective", "decision_threshold_effective", "threshold_source_effective", "threshold_policy_effective", "local_output_path"]
    parts = []
    for metric, title in [
        ("test_f1_effective", "Top 10 by test_f1"),
        ("test_recall_effective", "Top 10 by test_recall"),
        ("test_precision_effective", "Top 10 by test_precision"),
        ("test_pr_auc_effective", "Top 10 by test_pr_auc"),
    ]:
        ranked = build_rankings(rows, metric)
        header = "".join(f"<th>{esc(col)}</th>" for col in columns)
        body = []
        for row in ranked:
            body.append("<tr>" + "".join(f'<td class="{"path" if col == "local_output_path" else ""}">{esc(row.get(col))}</td>' for col in columns) + "</tr>")
        if not ranked:
            body.append(f'<tr><td colspan="{len(columns)}" class="muted">No ranked models.</td></tr>')
        parts.append(f'<div class="ranking"><h3>{esc(title)}</h3><table><thead><tr>{header}</tr></thead><tbody>{"".join(body)}</tbody></table></div>')
    return "\n".join(parts)


def render_criterion_analysis(rows: list[dict[str, Any]]) -> str:
    analysis = build_criterion_analysis(rows)
    items = [
        ("n_blocks", analysis["n_blocks"]),
        ("f1_choice_improves_f1_count", analysis["f1_choice_improves_f1_count"]),
        ("precision_choice_improves_precision_count", analysis["precision_choice_improves_precision_count"]),
        ("avg_recall_loss_if_select_f1", analysis["avg_recall_loss_if_select_f1"]),
        ("avg_recall_loss_if_select_precision", analysis["avg_recall_loss_if_select_precision"]),
        ("avg_f1_gain_if_select_f1", analysis["avg_f1_gain_if_select_f1"]),
        ("avg_precision_gain_if_select_precision", analysis["avg_precision_gain_if_select_precision"]),
    ]
    return f"""
    <section class="criterion-analysis">
      <h2>Criterion Analysis</h2>
      <p class="muted">Aggregated over outputs that include the new F05 validation candidate blocks.</p>
      <div class="summary-grid">
        {''.join(f'<div class="stat"><span>{esc(label)}</span><strong>{esc(value)}</strong></div>' for label, value in items)}
      </div>
    </section>
"""


def render_html(index: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    summary = build_global_summary(index, rows)
    grouped = group_models_by_comparison_block(rows)
    generated_at = datetime.now().isoformat(timespec="seconds")
    pipeline_counts = summary.get("pipelines") or {}
    pipeline_items = "".join(
        f'<div class="mini-stat {pipeline}"><span>{esc(PIPELINE_LABELS.get(pipeline, pipeline))}</span><strong>{esc(pipeline_counts.get(pipeline, 0))}</strong></div>'
        for pipeline in PIPELINE_ORDER
    )
    measure_groups: dict[str, list[tuple[tuple[str, str, str, str], dict[str, list[dict[str, Any]]]]]] = {}
    sorted_keys = sorted(grouped, key=lambda item: (item[0], direction_sort_value(item[1]), threshold_sort_value(item[2]), item[3]))
    for key in sorted_keys:
        measure_groups.setdefault(key[0], []).append((key, grouped[key]))
    blocks_html = (
        "\n".join(render_measure_details(measure, measure_groups[measure]) for measure in sorted(measure_groups))
        if measure_groups
        else '<p class="muted">No models in model_index.yaml.</p>'
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>MLOps4RT-Edge Comparison Report</title>
  <style>
    :root {{
      --ink: #1f2933; --muted: #687789; --line: #d8e0e8; --paper: #ffffff; --bg: #f5f7fa;
      --ts: #e8f1fb; --ts-border: #8eb6df; --events: #e8f6ee; --events-border: #8ac6a3;
      --uniEvents: #fff1e3; --uniEvents-border: #e3aa6f;
    }}
    body {{ margin: 0; font-family: Arial, sans-serif; color: var(--ink); background: var(--bg); }}
    header {{ padding: 28px 34px; background: #ffffff; border-bottom: 1px solid var(--line); }}
    h1 {{ margin: 0 0 6px; font-size: 28px; }}
    h2, h3, h4, h5 {{ color: #102a43; }}
    .muted {{ color: var(--muted); }}
    main {{ padding: 24px 34px 42px; }}
    .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-top: 18px; }}
    .stat, .mini-stat {{ background: var(--paper); border: 1px solid var(--line); border-radius: 8px; padding: 14px; }}
    .stat span, .mini-stat span {{ color: var(--muted); display: block; font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }}
    .stat strong, .mini-stat strong {{ display: block; font-size: 25px; margin-top: 5px; }}
    .mini-stat.ts {{ background: var(--ts); border-color: var(--ts-border); }}
    .mini-stat.events {{ background: var(--events); border-color: var(--events-border); }}
    .mini-stat.uniEvents {{ background: var(--uniEvents); border-color: var(--uniEvents-border); }}
    .rankings {{ margin-top: 24px; }}
    .ranking {{ background: var(--paper); border: 1px solid var(--line); border-radius: 8px; padding: 14px; margin-bottom: 16px; overflow-x: auto; }}
    .measure-details {{ background: var(--paper); border: 1px solid var(--line); border-radius: 10px; margin-top: 16px; box-shadow: 0 8px 22px rgba(20, 45, 70, .05); }}
    .measure-details > summary {{ display: flex; align-items: center; justify-content: space-between; gap: 16px; padding: 16px 18px; font-size: 15px; }}
    .measure-details > summary::marker {{ color: #627d98; }}
    .measure-name {{ color: #102a43; font-size: 18px; }}
    .measure-meta {{ color: var(--muted); font-size: 12px; font-weight: 600; }}
    .measure-blocks {{ border-top: 1px solid var(--line); padding: 0 18px 18px; }}
    .comparison-block {{ margin-top: 18px; background: var(--paper); border: 1px solid var(--line); border-radius: 10px; padding: 18px; box-shadow: 0 8px 22px rgba(20, 45, 70, .06); }}
    .block-title h2 {{ margin: 0; font-size: 21px; }}
    .block-title h2 span {{ margin-left: 8px; color: var(--muted); font-weight: 600; }}
    .block-title p {{ margin: 8px 0 0; color: var(--muted); }}
    .pipeline-grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; margin-top: 18px; }}
    .pipeline-column {{ border-radius: 9px; border: 1px solid var(--line); padding: 12px; min-height: 160px; }}
    .pipeline-column.ts {{ background: var(--ts); border-color: var(--ts-border); }}
    .pipeline-column.events {{ background: var(--events); border-color: var(--events-border); }}
    .pipeline-column.uniEvents {{ background: var(--uniEvents); border-color: var(--uniEvents-border); }}
    .column-title {{ font-weight: 700; margin-bottom: 10px; }}
    .model-card {{ background: rgba(255,255,255,.88); border: 1px solid rgba(30,60,90,.14); border-radius: 8px; padding: 12px; margin-bottom: 12px; box-shadow: 0 4px 12px rgba(20, 45, 70, .08); }}
    .card-head {{ display: flex; justify-content: space-between; gap: 10px; align-items: flex-start; }}
    .card-head h5 {{ margin: 0; font-size: 15px; }}
    .card-head p, .bands {{ margin: 4px 0 0; color: var(--muted); font-size: 12px; overflow-wrap: anywhere; }}
    .status, .badge {{ border: 1px solid var(--line); background: #fff; border-radius: 999px; padding: 3px 8px; font-size: 11px; white-space: nowrap; }}
    .badge.best {{ display: inline-block; margin: 9px 0; background: #fff7cc; border-color: #e2c044; font-weight: 700; }}
    .main-metrics {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin: 12px 0; }}
    .metric {{ background: #fff; border: 1px solid var(--line); border-radius: 7px; padding: 9px; }}
    .metric span {{ display: block; color: var(--muted); font-size: 11px; }}
    .metric strong {{ display: block; font-size: 18px; margin-top: 3px; }}
    .phase-compat-section {{ background: #f8fbff; border: 1px solid var(--line); border-radius: 7px; margin: 10px 0; padding: 9px; }}
    .phase-compat-warning {{ background: #fff8f1; border-color: #e3aa6f; }}
    .phase-compat-section h6 {{ margin: 0 0 8px; color: #102a43; font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }}
    .phase-compat-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 6px 10px; font-size: 12px; }}
    .phase-compat-grid span {{ display: block; color: var(--muted); }}
    .phase-compat-grid strong {{ overflow-wrap: anywhere; }}
    .criterion-analysis {{ margin-top: 24px; }}
    .criterion-details {{ border-top: 1px solid var(--line); padding-top: 9px; }}
    .criterion-note {{ font-size: 12px; line-height: 1.4; }}
    .criterion-grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px; margin-top: 8px; }}
    .criterion-card {{ background: #fff; border: 1px solid var(--line); border-radius: 7px; padding: 9px; }}
    .criterion-title {{ display: flex; align-items: center; justify-content: space-between; gap: 6px; font-size: 12px; color: #102a43; }}
    .criterion-badge {{ border: 1px solid #b8c2cc; background: #f4f7fb; border-radius: 999px; padding: 2px 6px; font-size: 10px; white-space: nowrap; }}
    .criterion-values {{ display: grid; gap: 4px; margin-top: 8px; font-size: 11px; }}
    .criterion-values div {{ display: flex; justify-content: space-between; gap: 8px; }}
    .criterion-values span, .criterion-delta span {{ color: var(--muted); }}
    .criterion-deltas {{ border-top: 1px dashed var(--line); margin-top: 7px; padding-top: 6px; display: grid; gap: 4px; font-size: 11px; }}
    .criterion-delta {{ display: flex; justify-content: space-between; gap: 8px; }}
    .criterion-delta.positive strong {{ color: #13795b; }}
    .criterion-delta.negative strong {{ color: #b45309; }}
    .kv-grid, .trace {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 6px 10px; font-size: 12px; }}
    .kv-grid span, .trace span {{ display: block; color: var(--muted); }}
    .kv-grid strong, .trace strong {{ overflow-wrap: anywhere; }}
    details {{ margin-top: 10px; }}
    summary {{ cursor: pointer; color: #334e68; font-weight: 700; font-size: 12px; }}
    .empty {{ background: rgba(255,255,255,.7); border: 1px dashed rgba(30,60,90,.25); color: var(--muted); border-radius: 8px; padding: 24px 12px; text-align: center; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 12px; margin-top: 10px; background: #fff; }}
    th, td {{ border: 1px solid var(--line); padding: 7px; text-align: left; vertical-align: top; }}
    th {{ background: #eef2f7; }}
    .path {{ font-family: Consolas, monospace; overflow-wrap: anywhere; }}
    @media (max-width: 1050px) {{ .pipeline-grid, .criterion-grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <header>
    <h1>MLOps4RT-Edge Comparison Report</h1>
    <p class="muted">Generated at {esc(generated_at)} from registry/model_index.yaml</p>
    <div class="summary-grid">
      <div class="stat"><span>Total models</span><strong>{esc(summary.get("n_models", 0))}</strong></div>
      <div class="stat"><span>Measures</span><strong>{esc(summary.get("n_measures", 0))}</strong></div>
      <div class="stat"><span>Target groups</span><strong>{esc(summary.get("n_target_groups", 0))}</strong></div>
      {pipeline_items}
    </div>
  </header>
  <main>
    <section class="rankings">
      <h2>Global Rankings</h2>
      {render_rankings(rows)}
    </section>
    {render_criterion_analysis(rows)}
    <section>
      <h2>Comparison Blocks</h2>
      {blocks_html}
    </section>
  </main>
</body>
</html>
"""


def write_report(html_text: str, path: Path = REPORT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html_text, encoding="utf-8")


def main() -> None:
    try:
        rebuild_model_index()
        index = load_model_index()
        rows = flatten_models_from_index(index)
        html_text = render_html(index, rows)
    except Exception as exc:
        print(f"Error: {exc}")
        html_text = render_html({"summary": {}, "models_by_measure": {}}, [])
        html_text = html_text.replace(
            "<main>",
            f'<main><div class="comparison-block"><strong>Error:</strong> {esc(exc)}</div>',
            1,
        )
    write_report(html_text)
    print(f"Wrote {REPORT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()