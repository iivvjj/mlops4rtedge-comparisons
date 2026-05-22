#!/usr/bin/env python3
"""Build a visual HTML comparison report from model_index.yaml and copied outputs."""

from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_INDEX_PATH = REPO_ROOT / "registry" / "model_index.yaml"
REPORT_PATH = REPO_ROOT / "reports" / "report.html"

PIPELINE_ORDER = ["ts", "events", "uniEvents"]
PIPELINE_LABELS = {"ts": "TS", "events": "Events", "uniEvents": "UniEvents"}

F05_METRIC_KEYS = [
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

BLOCK_TABLE_COLUMNS = [
    "pipeline",
    "bands_id",
    "f05_variant",
    "test_precision",
    "test_recall",
    "test_f1",
    "best_val_recall",
    "positive_ratio_train",
    "n_train",
    "n_val",
    "n_test",
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


def extract_f05_metrics(outputs: dict[str, Any]) -> dict[str, Any]:
    return {key: walk_values(outputs, key) for key in F05_METRIC_KEYS}


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


def load_metrics_for_model(model: dict[str, Any]) -> dict[str, Any]:
    output_root = local_path(model.get("local_output_path"))
    f05_outputs = load_yaml(output_root / "f05" / "outputs.yaml") if output_root else {}
    f04_outputs = load_yaml(output_root / "f04" / "outputs.yaml") if output_root else {}
    metrics = extract_f05_metrics(f05_outputs)
    metrics.update(extract_f04_metrics(f04_outputs))
    return metrics


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
                        for bands_key, model in (bands_map or {}).items():
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
                                "seed": model_info.get("seed"),
                                "job_key": model.get("job_key"),
                                "run_id": model.get("run_id"),
                                "status": model.get("status"),
                                "f05_variant": variants.get("f05"),
                                "local_output_path": normalize_path(model.get("local_output_path")),
                                "f05_path": normalize_path(variant_paths.get("f05")),
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


def build_rankings(rows: list[dict[str, Any]], metric: str, limit: int = 10) -> list[dict[str, Any]]:
    ranked = [row for row in rows if as_float(row.get(metric)) is not None]
    ranked.sort(key=lambda row: as_float(row.get(metric)) or float("-inf"), reverse=True)
    return ranked[:limit]


def best_f1_rows(rows: list[dict[str, Any]]) -> set[int]:
    scored = [(idx, as_float(row.get("test_f1"))) for idx, row in enumerate(rows)]
    scored = [(idx, score) for idx, score in scored if score is not None]
    if not scored:
        return set()
    best = max(score for _, score in scored)
    return {idx for idx, score in scored if score == best}


def model_sort_key(row: dict[str, Any]) -> tuple[str, float, str]:
    score = as_float(row.get("test_f1"))
    return (str(row.get("bands_id")), -(score if score is not None else -1.0), str(row.get("f05_variant")))


def render_metric(label: str, value: Any) -> str:
    return f'<div class="metric"><span>{esc(label)}</span><strong>{esc(value)}</strong></div>'


def render_kv(label: str, value: Any) -> str:
    return f"<div><span>{esc(label)}</span><strong>{esc(value)}</strong></div>"


def render_model_card(row: dict[str, Any], is_best: bool) -> str:
    pipeline = row.get("pipeline")
    class_name = f"model-card {pipeline}"
    best_badge = '<span class="badge best">Best F1 in block</span>' if is_best else ""
    bands_line = ""
    if pipeline != "ts":
        bands_line = f'<p class="bands">bands: {esc(row.get("bands"))}</p>'
    secondary = [
        ("best_val_recall", row.get("best_val_recall")),
        ("decision_threshold", row.get("decision_threshold")),
        ("positive_ratio_train", row.get("positive_ratio_train")),
        ("n_train", row.get("n_train")),
        ("n_val", row.get("n_val")),
        ("n_test", row.get("n_test")),
        ("tp", row.get("tp")),
        ("fp", row.get("fp")),
        ("tn", row.get("tn")),
        ("fn", row.get("fn")),
        ("execution_time", row.get("execution_time")),
    ]
    trace = [
        ("job_key", row.get("job_key")),
        ("run_id", row.get("run_id")),
        ("local_output_path", row.get("local_output_path")),
        ("f05_path", row.get("f05_path")),
    ]
    return f"""
<article class="{class_name}">
  <div class="card-head">
    <div>
      <h5>{esc(PIPELINE_LABELS.get(str(pipeline), pipeline))}</h5>
      <p>{esc(row.get("bands_id") or "no_bands")} · {esc(row.get("f05_variant"))}</p>
    </div>
    <span class="status">{esc(row.get("status"))}</span>
  </div>
  {best_badge}
  {bands_line}
  <div class="main-metrics">
    {render_metric("F1", row.get("test_f1"))}
    {render_metric("Recall", row.get("test_recall"))}
    {render_metric("Precision", row.get("test_precision"))}
  </div>
  <div class="kv-grid">
    {''.join(render_kv(label, value) for label, value in secondary)}
  </div>
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


def render_rankings(rows: list[dict[str, Any]]) -> str:
    columns = ["measure", "direction", "effective_threshold", "temporal_config_id", "pipeline", "bands_id", "f05_variant", "test_precision", "test_recall", "test_f1", "local_output_path"]
    parts = []
    for metric, title in [("test_f1", "Top 10 by test_f1"), ("test_recall", "Top 10 by test_recall"), ("test_precision", "Top 10 by test_precision")]:
        ranked = build_rankings(rows, metric)
        header = "".join(f"<th>{esc(col)}</th>" for col in columns)
        body = []
        for row in ranked:
            body.append("<tr>" + "".join(f'<td class="{"path" if col == "local_output_path" else ""}">{esc(row.get(col))}</td>' for col in columns) + "</tr>")
        if not ranked:
            body.append(f'<tr><td colspan="{len(columns)}" class="muted">No ranked models.</td></tr>')
        parts.append(f'<div class="ranking"><h3>{esc(title)}</h3><table><thead><tr>{header}</tr></thead><tbody>{"".join(body)}</tbody></table></div>')
    return "\n".join(parts)


def render_html(index: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    summary = build_global_summary(index, rows)
    grouped = group_models_by_comparison_block(rows)
    generated_at = datetime.now().isoformat(timespec="seconds")
    pipeline_counts = summary.get("pipelines") or {}
    pipeline_items = "".join(
        f'<div class="mini-stat {pipeline}"><span>{esc(PIPELINE_LABELS.get(pipeline, pipeline))}</span><strong>{esc(pipeline_counts.get(pipeline, 0))}</strong></div>'
        for pipeline in PIPELINE_ORDER
    )
    block_parts = []
    for key in sorted(grouped, key=lambda item: (item[0], direction_sort_value(item[1]), threshold_sort_value(item[2]), item[3])):
        block_parts.append(render_comparison_block(key, grouped[key]))
    blocks_html = "\n".join(block_parts) if block_parts else '<p class="muted">No models in model_index.yaml.</p>'

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
    .comparison-block {{ margin-top: 26px; background: var(--paper); border: 1px solid var(--line); border-radius: 10px; padding: 18px; box-shadow: 0 8px 22px rgba(20, 45, 70, .06); }}
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
    .main-metrics {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin: 12px 0; }}
    .metric {{ background: #fff; border: 1px solid var(--line); border-radius: 7px; padding: 9px; }}
    .metric span {{ display: block; color: var(--muted); font-size: 11px; }}
    .metric strong {{ display: block; font-size: 18px; margin-top: 3px; }}
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
    @media (max-width: 1050px) {{ .pipeline-grid {{ grid-template-columns: 1fr; }} }}
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
