from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import json
import numpy as np

from ._common import dump_csv, dump_json, timestamp_slug, to_jsonable


def _flatten_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    metrics = payload.get("metrics")
    if isinstance(metrics, dict):
        flat.update(metrics)
    extra_keys = (
        "threshold",
        "threshold_optimization",
        "baseline_metrics",
        "optimized_metrics",
        "catalog_coverage",
        "diversity",
        "candidate_count_mean",
        "mean_false_positives_per_document",
        "mean_missing_skills_per_document",
        "status",
        "document_count",
        "evaluated_document_count",
        "gold_skill_count",
        "predicted_skill_count",
        "total_examples",
        "real_positive_count",
        "real_negative_count",
        "predicted_positive_count",
        "predicted_negative_count",
        "positive_prevalence",
        "predicted_positive_rate",
        "label_cardinality_true",
        "label_cardinality_predicted",
        "mean_labels_per_example",
    )
    for key in extra_keys:
        if key in payload:
            flat[key] = payload[key]
    return flat


def _render_table(title: str, rows: list[dict[str, Any]]) -> str:
    if not rows:
        return f"<h2>{title}</h2><p>Aucune donnée.</p>"
    headers = list(rows[0].keys())
    header_html = "".join(f"<th>{header}</th>" for header in headers)
    body_rows = []
    for row in rows:
        cells = "".join(f"<td>{row.get(header, '')}</td>" for header in headers)
        body_rows.append(f"<tr>{cells}</tr>")
    return f"<h2>{title}</h2><table><thead><tr>{header_html}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def _plot_confusion_matrix(path: Path, matrix: list[list[int]]) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return
    array = np.asarray(matrix, dtype=float)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(array, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Prédit 0", "Prédit 1"])
    ax.set_yticks([0, 1], labels=["Vrai 0", "Vrai 1"])
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            ax.text(j, i, int(array[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_curve(path: Path, payload: dict[str, list[float]], *, kind: str) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    if kind == "roc" and payload.get("fpr") and payload.get("tpr"):
        ax.plot(payload["fpr"], payload["tpr"], label="ROC")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
    elif kind == "pr" and payload.get("recall") and payload.get("precision"):
        ax.plot(payload["recall"], payload["precision"], label="PR")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
    else:
        plt.close(fig)
        return
    ax.legend(loc="best")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_label_bars(path: Path, labels: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return
    if not labels:
        return
    names = [row.get("label", "") for row in labels]
    f1 = [float(row.get("f1", 0.0)) for row in labels]
    fig, ax = plt.subplots(figsize=(max(6, len(names) * 0.45), 4))
    ax.bar(names, f1)
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _report_html(payload: dict[str, Any], *, artifact_names: list[str], comparison: dict[str, Any] | None) -> str:
    metrics = payload.get("metrics") or {}
    label_rows = payload.get("per_label") or payload.get("per_class") or []
    warnings = payload.get("warnings") or []
    sections = [
        "<html><head><meta charset='utf-8'><title>Deepforma evaluation</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px;max-width:1200px}table{border-collapse:collapse;width:100%;margin:12px 0}th,td{border:1px solid #ddd;padding:6px 8px;text-align:left}th{background:#f5f5f5}code,pre{background:#f7f7f7;padding:8px;border-radius:4px} .warn{color:#8a4b00}</style>",
        "</head><body>",
        f"<h1>Evaluation {payload.get('task', 'model')}</h1>",
        _render_table("Métriques principales", [metrics]) if metrics else "<p>Aucune métrique.</p>",
    ]
    if comparison:
        sections.append(_render_table("Comparaison", [comparison]))
    if label_rows:
        if isinstance(label_rows, dict):
            normalized_rows = [dict({"label": key}, **to_jsonable(value)) for key, value in label_rows.items()]
        else:
            normalized_rows = [to_jsonable(row) for row in label_rows]
        sections.append(_render_table("Métriques par label", normalized_rows[:50]))
    if warnings:
        sections.append("<h2>Avertissements</h2>" + "".join(f"<p class='warn'>{warning}</p>" for warning in warnings))
    if artifact_names:
        sections.append("<h2>Fichiers</h2><ul>" + "".join(f"<li><a href='{name}'>{name}</a></li>" for name in artifact_names) + "</ul>")
    sections.append("<h2>JSON</h2><pre>" + json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2)[:20000] + "</pre>")
    sections.append("</body></html>")
    return "".join(sections)


def write_evaluation_artifacts(
    report: Any,
    output_dir: str | Path,
    *,
    model_name: str,
    task: str,
    version: str | None = None,
    comparison: dict[str, Any] | None = None,
) -> Path:
    root = Path(output_dir) / model_name / (version or timestamp_slug())
    root.mkdir(parents=True, exist_ok=True)
    payload = to_jsonable(report)
    if isinstance(payload, dict):
        payload.setdefault("model_name", model_name)
        payload.setdefault("task", task)
        payload.setdefault("generated_at", datetime.now(timezone.utc).isoformat())
    dump_json(root / "report.json", payload)
    summary_rows = [_flatten_metrics(payload)] if isinstance(payload, dict) else []
    if summary_rows:
        dump_csv(root / "summary.csv", summary_rows)
    artifact_names = ["report.json", "summary.csv"] if summary_rows else ["report.json"]
    if isinstance(payload, dict):
        if isinstance(payload.get("confusion_matrix"), list) and payload.get("confusion_matrix"):
            _plot_confusion_matrix(root / "confusion_matrix.png", payload["confusion_matrix"])
            artifact_names.append("confusion_matrix.png")
        if isinstance(payload.get("roc_curve"), dict) and payload["roc_curve"]:
            _plot_curve(root / "roc_curve.png", payload["roc_curve"], kind="roc")
            artifact_names.append("roc_curve.png")
        if isinstance(payload.get("pr_curve"), dict) and payload["pr_curve"]:
            _plot_curve(root / "pr_curve.png", payload["pr_curve"], kind="pr")
            artifact_names.append("pr_curve.png")
        if isinstance(payload.get("per_label"), list) and payload["per_label"]:
            _plot_label_bars(root / "per_label_f1.png", payload["per_label"])
            artifact_names.append("per_label_f1.png")
    html = _report_html(payload if isinstance(payload, dict) else {"task": task}, artifact_names=artifact_names, comparison=comparison)
    (root / "report.html").write_text(html, encoding="utf-8")
    return root


def latest_evaluation_dir(output_dir: str | Path, model_name: str) -> Path | None:
    root = Path(output_dir) / model_name
    if not root.exists():
        return None
    candidates = [path for path in root.iterdir() if path.is_dir()]
    if not candidates:
        return None
    return sorted(candidates, key=lambda path: path.name)[-1]


def load_evaluation_report(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
