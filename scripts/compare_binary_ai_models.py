#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deepforma.evaluation.binary_classification_metrics import evaluate_binary_classification  # noqa: E402
from deepforma.training.binary_ai_ml import load_binary_ai_ml  # noqa: E402
from deepforma.training.binary_ai_textcnn import load_binary_ai_textcnn  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("compare_binary_ai_models")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare les modèles IA from scratch")
    parser.add_argument("--test", type=Path, default=Path("data/training/binary_ai/test.parquet"))
    parser.add_argument("--ml-model-dir", type=Path, default=Path("models/binary_ai_ml"))
    parser.add_argument("--textcnn-model-dir", type=Path, default=Path("models/binary_ai_textcnn"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/binary_ai"))
    return parser


def _model_size_bytes(model_dir: Path) -> int:
    total = 0
    for path in model_dir.rglob("*"):
        if path.is_file():
            total += path.stat().st_size
    return total


def _predict_pipeline(model, frame: pd.DataFrame, *, model_name: str, model_dir: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    texts = frame["text"].fillna("").astype(str).tolist()
    start = __import__("time").perf_counter()
    scores = model.predict_proba(texts)[:, 1]
    latency_ms = (__import__("time").perf_counter() - start) * 1000.0
    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    threshold = float(metadata.get("threshold", 0.5))
    report = evaluate_binary_classification(
        frame["is_ai"].astype(int),
        scores,
        threshold=threshold,
        model_name=model_name,
        inference_time_ms=latency_ms / max(len(frame), 1),
        model_size_bytes=_model_size_bytes(model_dir),
    )
    return scores, report, metadata


def _predict_textcnn(model, payload: dict[str, Any], vocab: dict[str, int], frame: pd.DataFrame, *, model_name: str, model_dir: Path):
    import time
    import torch
    from deepforma.training.binary_ai_textcnn import encode_text

    device = torch.device("cpu")
    texts = frame["text"].fillna("").astype(str).tolist()
    scores: list[float] = []
    start = time.perf_counter()
    with torch.no_grad():
        for text in texts:
            encoded = torch.tensor([encode_text(text, vocab, max_length=payload["config"]["max_length"])], dtype=torch.long)
            logits = model(encoded.to(device))
            scores.append(float(torch.sigmoid(logits)[0].item()))
    latency_ms = (time.perf_counter() - start) * 1000.0
    threshold = float(payload.get("threshold", 0.5))
    report = evaluate_binary_classification(
        frame["is_ai"].astype(int),
        np.asarray(scores, dtype=float),
        threshold=threshold,
        model_name=model_name,
        inference_time_ms=latency_ms / max(len(frame), 1),
        model_size_bytes=_model_size_bytes(model_dir),
    )
    return np.asarray(scores, dtype=float), report


def _plot_confusion_matrix(report, output_path: Path, title: str) -> None:
    matrix = np.array(
        [
            [report.confusion_counts["tn"], report.confusion_counts["fp"]],
            [report.confusion_counts["fn"], report.confusion_counts["tp"]],
        ]
    )
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], labels=["non-IA", "IA"])
    ax.set_yticks([0, 1], labels=["non-IA", "IA"])
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Réel")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_curve(model_reports: list[tuple[str, Any]], output_path: Path, kind: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for label, report in model_reports:
        if kind == "roc":
            curve = report.roc_curve
            ax.plot(curve["fpr"], curve["tpr"], label=f"{label} (AUC={report.metrics.get('roc_auc')})")
            ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
            ax.set_xlabel("False positive rate")
            ax.set_ylabel("True positive rate")
        elif kind == "pr":
            curve = report.precision_recall_curve
            ax.plot(curve["recall"], curve["precision"], label=f"{label} (AP={report.metrics.get('pr_auc')})")
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
        elif kind == "calibration":
            curve = report.calibration_curve
            ax.plot(curve["predicted"], curve["observed"], marker="o", label=label)
            ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
            ax.set_xlabel("Probability mean")
            ax.set_ylabel("Observed rate")
        else:
            raise ValueError(kind)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_threshold_analysis(report, output_path: Path, label: str) -> None:
    rows = pd.DataFrame(report.threshold_curve)
    fig, ax = plt.subplots(figsize=(6, 4))
    if not rows.empty:
        ax.plot(rows["threshold"], rows["f1_ia"], label="F1 IA")
        ax.plot(rows["threshold"], rows["precision_ia"], label="Precision IA")
        ax.plot(rows["threshold"], rows["recall_ia"], label="Recall IA")
    ax.axvline(report.threshold, linestyle="--", color="black", label="Seuil retenu")
    ax.set_title(label)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Metric")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_probability_distributions(reports: list[tuple[str, Any]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, report in reports:
        probabilities = np.array(list(report.probability_distribution.keys()), dtype=float)
        counts = np.array(list(report.probability_distribution.values()), dtype=float)
        if probabilities.size == 0:
            continue
        ax.plot(probabilities, counts / counts.sum(), label=label)
    ax.set_xlabel("Probability")
    ax.set_ylabel("Frequency")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _write_comparison(report_rows: list[dict[str, Any]], output_dir: Path) -> None:
    df = pd.DataFrame(report_rows)
    df.to_csv(output_dir / "model_comparison.csv", index=False, encoding="utf-8")
    columns = list(df.columns)
    lines = ["# Binary AI model comparison", "", "| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in df.iterrows():
        values = ["" if pd.isna(row.get(column)) else str(row.get(column)) for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    output_dir.joinpath("model_comparison.md").write_text("\n".join(lines), encoding="utf-8")


def _report_row(name: str, report, metadata: dict[str, Any]) -> dict[str, Any]:
    metrics = report.metrics
    return {
        "model": name,
        "accuracy": metrics.get("accuracy"),
        "balanced_accuracy": metrics.get("balanced_accuracy"),
        "precision_ia": metrics.get("precision_ia"),
        "recall_ia": metrics.get("recall_ia"),
        "f1_ia": metrics.get("f1_ia"),
        "f1_macro": metrics.get("f1_macro"),
        "pr_auc": metrics.get("pr_auc"),
        "roc_auc": metrics.get("roc_auc"),
        "mcc": metrics.get("mcc"),
        "threshold": report.threshold,
        "predicted_positive_rate": report.predicted_positive_rate,
        "real_positive_rate": report.positive_rate,
        "inference_time_ms": report.inference_time_ms,
        "model_size_bytes": report.model_size_bytes,
        "train_rows": metadata.get("train_rows"),
        "validation_rows": metadata.get("validation_rows"),
        "test_rows": metadata.get("test_rows"),
        "alerts": " | ".join(report.alerts or []),
    }


def main() -> int:
    args = build_parser().parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    test_frame = pd.read_parquet(args.test)

    ml_pipeline, ml_metadata = load_binary_ai_ml(args.ml_model_dir)
    ml_scores, ml_report, _ = _predict_pipeline(ml_pipeline, test_frame, model_name="binary_ai_ml", model_dir=args.ml_model_dir)

    textcnn_model, textcnn_payload, vocab, _ = load_binary_ai_textcnn(args.textcnn_model_dir)
    textcnn_scores, textcnn_report = _predict_textcnn(textcnn_model, textcnn_payload, vocab, test_frame, model_name="binary_ai_textcnn", model_dir=args.textcnn_model_dir)

    (output_dir / "ml_metrics.json").write_text(json.dumps(ml_report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "textcnn_metrics.json").write_text(json.dumps(textcnn_report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    _write_comparison(
        [
            _report_row("binary_ai_ml", ml_report, ml_metadata),
            _report_row("binary_ai_textcnn", textcnn_report, textcnn_payload),
        ],
        output_dir,
    )

    _plot_confusion_matrix(ml_report, output_dir / "confusion_matrix_ml.png", "ML confusion matrix")
    _plot_confusion_matrix(textcnn_report, output_dir / "confusion_matrix_textcnn.png", "TextCNN confusion matrix")
    _plot_curve([("ML", ml_report), ("TextCNN", textcnn_report)], output_dir / "roc_curves.png", "roc")
    _plot_curve([("ML", ml_report), ("TextCNN", textcnn_report)], output_dir / "precision_recall_curves.png", "pr")
    _plot_curve([("ML", ml_report), ("TextCNN", textcnn_report)], output_dir / "calibration_curves.png", "calibration")
    _plot_threshold_analysis(ml_report, output_dir / "threshold_analysis_ml.png", "ML threshold analysis")
    _plot_threshold_analysis(textcnn_report, output_dir / "threshold_analysis_textcnn.png", "TextCNN threshold analysis")
    _plot_probability_distributions([("ML", ml_report), ("TextCNN", textcnn_report)], output_dir / "probability_distributions.png")

    history = textcnn_payload.get("training_history", [])
    if history:
        df_history = pd.DataFrame(history)
        if "train_loss" in df_history:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(df_history["epoch"], df_history["train_loss"], label="train_loss")
            if "validation_pr_auc" in df_history:
                ax.plot(df_history["epoch"], df_history["validation_pr_auc"], label="validation_pr_auc")
            ax.legend()
            fig.tight_layout()
            fig.savefig(output_dir / "textcnn_training_loss.png", dpi=180)
            plt.close(fig)
        if "validation_f1_ia" in df_history:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(df_history["epoch"], df_history["validation_f1_ia"], label="validation_f1_ia")
            ax.legend()
            fig.tight_layout()
            fig.savefig(output_dir / "textcnn_training_f1.png", dpi=180)
            plt.close(fig)

    chosen = "binary_ai_ml" if (ml_report.metrics.get("f1_ia") or 0.0) >= (textcnn_report.metrics.get("f1_ia") or 0.0) else "binary_ai_textcnn"
    summary = {
        "chosen_model": chosen,
        "criterion": "test f1_ia",
        "ml_f1_ia": ml_report.metrics.get("f1_ia"),
        "textcnn_f1_ia": textcnn_report.metrics.get("f1_ia"),
        "ml_pr_auc": ml_report.metrics.get("pr_auc"),
        "textcnn_pr_auc": textcnn_report.metrics.get("pr_auc"),
    }
    (output_dir / "comparison_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Rapports écrits dans %s", output_dir)
    LOGGER.info("Modèle recommandé: %s", chosen)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

