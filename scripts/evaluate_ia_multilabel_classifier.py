#!/usr/bin/env python3
"""Evalue le classifieur multilabel de competences IA sur le jeu de test.

Produit rapports JSON, CSV, MD avec micro-F1, macro-F1, precision, rappel,
metriques par label, matrice de cooccurrence, distribution des predictions.

Usage:
    python scripts/evaluate_ia_multilabel_classifier.py \\
        --model-dir models/ia-classifier-v2/final \\
        --test-file data/processed/ia_multilabel_test.jsonl \\
        --output-dir reports
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_ia_multilabel_classifier")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import numpy as np
    import pandas as pd
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
except ImportError as e:
    logger.error("Dependance manquante: %s", e)
    sys.exit(1)

from src.models.ia_classifier import IAClassifier


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    model_dir = Path(args.model_dir)
    test_path = Path(args.test_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Chargement du modele depuis %s...", model_dir)
    classifier = IAClassifier(
        model_dir=model_dir,
        taxonomy_path=args.taxonomy,
        device=args.device,
        fallback_threshold=0.50,
    )

    logger.info("Chargement du test depuis %s...", test_path)
    test_records = load_jsonl(test_path)
    logger.info("Echantillons de test: %d", len(test_records))

    texts = [r["text"] for r in test_records]
    y_true = np.array([r["multi_hot"] for r in test_records], dtype=np.float32)
    labels_ordered = classifier.labels

    logger.info("Inference sur %d echantillons...", len(texts))
    y_proba = classifier.predict_probas(texts)

    # Apply per-label thresholds
    y_pred = np.zeros_like(y_proba)
    for i, label in enumerate(labels_ordered):
        thr = classifier._get_threshold(label)
        y_pred[:, i] = (y_proba[:, i] >= thr).astype(int)

    # Also compute predictions at 0.5 for reference
    y_pred_05 = (y_proba >= 0.50).astype(int)

    # Global metrics at optimal thresholds
    f1_micro = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    prec_micro = float(precision_score(y_true, y_pred, average="micro", zero_division=0))
    prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))

    rec_micro = float(recall_score(y_true, y_pred, average="micro", zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    # Average precision
    ap_micro = float(average_precision_score(y_true, y_proba, average="micro"))
    ap_macro = float(average_precision_score(y_true, y_proba, average="macro"))

    # ROC-AUC (per label, where possible)
    roc_aucs = {}
    for i, label in enumerate(labels_ordered):
        try:
            auc = roc_auc_score(y_true[:, i], y_proba[:, i])
            roc_aucs[label] = round(float(auc), 4)
        except Exception:
            roc_aucs[label] = None

    # Per-label metrics
    per_label = []
    for i, label in enumerate(labels_ordered):
        y_true_i = y_true[:, i].astype(int)
        y_pred_i = y_pred[:, i].astype(int)
        per_label.append({
            "label": label,
            "support": int(y_true_i.sum()),
            "f1": round(float(f1_score(y_true_i, y_pred_i, zero_division=0)), 4),
            "precision": round(float(precision_score(y_true_i, y_pred_i, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true_i, y_pred_i, zero_division=0)), 4),
            "roc_auc": roc_aucs[label],
            "threshold": classifier.thresholds.get(label, 0.50),
            "n_predicted": int(y_pred_i.sum()),
        })

    # Empty predictions statistics
    n_empty = int((y_pred.sum(axis=1) == 0).sum())
    n_all = int((y_pred.sum(axis=1) == len(labels_ordered)).sum())

    # Prediction distribution
    pred_counts_per_label = [int(y_pred[:, i].sum()) for i in range(len(labels_ordered))]

    # Co-occurrence matrix
    n_labels = len(labels_ordered)
    cooc = np.zeros((n_labels, n_labels), dtype=int)
    for i in range(n_labels):
        for j in range(i, n_labels):
            both = ((y_pred[:, i] == 1) & (y_pred[:, j] == 1)).sum()
            cooc[i, j] = both
            cooc[j, i] = both

    cooc_df = pd.DataFrame(
        cooc, index=labels_ordered, columns=labels_ordered
    )

    # Probability distribution (positive vs negative)
    pos_probas = y_proba[y_true == 1]
    neg_probas = y_proba[y_true == 0]

    proba_hist = {
        "positive": {
            "mean": float(pos_probas.mean()) if len(pos_probas) > 0 else 0,
            "std": float(pos_probas.std()) if len(pos_probas) > 0 else 0,
            "min": float(pos_probas.min()) if len(pos_probas) > 0 else 0,
            "max": float(pos_probas.max()) if len(pos_probas) > 0 else 0,
        },
        "negative": {
            "mean": float(neg_probas.mean()) if len(neg_probas) > 0 else 0,
            "std": float(neg_probas.std()) if len(neg_probas) > 0 else 0,
            "min": float(neg_probas.min()) if len(neg_probas) > 0 else 0,
            "max": float(neg_probas.max()) if len(neg_probas) > 0 else 0,
        },
    }

    # Verify training signal
    per_label_std = [float(y_proba[:, i].std()) for i in range(n_labels)]
    mean_std = float(np.mean(per_label_std))

    training_signal_weak = mean_std < 0.03 or (
        float(np.mean(y_proba)) > 0.40 and float(np.mean(y_proba)) < 0.60
        and mean_std < 0.05
    )

    if training_signal_weak:
        logger.warning(
            "ATTENTION: Signal d'entrainement faible. "
            "Ecart-type moyen des probabilites: %.4f (< 0.03). "
            "Le modele n'a probablement pas appris.",
            mean_std,
        )

    # Error analysis
    errors = []
    for idx in range(min(len(test_records), args.max_errors)):
        yt = y_true[idx].astype(int)
        yp = y_pred[idx].astype(int)
        false_pos = [labels_ordered[i] for i in range(n_labels) if yp[i] == 1 and yt[i] == 0]
        false_neg = [labels_ordered[i] for i in range(n_labels) if yp[i] == 0 and yt[i] == 1]
        true_pos = [labels_ordered[i] for i in range(n_labels) if yp[i] == 1 and yt[i] == 1]

        if false_pos or false_neg:
            errors.append({
                "text": texts[idx][:200],
                "true_labels": [labels_ordered[i] for i in range(n_labels) if yt[i] == 1],
                "predicted_labels": [labels_ordered[i] for i in range(n_labels) if yp[i] == 1],
                "false_positives": false_pos,
                "false_negatives": false_neg,
                "true_positives": true_pos,
                "probabilities": {
                    labels_ordered[i]: round(float(y_proba[idx, i]), 4)
                    for i in range(n_labels)
                },
            })

    # Build report
    metrics = {
        "micro_f1": round(f1_micro, 4),
        "macro_f1": round(f1_macro, 4),
        "weighted_f1": round(f1_weighted, 4),
        "micro_precision": round(prec_micro, 4),
        "macro_precision": round(prec_macro, 4),
        "micro_recall": round(rec_micro, 4),
        "macro_recall": round(rec_macro, 4),
        "micro_average_precision": round(ap_micro, 4),
        "macro_average_precision": round(ap_macro, 4),
    }

    report = {
        "model_path": str(model_dir),
        "test_file": str(test_path),
        "num_samples": len(test_records),
        "num_labels": n_labels,
        "metrics": metrics,
        "per_label_metrics": per_label,
        "proba_distribution": proba_hist,
        "training_signal": {
            "mean_proba_std": round(mean_std, 4),
            "weak_signal_detected": training_signal_weak,
        },
        "prediction_statistics": {
            "empty_predictions": n_empty,
            "empty_pct": round(n_empty / len(test_records) * 100, 2),
            "all_labels_predicted": n_all,
            "all_labels_pct": round(n_all / len(test_records) * 100, 2),
            "per_label_predicted": pred_counts_per_label,
        },
        "num_errors_analyzed": len(errors),
    }

    # Save metrics JSON
    metrics_path = output_dir / "ia_classifier_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info("Metriques sauvegardees: %s", metrics_path)

    # Save per-label CSV
    per_label_df = pd.DataFrame(per_label)
    per_label_path = output_dir / "ia_classifier_per_label.csv"
    per_label_df.to_csv(per_label_path, index=False)
    logger.info("Metriques par label: %s", per_label_path)

    # Save errors CSV
    errors_df = pd.DataFrame(errors) if errors else pd.DataFrame()
    errors_path = output_dir / "ia_classifier_errors.csv"
    errors_df.to_csv(errors_path, index=False)
    logger.info("Erreurs sauvegardees: %s (%d)", errors_path, len(errors))

    # Save markdown report
    md = _build_markdown_report(report, labels_ordered)
    md_path = output_dir / "ia_classifier_report.md"
    md_path.write_text(md, encoding="utf-8")
    logger.info("Rapport markdown: %s", md_path)

    # Also save co-occurrence
    cooc_path = output_dir / "ia_classifier_cooccurrence.csv"
    cooc_df.to_csv(cooc_path)
    logger.info("Matrice de cooccurrence: %s", cooc_path)

    logger.info("Evaluation terminee.")
    logger.info("  Micro-F1: %.4f", f1_micro)
    logger.info("  Macro-F1: %.4f", f1_macro)
    logger.info("  Precision micro: %.4f", prec_micro)
    logger.info("  Rappel micro: %.4f", rec_micro)

    return report


def _build_markdown_report(report: dict, labels: list[str]) -> str:
    m = report["metrics"]
    lines = [
        "# Rapport d'evaluation du classifieur multilabel IA",
        "",
        f"- **Fichier de test** : {report['test_file']}",
        f"- **Echantillons** : {report['num_samples']}",
        f"- **Labels** : {report['num_labels']}",
        "",
        "## Metriques globales",
        "",
        "| Metrique | Valeur |",
        "|----------|--------|",
        f"| Micro-F1 | {m['micro_f1']} |",
        f"| Macro-F1 | {m['macro_f1']} |",
        f"| Weighted-F1 | {m['weighted_f1']} |",
        f"| Precision micro | {m['micro_precision']} |",
        f"| Precision macro | {m['macro_precision']} |",
        f"| Rappel micro | {m['micro_recall']} |",
        f"| Rappel macro | {m['macro_recall']} |",
        f"| Average precision micro | {m['micro_average_precision']} |",
        f"| Average precision macro | {m['macro_average_precision']} |",
        "",
        "## Distribution des probabilites",
        "",
        "| | Positive | Negative |",
        "|----------|----------|----------|",
    ]

    pd_ = report["proba_distribution"]
    for stat in ("mean", "std", "min", "max"):
        pos_v = pd_["positive"].get(stat, "N/A")
        neg_v = pd_["negative"].get(stat, "N/A")
        lines.append(f"| {stat.capitalize()} | {pos_v} | {neg_v} |")

    lines += [
        "",
        "## Signal d'entrainement",
        "",
        f"- Ecart-type moyen des probabilites : {report['training_signal']['mean_proba_std']}",
        f"- Signal faible detecte : {'OUI' if report['training_signal']['weak_signal_detected'] else 'NON'}",
        "",
        "## Statistiques de prediction",
        "",
        f"- Predictions vides : {report['prediction_statistics']['empty_predictions']} ({report['prediction_statistics']['empty_pct']}%)",
        f"- Tous les labels predits : {report['prediction_statistics']['all_labels_predicted']} ({report['prediction_statistics']['all_labels_pct']}%)",
        "",
        "## Metriques par label",
        "",
        "| Label | Support | F1 | Precision | Rappel | ROC-AUC | Seuil | Predits |",
        "|-------|---------|----|-----------|--------|---------|-------|---------|",
    ]

    for pl in report["per_label_metrics"]:
        auc = pl.get("roc_auc", "N/A")
        lines.append(
            f"| {pl['label']} | {pl['support']} | {pl['f1']} | "
            f"{pl['precision']} | {pl['recall']} | {auc} | "
            f"{pl['threshold']} | {pl['n_predicted']} |"
        )

    if report["num_errors_analyzed"] > 0:
        lines += [
            "",
            f"## Analyse d'erreurs ({report['num_errors_analyzed']} echantillons)",
            "",
            "Voir le fichier CSV `ia_classifier_errors.csv` pour le detail.",
        ]

    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluation du classifieur multilabel IA"
    )
    p.add_argument("--model-dir", type=str, required=True)
    p.add_argument("--test-file", type=str, required=True)
    p.add_argument("--output-dir", type=str, default="reports")
    p.add_argument("--taxonomy", type=str, default="config/ia_taxonomy_v2.json")
    p.add_argument("--device", type=str, default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    p.add_argument("--max-errors", type=int, default=100)
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
