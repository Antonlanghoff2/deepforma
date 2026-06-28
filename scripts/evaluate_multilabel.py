#!/usr/bin/env python3
"""Évaluation du classifieur multi-label avec métriques détaillées.

Usage:
    python scripts/evaluate_multilabel.py \\
        --model models/multilabel_v2/final \\
        --test data/multilabel/multilabel_dataset.csv

Calcule F1 micro/macro par label, précision, rappel, support,
et cherche le seuil global optimal (0.05-0.95).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_multilabel")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import numpy as np
    import pandas as pd
    import torch
    from ast import literal_eval
    from datasets import Dataset
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        f1_score,
        precision_recall_fscore_support,
    )
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
except ImportError as e:
    logger.error("Dépendance manquante: %s", e)
    sys.exit(1)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
#  Arguments
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Évaluation classifieur multi-label taxonomie IA")
    p.add_argument("--model", type=str, required=True,
                    help="Chemin du modèle (répertoire avec config.json)")
    p.add_argument("--test", type=str, required=True,
                    help="Chemin dataset de test CSV")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-seq-length", type=int, default=256)
    p.add_argument("--threshold", type=float, default=0.5,
                    help="Seuil de prédiction global")
    p.add_argument("--output", type=str, default="",
                    help="Répertoire pour sauvegarder le rapport")
    return p


# ---------------------------------------------------------------------------
#  Prédiction
# ---------------------------------------------------------------------------

def load_model_and_data(
    model_path: str,
    test_csv: str,
    batch_size: int,
    max_seq_length: int,
) -> tuple[Any, list[str], Dataset, dict]:
    """Charge le modèle et le dataset de test."""
    model_dir = Path(model_path)
    if not (model_dir / "config.json").exists():
        logger.error("config.json introuvable dans %s", model_dir)
        sys.exit(1)

    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    id2label: dict[str, str] = config.get("id2label", {})
    label2id: dict[str, int] = config.get("label2id", {})

    if not id2label or not label2id:
        # Try taxonomy_info.json
        tax_path = model_dir / "taxonomy_info.json"
        if tax_path.exists():
            tax_info = json.loads(tax_path.read_text(encoding="utf-8"))
            id2label = tax_info.get("id2label", {})
            label2id = tax_info.get("label2id", {})
        else:
            logger.error("Aucun mapping id2label/label2id trouvé dans le modèle")
            sys.exit(1)

    logger.info("Labels dans le modèle: %d", len(id2label))

    # Build ordered label list
    max_idx = max(int(k) for k in id2label)
    label_ids = [id2label[str(i)] for i in range(max_idx + 1)]

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_dir),
        num_labels=len(label_ids),
        problem_type="multi_label_classification",
    ).to(DEVICE)
    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

    # Load data
    df = pd.read_csv(test_csv)
    if "multi_hot" not in df.columns:
        # Try to load from train-style CSV
        info_path = Path(test_csv).parent / "multilabel_dataset_info.json"
        if info_path.exists():
            info = json.loads(info_path.read_text(encoding="utf-8"))
            label2id = info.get("label2id", {})

    texts = []
    label_vectors = []

    for _, row in df.iterrows():
        text = str(row.get("text", "") or "")
        multi_hot = row.get("multi_hot", "[]")
        if isinstance(multi_hot, str):
            multi_hot = literal_eval(multi_hot)
        texts.append(text)
        label_vectors.append(multi_hot)

    ds = Dataset.from_dict({"text": texts, "labels": label_vectors})

    # Tokenize
    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_seq_length)

    ds = ds.map(tokenize, batched=True)
    ds = ds.remove_columns(["text"])
    ds.set_format("torch")

    collator = DataCollatorWithPadding(tokenizer)
    loader = DataLoader(ds, batch_size=batch_size, collate_fn=collator)

    return model, label_ids, loader, {"id2label": id2label, "label2id": label2id}


# ---------------------------------------------------------------------------
#  Métriques
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray | None,
    label_ids: list[str],
    threshold: float,
) -> dict[str, Any]:
    """Calcule toutes les métriques d'évaluation."""
    n_labels = len(label_ids)

    # Global
    f1_micro = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # Per label
    precision, recall, f1_per_label, support = precision_recall_fscore_support(
        y_true, y_pred, zero_division=0, labels=range(n_labels)
    )

    per_label = {}
    for i, label_id in enumerate(label_ids):
        per_label[label_id] = {
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1": round(float(f1_per_label[i]), 4),
            "support": int(support[i]),
        }

    # Accuracy (subset accuracy)
    subset_acc = float(accuracy_score(y_true, y_pred))

    # Hamming loss
    hamming_loss = float(np.mean(y_true != y_pred))

    result: dict[str, Any] = {
        "threshold": threshold,
        "n_labels": n_labels,
        "n_samples": y_true.shape[0],
        "f1_micro": round(f1_micro, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
        "subset_accuracy": round(subset_acc, 4),
        "hamming_loss": round(hamming_loss, 4),
        "per_label": per_label,
        "label_order": label_ids,
    }

    # Score distribution
    if scores is not None:
        result["score_stats"] = {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
            "median": float(np.median(scores)),
        }

    return result


def find_optimal_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    label_ids: list[str],
    thresholds: list[float],
) -> dict[str, Any]:
    """Cherche le seuil global optimal."""
    best_threshold = 0.5
    best_f1 = 0.0
    results = []

    for thresh in thresholds:
        y_pred = (scores >= thresh).astype(int)
        f1_micro = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
        results.append({"threshold": thresh, "f1_micro": round(f1_micro, 4)})
        if f1_micro > best_f1:
            best_f1 = f1_micro
            best_threshold = thresh

    return {
        "best_threshold": best_threshold,
        "best_f1_micro": round(best_f1, 4),
        "threshold_sweep": results,
    }


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    logger.info("Device: %s", DEVICE)
    logger.info("Modèle: %s", args.model)
    logger.info("Test: %s", args.test)

    model, label_ids, loader, mapping = load_model_and_data(
        args.model, args.test, args.batch_size, args.max_seq_length
    )

    y_true_list = []
    y_pred_list = []
    scores_list = []

    with torch.no_grad():
        for batch in loader:
            batch = {
                k: v.to(DEVICE)
                for k, v in batch.items()
                if k in ("input_ids", "attention_mask", "labels")
            }

            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            ).logits

            scores = torch.sigmoid(logits).cpu().numpy()
            preds = (scores >= args.threshold).astype(int)

            y_true_list.append(batch["labels"].detach().cpu().numpy().astype(int))
            y_pred_list.append(preds)
            scores_list.append(scores)

    y_true = np.vstack(y_true_list)
    y_pred = np.vstack(y_pred_list)
    scores = np.vstack(scores_list) if scores_list else None

    # Métriques principales
    metrics = compute_metrics(y_true, y_pred, scores, label_ids, args.threshold)
    logger.info(
        "Seuil=%.2f | F1 micro=%.4f | F1 macro=%.4f | Subset acc=%.4f",
        args.threshold,
        metrics["f1_micro"],
        metrics["f1_macro"],
        metrics["subset_accuracy"],
    )

    # Résumé par label
    logger.info("\n  %-30s %8s %8s %8s %8s", "Label", "Prec", "Rappel", "F1", "Sup")
    for label_id in label_ids:
        pl = metrics["per_label"][label_id]
        logger.info(
            "  %-30s %8.4f %8.4f %8.4f %8d",
            label_id, pl["precision"], pl["recall"], pl["f1"], pl["support"],
        )

    # Recherche de seuil optimal
    thresholds = [round(t, 2) for t in np.arange(0.05, 0.96, 0.05)]
    threshold_search = find_optimal_threshold(y_true, scores, label_ids, thresholds)
    logger.info(
        "\nMeilleur seuil global: %.2f (F1 micro=%.4f)",
        threshold_search["best_threshold"],
        threshold_search["best_f1_micro"],
    )

    # Rapport final
    report = {
        "model": args.model,
        "dataset": args.test,
        "device": str(DEVICE),
        "metrics": metrics,
        "threshold_optimization": threshold_search,
    }

    # Sauvegarder
    if args.output:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "evaluation_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False)
        )
        logger.info("\nRapport sauvegardé: %s", out_dir / "evaluation_report.json")
    else:
        print(json.dumps(report, indent=2, ensure_ascii=False))

    return report


def main():
    parser = build_parser()
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
