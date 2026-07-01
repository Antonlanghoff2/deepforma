#!/usr/bin/env python3
"""Entraine le classifieur multilabel de competences IA.

Usage:
    python scripts/train_ia_multilabel_classifier.py \\
        --input-dir data/processed \\
        --output-dir models/ia-classifier-v2 \\
        --base-model camembert-base \\
        --epochs 10 --batch-size 16 --lr 2e-5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_ia_multilabel_classifier")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import numpy as np
    import torch
    import torch.nn as nn
    from sklearn.metrics import f1_score
    from torch.utils.data import DataLoader, Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        get_scheduler,
    )
except ImportError as e:
    logger.error("Dependance manquante: %s", e)
    sys.exit(1)

from src.models.ia_classifier import IAClassifier


class MultilabelDataset(Dataset):
    def __init__(self, texts: list[str], labels: np.ndarray):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {"text": self.texts[idx], "labels": self.labels[idx]}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def compute_pos_weights(y: np.ndarray, cap: float = 10.0) -> torch.Tensor:
    n_pos = y.sum(axis=0)
    n_neg = y.shape[0] - n_pos
    weights = np.where(n_pos > 0, n_neg / n_pos, cap)
    weights = np.clip(weights, None, cap)
    return torch.tensor(weights, dtype=torch.float32)


def find_optimal_thresholds(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    min_thr: float = 0.15,
    max_thr: float = 0.85,
    step: float = 0.01,
) -> list[float]:
    n_labels = y_true.shape[1]
    best_thresholds = []
    for i in range(n_labels):
        best_f1 = 0.0
        best_t = 0.50
        for t in np.arange(min_thr, max_thr + step, step):
            y_pred = (y_proba[:, i] >= t).astype(int)
            f1 = f1_score(y_true[:, i], y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        best_thresholds.append(float(best_t))
    return best_thresholds


def train(args: argparse.Namespace) -> dict[str, Any]:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    input_dir = Path(args.input_dir)

    # Load data
    train_records = load_jsonl(input_dir / "ia_multilabel_train.jsonl")
    val_records = load_jsonl(input_dir / "ia_multilabel_validation.jsonl")

    metadata_path = input_dir / "ia_multilabel_metadata.json"
    with open(metadata_path) as f:
        metadata = json.load(f)

    labels_ordered = metadata["labels"]
    num_labels = len(labels_ordered)
    label2id = metadata["label2id"]

    logger.info("Labels: %d", num_labels)
    logger.info("Train: %d, Validation: %d", len(train_records), len(val_records))

    # Extract texts and multi-hot vectors
    train_texts = [r["text"] for r in train_records]
    train_labels = np.array([r["multi_hot"] for r in train_records], dtype=np.float32)

    val_texts = [r["text"] for r in val_records]
    val_labels = np.array([r["multi_hot"] for r in val_records], dtype=np.float32)

    # Pos weights from training set
    if args.use_class_weights:
        pos_weight = compute_pos_weights(train_labels, cap=args.pos_weight_cap)
        logger.info("Pos weights calcules (%d labels)", len(pos_weight))
    else:
        pos_weight = None

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    def tokenize_fn(batch_texts: list[str]) -> dict[str, Any]:
        return tokenizer(
            batch_texts,
            truncation=True,
            max_length=args.max_seq_length,
            padding=False,
        )

    # Build datasets and loaders
    train_dataset = MultilabelDataset(train_texts, train_labels)
    val_dataset = MultilabelDataset(val_texts, val_labels)

    collator = DataCollatorWithPadding(tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: _collate_fn(batch, collator, tokenize_fn),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        collate_fn=lambda batch: _collate_fn(batch, collator, tokenize_fn),
    )

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=num_labels,
        problem_type="multi_label_classification",
        id2label={str(i): lbl for i, lbl in enumerate(labels_ordered)},
        label2id={lbl: i for i, lbl in enumerate(labels_ordered)},
        ignore_mismatched_sizes=True,
    ).to(args.device)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    model.train()

    # Optimizer
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    num_steps = args.epochs * len(train_loader)
    scheduler = get_scheduler(
        "linear",
        optimizer=optim,
        num_warmup_steps=int(num_steps * args.warmup_ratio),
        num_training_steps=num_steps,
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight.to(args.device) if pos_weight is not None else None
    )

    scaler = torch.amp.GradScaler() if (args.fp16 and torch.cuda.is_available()) else None

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_f1 = 0.0
    best_val_loss = float("inf")
    patience_counter = 0
    epoch_metrics: list[dict[str, Any]] = []
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses: list[float] = []

        for batch in train_loader:
            batch = {
                k: v.to(args.device)
                for k, v in batch.items()
                if k in ("input_ids", "attention_mask", "labels")
            }

            if scaler:
                with torch.amp.autocast("cuda"):
                    logits = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                    ).logits
                    loss = criterion(logits, batch["labels"])
            else:
                logits = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                ).logits
                loss = criterion(logits, batch["labels"])

            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                optim.step()

            scheduler.step()
            optim.zero_grad()

            train_losses.append(float(loss.item()))
            global_step += 1

        avg_train_loss = float(np.mean(train_losses))

        # Validation
        val_losses: list[float] = []
        all_val_preds: list[np.ndarray] = []
        all_val_labels: list[np.ndarray] = []

        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                batch = {
                    k: v.to(args.device)
                    for k, v in batch.items()
                    if k in ("input_ids", "attention_mask", "labels")
                }
                logits = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                ).logits
                loss = criterion(logits, batch["labels"])
                val_losses.append(float(loss.item()))

                probas = torch.sigmoid(logits).cpu().numpy()
                preds = (probas >= 0.5).astype(int)
                all_val_preds.append(preds)
                all_val_labels.append(batch["labels"].cpu().numpy().astype(int))

        avg_val_loss = float(np.mean(val_losses))
        y_val_true = np.vstack(all_val_labels)
        y_val_pred = np.vstack(all_val_preds)
        val_f1_micro = float(f1_score(y_val_true, y_val_pred, average="micro", zero_division=0))
        val_f1_macro = float(f1_score(y_val_true, y_val_pred, average="macro", zero_division=0))

        metrics = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(avg_val_loss, 4),
            "val_f1_micro": round(val_f1_micro, 4),
            "val_f1_macro": round(val_f1_macro, 4),
        }
        epoch_metrics.append(metrics)

        logger.info(
            "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | "
            "val_f1_micro=%.4f | val_f1_macro=%.4f",
            epoch, args.epochs, avg_train_loss, avg_val_loss,
            val_f1_micro, val_f1_macro,
        )

        # Save best based on macro-F1
        if val_f1_macro > best_val_f1:
            best_val_f1 = val_f1_macro
            best_val_loss = avg_val_loss
            patience_counter = 0
            model.save_pretrained(str(output_dir / "best"))
            tokenizer.save_pretrained(str(output_dir / "best"))
            logger.info("  -> Nouveau meilleur modele (val_f1_macro=%.4f)", val_f1_macro)
        else:
            patience_counter += 1

        if args.early_stopping_patience > 0 and patience_counter >= args.early_stopping_patience:
            logger.info(
                "Early stopping apres %d epochs sans amelioration.", patience_counter
            )
            break

    # Save final
    model.save_pretrained(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))
    logger.info("Modele final sauvegarde: %s/final", output_dir)

    # Compute optimal thresholds on validation set
    model.eval()
    all_val_probas: list[np.ndarray] = []
    with torch.no_grad():
        for batch in val_loader:
            batch = {
                k: v.to(args.device)
                for k, v in batch.items()
                if k in ("input_ids", "attention_mask")
            }
            logits = model(**batch).logits
            probas = torch.sigmoid(logits).cpu().numpy()
            all_val_probas.append(probas)
    y_val_proba = np.vstack(all_val_probas)

    optimal_thresholds = find_optimal_thresholds(
        y_val_true, y_val_proba,
        min_thr=args.min_threshold,
        max_thr=args.max_threshold,
    )

    # Save thresholds
    thresholds_dict = {
        labels_ordered[i]: round(optimal_thresholds[i], 4)
        for i in range(num_labels)
    }
    thr_path = output_dir / "final" / "thresholds.json"
    with open(thr_path, "w") as f:
        json.dump({"thresholds": thresholds_dict}, f, indent=2)
    logger.info("Seuils optimaux sauvegardes: %s", thr_path)

    # Save taxonomy info alongside model
    taxo_info = {
        "taxonomy_version": metadata["taxonomy_version"],
        "model_version": metadata.get("model_version", "1.0"),
        "num_labels": num_labels,
        "labels": labels_ordered,
        "label2id": label2id,
        "training_args": vars(args),
        "best_val_f1_macro": round(best_val_f1, 4),
    }
    taxo_path = output_dir / "final" / "taxonomy_info.json"
    with open(taxo_path, "w", encoding="utf-8") as f:
        json.dump(taxo_info, f, indent=2, ensure_ascii=False)

    # Training report
    report = {
        "output_dir": str(output_dir),
        "base_model": args.base_model,
        "num_labels": num_labels,
        "num_epochs": epoch,
        "best_val_f1_macro": round(best_val_f1, 4),
        "best_val_loss": round(best_val_loss, 4),
        "epoch_metrics": epoch_metrics,
        "optimal_thresholds": thresholds_dict,
    }
    report_path = output_dir / "training_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Rapport d'entrainement: %s", report_path)

    logger.info(
        "Entrainement termine. Best val macro-F1: %.4f", best_val_f1
    )
    return report


def _collate_fn(
    batch: list[dict[str, Any]], collator: Any, tokenize_fn: Any
) -> dict[str, Any]:
    texts = [b["text"] for b in batch]
    labels = torch.stack([b["labels"] for b in batch])
    tokenized = collator(tokenize_fn(texts))
    tokenized["labels"] = labels
    return tokenized


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Entrainement du classifieur multilabel IA"
    )
    p.add_argument("--input-dir", type=str, default="data/processed")
    p.add_argument("--output-dir", type=str, default="models/ia-classifier-v2")
    p.add_argument("--base-model", type=str, default="camembert-base")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--max-seq-length", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--fp16", action="store_true", default=True)
    p.add_argument("--no-fp16", action="store_false", dest="fp16")
    p.add_argument("--gradient-checkpointing", action="store_true", default=True)
    p.add_argument("--no-gradient-checkpointing", action="store_false", dest="gradient_checkpointing")
    p.add_argument("--pos-weight-cap", type=float, default=10.0)
    p.add_argument("--use-class-weights", action="store_true", default=True)
    p.add_argument("--no-class-weights", action="store_false", dest="use_class_weights")
    p.add_argument("--early-stopping-patience", type=int, default=5)
    p.add_argument("--min-threshold", type=float, default=0.15)
    p.add_argument("--max-threshold", type=float, default=0.85)
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
