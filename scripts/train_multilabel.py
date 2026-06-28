#!/usr/bin/env python3
"""Entraînement du classifieur multi-label avec la nouvelle taxonomie.

Usage:
    python scripts/train_multilabel.py \\
        --train data/multilabel/multilabel_dataset.csv \\
        --output models/multilabel_v2 \\
        --epochs 10 --batch-size 16 --lr 2e-5

Valide que le mapping id2label du dataset correspond à celui du modèle
(échec explicite si incohérent). Supporte fp16 + gradient checkpointing
pour fonctionnement sur ~8 Go VRAM.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_multilabel")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    from datasets import Dataset
    from sklearn.metrics import f1_score
    from torch.utils.data import DataLoader
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        get_scheduler,
    )
except ImportError as e:
    logger.error("Dépendance manquante: %s", e)
    sys.exit(1)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
#  Arguments
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Entraînement classifieur multi-label taxonomie IA")
    p.add_argument("--train", type=str, required=True,
                    help="Chemin dataset multi-label CSV")
    p.add_argument("--validation", type=str, default="",
                    help="Chemin dataset validation CSV (optionnel)")
    p.add_argument("--output", type=str, default="models/multilabel_v2",
                    help="Répertoire de sortie")
    p.add_argument("--base-model", type=str, default="camembert-base")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--max-seq-length", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=str(DEVICE))
    p.add_argument("--gradient-accumulation", type=int, default=1)
    p.add_argument("--fp16", action="store_true", default=True,
                    help="Utiliser mixed precision (fp16)")
    p.add_argument("--gradient-checkpointing", action="store_true", default=True,
                    help="Activer gradient checkpointing")
    p.add_argument("--pos-weight-cap", type=float, default=10.0,
                    help="Valeur max pour pos_weight")
    p.add_argument("--save-every-epoch", action="store_true", default=False)
    p.add_argument("--checkpoint-path", type=str, default="",
                    help="Reprendre depuis un checkpoint")
    return p


# ---------------------------------------------------------------------------
#  Dataset
# ---------------------------------------------------------------------------

def validate_label_mapping(dataset_info: dict, model_info: dict | None) -> None:
    """Vérifie que le mapping id2label du dataset correspond à celui du modèle.
    
    Lève une exception explicite si les mappings divergent.
    """
    ds_id2label = dataset_info["id2label"]
    if model_info is None:
        return
    model_id2label = model_info.get("id2label", {})
    if ds_id2label != model_id2label:
        msg = (
            f"MAPPING INCOHÉRENT: le dataset a {len(ds_id2label)} labels "
            f"(hash: {dataset_info.get('taxonomy_hash', 'N/A')}) "
            f"mais le modèle en a {len(model_id2label)}. "
            "Ré-entraînez avec --checkpoint-path vide ou mettez à jour le modèle."
        )
        logger.error(msg)
        raise ValueError(msg)


def parse_multi_values(value: str) -> list[str]:
    if pd.isna(value) or not str(value).strip():
        return []
    return [s.strip() for s in str(value).split("|") if s.strip()]


def load_multilabel_dataset(
    csv_path: str,
    info_path: str,
    n_samples: int | None = None,
) -> tuple[Dataset, list[str], dict, torch.Tensor | None]:
    """Charge le dataset multi-label avec ses métadonnées."""
    df = pd.read_csv(csv_path)
    info = json.loads(Path(info_path).read_text(encoding="utf-8"))

    label_ids: list[str] = info["label_ids"]
    pos_weight_list: list[float] = info["pos_weight"]
    id2label: dict[str, str] = info["id2label"]
    label2id: dict[str, int] = info["label2id"]

    if n_samples:
        df = df.head(n_samples)

    texts = []
    label_vectors = []
    from ast import literal_eval

    for _, row in df.iterrows():
        text = str(row.get("text", "") or "")
        multi_hot = row.get("multi_hot", "[]")
        if isinstance(multi_hot, str):
            multi_hot = literal_eval(multi_hot)
        label_vectors.append(multi_hot)
        texts.append(text)

    ds = Dataset.from_dict({"text": texts, "labels": label_vectors})

    pos_weight = None
    if pos_weight_list:
        pos_weight = torch.tensor(pos_weight_list, dtype=torch.float32)

    return ds, label_ids, info, pos_weight


# ---------------------------------------------------------------------------
#  Entraînement
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> dict[str, Any]:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available() and "cuda" in args.device:
        torch.cuda.empty_cache()

    train_csv = Path(args.train)
    info_path = train_csv.parent / "multilabel_dataset_info.json"
    if not info_path.exists():
        # Fallback: look next to CSV
        info_path = train_csv.with_suffix(".info.json")
        if not info_path.exists():
            logger.error("Fichier info introuvable: %s", info_path)
            sys.exit(1)

    # Load dataset
    ds, label_ids, ds_info, pos_weight = load_multilabel_dataset(
        str(train_csv), str(info_path)
    )
    num_labels = len(label_ids)
    logger.info("Device: %s", args.device)
    logger.info("Labels: %d", num_labels)
    logger.info("Échantillons: %d", len(ds))

    # Validate mapping vs existing checkpoint
    model_id2label = None
    if args.checkpoint_path:
        ckpt_path = Path(args.checkpoint_path)
        if (ckpt_path / "config.json").exists():
            config = json.loads((ckpt_path / "config.json").read_text())
            model_id2label = config.get("id2label")
            logger.info("Checkpoint trouvé: %s (%d labels)", ckpt_path, len(model_id2label or {}))
    validate_label_mapping(ds_info, model_id2label)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_seq_length,
        )

    ds = ds.map(tokenize, batched=True)
    ds = ds.remove_columns(["text"])
    ds.set_format("torch")

    collator = DataCollatorWithPadding(tokenizer)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
    )

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model if not args.checkpoint_path else args.checkpoint_path,
        num_labels=num_labels,
        problem_type="multi_label_classification",
        id2label=ds_info["id2label"],
        label2id=ds_info["label2id"],
        ignore_mismatched_sizes=bool(args.checkpoint_path),
    ).to(args.device)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing activé")

    model.train()

    # Optimizer
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    num_steps = args.epochs * len(loader)
    scheduler = get_scheduler(
        "linear",
        optimizer=optim,
        num_warmup_steps=int(num_steps * args.warmup_ratio),
        num_training_steps=num_steps,
    )

    # Loss
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(args.device) if pos_weight is not None else None)

    # Mixed precision
    scaler = torch.amp.GradScaler() if (args.fp16 and DEVICE.type == "cuda") else None
    if scaler:
        logger.info("Mixed precision (fp16) activée")

    # Training loop
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    best_f1 = 0.0
    best_loss = float("inf")
    epoch_metrics: list[dict] = []

    for epoch in range(args.epochs):
        epoch_losses = []
        all_preds = []
        all_labels = []

        for step, batch in enumerate(loader):
            batch = {
                k: v.to(args.device)
                for k, v in batch.items()
                if k in ("input_ids", "attention_mask", "labels")
            }

            if scaler:
                with torch.amp.autocast(DEVICE.type):
                    logits = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                    ).logits
                    loss = criterion(logits, batch["labels"].float())
            else:
                logits = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                ).logits
                loss = criterion(logits, batch["labels"].float())

            loss = loss / args.gradient_accumulation

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % args.gradient_accumulation == 0:
                if scaler:
                    scaler.step(optim)
                    scaler.update()
                else:
                    optim.step()
                scheduler.step()
                optim.zero_grad()

            epoch_losses.append(float(loss.item()) * args.gradient_accumulation)

            with torch.no_grad():
                scores = torch.sigmoid(logits).detach().cpu().numpy()
                preds = (scores >= 0.5).astype(int)
                all_preds.append(preds)
                all_labels.append(batch["labels"].detach().cpu().numpy().astype(int))

            global_step += 1

        # Epoch metrics
        avg_loss = float(np.mean(epoch_losses))
        y_true = np.vstack(all_labels)
        y_pred = np.vstack(all_preds)
        f1_micro = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
        f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

        metrics = {
            "epoch": epoch + 1,
            "loss": round(avg_loss, 4),
            "f1_micro": round(f1_micro, 4),
            "f1_macro": round(f1_macro, 4),
        }
        epoch_metrics.append(metrics)
        logger.info(
            "Epoch %d/%d | loss=%.4f | F1 micro=%.4f | F1 macro=%.4f",
            epoch + 1, args.epochs, avg_loss, f1_micro, f1_macro,
        )

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_f1 = f1_micro
            model.save_pretrained(str(output_dir / "best"))
            tokenizer.save_pretrained(str(output_dir / "best"))
            logger.info("  → Nouveau meilleur modèle (loss=%.4f)", avg_loss)

        if f1_micro > best_f1:
            best_f1 = f1_micro

        if args.save_every_epoch:
            epoch_dir = output_dir / f"epoch_{epoch + 1}"
            model.save_pretrained(str(epoch_dir))
            tokenizer.save_pretrained(str(epoch_dir))

    # Save final
    model.save_pretrained(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))

    # Save taxonomy info alongside model
    taxonomy_info = {
        "taxonomy_version": ds_info.get("taxonomy_version", "N/A"),
        "taxonomy_hash": ds_info.get("taxonomy_hash", ""),
        "num_labels": num_labels,
        "label_ids": label_ids,
        "id2label": ds_info["id2label"],
        "label2id": ds_info["label2id"],
    }
    (output_dir / "final" / "taxonomy_info.json").write_text(
        json.dumps(taxonomy_info, indent=2, ensure_ascii=False)
    )

    # Summary report
    report = {
        "output_dir": str(output_dir),
        "base_model": args.base_model,
        "num_labels": num_labels,
        "num_epochs": args.epochs,
        "best_loss": round(best_loss, 4),
        "best_f1_micro": round(best_f1, 4),
        "epoch_metrics": epoch_metrics,
        "taxonomy_hash": ds_info.get("taxonomy_hash", ""),
        "label_ids": label_ids,
    }
    (output_dir / "training_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False)
    )

    logger.info("Entraînement terminé.")
    logger.info("  Meilleure loss: %.4f", best_loss)
    logger.info("  Best F1 micro: %.4f", best_f1)
    logger.info("  Modèles sauvegardés: %s", output_dir)

    return report


def main():
    parser = build_parser()
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
