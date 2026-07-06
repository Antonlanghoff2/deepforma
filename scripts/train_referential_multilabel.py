#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / 'src'
for _path in (ROOT_DIR, SRC_DIR):
    _text = str(_path)
    if _text not in sys.path:
        sys.path.insert(0, _text)

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

LABELS = ['Machine Learning', 'Deep Learning', 'NLP', 'MLOps', 'Other']


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Entraîne le classifieur multilabel référentiel')
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--validation', type=Path, required=True)
    parser.add_argument('--test', type=Path, required=True)
    parser.add_argument('--base-model', type=str, default='camembert-base')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--learning-rate', type=float, default=2e-5)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-length', type=int, default=256)
    return parser


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding='utf-8').splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _labels_from_row(row: dict[str, Any]) -> list[str]:
    labels = row.get('approved_labels') or row.get('predicted_labels') or []
    cleaned = []
    for label in labels:
        value = str(label or '')
        if value in LABELS and value not in cleaned:
            cleaned.append(value)
    if not cleaned:
        cleaned = ['Other']
    return cleaned


def _multi_hot(labels: list[str]) -> list[int]:
    return [1 if label in labels else 0 for label in LABELS]


def _predict_scores(model, tokenizer, texts: list[str], *, max_length: int, batch_size: int, device: str) -> np.ndarray:
    import torch

    model = model.to(device)
    model.eval()
    all_probs: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        enc = tokenizer(batch, truncation=True, padding=True, max_length=max_length, return_tensors='pt').to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
    return np.vstack(all_probs) if all_probs else np.zeros((0, len(LABELS)))


def _tune_thresholds(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
    for index, label in enumerate(LABELS):
        best_threshold = 0.5
        best_f1 = -1.0
        gold = y_true[:, index]
        scores = y_score[:, index]
        if gold.sum() == 0:
            thresholds[label] = 0.5
            continue
        for threshold in grid:
            pred = (scores >= threshold).astype(int)
            f1 = f1_score(gold, pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = float(threshold)
        thresholds[label] = round(best_threshold, 4)
    return thresholds


def _apply_thresholds(scores: np.ndarray, thresholds: dict[str, float]) -> np.ndarray:
    preds = np.zeros_like(scores, dtype=int)
    for index, label in enumerate(LABELS):
        preds[:, index] = (scores[:, index] >= float(thresholds.get(label, 0.5))).astype(int)
    return preds


def _metrics_from_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    per_label = {}
    for index, label in enumerate(LABELS):
        per_label[label] = {
            'precision': round(float(precision_score(y_true[:, index], y_pred[:, index], zero_division=0)), 4),
            'recall': round(float(recall_score(y_true[:, index], y_pred[:, index], zero_division=0)), 4),
            'f1': round(float(f1_score(y_true[:, index], y_pred[:, index], zero_division=0)), 4),
            'support': int(y_true[:, index].sum()),
        }
    return {
        'f1_micro': round(float(f1_score(y_true, y_pred, average='micro', zero_division=0)), 4),
        'f1_macro': round(float(f1_score(y_true, y_pred, average='macro', zero_division=0)), 4),
        'per_label': per_label,
        'precision_micro': round(float(precision_score(y_true, y_pred, average='micro', zero_division=0)), 4),
        'recall_micro': round(float(recall_score(y_true, y_pred, average='micro', zero_division=0)), 4),
    }


def main() -> None:
    args = build_parser().parse_args()
    try:
        from datasets import Dataset  # type: ignore
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments  # type: ignore
    except Exception as exc:
        raise SystemExit(f'Dépendances d entraînement multilabel manquantes: {exc}')

    train_rows = _read_jsonl(args.train)
    validation_rows = _read_jsonl(args.validation)
    test_rows = _read_jsonl(args.test)
    if not train_rows and not validation_rows and not test_rows:
        raise SystemExit('Aucun exemple multilabel à entraîner.')

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=len(LABELS),
        problem_type='multi_label_classification',
        id2label={i: label for i, label in enumerate(LABELS)},
        label2id={label: i for i, label in enumerate(LABELS)},
    )

    def encode(batch: dict[str, list[Any]]) -> dict[str, Any]:
        texts = [str(text or '') for text in batch['text']]
        tokenized = tokenizer(texts, truncation=True, max_length=args.max_length, padding=False)
        tokenized['labels'] = [_multi_hot(_labels_from_row({'approved_labels': labels})) for labels in batch['approved_labels']]
        return tokenized

    def prepare_dataset(rows: list[dict[str, Any]]) -> Dataset:
        dataset_rows = []
        for row in rows:
            copy = dict(row)
            copy['approved_labels'] = _labels_from_row(row)
            copy['labels'] = _multi_hot(copy['approved_labels'])
            dataset_rows.append(copy)
        return Dataset.from_list(dataset_rows)

    train_ds = prepare_dataset(train_rows)
    val_ds = prepare_dataset(validation_rows) if validation_rows else prepare_dataset(train_rows[:1])
    test_ds = prepare_dataset(test_rows) if test_rows else prepare_dataset(train_rows[:1])

    def tokenize_dataset(dataset: Dataset) -> Dataset:
        def _encode(batch: dict[str, list[Any]]) -> dict[str, Any]:
            texts = [str(text or '') for text in batch['text']]
            tokenized = tokenizer(texts, truncation=True, max_length=args.max_length, padding=False)
            tokenized['labels'] = [list(map(float, labels)) for labels in batch['labels']]
            return tokenized
        return dataset.map(_encode, batched=True, remove_columns=dataset.column_names)

    train_ds = tokenize_dataset(train_ds)
    val_ds = tokenize_dataset(val_ds)
    test_ds = tokenize_dataset(test_ds)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = 1 / (1 + np.exp(-logits))
        preds = (probs >= 0.5).astype(int)
        return _metrics_from_predictions(labels.astype(int), preds)

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        evaluation_strategy='epoch',
        save_strategy='epoch',
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        seed=args.seed,
        fp16=args.fp16,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    val_output = trainer.predict(val_ds)
    val_scores = 1 / (1 + np.exp(-val_output.predictions))
    val_labels = np.array(val_output.label_ids)
    thresholds = _tune_thresholds(val_labels, val_scores)

    test_output = trainer.predict(test_ds)
    test_scores = 1 / (1 + np.exp(-test_output.predictions))
    test_labels = np.array(test_output.label_ids)
    test_preds = _apply_thresholds(test_scores, thresholds)
    metrics = _metrics_from_predictions(test_labels, test_preds)
    metrics['thresholds'] = thresholds
    metrics['validation_thresholds'] = thresholds

    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_dir = args.output_dir / 'final'
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    (final_dir / 'label_classes.json').write_text(json.dumps(LABELS, ensure_ascii=False, indent=2), encoding='utf-8')
    (final_dir / 'thresholds.json').write_text(json.dumps(thresholds, ensure_ascii=False, indent=2), encoding='utf-8')
    (args.output_dir / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
