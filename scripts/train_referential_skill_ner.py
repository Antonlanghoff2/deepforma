#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common.text import normalize_for_match
from referential_learning.section_labels import NER_LABELS

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Entraîne un modèle NER pour les compétences référentielles')
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--validation', type=Path, required=True)
    parser.add_argument('--test', type=Path, required=True)
    parser.add_argument('--base-model', type=str, default='camembert-base')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--gradient-accumulation-steps', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--learning-rate', type=float, default=2e-5)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-length', type=int, default=256)
    parser.add_argument('--stride', type=int, default=64)
    return parser

def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding='utf-8').splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows

def _spans_to_bio(text: str, entities: list[dict[str, Any]], labels: list[str]) -> tuple[list[str], list[int], list[tuple[int, int]]]:
    try:
        from transformers import AutoTokenizer  # type: ignore
    except Exception as exc:
        raise SystemExit(f'Transformers manquant: {exc}')
    tokenizer = AutoTokenizer.from_pretrained('camembert-base', use_fast=True)
    encoding = tokenizer(text, return_offsets_mapping=True, truncation=True, max_length=256)
    offsets = encoding['offset_mapping']
    tokens = encoding['input_ids']
    bio_labels = [-100] * len(tokens)
    label_to_id = {label: idx for idx, label in enumerate(['O', *[f'B-{label}' for label in labels], *[f'I-{label}' for label in labels]])}
    for entity in entities:
        start = int(entity.get('start', 0))
        end = int(entity.get('end', 0))
        label = str(entity.get('approved_label') or entity.get('predicted_label') or 'OTHER')
        if label not in labels:
            label = 'OTHER'
        b_label = f'B-{label}'
        i_label = f'I-{label}'
        for idx, (offset_start, offset_end) in enumerate(offsets):
            if offset_start == offset_end:
                continue
            if offset_start >= start and offset_end <= end:
                bio_labels[idx] = label_to_id[b_label if offset_start == start else i_label]
    return tokens, bio_labels, offsets

def main() -> None:
    args = build_parser().parse_args()
    try:
        from datasets import Dataset  # type: ignore
        from seqeval.metrics import classification_report, f1_score, precision_score, recall_score  # type: ignore
        from transformers import (  # type: ignore
            AutoModelForTokenClassification,
            AutoTokenizer,
            DataCollatorForTokenClassification,
            Trainer,
            TrainingArguments,
        )
    except Exception as exc:
        raise SystemExit(f'Dépendances d entraînement NER manquantes: {exc}')

    train_rows = _read_jsonl(args.train)
    validation_rows = _read_jsonl(args.validation)
    test_rows = _read_jsonl(args.test)
    labels = ['OTHER', *NER_LABELS]
    label_list = ['O', *[f'B-{label}' for label in labels], *[f'I-{label}' for label in labels]]
    label2id = {label: idx for idx, label in enumerate(label_list)}
    id2label = {idx: label for label, idx in label2id.items()}
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(args.base_model, num_labels=len(label_list), id2label=id2label, label2id=label2id)

    def encode(example: dict[str, Any]) -> dict[str, Any]:
        encoding = tokenizer(
            example['text'],
            truncation=True,
            max_length=args.max_length,
            return_offsets_mapping=True,
            return_overflowing_tokens=False,
            stride=args.stride,
        )
        offsets = encoding.pop('offset_mapping')
        labels_ids = [-100] * len(encoding['input_ids'])
        for entity in example.get('entities', []):
            start = int(entity.get('start', 0))
            end = int(entity.get('end', 0))
            label = str(entity.get('approved_label') or entity.get('predicted_label') or 'OTHER')
            if label not in labels:
                label = 'OTHER'
            for idx, (offset_start, offset_end) in enumerate(offsets):
                if offset_start == offset_end:
                    continue
                if offset_start >= start and offset_end <= end:
                    labels_ids[idx] = label2id[f'B-{label}' if offset_start == start else f'I-{label}']
        encoding['labels'] = labels_ids
        return encoding

    train_ds = Dataset.from_list(train_rows).map(encode)
    val_ds = Dataset.from_list(validation_rows).map(encode)
    test_ds = Dataset.from_list(test_rows).map(encode)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        true_labels = []
        pred_labels = []
        for pred_row, label_row in zip(preds, labels):
            row_true = []
            row_pred = []
            for pred_id, label_id in zip(pred_row, label_row):
                if label_id == -100:
                    continue
                row_true.append(id2label[int(label_id)])
                row_pred.append(id2label[int(pred_id)])
            true_labels.append(row_true)
            pred_labels.append(row_pred)
        return {
            'precision': float(precision_score(true_labels, pred_labels)),
            'recall': float(recall_score(true_labels, pred_labels)),
            'f1': float(f1_score(true_labels, pred_labels)),
        }

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        evaluation_strategy='epoch',
        save_strategy='epoch',
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=True,
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
    metrics = trainer.evaluate(test_ds)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(args.output_dir / 'final')
    tokenizer.save_pretrained(args.output_dir / 'final')
    (args.output_dir / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')

if __name__ == '__main__':
    main()
