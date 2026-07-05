#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Entraîne un classifieur de sections référentielles')
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--validation', type=Path, required=True)
    parser.add_argument('--test', type=Path, required=True)
    parser.add_argument('--base-model', type=str, default='camembert-base')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--learning-rate', type=float, default=2e-5)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    return parser

def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding='utf-8').splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows

def main() -> None:
    args = build_parser().parse_args()
    try:
        from datasets import Dataset  # type: ignore
        from sklearn.metrics import accuracy_score, f1_score  # type: ignore
        from transformers import (  # type: ignore
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
        )
    except Exception as exc:
        raise SystemExit(f'Dépendances d entraînement manquantes: {exc}')

    train_rows = _read_jsonl(args.train)
    validation_rows = _read_jsonl(args.validation)
    test_rows = _read_jsonl(args.test)
    labels = sorted({str(row['label']) for row in (*train_rows, *validation_rows, *test_rows) if row.get('label')})
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(args.base_model, num_labels=len(labels), id2label=id2label, label2id=label2id)

    def encode(batch: dict[str, list[object]]) -> dict[str, object]:
        texts = [str(text) for text in batch['text']]
        tokens = tokenizer(texts, truncation=True, max_length=256)
        tokens['labels'] = [label2id[str(label)] for label in batch['label']]
        return tokens

    train_ds = Dataset.from_list(train_rows).map(encode, batched=True)
    val_ds = Dataset.from_list(validation_rows).map(encode, batched=True)
    test_ds = Dataset.from_list(test_rows).map(encode, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        return {
            'accuracy': float(accuracy_score(labels, preds)),
            'macro_f1': float(f1_score(labels, preds, average='macro')),
        }

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
        metric_for_best_model='macro_f1',
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
