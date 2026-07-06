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
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ALLOWED_LABELS = ('SKILL', 'METHOD', 'TOOL', 'DOMAIN')
BIO_LABELS = ['O', *[f'B-{label}' for label in ALLOWED_LABELS], *[f'I-{label}' for label in ALLOWED_LABELS]]
LABEL2ID = {label: idx for idx, label in enumerate(BIO_LABELS)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Entraîne un NER CamemBERT pour les compétences référentielles')
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
    if not path.exists():
        return rows
    for line in path.read_text(encoding='utf-8').splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _normalize_label(label: str | None) -> str | None:
    value = str(label or '').upper().strip()
    if value in ALLOWED_LABELS:
        return value
    if value in {'SOFT_SKILL', 'KNOWLEDGE', 'OTHER', 'PRICE', 'DURATION', 'REFERENCE', 'PROVIDER', 'CERTIFICATION'}:
        return None
    return value if value in ALLOWED_LABELS else None


def _gold_entities(example: dict[str, Any]) -> list[dict[str, Any]]:
    entities: list[dict[str, Any]] = []
    for entity in example.get('entities', []):
        label = _normalize_label(entity.get('approved_label') or entity.get('predicted_label'))
        if not label:
            continue
        start = int(entity.get('start', 0))
        end = int(entity.get('end', 0))
        if end <= start:
            continue
        entities.append({'start': start, 'end': end, 'label': label})
    return entities


def _sample_key(document_id: str, page: int, block_id: Any, text: str) -> str:
    stable_block = str(block_id or hashlib.sha1(str(text).encode('utf-8')).hexdigest()[:12])
    return f'{document_id}::{page}::{stable_block}'


def _tokenize_with_alignment(tokenizer, examples: list[dict[str, Any]], *, max_length: int, stride: int) -> list[dict[str, Any]]:
    features: list[dict[str, Any]] = []
    for example in examples:
        text = str(example.get('text') or '')
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            stride=stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding=False,
        )
        offsets_batch = tokenized.pop('offset_mapping')
        overflow_map = tokenized.pop('overflow_to_sample_mapping')
        for feature_idx, input_ids in enumerate(tokenized['input_ids']):
            offsets = offsets_batch[feature_idx]
            labels = [-100] * len(input_ids)
            for entity in _gold_entities(example):
                b_label = LABEL2ID[f"B-{entity['label']}"]
                i_label = LABEL2ID[f"I-{entity['label']}"]
                for token_idx, (start, end) in enumerate(offsets):
                    if start == end:
                        continue
                    if end <= entity['start'] or start >= entity['end']:
                        continue
                    labels[token_idx] = b_label if start == entity['start'] else i_label
            features.append({
                'input_ids': input_ids,
                'attention_mask': tokenized['attention_mask'][feature_idx],
                'labels': labels,
                'document_id': example.get('document_id', ''),
                'page': int(example.get('page', 0)),
                'section': example.get('section', 'OTHER'),
                'text': text,
                'sample_id': _sample_key(str(example.get('document_id', '')), int(example.get('page', 0)), example.get('block_id'), text),
                'offset_mapping': offsets,
            })
    return features


def _spans_from_label_ids(label_ids: list[int], offsets: list[tuple[int, int]]) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for label_id, (start, end) in zip(label_ids, offsets):
        if label_id == -100 or start == end:
            if current is not None:
                spans.append(current)
                current = None
            continue
        label = ID2LABEL.get(int(label_id), 'O')
        if label == 'O':
            if current is not None:
                spans.append(current)
                current = None
            continue
        prefix, base = label.split('-', 1)
        if current is None:
            current = {'start': start, 'end': end, 'label': base}
            continue
        if current['label'] == base and prefix == 'I':
            current['end'] = end
            continue
        spans.append(current)
        current = {'start': start, 'end': end, 'label': base}
    if current is not None:
        spans.append(current)
    return spans


def _gold_span_set(example: dict[str, Any]) -> set[tuple[int, int, str]]:
    return {(entity['start'], entity['end'], entity['label']) for entity in _gold_entities(example)}


def _predict_sample_spans(model, tokenizer, example: dict[str, Any], *, max_length: int, stride: int, device: str) -> set[tuple[int, int, str]]:
    import torch

    text = str(example.get('text') or '')
    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding=True,
        return_tensors='pt',
    )
    offsets_batch = tokenized.pop('offset_mapping')
    tokenized.pop('overflow_to_sample_mapping', None)
    tokenized = {key: value.to(device) for key, value in tokenized.items()}
    model = model.to(device)
    model.eval()
    span_set: set[tuple[int, int, str]] = set()
    with torch.no_grad():
        outputs = model(**tokenized)
        logits = outputs.logits.detach().cpu().numpy()
    for feature_idx, feature_logits in enumerate(logits):
        offsets = [tuple(map(int, pair)) for pair in offsets_batch[feature_idx].tolist()]
        label_ids = feature_logits.argmax(axis=-1).tolist()
        for span in _spans_from_label_ids(label_ids, offsets):
            span_set.add((int(span['start']), int(span['end']), str(span['label'])))
    return span_set


def _compute_span_metrics(examples: list[dict[str, Any]], predictions: dict[str, set[tuple[int, int, str]]]) -> dict[str, Any]:
    tp = fp = fn = 0
    per_type: dict[str, dict[str, int]] = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    false_positives: list[dict[str, Any]] = []
    false_negatives: list[dict[str, Any]] = []

    for example in examples:
        sample_id = example['sample_id']
        gold = _gold_span_set(example)
        pred = predictions.get(sample_id, set())
        tp_set = gold & pred
        fp_set = pred - gold
        fn_set = gold - pred
        tp += len(tp_set)
        fp += len(fp_set)
        fn += len(fn_set)
        for _, _, label in tp_set:
            per_type[label]['tp'] += 1
        for _, _, label in fp_set:
            per_type[label]['fp'] += 1
            false_positives.append({
                'document_id': example.get('document_id', ''),
                'page': example.get('page', 0),
                'section': example.get('section', 'OTHER'),
                'text': example.get('text', ''),
                'kind': 'fp',
                'label': label,
            })
        for _, _, label in fn_set:
            per_type[label]['fn'] += 1
            false_negatives.append({
                'document_id': example.get('document_id', ''),
                'page': example.get('page', 0),
                'section': example.get('section', 'OTHER'),
                'text': example.get('text', ''),
                'kind': 'fn',
                'label': label,
            })

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    exact_f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    per_type_metrics = {}
    for label, counts in per_type.items():
        label_precision = counts['tp'] / (counts['tp'] + counts['fp']) if counts['tp'] + counts['fp'] else 0.0
        label_recall = counts['tp'] / (counts['tp'] + counts['fn']) if counts['tp'] + counts['fn'] else 0.0
        label_f1 = (2 * label_precision * label_recall / (label_precision + label_recall)) if label_precision + label_recall else 0.0
        per_type_metrics[label] = {
            'precision': round(label_precision, 4),
            'recall': round(label_recall, 4),
            'f1': round(label_f1, 4),
            'support': counts['tp'] + counts['fn'],
            'false_positives': counts['fp'],
            'false_negatives': counts['fn'],
        }

    return {
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'exact_f1': round(exact_f1, 4),
        'f1_exact': round(exact_f1, 4),
        'per_type': per_type_metrics,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
    }


def main() -> None:
    args = build_parser().parse_args()
    try:
        from datasets import Dataset  # type: ignore
        from transformers import AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification, Trainer, TrainingArguments  # type: ignore
    except Exception as exc:
        raise SystemExit(f'Dépendances d entraînement NER manquantes: {exc}')

    train_rows = _read_jsonl(args.train)
    validation_rows = _read_jsonl(args.validation)
    test_rows = _read_jsonl(args.test)
    all_rows = train_rows + validation_rows + test_rows
    if not all_rows:
        raise SystemExit('Aucun exemple NER à entraîner.')

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(
        args.base_model,
        num_labels=len(BIO_LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    train_ds = Dataset.from_list(train_rows)
    val_ds = Dataset.from_list(validation_rows) if validation_rows else Dataset.from_list(train_rows[:1])
    test_ds = Dataset.from_list(test_rows) if test_rows else Dataset.from_list(train_rows[:1])

    def encode(batch: dict[str, list[Any]]) -> dict[str, Any]:
        texts = [str(text or '') for text in batch['text']]
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=args.max_length,
            stride=args.stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding=False,
        )
        sample_map = tokenized.pop('overflow_to_sample_mapping')
        offsets_batch = tokenized.pop('offset_mapping')
        labels: list[list[int]] = []
        document_ids: list[str] = []
        pages: list[int] = []
        sections: list[str] = []
        texts_out: list[str] = []
        sample_ids: list[str] = []
        for feature_idx, input_ids in enumerate(tokenized['input_ids']):
            sample_idx = int(sample_map[feature_idx])
            entities = _gold_entities({
                'entities': batch.get('entities', [])[sample_idx],
            })
            offsets = offsets_batch[feature_idx]
            label_ids = [-100] * len(input_ids)
            for entity in entities:
                b_id = LABEL2ID[f"B-{entity['label']}"]
                i_id = LABEL2ID[f"I-{entity['label']}"]
                for token_idx, (start, end) in enumerate(offsets):
                    if start == end:
                        continue
                    if end <= entity['start'] or start >= entity['end']:
                        continue
                    label_ids[token_idx] = b_id if start == entity['start'] else i_id
            labels.append(label_ids)
            document_id = str(batch.get('document_id', [''])[sample_idx])
            page = int(batch.get('page', [0])[sample_idx])
            section = str(batch.get('section', ['OTHER'])[sample_idx])
            text = str(batch.get('text', [''])[sample_idx])
            document_ids.append(document_id)
            pages.append(page)
            sections.append(section)
            texts_out.append(text)
            sample_ids.append(_sample_key(document_id, page, batch.get('block_id', [None])[sample_idx], text))
        tokenized['labels'] = labels
        tokenized['document_id'] = document_ids
        tokenized['page'] = pages
        tokenized['section'] = sections
        tokenized['text'] = texts_out
        tokenized['sample_id'] = sample_ids
        return tokenized

    def prepare_dataset(dataset: Dataset) -> Dataset:
        dataset = dataset.map(encode, batched=True, remove_columns=dataset.column_names)
        dataset = dataset.filter(lambda row: len(row['input_ids']) > 0)
        return dataset

    train_features = prepare_dataset(train_ds)
    val_features = prepare_dataset(val_ds)
    test_features = prepare_dataset(test_ds)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        valid = labels != -100
        correct = (preds == labels) & valid
        precision = float(correct.sum() / max(((preds != LABEL2ID['O']) & valid).sum(), 1))
        recall = float(correct.sum() / max(valid.sum(), 1))
        f1 = float(2 * precision * recall / max(precision + recall, 1e-12))
        return {'precision': precision, 'recall': recall, 'f1': f1}

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
        train_dataset=train_features,
        eval_dataset=val_features,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    eval_metrics = trainer.evaluate(test_features)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_dir = args.output_dir / 'final'
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    (final_dir / 'label_list.json').write_text(json.dumps(BIO_LABELS, ensure_ascii=False, indent=2), encoding='utf-8')
    (final_dir / 'label_classes.json').write_text(json.dumps(list(ALLOWED_LABELS), ensure_ascii=False, indent=2), encoding='utf-8')

    model = AutoModelForTokenClassification.from_pretrained(final_dir)
    tokenizer = AutoTokenizer.from_pretrained(final_dir, use_fast=True)

    gold_examples = [
        {
            'document_id': row.get('document_id', ''),
            'page': int(row.get('page', 0)),
            'section': row.get('section', 'OTHER'),
            'text': row.get('text', ''),
            'sample_id': f"{row.get('document_id', '')}::{int(row.get('page', 0))}::{idx}",
            'entities': row.get('entities', []),
        }
        for idx, row in enumerate(test_rows)
    ]

    predictions: dict[str, set[tuple[int, int, str]]] = defaultdict(set)
    for example in gold_examples:
        pred_spans = _predict_sample_spans(model, tokenizer, example, max_length=args.max_length, stride=args.stride, device=args.device)
        predictions[example['sample_id']].update(pred_spans)

    metrics = _compute_span_metrics(gold_examples, predictions)
    metrics['trainer'] = {key: float(value) if isinstance(value, (int, float, np.floating)) else value for key, value in eval_metrics.items() if key.startswith('eval_') or key in {'precision', 'recall', 'f1'}}
    (args.output_dir / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    (args.output_dir / 'ner_metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
