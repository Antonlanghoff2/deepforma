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
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from scripts.train_referential_skill_ner import ALLOWED_LABELS as NER_ALLOWED_LABELS
from scripts.train_referential_skill_ner import _compute_span_metrics, _predict_sample_spans, _read_jsonl as read_ner_jsonl, _sample_key
from scripts.train_referential_multilabel import LABELS as MULTILABEL_LABELS
from scripts.train_referential_multilabel import _apply_thresholds, _metrics_from_predictions, _predict_scores, _read_jsonl as read_multilabel_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Évalue les modèles référentiels NER et multilabel')
    parser.add_argument('--ner-test', type=Path, required=True)
    parser.add_argument('--multilabel-test', type=Path, required=True)
    parser.add_argument('--ner-model', type=Path, default=Path('models/referential-skill-ner/final'))
    parser.add_argument('--multilabel-model', type=Path, default=Path('models/referential-multilabel/final'))
    parser.add_argument('--output-dir', type=Path, default=Path('reports'))
    parser.add_argument('--max-length', type=int, default=256)
    parser.add_argument('--stride', type=int, default=64)
    parser.add_argument('--device', type=str, default='cpu')
    return parser


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding='utf-8').splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


def _write_errors_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('document_id,page,section,kind,label,text,details\n', encoding='utf-8')
        return
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['document_id', 'page', 'section', 'kind', 'label', 'text', 'details'])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _gold_ner_examples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        entities = []
        for entity in row.get('entities', []):
            label = str(entity.get('approved_label') or entity.get('predicted_label') or '').upper()
            if label not in NER_ALLOWED_LABELS:
                continue
            entities.append({
                'start': int(entity.get('start', 0)),
                'end': int(entity.get('end', 0)),
                'label': label,
            })
        examples.append({
            'document_id': row.get('document_id', ''),
            'page': int(row.get('page', 0)),
            'section': row.get('section', 'OTHER'),
            'text': row.get('text', ''),
            'sample_id': _sample_key(str(row.get('document_id', '')), int(row.get('page', 0)), row.get('block_id'), str(row.get('text', ''))),
            'entities': entities,
        })
    return examples


def _predict_ner_metrics(rows: list[dict[str, Any]], model_dir: Path, *, max_length: int, stride: int, device: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not model_dir.exists():
        return {'status': 'unavailable', 'reason': f'Model NER introuvable: {model_dir}'}, []
    try:
        from transformers import AutoModelForTokenClassification, AutoTokenizer  # type: ignore
    except Exception as exc:
        return {'status': 'unavailable', 'reason': str(exc)}, []

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    gold_examples = _gold_ner_examples(rows)
    predictions: dict[str, set[tuple[int, int, str]]] = defaultdict(set)
    errors: list[dict[str, Any]] = []
    for example in gold_examples:
        pred = _predict_sample_spans(model, tokenizer, example, max_length=max_length, stride=stride, device=device)
        predictions[example['sample_id']].update(pred)
    metrics = _compute_span_metrics(gold_examples, predictions)
    for example in gold_examples:
        sample_pred = predictions.get(example['sample_id'], set())
        gold = {(item['start'], item['end'], item['label']) for item in example['entities']}
        for start, end, label in sample_pred - gold:
            errors.append({
                'document_id': example['document_id'],
                'page': example['page'],
                'section': example['section'],
                'kind': 'fp',
                'label': label,
                'text': example['text'][start:end],
                'details': json.dumps({'start': start, 'end': end}, ensure_ascii=False),
            })
        for start, end, label in gold - sample_pred:
            errors.append({
                'document_id': example['document_id'],
                'page': example['page'],
                'section': example['section'],
                'kind': 'fn',
                'label': label,
                'text': example['text'][start:end],
                'details': json.dumps({'start': start, 'end': end}, ensure_ascii=False),
            })
    metrics['status'] = 'ok'
    return metrics, errors


def _cooccurrence_matrix(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    matrix: dict[str, dict[str, int]] = {label: {other: 0 for other in MULTILABEL_LABELS} for label in MULTILABEL_LABELS}
    for row in rows:
        labels = [str(label) for label in (row.get('approved_labels') or row.get('predicted_labels') or []) if str(label) in MULTILABEL_LABELS]
        unique = list(dict.fromkeys(labels))
        for label in unique:
            for other in unique:
                matrix[label][other] += 1
    return matrix


def _threshold_curve(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, list[dict[str, float]]]:
    curves: dict[str, list[dict[str, float]]] = {}
    grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
    for index, label in enumerate(MULTILABEL_LABELS):
        points = []
        gold = y_true[:, index]
        scores = y_score[:, index]
        for threshold in grid:
            pred = (scores >= threshold).astype(int)
            metrics = _metrics_from_predictions(y_true, pred)
            points.append({'threshold': float(threshold), 'precision': metrics['per_label'][label]['precision'], 'recall': metrics['per_label'][label]['recall'], 'f1': metrics['per_label'][label]['f1']})
        curves[label] = points
    return curves


def _predict_multilabel_metrics(rows: list[dict[str, Any]], model_dir: Path, *, max_length: int, device: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not model_dir.exists():
        return {'status': 'unavailable', 'reason': f'Model multilabel introuvable: {model_dir}'}, []
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore
    except Exception as exc:
        return {'status': 'unavailable', 'reason': str(exc)}, []

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    texts = [str(row.get('text', '')) for row in rows]
    scores = _predict_scores(model, tokenizer, texts, max_length=max_length, batch_size=8, device=device)
    y_true = []
    errors: list[dict[str, Any]] = []
    for row in rows:
        labels = [str(label) for label in (row.get('approved_labels') or row.get('predicted_labels') or []) if str(label) in MULTILABEL_LABELS]
        y_true.append([1 if label in labels else 0 for label in MULTILABEL_LABELS])
    y_true_array = np.array(y_true)
    thresholds_path = model_dir / 'thresholds.json'
    if thresholds_path.exists():
        thresholds = json.loads(thresholds_path.read_text(encoding='utf-8'))
    else:
        thresholds = {label: 0.5 for label in MULTILABEL_LABELS}
    y_pred = _apply_thresholds(scores, thresholds)
    metrics = _metrics_from_predictions(y_true_array, y_pred)
    metrics['thresholds'] = thresholds
    metrics['cooccurrence_matrix'] = _cooccurrence_matrix(rows)
    metrics['threshold_curve'] = _threshold_curve(y_true_array, scores)
    for row, truth, pred in zip(rows, y_true_array, y_pred):
        gold_labels = {label for label, flag in zip(MULTILABEL_LABELS, truth) if flag}
        pred_labels = {label for label, flag in zip(MULTILABEL_LABELS, pred) if flag}
        for label in pred_labels - gold_labels:
            errors.append({
                'document_id': row.get('document_id', ''),
                'page': row.get('page', 0),
                'section': row.get('section', 'OTHER'),
                'kind': 'fp_multilabel',
                'label': label,
                'text': row.get('text', ''),
                'details': json.dumps({}, ensure_ascii=False),
            })
        for label in gold_labels - pred_labels:
            errors.append({
                'document_id': row.get('document_id', ''),
                'page': row.get('page', 0),
                'section': row.get('section', 'OTHER'),
                'kind': 'fn_multilabel',
                'label': label,
                'text': row.get('text', ''),
                'details': json.dumps({}, ensure_ascii=False),
            })
    metrics['status'] = 'ok'
    return metrics, errors


def main() -> None:
    args = build_parser().parse_args()
    ner_rows = read_ner_jsonl(args.ner_test)
    multilabel_rows = read_multilabel_jsonl(args.multilabel_test)

    ner_metrics, ner_errors = _predict_ner_metrics(ner_rows, args.ner_model, max_length=args.max_length, stride=args.stride, device=args.device)
    multilabel_metrics, multilabel_errors = _predict_multilabel_metrics(multilabel_rows, args.multilabel_model, max_length=args.max_length, device=args.device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_dir / 'referential_ner_metrics.json', ner_metrics)
    _write_json(args.output_dir / 'referential_multilabel_metrics.json', multilabel_metrics)
    _write_errors_csv(args.output_dir / 'referential_model_errors.csv', [*ner_errors, *multilabel_errors])


if __name__ == '__main__':
    main()
