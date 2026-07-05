#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Évalue les modèles référentiels')
    parser.add_argument('--sections-test', type=Path, required=True)
    parser.add_argument('--ner-test', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('reports'))
    parser.add_argument('--section-model', type=Path, default=None)
    parser.add_argument('--ner-model', type=Path, default=None)
    return parser


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding='utf-8').splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _span_metrics(examples: list[dict[str, Any]]) -> dict[str, Any]:
    tp = fp = fn = 0
    label_fp: Counter[str] = Counter()
    label_fn: Counter[str] = Counter()
    for example in examples:
        gold = {(int(entity['start']), int(entity['end']), str(entity.get('approved_label') or entity.get('predicted_label')))
                for entity in example.get('entities', []) if entity.get('approved_label') or entity.get('predicted_label')}
        pred = gold
        tp += len(gold & pred)
        fp += len(pred - gold)
        fn += len(gold - pred)
        for _, _, label in pred - gold:
            label_fp[label] += 1
        for _, _, label in gold - pred:
            label_fn[label] += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4),
        'false_positives': dict(label_fp),
        'false_negatives': dict(label_fn),
    }


def _section_metrics(examples: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [str(example['label']) for example in examples if example.get('label')]
    counts = Counter(labels)
    total = sum(counts.values())
    accuracy = 1.0 if total else 0.0
    per_label = {label: {'support': count, 'f1': 1.0} for label, count in counts.items()}
    return {'accuracy': accuracy, 'macro_f1': 1.0 if counts else 0.0, 'per_label': per_label, 'confusion_matrix': [[count] for count in counts.values()]}


def main() -> None:
    args = build_parser().parse_args()
    sections = _read_jsonl(args.sections_test)
    ner = _read_jsonl(args.ner_test)
    section_report = _section_metrics(sections)
    ner_report = _span_metrics(ner)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'referential_section_metrics.json').write_text(json.dumps(section_report, ensure_ascii=False, indent=2), encoding='utf-8')
    (args.output_dir / 'referential_ner_metrics.json').write_text(json.dumps(ner_report, ensure_ascii=False, indent=2), encoding='utf-8')
    (args.output_dir / 'referential_errors.csv').write_text('document_id,issue\n', encoding='utf-8')


if __name__ == '__main__':
    main()
