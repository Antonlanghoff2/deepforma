#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from transformers import AutoModelForTokenClassification, AutoTokenizer

from common.text import clean_text, normalize_for_match

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('compare_model_versions')

LABEL2ID = {'O': 0, 'B-SKILL': 1, 'I-SKILL': 2}


@dataclass(frozen=True)
class Example:
    id: str
    text: str
    entities: list[dict[str, Any]]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class PredEntity:
    start: int
    end: int
    label: str
    surface_form: str
    canonical_name: str
    confidence: float


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_examples(path: Path) -> list[Example]:
    return [
        Example(
            id=str(record.get('id')),
            text=str(record.get('text', '')),
            entities=[e for e in record.get('entities', []) if isinstance(e, dict)],
            metadata=record.get('metadata', {}) or {},
        )
        for record in load_jsonl(path)
    ]


def load_skill_inventory(base_dataset: Path) -> set[str]:
    inventory = set()
    for example in load_examples(base_dataset):
        for entity in example.entities:
            name = normalize_for_match(entity.get('canonical_name') or entity.get('surface_form') or '')
            if name:
                inventory.add(name)
    return inventory


def load_model(model_dir: Path, device: str | None = None):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    if device:
        model.to(device)
    model.eval()
    return tokenizer, model


def predict_entities(text: str, tokenizer: Any, model: Any, device: str | None = None) -> list[PredEntity]:
    encoded = tokenizer(text, return_tensors='pt', truncation=True, max_length=256, return_offsets_mapping=True)
    offsets = encoded.pop('offset_mapping')[0].tolist()
    if device:
        encoded = {k: v.to(device) for k, v in encoded.items()}
    with np.errstate(over='ignore'):
        import torch
        with torch.no_grad():
            logits = model(**encoded).logits[0]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            pred_ids = probs.argmax(axis=-1)
    entities: list[PredEntity] = []
    current: dict[str, Any] | None = None
    for idx, pred_id in enumerate(pred_ids):
        label = model.config.id2label.get(str(int(pred_id)), model.config.id2label.get(int(pred_id), 'O'))
        start, end = offsets[idx]
        if end <= start:
            continue
        score = float(probs[idx][int(pred_id)])
        if label == 'B-SKILL':
            if current:
                entities.append(PredEntity(**current))
            current = {'start': start, 'end': end, 'label': 'SKILL', 'surface_form': text[start:end], 'canonical_name': text[start:end], 'confidence': score}
        elif label == 'I-SKILL' and current and start <= current['end']:
            current['end'] = end
            current['surface_form'] = text[current['start']:end]
            current['canonical_name'] = current['surface_form']
            current['confidence'] = min(current['confidence'], score)
        else:
            if current:
                entities.append(PredEntity(**current))
                current = None
    if current:
        entities.append(PredEntity(**current))
    return entities


def gold_entities(example: Example) -> list[dict[str, Any]]:
    return [e for e in example.entities if e.get('start') is not None and e.get('end') is not None]


def exact_key(entity: dict[str, Any]) -> tuple[Any, ...]:
    return (int(entity['start']), int(entity['end']), clean_text(entity.get('label') or 'SKILL'))


def normalized_key(entity: dict[str, Any]) -> tuple[Any, ...]:
    return (normalize_for_match(entity.get('canonical_name') or entity.get('surface_form') or ''), clean_text(entity.get('label') or 'SKILL'))


def pred_exact_key(entity: PredEntity) -> tuple[Any, ...]:
    return (entity.start, entity.end, entity.label)


def pred_normalized_key(entity: PredEntity) -> tuple[Any, ...]:
    return (normalize_for_match(entity.canonical_name or entity.surface_form), entity.label)


def entity_metrics(gold: list[Any], pred: list[Any], normalized: bool = False) -> dict[str, Any]:
    gold_keys = {normalized_key(e) if normalized else exact_key(e) for e in gold}
    pred_keys = {pred_normalized_key(e) if normalized else pred_exact_key(e) for e in pred}
    tp = len(gold_keys & pred_keys)
    fp = len(pred_keys - gold_keys)
    fn = len(gold_keys - pred_keys)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'false_positives': fp,
        'false_negatives': fn,
        'true_positives': tp,
    }


def group_metrics(rows: list[dict[str, Any]], key_name: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row.get(key_name) or 'unknown'].append(row)
    output = []
    for key, items in grouped.items():
        gold = [item['gold'] for item in items]
        pred = [item['pred'] for item in items]
        exact = entity_metrics(gold, pred, normalized=False)
        normalized = entity_metrics(gold, pred, normalized=True)
        output.append({
            key_name: key,
            'samples': len(items),
            'exact_f1': exact['f1'],
            'normalized_f1': normalized['f1'],
            'precision': exact['precision'],
            'recall': exact['recall'],
            'false_positives': exact['false_positives'],
            'false_negatives': exact['false_negatives'],
        })
    output.sort(key=lambda row: (-row['samples'], row[key_name]))
    return output


def evaluate_model(model_dir: Path, examples: list[Example], base_skill_inventory: set[str], device: str | None = None) -> dict[str, Any]:
    tokenizer, model = load_model(model_dir, device=device)
    rows = []
    pred_entity_counts = []
    justification_issues = 0
    total_pred_entities = 0
    for example in examples:
        preds = predict_entities(example.text, tokenizer, model, device=device)
        gold = gold_entities(example)
        rows.append({
            'gold': gold,
            'pred': preds,
            'skill': normalize_for_match(gold[0].get('canonical_name') or gold[0].get('surface_form') or '') if gold else 'unknown',
            'family': clean_text(example.metadata.get('job_family')) or 'unknown',
            'territory': clean_text(example.metadata.get('territory')) or 'unknown',
            'is_new_skill': bool(gold and normalize_for_match(gold[0].get('canonical_name') or gold[0].get('surface_form') or '') not in base_skill_inventory),
        })
        total_pred_entities += len(preds)
        for pred in preds:
            normalized_surface = normalize_for_match(pred.surface_form)
            if normalized_surface not in normalize_for_match(example.text):
                justification_issues += 1
        pred_entity_counts.append(len(preds))

    gold_entities_all = [entity for example in examples for entity in gold_entities(example)]
    pred_entities_all = [pred for example in examples for pred in predict_entities(example.text, tokenizer, model, device=device)]

    exact = entity_metrics(gold_entities_all, pred_entities_all, normalized=False)
    normalized = entity_metrics(gold_entities_all, pred_entities_all, normalized=True)

    fp_rate = exact['false_positives'] / max(len(pred_entities_all), 1)
    fn_rate = exact['false_negatives'] / max(len(gold_entities_all), 1)
    justification_rate = justification_issues / max(total_pred_entities, 1)

    skill_rows = []
    for row in rows:
        gold = row['gold']
        pred = row['pred']
        skill_name = row['skill']
        skill_rows.append({**row, 'skill': skill_name})

    by_skill = group_metrics(skill_rows, 'skill')
    by_family = group_metrics(skill_rows, 'family')
    by_territory = group_metrics(skill_rows, 'territory')

    return {
        'exact': exact,
        'normalized': normalized,
        'false_positive_rate': fp_rate,
        'false_negative_rate': fn_rate,
        'no_textual_justification_rate': justification_rate,
        'by_skill': by_skill,
        'by_family': by_family,
        'by_territory': by_territory,
        'rows': skill_rows,
    }


def compare_models(candidate: dict[str, Any], production: dict[str, Any]) -> dict[str, Any]:
    promotion = {
        'eligible': False,
        'reason': '',
    }
    exact_f1_delta = candidate['exact']['f1'] - production['exact']['f1']
    precision_delta = candidate['exact']['precision'] - production['exact']['precision']
    if exact_f1_delta < 0.01:
        promotion['reason'] = 'F1 exact insufficient'
    elif precision_delta < -0.02:
        promotion['reason'] = 'Precision regression too large'
    elif candidate['false_positive_rate'] > production['false_positive_rate'] + 0.01:
        promotion['reason'] = 'Invented skills rate increased'
    else:
        promotion['eligible'] = True
        promotion['reason'] = 'Promotion criteria satisfied'
    return {
        'candidate': candidate,
        'production': production,
        'delta': {
            'exact_f1': exact_f1_delta,
            'precision': precision_delta,
            'recall': candidate['exact']['recall'] - production['exact']['recall'],
            'normalized_f1': candidate['normalized']['f1'] - production['normalized']['f1'],
            'false_positive_rate': candidate['false_positive_rate'] - production['false_positive_rate'],
            'false_negative_rate': candidate['false_negative_rate'] - production['false_negative_rate'],
            'no_textual_justification_rate': candidate['no_textual_justification_rate'] - production['no_textual_justification_rate'],
        },
        'promotion': promotion,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Compare model versions for continual learning')
    parser.add_argument('--candidate-model', type=Path, required=True)
    parser.add_argument('--production-model', type=Path, required=True)
    parser.add_argument('--test-dataset', type=Path, required=True)
    parser.add_argument('--base-dataset', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('reports'))
    parser.add_argument('--device', default=None)
    parser.add_argument('--critical-regression', type=float, default=0.05)
    parser.add_argument('--min-f1-improvement', type=float, default=0.01)
    parser.add_argument('--max-precision-drop', type=float, default=0.02)
    parser.add_argument('--max-invented-skill-rate-increase', type=float, default=0.01)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    examples = load_examples(args.test_dataset)
    base_skill_inventory = load_skill_inventory(args.base_dataset)

    candidate = evaluate_model(args.candidate_model, examples, base_skill_inventory, device=args.device)
    production = evaluate_model(args.production_model, examples, base_skill_inventory, device=args.device)

    comparison = compare_models(candidate, production)
    comparison['base_skill_inventory_size'] = len(base_skill_inventory)
    comparison['new_skill_coverage'] = {
        'candidate': sum(1 for row in candidate['rows'] if row.get('is_new_skill')),
        'production': sum(1 for row in production['rows'] if row.get('is_new_skill')),
    }
    comparison['promotion']['eligible'] = bool(
        comparison['promotion']['eligible']
        and comparison['delta']['exact_f1'] >= args.min_f1_improvement
        and comparison['delta']['precision'] >= -args.max_precision_drop
        and comparison['delta']['false_positive_rate'] <= args.max_invented_skill_rate_increase
        and comparison['delta']['exact_f1'] >= -args.critical_regression
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'model_comparison.json').write_text(json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True), encoding='utf-8')

    rows = []
    for section, entries in [('skill', comparison['candidate']['by_skill']), ('family', comparison['candidate']['by_family']), ('territory', comparison['candidate']['by_territory'])]:
        for entry in entries:
            row = {'section': section, **entry}
            rows.append(row)
    with (output_dir / 'model_comparison_errors.csv').open('w', encoding='utf-8', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=sorted(rows[0].keys()) if rows else ['section'])
        writer.writeheader()
        writer.writerows(rows)

    md = []
    md.append('# Model Comparison')
    md.append(f"- Candidate exact F1: {comparison['candidate']['exact']['f1']:.4f}")
    md.append(f"- Production exact F1: {comparison['production']['exact']['f1']:.4f}")
    md.append(f"- Delta exact F1: {comparison['delta']['exact_f1']:.4f}")
    md.append(f"- Candidate precision: {comparison['candidate']['exact']['precision']:.4f}")
    md.append(f"- Production precision: {comparison['production']['exact']['precision']:.4f}")
    md.append(f"- Delta precision: {comparison['delta']['precision']:.4f}")
    md.append(f"- Candidate no textual justification rate: {comparison['candidate']['no_textual_justification_rate']:.4f}")
    md.append(f"- Production no textual justification rate: {comparison['production']['no_textual_justification_rate']:.4f}")
    md.append(f"- Promotion eligible: {'yes' if comparison['promotion']['eligible'] else 'no'}")
    md.append(f"- Reason: {comparison['promotion']['reason']}")
    (output_dir / 'model_card.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    logger.info('Comparison complete: %s', comparison['promotion'])


if __name__ == '__main__':
    main()
