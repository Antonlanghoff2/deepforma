#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / 'src'):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from common.text import clean_text, normalize_for_match, stable_hash


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Construit le dataset d entraînement pour l extraction de compétences.')
    parser.add_argument('--input', type=Path, default=Path('data/training/skill_extraction/offers_enriched.jsonl'))
    parser.add_argument('--output-dir', type=Path, default=Path('data/training/skill_extraction'))
    parser.add_argument('--include-rome-context', action='store_true')
    return parser


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.is_dir():
        rows: list[dict[str, Any]] = []
        for file in sorted(path.rglob('*.jsonl')):
            rows.extend(_read_jsonl(file))
        return rows
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = '\n'.join(json.dumps(row, ensure_ascii=False) for row in rows)
    path.write_text(payload + ('\n' if rows else ''), encoding='utf-8')


def _evidence_span(text: str, evidence: str) -> tuple[int, int]:
    if not evidence:
        return 0, 0
    idx = normalize_for_match(text).find(normalize_for_match(evidence))
    if idx >= 0:
        return idx, idx + len(normalize_for_match(evidence))
    pos = text.lower().find(evidence.lower())
    if pos >= 0:
        return pos, pos + len(evidence)
    return 0, min(len(text), max(1, len(evidence)))


def _group_id(row: dict[str, Any]) -> str:
    rncp_codes = sorted({str(item.get('rncp_code')) for item in row.get('rncp_candidates', []) if item.get('rncp_code')})
    rome = row.get('rome', {}) if isinstance(row.get('rome'), dict) else {}
    return stable_hash(row.get('offer_id'), row.get('title'), rome.get('code') if isinstance(rome, dict) else '', '|'.join(rncp_codes), length=24)


def main() -> None:
    args = build_parser().parse_args()
    rows = _read_jsonl(args.input)
    examples: list[dict[str, Any]] = []
    split_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    label_counts: Counter[str] = Counter()
    for row in rows:
        title = clean_text(row.get('title') or '')
        description = clean_text(row.get('description') or '')
        rome = row.get('rome', {}) if isinstance(row.get('rome'), dict) else {}
        rome_label = clean_text(rome.get('label') or '')
        text_parts = [title]
        if args.include_rome_context and rome_label:
            text_parts.append(rome_label)
        if description:
            text_parts.append(description)
        text = '\n'.join(part for part in text_parts if part)
        labels = []
        for skill in row.get('skills', []) or []:
            canonical_id = clean_text(skill.get('canonical_skill_id') or '')
            canonical_label = clean_text(skill.get('canonical_label') or '')
            evidence = clean_text(skill.get('evidence') or '')
            if not canonical_id or not canonical_label or not evidence:
                continue
            start, end = _evidence_span(text, evidence)
            labels.append({
                'canonical_skill_id': canonical_id,
                'canonical_label': canonical_label,
                'evidence_start': start,
                'evidence_end': end,
                'evidence_text': evidence,
                'source_links': skill.get('source_links', []),
            })
            label_counts[canonical_label] += 1
        example = {
            'id': stable_hash(row.get('offer_id'), text, length=24),
            'text': text,
            'title': title,
            'rome_code': rome.get('code') if isinstance(rome, dict) else row.get('rome_code'),
            'rome_label': rome_label,
            'labels': labels,
        }
        bucket = int(stable_hash(_group_id(row), length=8), 16) % 100
        split = 'train' if bucket < 70 else 'validation' if bucket < 85 else 'test'
        split_buckets[split].append(example)
        examples.append(example)
    report = {
        'total_examples': len(examples),
        'train': len(split_buckets['train']),
        'validation': len(split_buckets['validation']),
        'test': len(split_buckets['test']),
        'label_counts': label_counts.most_common(25),
        'rome_codes': sorted({str((row.get('rome', {}) or {}).get('code')) for row in rows if isinstance(row.get('rome'), dict) and (row.get('rome', {}) or {}).get('code')}),
        'negative_examples': sum(1 for row in examples if not row['labels']),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    out = args.output_dir
    _write_jsonl(out / 'train.jsonl', split_buckets['train'])
    _write_jsonl(out / 'validation.jsonl', split_buckets['validation'])
    _write_jsonl(out / 'test.jsonl', split_buckets['test'])
    (out / 'dataset_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding='utf-8')


if __name__ == '__main__':
    main()
