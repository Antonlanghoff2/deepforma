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
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

CANDIDATE_FILES = {
    'ner': Path('data/annotation/referential_ner_candidates.jsonl'),
    'multilabel': Path('data/annotation/referential_multilabel_candidates.jsonl'),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Exporte les annotations référentielles approuvées vers les splits d entraînement')
    parser.add_argument('--ner-input', type=Path, default=CANDIDATE_FILES['ner'])
    parser.add_argument('--multilabel-input', type=Path, default=CANDIDATE_FILES['multilabel'])
    parser.add_argument('--output-dir', type=Path, default=Path('data/training'))
    parser.add_argument('--seed', type=int, default=42)
    return parser


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _approved_document(row: dict[str, Any]) -> bool:
    status = str(row.get('status') or '').lower()
    return status in {'approved', 'validated', 'done'}


def _document_ids(rows: list[dict[str, Any]]) -> list[str]:
    docs = []
    seen = set()
    for row in rows:
        doc_id = str(row.get('document_id') or '')
        if doc_id and doc_id not in seen:
            seen.add(doc_id)
            docs.append(doc_id)
    return docs


def _split_documents(document_ids: list[str], seed: int) -> dict[str, list[str]]:
    rng = random.Random(seed)
    ids = list(document_ids)
    rng.shuffle(ids)
    n = len(ids)
    train_end = max(1, int(round(n * 0.70))) if n else 0
    validation_end = max(train_end, int(round(n * 0.85))) if n else 0
    validation_end = min(validation_end, n)
    return {
        'train': ids[:train_end],
        'validation': ids[train_end:validation_end],
        'test': ids[validation_end:],
    }


def _split_lookup(splits: dict[str, list[str]]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for split, document_ids in splits.items():
        for document_id in document_ids:
            lookup[document_id] = split
    return lookup


def _approved_ner_examples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for row in rows:
        if not _approved_document(row):
            continue
        entities = [entity for entity in row.get('entities', []) if entity.get('approved_label')]
        if not entities:
            continue
        examples.append({
            'document_id': row['document_id'],
            'source_file': row.get('source_file', ''),
            'page': row.get('page', 0),
            'section': row.get('section', 'OTHER'),
            'block_id': row.get('block_id'),
            'text': row.get('text', ''),
            'entities': entities,
            'status': 'approved',
        })
    return examples


def _approved_multilabel_examples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for row in rows:
        if not _approved_document(row):
            continue
        labels = row.get('approved_labels') or row.get('predicted_labels') or []
        if not labels:
            continue
        examples.append({
            'document_id': row['document_id'],
            'source_file': row.get('source_file', ''),
            'page': row.get('page', 0),
            'section': row.get('section', 'OTHER'),
            'block_id': row.get('block_id'),
            'text': row.get('text', ''),
            'predicted_labels': row.get('predicted_labels', []),
            'approved_labels': labels,
            'status': 'approved',
        })
    return examples


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(json.dumps(row, ensure_ascii=False) for row in rows) + ('\n' if rows else ''), encoding='utf-8')


def _split_rows(rows: list[dict[str, Any]], split_lookup: dict[str, str]) -> dict[str, list[dict[str, Any]]]:
    output = {'train': [], 'validation': [], 'test': []}
    for row in rows:
        split = split_lookup.get(str(row.get('document_id') or ''))
        if split:
            output[split].append(row)
    return output


def main() -> None:
    args = build_parser().parse_args()
    ner_rows = _read_jsonl(args.ner_input)
    multilabel_rows = _read_jsonl(args.multilabel_input)
    document_ids = list(dict.fromkeys(_document_ids(ner_rows) + _document_ids(multilabel_rows)))
    splits = _split_documents(document_ids, args.seed)
    split_lookup = _split_lookup(splits)

    ner_examples = _approved_ner_examples(ner_rows)
    multilabel_examples = _approved_multilabel_examples(multilabel_rows)
    ner_split = _split_rows(ner_examples, split_lookup)
    multilabel_split = _split_rows(multilabel_examples, split_lookup)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.output_dir / 'referential_ner_train.jsonl', ner_split['train'])
    _write_jsonl(args.output_dir / 'referential_ner_validation.jsonl', ner_split['validation'])
    _write_jsonl(args.output_dir / 'referential_ner_test.jsonl', ner_split['test'])
    _write_jsonl(args.output_dir / 'referential_multilabel_train.jsonl', multilabel_split['train'])
    _write_jsonl(args.output_dir / 'referential_multilabel_validation.jsonl', multilabel_split['validation'])
    _write_jsonl(args.output_dir / 'referential_multilabel_test.jsonl', multilabel_split['test'])

    manifest = {
        'seed': args.seed,
        'documents': splits,
        'counts': {
            'ner': {split: len(rows) for split, rows in ner_split.items()},
            'multilabel': {split: len(rows) for split, rows in multilabel_split.items()},
        },
    }
    (args.output_dir / 'referential_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
