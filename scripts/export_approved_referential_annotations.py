#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from referential_learning.store import AnnotationStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Exporte les annotations validées vers les splits d entraînement')
    parser.add_argument('--input', type=Path, default=Path('data/annotation/referential_candidates.jsonl'))
    parser.add_argument('--output-dir', type=Path, default=Path('data/training'))
    parser.add_argument('--seed', type=int, default=42)
    return parser


def _approved(document: dict[str, Any]) -> bool:
    return str(document.get('status') or '').lower() in {'approved', 'validated', 'done'}


def _split_documents(documents: list[dict[str, Any]], seed: int) -> dict[str, list[dict[str, Any]]]:
    rng = random.Random(seed)
    items = list(documents)
    rng.shuffle(items)
    n = len(items)
    train_end = max(1, int(round(n * 0.7))) if n else 0
    val_end = max(train_end + (1 if n - train_end > 1 else 0), int(round(n * 0.85))) if n else 0
    return {
        'train': items[:train_end],
        'validation': items[train_end:val_end],
        'test': items[val_end:],
    }


def _section_examples(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for document in documents:
        for block in document.get('blocks', []):
            label = block.get('approved_section') or block.get('predicted_section')
            if not label:
                continue
            examples.append({
                'document_id': document['document_id'],
                'source_file': document.get('source_file', ''),
                'page': block.get('page', 0),
                'block_id': block.get('block_id', ''),
                'text': block.get('text', ''),
                'label': label,
                'bbox': block.get('bbox'),
                'confidence': block.get('confidence', 0.0),
                'status': block.get('status', 'pending'),
            })
    return examples


def _ner_examples(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for document in documents:
        for page in document.get('pages', []):
            entities = [entity for entity in page.get('entities', []) if entity.get('approved_label')]
            if not entities:
                continue
            examples.append({
                'document_id': document['document_id'],
                'source_file': document.get('source_file', ''),
                'page': page.get('number', 0),
                'text': page.get('text', ''),
                'entities': entities,
            })
    return examples


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(json.dumps(row, ensure_ascii=False) for row in rows) + ('\n' if rows else ''), encoding='utf-8')


def main() -> None:
    args = build_parser().parse_args()
    store = AnnotationStore(args.input)
    documents = [row for row in store.load() if _approved(row)]
    splits = _split_documents(documents, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    section_manifest = {split: [doc['document_id'] for doc in docs] for split, docs in splits.items()}
    ner_manifest = dict(section_manifest)

    _write_jsonl(args.output_dir / 'referential_sections_train.jsonl', _section_examples(splits['train']))
    _write_jsonl(args.output_dir / 'referential_sections_validation.jsonl', _section_examples(splits['validation']))
    _write_jsonl(args.output_dir / 'referential_sections_test.jsonl', _section_examples(splits['test']))
    _write_jsonl(args.output_dir / 'referential_ner_train.jsonl', _ner_examples(splits['train']))
    _write_jsonl(args.output_dir / 'referential_ner_validation.jsonl', _ner_examples(splits['validation']))
    _write_jsonl(args.output_dir / 'referential_ner_test.jsonl', _ner_examples(splits['test']))
    (args.output_dir / 'referential_manifest.json').write_text(json.dumps({'sections': section_manifest, 'ner': ner_manifest}, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
