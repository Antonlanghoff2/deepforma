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

from common.text import clean_text, stable_hash
from referential_learning.pdf_loader import load_pdf_document
from referential_learning.section_labels import classify_section_label
from referential_learning.ml_dl_taxonomy import find_mentions, section_for_text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Construit les candidats NER référentiels ML/DL')
    parser.add_argument('--input-dir', type=Path, default=Path('data/raw/referentiel'))
    parser.add_argument('--output', type=Path, default=Path('data/annotation/referential_ner_candidates.jsonl'))
    return parser


def _section_label(text: str) -> str:
    match = classify_section_label(text)
    if match.label != 'OTHER':
        return match.label
    return section_for_text(text)


def _build_entity(entity: dict[str, Any], *, source_file: str, document_id: str, page: int) -> dict[str, Any]:
    start = int(entity['start'])
    end = int(entity['end'])
    label = entity['entity_type']
    return {
        'entity_id': f'{document_id}:{page}:{start}:{end}:{label}',
        'start': start,
        'end': end,
        'text': clean_text(entity['text']),
        'predicted_label': label,
        'approved_label': None,
        'canonical_name': entity['canonical_name'],
        'source_file': source_file,
        'document_id': document_id,
        'page': page,
        'status': 'pending',
    }


def _page_records(document) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for page in document.pages:
        text = clean_text(page.text)
        if not text:
            continue
        entities = [_build_entity(entity, source_file=document.source_file, document_id=document.document_id, page=page.number) for entity in find_mentions(text)]
        rows.append({
            'record_id': stable_hash(document.document_id, page.number, text, length=24),
            'document_id': document.document_id,
            'source_file': document.source_file,
            'page': page.number,
            'section': _section_label(text),
            'block_id': page.blocks[0].block_id if page.blocks else None,
            'text': text,
            'entities': entities,
            'status': 'pending',
        })
    return rows


def main() -> None:
    args = build_parser().parse_args()
    rows: list[dict[str, Any]] = []
    for pdf_path in sorted([path for path in args.input_dir.rglob('*') if path.suffix.lower() == '.pdf']):
        document = load_pdf_document(pdf_path)
        rows.extend(_page_records(document))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text('\n'.join(json.dumps(row, ensure_ascii=False) for row in rows) + ('\n' if rows else ''), encoding='utf-8')


if __name__ == '__main__':
    main()
