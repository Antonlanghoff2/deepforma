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
from referential_learning.ml_dl_taxonomy import infer_families, section_for_text
from referential_learning.pdf_loader import load_pdf_document
from referential_learning.section_labels import classify_section_label

ORDER = ['Machine Learning', 'Deep Learning', 'NLP', 'MLOps', 'Other']


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Construit les candidats multilabel référentiels ML/DL')
    parser.add_argument('--input-dir', type=Path, default=Path('data/raw/referentiel'))
    parser.add_argument('--output', type=Path, default=Path('data/annotation/referential_multilabel_candidates.jsonl'))
    return parser


def _section_label(text: str) -> str:
    match = classify_section_label(text)
    if match.label != 'OTHER':
        return match.label
    return section_for_text(text)


def _family_labels(text: str) -> list[str]:
    labels = infer_families(text)
    if not labels:
        return ['Other']
    ordered = [label for label in ORDER if label in labels]
    return ordered or ['Other']


def _page_records(document) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for page in document.pages:
        text = clean_text(page.text)
        if not text:
            continue
        rows.append({
            'record_id': stable_hash(document.document_id, page.number, text, length=24),
            'document_id': document.document_id,
            'source_file': document.source_file,
            'page': page.number,
            'section': _section_label(text),
            'block_id': page.blocks[0].block_id if page.blocks else None,
            'text': text,
            'predicted_labels': _family_labels(text),
            'approved_labels': None,
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
