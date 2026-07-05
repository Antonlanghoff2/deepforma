#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from referential_learning.pdf_loader import load_pdf_document
from referential_learning.pipeline import build_annotation_document
from referential_learning.store import AnnotationStore

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Génère des candidats d annotation pour les référentiels')
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('data/annotation/referential_candidates.jsonl'))
    return parser

def main() -> None:
    args = build_parser().parse_args()
    store = AnnotationStore(args.output)
    records = []
    for pdf_path in sorted([path for path in args.input_dir.rglob('*') if path.suffix.lower() == '.pdf']):
        document = load_pdf_document(pdf_path)
        annotation = build_annotation_document(document).to_dict()
        annotation['status'] = 'pending'
        annotation['source_file'] = pdf_path.name
        annotation['document_id'] = document.document_id
        annotation['sha256'] = document.sha256
        annotation['needs_ocr'] = document.needs_ocr
        records.append(annotation)
    store.save(records)

if __name__ == '__main__':
    main()
