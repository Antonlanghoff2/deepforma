#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from common.text import clean_text
from referential_learning.pdf_loader import load_pdf_document
from referential_learning.section_labels import classify_section_label

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Audit du corpus de PDFs de référentiel')
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-report', type=Path, required=True)
    return parser

def _detect_title(document) -> str:
    for page in document.pages[:2]:
        for block in page.blocks[:5]:
            text = clean_text(block.text)
            if len(text) >= 6 and not text.lower().startswith(('référentiel', 'referentiel', 'bloc ', 'activité', 'activite')):
                return text[:120]
    return ''

def _section_candidates(document) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for page in document.pages:
        for block in page.blocks:
            match = classify_section_label(block.text)
            if match.label != 'OTHER' or match.confidence >= 0.7:
                candidates.append({
                    'page': page.number,
                    'text': block.text[:200],
                    'label': match.label,
                    'confidence': match.confidence,
                })
    return candidates[:50]

def main() -> None:
    args = build_parser().parse_args()
    records: list[dict[str, object]] = []
    pdfs = sorted([path for path in args.input_dir.rglob('*') if path.suffix.lower() == '.pdf'])
    for path in pdfs:
        try:
            document = load_pdf_document(path)
            page_texts = [page.text for page in document.pages]
            total_chars = sum(len(text) for text in page_texts)
            densities = [page.density for page in document.pages if page.density]
            record = {
                'source_file': path.name,
                'path': str(path),
                'document_id': document.document_id,
                'sha256': document.sha256,
                'file_size': document.file_size,
                'page_count': document.page_count,
                'text_layer_present': any(page.has_text_layer for page in document.pages),
                'text_characters': total_chars,
                'mean_density': round(mean(densities), 8) if densities else 0.0,
                'needs_ocr': document.needs_ocr or not any(page.has_text_layer for page in document.pages),
                'title_detected': _detect_title(document),
                'section_candidates': _section_candidates(document),
                'page_summaries': [
                    {
                        'page': page.number,
                        'text_length': page.text_length,
                        'density': round(page.density, 8),
                        'block_count': len(page.blocks),
                    }
                    for page in document.pages
                ],
                'warnings': document.warnings,
                'errors': [],
            }
        except Exception as exc:
            record = {
                'source_file': path.name,
                'path': str(path),
                'page_count': 0,
                'text_layer_present': False,
                'text_characters': 0,
                'mean_density': 0.0,
                'needs_ocr': True,
                'title_detected': '',
                'section_candidates': [],
                'page_summaries': [],
                'warnings': [],
                'errors': [str(exc)],
            }
        records.append(record)

    summary = {
        'input_dir': str(args.input_dir),
        'pdf_count': len(pdfs),
        'text_layer_count': sum(1 for record in records if record.get('text_layer_present')),
        'needs_ocr_count': sum(1 for record in records if record.get('needs_ocr')),
        'average_pages': round(sum(int(record.get('page_count', 0)) for record in records) / max(len(records), 1), 2),
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps({'summary': summary, 'documents': records}, ensure_ascii=False, indent=2), encoding='utf-8')

if __name__ == '__main__':
    main()
