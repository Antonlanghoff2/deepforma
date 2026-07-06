#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from common.text import clean_text, normalize_for_match
from referential_import.import_service import ReferentialImportService
from referential_import.store import ReferentialImportStore
from referential_learning.section_labels import classify_section_label


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Diagnostic détaillé du pipeline référentiel')
    parser.add_argument('--file', type=Path, required=True)
    parser.add_argument('--store-path', type=Path, default=Path('data/referentials/referential_imports.sqlite3'))
    parser.add_argument('--approve', action='store_true', help='Persiste l import après le diagnostic')
    parser.add_argument('--validated-by', type=str, default='debug')
    return parser


def _extract_section_lines(page_text: str) -> dict[str, list[str]]:
    headings = {
        'PUBLIC': ('public', 'prérequis', 'pre requis', 'public cible'),
        'OBJECTIVES': ('objectifs',),
        'PROGRAM': ('programme', 'contenu'),
        'SKILLS': ('compétences', 'competences'),
    }
    current = 'OTHER'
    collected: dict[str, list[str]] = {key: [] for key in headings}
    collected['OTHER'] = []
    for raw_line in page_text.splitlines():
        line = clean_text(raw_line)
        if not line:
            continue
        normalized = normalize_for_match(line)
        if any(normalized.startswith(value) for value in ('public', 'prerequis', 'pre requis')):
            current = 'PUBLIC'
            continue
        if normalized.startswith('objectifs'):
            current = 'OBJECTIVES'
            continue
        if normalized.startswith('programme') or normalized.startswith('contenu'):
            current = 'PROGRAM'
            continue
        if normalized.startswith('competences') or normalized.startswith('competences acquises'):
            current = 'SKILLS'
            continue
        if current == 'SKILLS' and line.startswith('•'):
            collected['SKILLS'].append(line.lstrip('•').strip())
        elif current in collected:
            collected[current].append(line)
        else:
            collected['OTHER'].append(line)
    return collected


def main() -> None:
    args = build_parser().parse_args()
    service = ReferentialImportService(store=ReferentialImportStore(args.store_path))
    analysis = service.analyze(args.file)
    report = analysis['report']
    metadata = analysis.get('metadata', {})
    document = analysis['source_document']
    page_counts = [len(page.blocks) for page in document.pages]
    total_chars = sum(len(clean_text(page.text)) for page in document.pages)
    line_count = sum(page_counts)
    sections = Counter()
    section_lines: dict[str, list[str]] = {'PUBLIC': [], 'OBJECTIVES': [], 'PROGRAM': [], 'SKILLS': [], 'OTHER': []}
    for page in document.pages:
        for block in page.blocks:
            sections[classify_section_label(block.text).label] += 1
        page_sections = _extract_section_lines(page.text)
        for key, values in page_sections.items():
            section_lines.setdefault(key, []).extend(values)

    semantic_annotation = analysis.get('semantic_annotation') or {}
    semantic_entities = semantic_annotation.get('entities', []) if isinstance(semantic_annotation, dict) else []
    derived_skills = analysis.get('derived_skills', [])
    payload: dict[str, Any] = {
        'file': str(args.file),
        'page_count': len(document.pages),
        'total_characters': total_chars,
        'line_count': line_count,
        'block_counts': page_counts,
        'sections': dict(sections),
        'metadata': {
            'provider': metadata.get('provider') or analysis['document'].provider,
            'title': metadata.get('title') or analysis['document'].title,
            'reference': metadata.get('reference') or analysis['document'].reference,
            'duration_hours': metadata.get('duration_hours') or analysis['document'].duration_hours,
            'cpf_eligible': metadata.get('cpf_eligible') if metadata.get('cpf_eligible') is not None else analysis['document'].cpf_eligible,
        },
        'objectives': section_lines.get('OBJECTIVES', []),
        'modules': section_lines.get('PROGRAM', []),
        'skills_count': len(derived_skills),
        'skills': [f"{skill.category}: {skill.canonical_label}" for skill in derived_skills],
        'semantic_entity_count': len(semantic_entities),
        'warnings': [item.to_dict() for item in report.warnings],
        'errors': [item.to_dict() for item in report.errors],
        'validation_status': report.status,
        'persistence_status': 'not_attempted',
        'saved': False,
    }

    if args.approve:
        output_path = service.approve(analysis, validated_by=args.validated_by)
        payload['persistence_status'] = 'saved'
        payload['saved'] = True
        payload['output_path'] = str(output_path)

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
