#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / 'src'):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from common.text import clean_text, normalize_for_match, stable_hash
from referentials.unified_skill_referential import UnifiedSkillReferential


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Enrichit les offres avec ROME et RNCP.')
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--unified-referential', type=Path, default=Path('data/referentials/unified/skills.jsonl'))
    parser.add_argument('--mappings', type=Path, default=Path('data/referentials/mappings/rncp_rome_links.jsonl'))
    parser.add_argument('--output', type=Path, default=Path('data/training/skill_extraction/offers_enriched.jsonl'))
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--write', action='store_true')
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


def _sentences(text: str) -> list[str]:
    return [clean_text(part) for part in re.split(r'(?<=[.!?])\s+|\n+|[•·;]+', text or '') if clean_text(part)]


def _evidence_for_skill(text: str, skill: dict[str, Any]) -> str:
    candidates = [skill.get('canonical_label', ''), *skill.get('aliases', [])]
    for sentence in _sentences(text):
        norm_sentence = normalize_for_match(sentence)
        for candidate in candidates:
            candidate_norm = normalize_for_match(candidate)
            if candidate_norm and candidate_norm in norm_sentence:
                return sentence
    return ''


def _match_skill(text: str, unified_skills: list[dict[str, Any]], label: str) -> dict[str, Any] | None:
    norm = normalize_for_match(label)
    for skill in unified_skills:
        if norm == normalize_for_match(skill.get('canonical_label', '')):
            return skill
        if norm in {normalize_for_match(alias) for alias in skill.get('aliases', [])}:
            return skill
        evidence = _evidence_for_skill(text, skill)
        if evidence and norm and norm in normalize_for_match(evidence):
            return skill
    return None


def main() -> None:
    args = build_parser().parse_args()
    unified = UnifiedSkillReferential(args.unified_referential)
    unified_skills = unified.get_all_skills()
    source_by_label = {normalize_for_match(item['canonical_label']): item for item in unified_skills}
    links = _read_jsonl(args.mappings)
    rows = _read_jsonl(args.input)
    enriched: list[dict[str, Any]] = []
    for row in rows:
        title = clean_text(row.get('title') or row.get('intitule_poste') or '')
        description = clean_text(row.get('description') or row.get('description_original') or row.get('text') or row.get('offer_text') or '')
        rome_code = clean_text(row.get('rome_code') or (row.get('rome', {}).get('code') if isinstance(row.get('rome'), dict) else ''))
        rome_label = clean_text(row.get('rome_label') or (row.get('rome', {}).get('label') if isinstance(row.get('rome'), dict) else ''))
        rncp_candidates = []
        for link in links:
            if link.get('rome_code') == rome_code:
                rncp_candidates.append({'rncp_code': link.get('rncp_code'), 'mapping_score': float(link.get('score', 0.0))})
        skills: list[dict[str, Any]] = []
        seen_skill_labels: set[str] = set()
        source_fields = row.get('competences') or row.get('skills') or row.get('merged_skills') or row.get('normalized_skills') or row.get('structured_skills') or []
        if isinstance(source_fields, list):
            for entry in source_fields:
                if isinstance(entry, str):
                    label = clean_text(entry)
                else:
                    label = clean_text(entry.get('libelle') or entry.get('canonical_label') or entry.get('official_label') or entry.get('label') or '')
                if not label:
                    continue
                label_key = normalize_for_match(label)
                if label_key and label_key in seen_skill_labels:
                    continue
                seen_skill_labels.add(label_key) if label_key else None
                canonical = source_by_label.get(label_key) or _match_skill(description, unified_skills, label)
                if not canonical:
                    evidence = _evidence_for_skill(description, {'canonical_label': label, 'aliases': []})
                    if not evidence:
                        continue
                    skills.append({
                        'canonical_skill_id': stable_hash('merged', label, length=24),
                        'canonical_label': label,
                        'evidence': evidence,
                        'confidence': 0.8,
                        'source_links': entry.get('sources', []) if isinstance(entry, dict) else [],
                    })
                    continue
                evidence = _evidence_for_skill(description, canonical)
                if not evidence:
                    continue
                skills.append({
                    'canonical_skill_id': canonical['canonical_skill_id'],
                    'canonical_label': canonical['canonical_label'],
                    'evidence': evidence,
                    'confidence': 0.91,
                    'source_links': canonical['sources'],
                })
        enriched.append({
            'offer_id': row.get('offer_id') or row.get('id'),
            'title': title,
            'description': description,
            'rome': {'code': rome_code, 'label': rome_label, 'source': 'france_travail'},
            'rncp_candidates': rncp_candidates,
            'skills': skills,
        })
    print(json.dumps({'offers': len(enriched), 'with_skills': sum(1 for row in enriched if row['skills']), 'sample': enriched[:2]}, ensure_ascii=False, indent=2))
    if args.dry_run or not args.write:
        return
    _write_jsonl(args.output, enriched)


if __name__ == '__main__':
    main()
