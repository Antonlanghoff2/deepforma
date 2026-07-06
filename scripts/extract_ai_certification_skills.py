#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
for path in (ROOT, SRC):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from common.text import clean_text
from continual_learning.store import ContinualLearningStore
from skill_extraction.ai_certification_extractor import AICertificationSkillExtractor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Extrait les compétences de certification IA depuis les offres existantes.')
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--referential', type=Path, default=Path('data/referentials/ai_engineer_certification_2025.json'))
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--write', action='store_true')
    return parser


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = '\n'.join(json.dumps(row, ensure_ascii=False) for row in rows)
    path.write_text(payload + ('\n' if rows else ''), encoding='utf-8')


def _rows_from_source(source: Path) -> tuple[str, list[dict[str, Any]]]:
    if source.is_dir():
        rows: list[dict[str, Any]] = []
        for path in sorted(source.rglob('*.jsonl')):
            rows.extend(_read_jsonl(path))
        return 'jsonl-directory', rows
    if source.suffix.lower() == '.jsonl':
        return 'jsonl', _read_jsonl(source)
    if source.suffix.lower() in {'.sqlite', '.sqlite3', '.db'}:
        store = ContinualLearningStore(source)
        return 'sqlite', store.list_offers()
    raise ValueError(f'Source non supportée: {source}')


def _offer_text(row: dict[str, Any]) -> tuple[str | None, str]:
    title = clean_text(row.get('title') or row.get('intitule') or row.get('job_title'))
    description = clean_text(row.get('description') or row.get('description_original') or row.get('offer_text') or '')
    if not description and isinstance(row.get('raw_payload_json'), str):
        try:
            payload = json.loads(row['raw_payload_json'])
            description = clean_text(payload.get('description') or payload.get('offer_text') or '')
            if not title:
                title = clean_text(payload.get('title') or payload.get('intitule') or '')
        except Exception:
            pass
    return (title or None, description)


def _update_row(row: dict[str, Any], extraction: dict[str, Any]) -> dict[str, Any]:
    updated = dict(row)
    if extraction['intitule_poste']:
        updated['title'] = extraction['intitule_poste']
    updated['competences'] = extraction['competences']
    if isinstance(updated.get('raw_payload_json'), str):
        try:
            payload = json.loads(updated['raw_payload_json'])
            if extraction['intitule_poste']:
                payload['title'] = extraction['intitule_poste']
            payload['competences'] = extraction['competences']
            updated['raw_payload_json'] = json.dumps(payload, ensure_ascii=False)
        except Exception:
            pass
    return updated


def _store_results(source_type: str, source: Path, rows: list[dict[str, Any]], updated_rows: list[dict[str, Any]], *, write: bool) -> None:
    if not write:
        return
    if source_type == 'jsonl':
        _write_jsonl(source, updated_rows)
        return
    if source_type == 'jsonl-directory':
        raise ValueError("L'écriture en répertoire JSONL nécessite un fichier unique.")
    if source_type == 'sqlite':
        store = ContinualLearningStore(source)
        for original, updated in zip(rows, updated_rows):
            offer_row_id = int(original['id'])
            title = updated.get('title')
            competences = updated.get('competences') or []
            store.update_offer_title_and_competences(
                offer_row_id,
                title=title if title else None,
                competences=competences,
            )
        return
    raise ValueError(f'Écriture non prise en charge pour {source_type}')


def main() -> None:
    args = build_parser().parse_args()
    source_type, rows = _rows_from_source(args.input)
    extractor = AICertificationSkillExtractor(referential_path=args.referential)
    analyzed = 0
    with_skills = 0
    without_skills = 0
    total_skills = 0
    label_counts: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    updated_rows: list[dict[str, Any]] = []

    for index, row in enumerate(rows, start=1):
        title, description = _offer_text(row)
        if not description and not title:
            updated_rows.append(dict(row))
            continue
        extraction = extractor.extract(title=title, description=description)
        analyzed += 1
        competences = extraction['competences']
        total_skills += len(competences)
        if competences:
            with_skills += 1
            for item in competences:
                label_counts[item['libelle']] += 1
                if len(examples) < 5:
                    examples.append(
                        {
                            'title': extraction['intitule_poste'],
                            'referential_id': item['referential_id'],
                            'code': item['code'],
                            'evidence': item['evidence'],
                            'match_type': item['match_type'],
                        }
                    )
        else:
            without_skills += 1
        updated_rows.append(_update_row(row, extraction))

        if index % max(args.batch_size, 1) == 0:
            pass

    avg = round(total_skills / analyzed, 2) if analyzed else 0.0
    report = {
        'analyzed_offers': analyzed,
        'offers_with_skills': with_skills,
        'average_skills_per_offer': avg,
        'top_detected_skills': label_counts.most_common(10),
        'examples': examples,
        'offers_without_skills': without_skills,
        'write_mode': bool(args.write and not args.dry_run),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.write and not args.dry_run:
        _store_results(source_type, args.input, rows, updated_rows, write=True)


if __name__ == '__main__':
    main()
