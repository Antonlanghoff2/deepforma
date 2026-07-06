#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / 'src'):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from referentials.france_competences import FranceCompetencesApiClient, FranceCompetencesOpenDataImporter, FranceCompetencesSource


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Importe le référentiel France Compétences (RNCP/RS).')
    parser.add_argument('--source', default=None, choices=['open_data', 'api'])
    parser.add_argument('--data-path', type=Path, default=Path('data/raw/france_competences'))
    parser.add_argument('--output-dir', type=Path, default=Path('data/referentials/france_competences'))
    parser.add_argument('--active-only', action='store_true', default=True)
    parser.add_argument('--include-inactive', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--write', action='store_true')
    return parser


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = '\n'.join(json.dumps(row, ensure_ascii=False) for row in rows)
    path.write_text(payload + ('\n' if rows else ''), encoding='utf-8')


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding='utf-8')


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    with path.open('w', encoding='utf-8', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _select_source(args: argparse.Namespace) -> FranceCompetencesSource:
    mode = args.source or 'open_data'
    if mode == 'api':
        return FranceCompetencesApiClient()
    return FranceCompetencesOpenDataImporter(args.data_path, active_only=not args.include_inactive)


def main() -> None:
    args = build_parser().parse_args()
    source = _select_source(args)
    payload = source.load(active_only=not args.include_inactive)
    certifications = payload.get('certifications', [])
    blocks = payload.get('blocks', [])
    skills = payload.get('skills', [])
    metadata = {
        'source': args.source or 'open_data',
        'active_only': not args.include_inactive,
        'data_path': str(args.data_path),
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'counts': {'certifications': len(certifications), 'blocks': len(blocks), 'skills': len(skills)},
    }
    report = {'metadata': metadata, 'sample_certification': certifications[:1], 'sample_block': blocks[:1], 'sample_skill': skills[:1]}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.dry_run or not args.write:
        return
    out = args.output_dir
    _write_jsonl(out / 'certifications.jsonl', certifications)
    _write_jsonl(out / 'blocks.jsonl', blocks)
    _write_jsonl(out / 'skills.jsonl', skills)
    _write_json(out / 'metadata.json', metadata)


if __name__ == '__main__':
    main()
