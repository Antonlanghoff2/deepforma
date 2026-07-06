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

from referentials.rome_referential import RomeReferentialImporter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Importe le référentiel ROME.')
    parser.add_argument('--input', type=Path, default=Path('data/raw/rome'))
    parser.add_argument('--output-dir', type=Path, default=Path('data/referentials/rome'))
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


def main() -> None:
    args = build_parser().parse_args()
    importer = RomeReferentialImporter(args.input)
    payload = importer.load()
    metadata = {
        'source_path': str(args.input),
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'counts': {key: len(value) for key, value in payload.items()},
    }
    print(json.dumps({'metadata': metadata, 'sample_job': payload.get('jobs', [])[:1], 'sample_skill': payload.get('skills', [])[:1]}, ensure_ascii=False, indent=2))
    if args.dry_run or not args.write:
        return
    out = args.output_dir
    _write_jsonl(out / 'jobs.jsonl', payload.get('jobs', []))
    _write_jsonl(out / 'job_titles.jsonl', payload.get('job_titles', []))
    _write_jsonl(out / 'skills.jsonl', payload.get('skills', []))
    _write_jsonl(out / 'job_skill_links.jsonl', payload.get('job_skill_links', []))
    _write_json(out / 'metadata.json', metadata)


if __name__ == '__main__':
    main()
