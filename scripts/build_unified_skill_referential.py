#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / 'src'):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from referentials.france_competences import FranceCompetencesOpenDataImporter
from referentials.unified_skill_referential import build_unified_skill_referential, write_unified_skill_referential
from referentials.rome_referential import RomeReferentialImporter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Construit le référentiel unifié des compétences.')
    parser.add_argument('--rncp-path', type=Path, default=Path('data/raw/france_competences'))
    parser.add_argument('--rome-path', type=Path, default=Path('data/raw/rome'))
    parser.add_argument('--mappings-path', type=Path, default=Path('data/referentials/mappings/rncp_rome_links.jsonl'))
    parser.add_argument('--output-dir', type=Path, default=Path('data/referentials/unified'))
    parser.add_argument('--write', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    return parser


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
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


def main() -> None:
    args = build_parser().parse_args()
    rncp = FranceCompetencesOpenDataImporter(args.rncp_path).load(active_only=True)
    rome = RomeReferentialImporter(args.rome_path).load()
    mappings = _read_jsonl(args.mappings_path)
    unified, source_links = build_unified_skill_referential(france_competences=rncp, rome=rome, mappings=mappings)
    print(json.dumps({'unified_skills': len(unified), 'source_links': len(source_links), 'sample': unified[:3]}, ensure_ascii=False, indent=2))
    if args.dry_run or not args.write:
        return
    out = args.output_dir
    write_unified_skill_referential(out / 'skills.jsonl', unified)
    _write_jsonl(out / 'source_links.jsonl', source_links)


if __name__ == '__main__':
    main()
