#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / 'src'):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from scripts.train_continual_skill_extractor import train as train_continual


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Entraîne l extracteur de compétences basé sur le dataset RNCP/ROME.')
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--validation', type=Path, required=True)
    parser.add_argument('--test', type=Path, required=True)
    parser.add_argument('--base-model', default='camembert-base')
    parser.add_argument('--output-dir', type=Path, default=Path('models/skill-extractor'))
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--learning-rate', type=float, default=2e-5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default=None)
    parser.add_argument('--fp16', action='store_true')
    return parser


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _to_ner_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in rows:
        entities = []
        for label in row.get('labels', []) or []:
            entities.append({
                'start': label.get('evidence_start', 0),
                'end': label.get('evidence_end', 0),
                'text': label.get('evidence_text', ''),
                'provenance': 'human_review',
            })
        records.append({
            'id': row.get('id'),
            'text': row.get('text', ''),
            'entities': entities,
            'document_skills': [],
            'metadata': {'title': row.get('title'), 'rome_code': row.get('rome_code'), 'rome_label': row.get('rome_label')},
            'source_path': 'derived',
        })
    return records


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = '\n'.join(json.dumps(row, ensure_ascii=False) for row in rows)
    path.write_text(payload + ('\n' if rows else ''), encoding='utf-8')


def main() -> None:
    args = build_parser().parse_args()
    train_rows = _read_jsonl(args.train)
    validation_rows = _read_jsonl(args.validation)
    test_rows = _read_jsonl(args.test)
    with tempfile.TemporaryDirectory(prefix='skill_extractor_') as tmp:
        tmpdir = Path(tmp)
        base_path = tmpdir / 'base.jsonl'
        incremental_path = tmpdir / 'incremental.jsonl'
        validation_path = tmpdir / 'validation.jsonl'
        test_path = tmpdir / 'test.jsonl'
        _write_jsonl(base_path, _to_ner_records(train_rows))
        _write_jsonl(incremental_path, [])
        _write_jsonl(validation_path, _to_ner_records(validation_rows))
        _write_jsonl(test_path, _to_ner_records(test_rows))
        train_continual(SimpleNamespace(
            base_dataset=base_path,
            incremental_dataset=incremental_path,
            validation_dataset=validation_path,
            test_dataset=test_path,
            base_model=args.base_model,
            output_dir=args.output_dir,
            resume_from_model=None,
            max_samples=None,
            seed=args.seed,
            device=args.device,
            fp16=args.fp16,
            batch_size=args.batch_size,
            epochs=args.epochs,
        ))


if __name__ == '__main__':
    main()
