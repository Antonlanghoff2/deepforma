#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from hashlib import sha1
from pathlib import Path
import sys
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / 'src'):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Construit les datasets d'entraînement à partir des référentiels France Compétences.")
    parser.add_argument('--skills', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('data/training/france_competences'))
    parser.add_argument('--blocks', type=Path, default=None)
    parser.add_argument('--certifications', type=Path, default=None)
    parser.add_argument('--review-queue', type=Path, default=None)
    return parser


def _split_for_certification(certification_id: str) -> str:
    score = int(sha1(certification_id.encode('utf-8')).hexdigest(), 16) % 100
    if score < 70:
        return 'train'
    if score < 85:
        return 'validation'
    return 'test'


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + '\n')
def _entity(text: str, label: str) -> dict[str, Any]:
    return {'start': 0, 'end': len(text), 'text': text, 'label': label}


def _safe_frame(path: Path | None) -> pd.DataFrame:
    if path and path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def _safe_csv(path: Path | None) -> pd.DataFrame:
    if path and path.exists():
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()
    return pd.DataFrame()


def main() -> None:
    args = build_parser().parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    skills = pd.read_parquet(args.skills)
    certifications = _safe_frame(args.certifications or (args.skills.parent / 'certifications.parquet'))
    blocks = _safe_frame(args.blocks or (args.skills.parent / 'blocks.parquet'))
    review = _safe_csv(args.review_queue or (args.skills.parent / 'review_queue.csv'))

    datasets = {
        'ner': {'train': [], 'validation': [], 'test': []},
        'skill_classification': {'train': [], 'validation': [], 'test': []},
        'skill_normalization': {'train': [], 'validation': [], 'test': []},
        'semantic_pairs': {'train': [], 'validation': [], 'test': []},
    }

    for _, row in skills.fillna('').iterrows():
        cert_id = str(row.get('certification_id') or row.get('block_id') or row.get('referential_id') or 'unknown')
        split = _split_for_certification(cert_id)
        text = str(row.get('skill_original') or row.get('libelle_officiel') or row.get('libelle') or '').strip()
        if not text:
            continue
        skill_category = str(row.get('skill_category') or '').lower()
        label = 'METHOD' if skill_category in {'method', 'méthode'} else 'TOOL' if skill_category in {'tool', 'outils'} else 'SKILL'
        datasets['ner'][split].append({
            'id': str(row.get('referential_id') or row.get('skill_id') or row.get('block_id') or cert_id),
            'text': text,
            'title': str(row.get('certification_title') or ''),
            'labels': [_entity(text, label)],
            'certification_id': cert_id,
            'block_id': str(row.get('block_id') or ''),
        })
        datasets['skill_classification'][split].append({'text': text, 'label': 1, 'certification_id': cert_id})
        datasets['skill_normalization'][split].append({'text': text, 'canonical_label': str(row.get('skill_normalized') or text), 'label': 1, 'certification_id': cert_id})
        datasets['semantic_pairs'][split].append({'text_a': text, 'text_b': str(row.get('skill_short') or text), 'label': 1, 'pair_type': 'same_skill', 'certification_id': cert_id})

    for _, row in review.fillna('').iterrows():
        cert_id = str(row.get('certification_code') or row.get('certification_id') or 'unknown')
        split = _split_for_certification(cert_id)
        text = str(row.get('source_text') or row.get('text') or '').strip()
        if not text:
            continue
        datasets['ner'][split].append({'id': cert_id, 'text': text, 'title': '', 'labels': [_entity(text, 'NOT_SKILL')], 'certification_id': cert_id, 'block_id': str(row.get('block_title') or '')})
        datasets['skill_classification'][split].append({'text': text, 'label': 0, 'certification_id': cert_id})
        datasets['skill_normalization'][split].append({'text': text, 'canonical_label': '', 'label': 0, 'certification_id': cert_id})

    for split in ['train', 'validation', 'test']:
        _write_jsonl(out / f'ner_{split}.jsonl', datasets['ner'][split])
        _write_jsonl(out / f'skill_classification_{split}.jsonl', datasets['skill_classification'][split])
        _write_jsonl(out / f'skill_normalization_{split}.jsonl', datasets['skill_normalization'][split])
        _write_jsonl(out / f'semantic_pairs_{split}.jsonl', datasets['semantic_pairs'][split])

    report = {
        'ner': {split: len(rows) for split, rows in datasets['ner'].items()},
        'skill_classification': {split: len(rows) for split, rows in datasets['skill_classification'].items()},
        'skill_normalization': {split: len(rows) for split, rows in datasets['skill_normalization'].items()},
        'semantic_pairs': {split: len(rows) for split, rows in datasets['semantic_pairs'].items()},
        'certifications': int(certifications.shape[0]),
        'blocks': int(blocks.shape[0]),
        'skills': int(skills.shape[0]),
    }
    (out / 'dataset_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
