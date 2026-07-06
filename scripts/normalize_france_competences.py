#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
import sys
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / 'src'):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from data_sources.france_competences.rncp_parser import FranceCompetencesRncpParser
from data_sources.france_competences.rs_parser import FranceCompetencesRsParser
from deepforma.cpf.io import ensure_parent


LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Normalise les référentiels France Compétences vers Parquet/JSON/CSV.')
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('data/processed/france_competences'))
    parser.add_argument('--keep-evaluation', action='store_true', default=False)
    return parser


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding='utf-8')


def _write_csv(path: Path, rows: list[dict[str, Any]], *, fieldnames: list[str] | None = None) -> None:
    ensure_parent(path)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with path.open('w', encoding='utf-8', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def _normalize_text(value: str) -> str:
    return ' '.join(value.split()).strip()


def _short_label(value: str) -> str:
    text = _normalize_text(value)
    if len(text) <= 120:
        return text
    return ' '.join(text.split()[:16]).rstrip(' ,;:-')


def _action_object(label: str) -> tuple[str, str]:
    text = _normalize_text(label)
    if not text:
        return '', ''
    verbs = ['mettre en œuvre', 'mettre en place', 'analyser', 'concevoir', 'développer', 'utiliser', 'maîtriser', 'assurer', 'piloter', 'réaliser', 'préparer', 'organiser', 'adapter', 'définir']
    lowered = text.lower()
    for verb in verbs:
        if lowered.startswith(verb):
            remainder = text[len(verb):].strip(' ,:-')
            return verb, remainder
    first, _, rest = text.partition(' ')
    return first, rest


def _parquet(df: pd.DataFrame, path: Path) -> None:
    ensure_parent(path)
    df.to_parquet(path, index=False)


def _parse_input(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(item for item in path.iterdir() if item.suffix.lower() in {'.zip', '.xml', '.csv'})
    return [path]


def _safe_get_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s %(message)s')
    args = build_parser().parse_args()
    parser_rncp = FranceCompetencesRncpParser()
    parser_rs = FranceCompetencesRsParser()

    parsed_certifications: list[dict[str, Any]] = []
    parsed_blocks: list[dict[str, Any]] = []
    parsed_skills: list[dict[str, Any]] = []
    parsed_negatives: list[dict[str, Any]] = []
    parsed_activities: list[dict[str, Any]] = []

    for archive in _parse_input(args.input):
        name = archive.name.lower()
        if 'rs' in name and 'rncp' not in name:
            parsed = parser_rs.parse_archive(archive)
        else:
            parsed = parser_rncp.parse_archive(archive)
        parsed_certifications.extend(parsed.certifications)
        parsed_blocks.extend(parsed.blocks)
        parsed_skills.extend(parsed.skills)
        parsed_activities.extend(parsed.activities)
        parsed_negatives.extend(parsed.negatives)

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    cert_df = _safe_get_frame(parsed_certifications).drop_duplicates(subset=['certification_id', 'certification_code'], keep='first')
    if not cert_df.empty and 'source_file' not in cert_df.columns:
        cert_df['source_file'] = str(args.input)
    _parquet(cert_df, out / 'certifications.parquet')

    block_df = _safe_get_frame(parsed_blocks).drop_duplicates(subset=['block_id'], keep='first')
    _parquet(block_df, out / 'blocks.parquet')

    skill_df = _safe_get_frame(parsed_skills).drop_duplicates(subset=['referential_id', 'evidence'], keep='first')
    if not skill_df.empty:
        if 'skill_original' not in skill_df.columns:
            skill_df['skill_original'] = skill_df['libelle_officiel'] if 'libelle_officiel' in skill_df.columns else skill_df.get('libelle', '')
        skill_df['skill_short'] = skill_df['skill_original'].fillna('').map(_short_label)
        skill_df['skill_normalized'] = skill_df['canonical_name'] if 'canonical_name' in skill_df.columns else skill_df['skill_short']
        skill_df['skill_category'] = skill_df['category'] if 'category' in skill_df.columns else ''
        skill_df['skill_subcategory'] = skill_df['subcategory'] if 'subcategory' in skill_df.columns else ''
        action_object = skill_df['skill_short'].map(_action_object)
        skill_df['action_verb'] = [item[0] for item in action_object]
        skill_df['skill_object'] = [item[1] for item in action_object]
        block_title = skill_df['block_title'] if 'block_title' in skill_df.columns else pd.Series([''] * len(skill_df), index=skill_df.index)
        activity_name = skill_df['activity_name'] if 'activity_name' in skill_df.columns else pd.Series([''] * len(skill_df), index=skill_df.index)
        skill_df['context'] = block_title.fillna('') + ' | ' + activity_name.fillna('')
        if 'source_url' not in skill_df.columns:
            skill_df['source_url'] = None
        if 'is_active' not in skill_df.columns:
            skill_df['is_active'] = True
        if 'confidence' not in skill_df.columns:
            skill_df['confidence'] = 0.0
        if 'normalization_method' not in skill_df.columns:
            skill_df['normalization_method'] = 'exact'
    skill_columns = [
        'certification_id', 'certification_code', 'certification_title', 'block_id', 'block_code', 'block_title',
        'skill_id', 'code', 'skill_original', 'skill_short', 'skill_normalized', 'skill_category', 'skill_subcategory',
        'action_verb', 'skill_object', 'context', 'rome_codes', 'nsf_codes', 'source_url', 'is_active', 'confidence', 'normalization_method',
        'referential_id', 'libelle', 'libelle_officiel', 'evidence', 'match_type', 'canonical_name', 'technical_keywords',
        'origin_document', 'activity_code', 'activity_name', 'source_page', 'source_order', 'source_file',
    ]
    for column in skill_columns:
        if column not in skill_df.columns:
            skill_df[column] = None
    _parquet(skill_df[skill_columns], out / 'skills.parquet')

    rome_links: list[dict[str, Any]] = []
    for row in parsed_certifications:
        for rome_code in row.get('codes_rome', []) or []:
            rome_links.append(
                {
                    'certification_id': row.get('certification_id'),
                    'certification_code': row.get('certification_code'),
                    'rome_code': rome_code,
                    'relation_source': 'official',
                    'is_active': row.get('status') == 'active',
                }
            )
    _parquet(_safe_get_frame(rome_links), out / 'certification_rome_links.parquet')

    quality_report = {
        'certifications': int(len(cert_df)),
        'blocks': int(len(block_df)),
        'skills': int(len(skill_df)),
        'activities': int(len(parsed_activities)),
        'negative_examples': int(len(parsed_negatives)),
        'keep_evaluation': bool(args.keep_evaluation),
        'source': str(args.input),
    }
    _write_json(out / 'quality_report.json', quality_report)
    _write_csv(out / 'review_queue.csv', parsed_negatives, fieldnames=['text', 'label', 'block_code', 'activity_code', 'source_page', 'origin_document'])
    _write_json(out / 'metadata.json', {
        'source': str(args.input),
        'generated_at': pd.Timestamp.now(tz='UTC').isoformat(),
        'counts': quality_report,
    })
    print(json.dumps(quality_report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
