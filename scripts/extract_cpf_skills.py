
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from common.text import clean_text, normalize_for_match
from deepforma.cpf.skill_extractor import CPFSkillExtractor
from deepforma.skills.normalizer import SkillTaxonomyNormalizer


LOGGER = logging.getLogger(__name__)
DEFAULT_INPUT = Path('data/processed/cpf/formations_normalized.parquet')
DEFAULT_OUTPUT = Path('data/processed/cpf/formations_with_skills.parquet')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Enrichit le catalogue CPF normalisé avec des compétences structurées')
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--config', type=Path, default=Path('data/referentials/skills.json'))
    parser.add_argument('--limit', type=int, default=None)
    return parser


def _ensure_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [clean_text(item) for item in value if clean_text(item)]
    if isinstance(value, tuple) or isinstance(value, set):
        return [clean_text(item) for item in value if clean_text(item)]
    if hasattr(value, 'tolist') and not isinstance(value, (str, bytes, bytearray)):
        try:
            converted = value.tolist()
            if isinstance(converted, list):
                return _ensure_list(converted)
        except Exception:
            pass
    text = clean_text(value)
    if not text:
        return []
    if text.startswith('[') and text.endswith(']'):
        try:
            loaded = json.loads(text)
            if isinstance(loaded, list):
                return _ensure_list(loaded)
        except Exception:
            pass
    parts = [clean_text(part) for part in text.split('|') if clean_text(part)]
    return parts or [text]


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        key = normalize_for_match(value)
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(value)
    return deduped


def _skill_payload(values: list[str], normalizer: SkillTaxonomyNormalizer, source: str) -> list[dict[str, Any]]:
    matches = normalizer.normalize_many(values, extraction_source=source, confidence_floor=0.0)
    payload: list[dict[str, Any]] = []
    seen: set[str] = set()
    for value in values:
        match = next((item for item in matches if normalize_for_match(item.original_label) == normalize_for_match(value)), None)
        if match:
            if match.canonical_id in seen:
                continue
            seen.add(match.canonical_id)
            payload.append(
                {
                    'canonical_id': match.canonical_id,
                    'canonical_label': match.canonical_label,
                    'aliases': match.aliases,
                    'original_label': value,
                    'confidence': round(float(match.confidence), 4),
                    'extraction_source': match.extraction_source,
                }
            )
            continue
        key = normalize_for_match(value)
        if key in seen:
            continue
        seen.add(key)
        payload.append(
            {
                'canonical_id': key,
                'canonical_label': value,
                'aliases': [],
                'original_label': value,
                'confidence': 0.0,
                'extraction_source': source,
            }
        )
    return payload


def _merge_skill_payloads(*payloads: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, float]]:
    merged: dict[str, dict[str, Any]] = {}
    confidences: dict[str, float] = {}
    for payload in payloads:
        for item in payload:
            key = str(item.get('canonical_id') or normalize_for_match(item.get('canonical_label') or item.get('original_label') or ''))
            if not key:
                continue
            current = merged.setdefault(
                key,
                {
                    'canonical_id': key,
                    'canonical_label': item.get('canonical_label') or item.get('original_label') or key,
                    'aliases': list(item.get('aliases') or []),
                    'original_label': item.get('original_label') or item.get('canonical_label') or key,
                    'confidence': 0.0,
                    'extraction_source': item.get('extraction_source') or 'cpf_v3',
                },
            )
            current['confidence'] = max(float(current['confidence']), float(item.get('confidence') or 0.0))
            if item.get('canonical_label'):
                current['canonical_label'] = item.get('canonical_label')
            if item.get('original_label'):
                current['original_label'] = item.get('original_label')
            current['aliases'] = list(dict.fromkeys([*(current.get('aliases') or []), *(item.get('aliases') or [])]))
            confidences[key] = max(confidences.get(key, 0.0), float(item.get('confidence') or 0.0))
    ordered = sorted(merged.values(), key=lambda item: (-float(item['confidence']), str(item['canonical_label'])))
    for item in ordered:
        item['confidence'] = round(float(item.get('confidence') or 0.0), 4)
    return ordered, {key: round(value, 4) for key, value in confidences.items()}


def _extract_row(row: dict[str, Any], extractor: CPFSkillExtractor, normalizer: SkillTaxonomyNormalizer) -> dict[str, Any]:
    competences = _dedupe(_ensure_list(row.get('competences')))
    tags = _dedupe(_ensure_list(row.get('tags')))
    explicit_values = _dedupe(competences + [item for item in tags if item not in competences])
    explicit_payload = _skill_payload(explicit_values, normalizer, 'cpf_v3:explicit')
    text_result = extractor.extract(
        {
            'title': row.get('titre'),
            'certification': row.get('certification'),
            'description': row.get('description'),
            'objectives': row.get('objectives'),
            'nsf': row.get('nsf'),
        }
    )
    text_payload = text_result.skills_normalized or text_result.skills_inferred or []
    skills_normalized, skills_confidence = _merge_skill_payloads(explicit_payload, text_payload)
    skills_evidence = dict(text_result.skills_evidence)
    for item in explicit_payload:
        skills_evidence.setdefault(item['canonical_id'], []).append(
            {
                'field': 'competences',
                'evidence': item['original_label'],
                'source': item['extraction_source'],
            }
        )
    normalized = dict(row)
    normalized['competences'] = competences
    normalized['competences_normalisees'] = skills_normalized
    normalized['skills_explicit'] = explicit_payload
    normalized['skills_inferred'] = text_result.skills_inferred
    normalized['skills_normalized'] = skills_normalized
    normalized['skills_confidence'] = skills_confidence
    normalized['skills_evidence'] = skills_evidence
    normalized['tags'] = tags
    normalized['search_text'] = clean_text(row.get('texte_modele') or row.get('search_text') or row.get('titre'))
    normalized['certification_label'] = clean_text(row.get('certification_label') or row.get('certification') or row.get('code_certification'))
    normalized['certification_code'] = clean_text(row.get('certification_code') or row.get('code_certification') or row.get('code_rncp') or row.get('code_rs'))
    normalized['referential_type'] = clean_text(row.get('referential_type'))
    normalized['level'] = clean_text(row.get('level') or row.get('exit_level') or row.get('niveau'))
    normalized['organization'] = clean_text(row.get('organization') or row.get('organisme'))
    normalized['siret'] = clean_text(row.get('siret') or row.get('organization_siret'))
    normalized['region'] = clean_text(row.get('region'))
    normalized['department'] = clean_text(row.get('department'))
    normalized['region_code'] = clean_text(row.get('region_code'))
    normalized['department_code'] = clean_text(row.get('department_code'))
    return normalized


def extract_cpf_skills(input_path: Path, output_path: Path, *, config_path: Path | None = None, limit: int | None = None) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f'Catalogue CPF introuvable: {input_path}')
    df = pd.read_parquet(input_path)
    if df.empty:
        raise ValueError(f'Le parquet CPF est vide: {input_path}')
    if limit is not None:
        df = df.head(limit)
    normalizer = SkillTaxonomyNormalizer(config_path)
    extractor = CPFSkillExtractor(normalizer=normalizer)
    rows = [_extract_row(row, extractor, normalizer) for row in df.where(pd.notnull(df), None).to_dict(orient='records')]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    enriched = pd.DataFrame.from_records(rows)
    enriched.to_parquet(output_path, index=False)
    return enriched


def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s %(message)s')
    args = build_parser().parse_args()
    df = extract_cpf_skills(args.input, args.output, config_path=args.config, limit=args.limit)
    LOGGER.info('Parquet enrichi écrit dans %s (%s lignes)', args.output, len(df))


if __name__ == '__main__':
    main()
