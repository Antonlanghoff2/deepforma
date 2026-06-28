from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from common.text import clean_text
from deepforma.cpf.skill_extractor import CPFSkillExtractor
from deepforma.skills.normalizer import SkillTaxonomyNormalizer
from deepforma.training.cpf_dataset import canonicalize_formation_row


LOGGER = logging.getLogger(__name__)
DEFAULT_INPUT = Path('data/processed/cpf/formations.parquet')
DEFAULT_OUTPUT = Path('data/processed/cpf/formations_with_skills.parquet')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Enrichit le catalogue CPF avec des compétences normalisées")
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--config', type=Path, default=Path('data/referentials/skills.json'))
    parser.add_argument('--limit', type=int, default=None)
    return parser


def _extract_row(row: dict[str, Any], extractor: CPFSkillExtractor) -> dict[str, Any]:
    canonical = canonicalize_formation_row(row)
    extraction = extractor.extract(
        {
            'title': canonical['title'],
            'certification': canonical['certification_label'],
            'description': canonical['description'],
            'objectives': canonical['objectives'],
            'nsf': canonical['nsf'],
        }
    )
    normalized = canonical | {
        'skills_explicit': extraction.skills_explicit,
        'skills_inferred': extraction.skills_inferred,
        'skills_normalized': extraction.skills_normalized,
        'skills_confidence': extraction.skills_confidence,
        'skills_evidence': extraction.skills_evidence,
    }
    # Keep an easily serializable search text for later use.
    normalized['search_text'] = clean_text(canonical['search_text'])
    return normalized


def extract_cpf_skills(input_path: Path, output_path: Path, *, config_path: Path | None = None, limit: int | None = None) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f'Catalogue CPF introuvable: {input_path}')
    df = pd.read_parquet(input_path)
    if limit is not None:
        df = df.head(limit)
    normalizer = SkillTaxonomyNormalizer(config_path)
    extractor = CPFSkillExtractor(normalizer=normalizer)
    rows = [_extract_row(row, extractor) for row in df.fillna('').to_dict(orient='records')]
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
