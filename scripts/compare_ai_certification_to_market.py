#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
for candidate in (ROOT, SRC):
    text = str(candidate)
    if text not in sys.path:
        sys.path.insert(0, text)

from france_travail.client import FranceTravailClient
from services.certification_market_comparison import (
    CertificationMarketComparator,
    collect_market_offers,
    write_comparison_outputs,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_OUTPUT_DIR = ROOT / 'data' / 'reports' / 'ai_certification_market'
DEFAULT_INPUT_JSONL = ROOT / 'data' / 'france_travail' / 'normalized' / 'offers.jsonl'


def _split_values(value: str | None) -> list[str]:
    if not value:
        return []
    parts = [part.strip() for part in value.replace(';', ',').replace('|', ',').split(',')]
    return [part for part in parts if part]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f'Fichier JSONL introuvable: {path}')
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Compare le référentiel IA avec le marché France Travail.')
    parser.add_argument('--territory', default='75056', help='Code commune ou département.')
    parser.add_argument('--commune', default=None, help='Code commune INSEE, ex: 75056')
    parser.add_argument('--departement', default=None, help='Code département, ex: 75')
    parser.add_argument('--radius-km', type=int, default=None, help='Rayon de recherche.')
    parser.add_argument('--date-min', default=None, help='Date minimale (YYYY-MM-DD).')
    parser.add_argument('--date-max', default=None, help='Date maximale (YYYY-MM-DD).')
    parser.add_argument('--job-titles', default='ingénieur intelligence artificielle,AI Engineer,Machine Learning Engineer,Data Scientist,MLOps Engineer,ingénieur Machine Learning,ingénieur NLP,ingénieur Deep Learning,ingénieur IA générative,Data Engineer IA,chef de projet IA')
    parser.add_argument('--rome-codes', default='M1805', help='Codes ROME séparés par virgule.')
    parser.add_argument('--max-pages', type=int, default=3)
    parser.add_argument('--max-offers', type=int, default=200)
    parser.add_argument('--page-size', type=int, default=20)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--input-jsonl', type=Path, default=None, help='Offres normalisées déjà présentes en JSONL.')
    parser.add_argument('--referential', type=Path, default=None)
    parser.add_argument('--embedding-model', default=None)
    parser.add_argument('--dry-run', action='store_true')
    return parser


def _territory_label(commune: str | None, departement: str | None, territory: str) -> str:
    parts = [part for part in [commune, departement, territory] if part]
    return ' / '.join(dict.fromkeys(parts))


def main() -> int:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s %(message)s')
    parser = build_parser()
    args = parser.parse_args()

    commune = args.commune
    departement = args.departement
    territory = args.territory.strip()
    if not commune and not departement:
        if len(territory) == 5 and territory.isdigit():
            commune = territory
        elif len(territory) == 2 and territory.isdigit():
            departement = territory

    comparator = CertificationMarketComparator(referential_path=args.referential, embedding_model=args.embedding_model)
    job_titles = _split_values(args.job_titles)
    rome_codes = _split_values(args.rome_codes)

    if args.input_jsonl:
        offers = _load_jsonl(args.input_jsonl)
        source_queries = [str(args.input_jsonl)]
    else:
        client = FranceTravailClient()
        offers, source_queries = collect_market_offers(
            client,
            commune=commune,
            departement=departement,
            distance_km=args.radius_km,
            date_min=args.date_min,
            date_max=args.date_max,
            job_titles=job_titles,
            rome_codes=rome_codes,
            max_pages=args.max_pages,
            max_offers=args.max_offers,
            page_size=args.page_size,
        )

    report = comparator.compare(
        offers,
        territory=_territory_label(commune, departement, territory),
        radius_km=args.radius_km,
        date_min=args.date_min,
        date_max=args.date_max,
        job_titles=job_titles,
        rome_codes=rome_codes,
        source_queries=source_queries,
    )
    paths = write_comparison_outputs(report, args.output_dir)

    print(f"Offres analysées: {report.offer_count}")
    print(f"Compétences couvertes: {len(report.covered_skills)}")
    print(f"Compétences manquantes: {len(report.missing_skills)}")
    print(f"Score global de couverture: {report.global_coverage_score:.2f}%")
    print(f"JSON: {paths['json']}")
    print(f"CSV validation: {paths['validation_csv']}")
    print(f"CSV écarts: {paths['gaps_csv']}")

    if args.dry_run:
        top = report.top_demanded_skills[:10]
        if top:
            print('Top compétences demandées:')
            for row in top:
                print(f"- {row.label} ({row.offer_count} offres, {row.share_percent:.1f}%)")
        if report.example_offers:
            print("Exemples d'offres:")
            for offer in report.example_offers[:5]:
                print(f"- {offer.title}: {', '.join(offer.matches[:5])}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
