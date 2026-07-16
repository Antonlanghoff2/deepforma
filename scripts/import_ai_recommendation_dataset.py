#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / 'src'):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from ai_recommendations.loader import import_ai_recommendation_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Importe le dataset de recommandations IA et génère les sorties normalisées.')
    parser.add_argument('--input', type=Path, default=Path('data/raw/dataset_recommandations_IA_complet.csv'))
    parser.add_argument('--output-csv', type=Path, default=Path('data/referentials/ai_recommendation_rules.csv'))
    parser.add_argument('--output-json', type=Path, default=Path('data/referentials/ai_recommendation_rules.json'))
    parser.add_argument('--review-output', type=Path, default=Path('data/review/ai_recommendation_rules_review.csv'))
    parser.add_argument('--dry-run', action='store_true')
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = import_ai_recommendation_dataset(args.input, args.output_csv, args.output_json, args.review_output)
    report = result['report']
    print(f"Lignes lues:              {report.total_lines}")
    print(f"Lignes importées:         {result['rules_count']}")
    print(f"Lignes envoyées en revue:  {result['review_count']}")
    print(f"Doublons détectés:         {report.duplicate_lines}")
    if report.anomalies:
        print('Anomalies:')
        for name, count in sorted(report.anomalies.items()):
            print(f'  - {name}: {count}')
    if args.dry_run:
        print('[DRY RUN] aucun fichier écrit.')
        return 0
    print(f"CSV normalisé:            {result['csv_path']}")
    print(f"JSON normalisé:           {result['json_path']}")
    print(f"Revue humaine:            {result['review_path']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
