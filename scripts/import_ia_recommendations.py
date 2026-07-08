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

from data_sources.ia_recommendations import (
    IARecommendationQualityReport,
    load_ia_recommendations_csv,
    validate_ia_recommendations,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Importe le dataset de recommandations IA.')
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('data/processed/ia_recommendations'))
    parser.add_argument('--quality-report', type=Path, default=None)
    parser.add_argument('--write', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    return parser


def main() -> None:
    args = build_parser().parse_args()
    records, report = load_ia_recommendations_csv(args.input)

    print(f'Lignes lues:                {report.total_lines}')
    print(f'Lignes valides:             {report.valid_lines}')
    print(f'Lignes rejetees:            {report.rejected_lines}')
    if report.empty_keyword:
        print(f'  - mot-cle vide:           {report.empty_keyword}')
    if report.empty_recommendation:
        print(f'  - recommandation vide:    {report.empty_recommendation}')
    if report.ambiguous_lines:
        print(f'  - lignes ambigues:        {report.ambiguous_lines}')
    if report.exact_duplicates:
        print(f'Doublons exacts:            {report.exact_duplicates}')
    if report.normalized_duplicates:
        print(f'Doublons normalises:        {report.normalized_duplicates}')
    if report.default_rules:
        print(f'Regles par defaut:          {report.default_rules}')
    print(f'Recommandations actives:    {sum(1 for r in records if r.get("is_active"))}')

    if report.rejected_lines > 0:
        print(f'\nATTENTION: {report.rejected_lines} lignes rejetees.')
        for s in report.rejected_samples[:5]:
            print(f'  Ligne {s["line"]}: {s["reason"]} -> {s["preview"][:80]}')

    if report.exact_duplicates or report.normalized_duplicates:
        print(f'\nATTENTION: {report.exact_duplicates + report.normalized_duplicates} doublons detectes.')

    qa_report_path = args.quality_report or args.output_dir / 'quality_report.json'
    if args.dry_run or not args.write:
        print('\n[DRY RUN] Aucun fichier ecrit.')
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    qa_report_path.parent.mkdir(parents=True, exist_ok=True)

    qa_data = {
        'import_date': report.valid_lines > 0 and next(iter(records), {}).get('created_at', ''),
        'total_lines': report.total_lines,
        'valid_lines': report.valid_lines,
        'rejected_lines': report.rejected_lines,
        'empty_keyword': report.empty_keyword,
        'empty_recommendation': report.empty_recommendation,
        'exact_duplicates': report.exact_duplicates,
        'normalized_duplicates': report.normalized_duplicates,
        'ambiguous_lines': report.ambiguous_lines,
        'default_rules': report.default_rules,
        'rejected_samples': report.rejected_samples[:20],
        'duplicate_groups': report.duplicate_groups[:10],
    }
    qa_report_path.write_text(json.dumps(qa_data, ensure_ascii=False, indent=2), encoding='utf-8')

    import pandas as pd
    df = pd.DataFrame(records)
    parquet_path = args.output_dir / 'recommendations.parquet'
    df.to_parquet(parquet_path, index=False)
    print(f'\nFichier Parquet ecrit:      {parquet_path}')
    print(f'Rapport qualite ecrit:      {qa_report_path}')

    clean_csv_path = args.output_dir / 'recommendations_clean.csv'
    df.to_csv(clean_csv_path, index=False, encoding='utf-8-sig')
    print(f'Fichier CSV nettoye:        {clean_csv_path}')


if __name__ == '__main__':
    main()
