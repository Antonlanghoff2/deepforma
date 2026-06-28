from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from data.cpf_loader import inspect_cpf_source, resolve_cpf_source
from deepforma.cpf.io import json_dump


LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Inspecte le catalogue CPF V3 ou ses variantes historiques')
    parser.add_argument('--input', type=Path, default=None, help='Fichier CPF à inspecter')
    parser.add_argument('--sheet', type=str, default=None, help='Feuille Excel à utiliser')
    parser.add_argument('--config', type=Path, default=Path('config/cpf_columns.yaml'))
    parser.add_argument('--output', type=Path, default=Path('data/processed/reports/cpf_v3_inspection.json'))
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s %(message)s')
    args = build_parser().parse_args()
    source = resolve_cpf_source(args.input)
    report = inspect_cpf_source(source, sheet_name=args.sheet, config_path=args.config)
    payload = {
        'path': report.path,
        'resolved_path': report.resolved_path,
        'sheet_names': report.sheet_names,
        'selected_sheet': report.selected_sheet,
        'row_count': report.row_count,
        'column_count': report.column_count,
        'columns': report.columns,
        'non_null_by_column': report.non_null_by_column,
        'examples': report.examples,
        'candidate_columns': report.candidate_columns,
        'missing_required_columns': report.missing_required_columns,
        'warnings': report.warnings,
    }
    json_dump(args.output, payload)
    LOGGER.info("Rapport d'inspection écrit dans %s", args.output)
    LOGGER.info('Feuille utilisée: %s', report.selected_sheet)
    LOGGER.info('Colonnes candidates: %s', json.dumps(report.candidate_columns, ensure_ascii=False))


if __name__ == '__main__':
    main()
