from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from data.cpf_loader import prepare_cpf_v3_dataset, resolve_cpf_source


LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Prépare et normalise le catalogue CPF V3')
    parser.add_argument('--input', type=Path, default=None, help='Fichier CPF source')
    parser.add_argument('--output-dir', type=Path, default=Path('data/processed'))
    parser.add_argument('--sheet', type=str, default=None, help='Feuille Excel à utiliser')
    parser.add_argument('--config', type=Path, default=Path('config/cpf_columns.yaml'))
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s %(message)s')
    args = build_parser().parse_args()
    source = resolve_cpf_source(args.input)
    prepared = prepare_cpf_v3_dataset(source, args.output_dir, sheet_name=args.sheet, config_path=args.config)
    LOGGER.info('Catalogue préparé depuis %s', source)
    LOGGER.info('Lignes conservées: %s', len(prepared.frame))
    LOGGER.info('Doublons détectés: %s', len(prepared.duplicates))
    LOGGER.info('Fichiers générés: %s', json.dumps(prepared.report.get('output_files', {}), ensure_ascii=False))


if __name__ == '__main__':
    main()
