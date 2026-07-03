#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from referential_import import ReferentialImportService
from referential_import.store import ReferentialImportStore


LOGGER = logging.getLogger('import_referential')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Importeur spécialisé des référentiels RNCP / France Compétences')
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true', help='Analyse sans persister dans la base')
    parser.add_argument('--approve', action='store_true', help='Valide et persiste l import')
    parser.add_argument('--report', type=Path, default=Path('reports/referential_import.json'))
    parser.add_argument('--output', type=Path, default=Path('data/referentials/imported/'))
    parser.add_argument('--store-path', type=Path, default=Path('data/referentials/referential_imports.sqlite3'))
    parser.add_argument('--validated-by', type=str, default='human_review')
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    args = build_parser().parse_args()
    store = ReferentialImportStore(args.store_path)
    service = ReferentialImportService(store=store, output_dir=args.output)
    analysis = service.analyze(args.input)
    report = analysis['report']
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding='utf-8')
    LOGGER.info('Rapport d import écrit dans %s', args.report)
    LOGGER.info('Blocs=%s Activités=%s Compétences=%s Critères=%s Dérivées=%s', report.blocks, report.activities, report.competencies, report.criteria, report.derived_skills)

    if args.approve:
        output_path = service.approve(analysis, validated_by=args.validated_by)
        LOGGER.info('Import approuvé: %s', output_path)
    else:
        LOGGER.info('Analyse terminée en mode dry-run: aucune écriture de production')


if __name__ == '__main__':
    main()
