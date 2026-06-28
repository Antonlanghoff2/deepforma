from __future__ import annotations

import argparse
import logging
from pathlib import Path

from deepforma.cpf.io import json_dump
from deepforma.cpf.schema import inspect_catalog, save_schema_report


LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspecte le schéma du catalogue CPF")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config/cpf_columns.yaml"))
    parser.add_argument("--output", type=Path, default=Path("data/reports/cpf_schema_report.json"))
    parser.add_argument("--chunksize", type=int, default=50_000)
    parser.add_argument("--sample-limit", type=int, default=5_000)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = build_parser().parse_args()
    report = inspect_catalog(
        args.input,
        args.config,
        chunksize=args.chunksize,
        sample_limit=args.sample_limit,
    )
    save_schema_report(report, args.output)
    LOGGER.info("Rapport de schéma écrit dans %s", args.output)
    LOGGER.info("Colonnes détectées: %s", report.get("resolved_columns", {}))


if __name__ == "__main__":
    main()

