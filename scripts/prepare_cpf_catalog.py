from __future__ import annotations

import argparse
import logging
from pathlib import Path

from deepforma.cpf.prepare import prepare_catalog


LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Nettoie et normalise le catalogue CPF")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--config", type=Path, default=Path("config/cpf_columns.yaml"))
    parser.add_argument("--chunksize", type=int, default=25_000)
    parser.add_argument("--sample-limit", type=int, default=1_000)
    parser.add_argument("--similarity-threshold", type=float, default=0.96)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = build_parser().parse_args()
    result = prepare_catalog(
        args.input,
        args.output_dir,
        config_path=args.config,
        chunksize=args.chunksize,
        sample_limit=args.sample_limit,
        similarity_threshold=args.similarity_threshold,
    )
    stats = result["stats"]
    LOGGER.info(
        "Préparation terminée: lues=%s conservées=%s rejetées=%s doublons_exacts=%s doublons_proches=%s enrichies=%s",
        stats.rows_read,
        stats.rows_kept,
        stats.rows_rejected,
        stats.exact_duplicates,
        stats.near_duplicates,
        stats.enriched_rows,
    )


if __name__ == "__main__":
    main()

