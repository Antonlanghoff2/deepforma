#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deepforma.training.binary_ai_dataset import (  # noqa: E402
    DEFAULT_INPUTS,
    build_binary_ai_dataset,
    group_stratified_split,
    discover_default_inputs,
    write_dataset_outputs,
    write_split_outputs,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("build_binary_ai_dataset")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Construit le dataset binaire IA/non-IA from scratch")
    parser.add_argument("--inputs", nargs="*", default=None, help="Fichiers source (CSV, XLSX, JSONL, Parquet)")
    parser.add_argument("--output-dataset", type=Path, default=Path("data/processed/binary_ai/dataset.parquet"))
    parser.add_argument("--audit-output", type=Path, default=Path("reports/binary_ai/dataset_audit.json"))
    parser.add_argument("--duplicates-output", type=Path, default=Path("reports/binary_ai/dataset_duplicates.csv"))
    parser.add_argument("--conflicts-output", type=Path, default=Path("reports/binary_ai/dataset_conflicts.csv"))
    parser.add_argument("--splits-output-dir", type=Path, default=Path("data/training/binary_ai"))
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    inputs = [Path(path) for path in args.inputs] if args.inputs else discover_default_inputs("data")
    if not inputs:
        inputs = list(DEFAULT_INPUTS)
    LOGGER.info("Sources: %s", ", ".join(str(path) for path in inputs))
    frame, audit, duplicates, conflicts = build_binary_ai_dataset(inputs)
    write_dataset_outputs(
        frame,
        audit,
        duplicates,
        conflicts,
        dataset_path=args.output_dataset,
        audit_path=args.audit_output,
        duplicates_path=args.duplicates_output,
        conflicts_path=args.conflicts_output,
    )
    splits, manifest = group_stratified_split(frame, seed=args.seed)
    write_split_outputs(splits, manifest, output_dir=args.splits_output_dir)
    LOGGER.info("Dataset final: %d lignes | IA=%d | non-IA=%d", audit.rows_kept, audit.positives, audit.negatives)
    LOGGER.info("Splits: %s", manifest.sizes)
    LOGGER.info("Manifest: %s", args.splits_output_dir / "split_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

