#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deepforma.training.binary_ai_ml import BinaryAITrainingConfig, fit_binary_ai_ml  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("train_binary_ai_ml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Entraîne le modèle ML binaire IA from scratch")
    parser.add_argument("--train", type=Path, default=Path("data/training/binary_ai/train.parquet"))
    parser.add_argument("--validation", type=Path, default=Path("data/training/binary_ai/validation.parquet"))
    parser.add_argument("--test", type=Path, default=Path("data/training/binary_ai/test.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/binary_ai_ml"))
    parser.add_argument("--classifier", type=str, default="logistic")
    parser.add_argument("--word-min-df", type=int, default=2)
    parser.add_argument("--word-max-features", type=int, default=50_000)
    parser.add_argument("--char-min-df", type=int, default=2)
    parser.add_argument("--char-max-features", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold-mode", type=str, default="maximize_f1")
    parser.add_argument("--min-recall", type=float, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = BinaryAITrainingConfig(
        seed=args.seed,
        classifier=args.classifier,
        word_min_df=args.word_min_df,
        word_max_features=args.word_max_features,
        char_min_df=args.char_min_df,
        char_max_features=args.char_max_features,
        threshold_mode=args.threshold_mode,
        min_recall=args.min_recall,
    )
    train_frame = pd.read_parquet(args.train)
    validation_frame = pd.read_parquet(args.validation)
    test_frame = pd.read_parquet(args.test)
    artifacts = fit_binary_ai_ml(
        train_frame,
        validation_frame,
        test_frame,
        output_dir=args.output_dir,
        config=config,
    )
    LOGGER.info("Modèle ML entraîné: %s", artifacts.model_dir)
    LOGGER.info("Seuil retenu: %.4f", artifacts.threshold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

