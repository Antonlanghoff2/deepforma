#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deepforma.training.binary_ai_textcnn import BinaryAITextCNNConfig, fit_binary_ai_textcnn  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("train_binary_ai_textcnn")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Entraîne le TextCNN binaire IA from scratch")
    parser.add_argument("--train", type=Path, default=Path("data/training/binary_ai/train.parquet"))
    parser.add_argument("--validation", type=Path, default=Path("data/training/binary_ai/validation.parquet"))
    parser.add_argument("--test", type=Path, default=Path("data/training/binary_ai/test.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/binary_ai_textcnn"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vocab-size", type=int, default=30_000)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--num-filters", type=int, default=128)
    parser.add_argument("--dense-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--threshold-mode", type=str, default="maximize_f1")
    parser.add_argument("--min-recall", type=float, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = BinaryAITextCNNConfig(
        seed=args.seed,
        vocab_size=args.vocab_size,
        max_length=args.max_length,
        embedding_dim=args.embedding_dim,
        num_filters=args.num_filters,
        dense_dim=args.dense_dim,
        dropout=args.dropout,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        patience=args.patience,
        grad_clip=args.grad_clip,
        device=args.device,
        threshold_mode=args.threshold_mode,
        min_recall=args.min_recall,
    )
    train_frame = pd.read_parquet(args.train)
    validation_frame = pd.read_parquet(args.validation)
    test_frame = pd.read_parquet(args.test)
    artifacts = fit_binary_ai_textcnn(
        train_frame,
        validation_frame,
        test_frame,
        output_dir=args.output_dir,
        config=config,
    )
    LOGGER.info("TextCNN entraîné: %s", artifacts.model_dir)
    LOGGER.info("Seuil retenu: %.4f", artifacts.threshold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

