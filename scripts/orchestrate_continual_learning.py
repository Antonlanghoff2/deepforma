#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
from pathlib import Path

from continual_learning.store import ContinualLearningStore

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('orchestrate_continual_learning')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Orchestrate the continual learning maintenance cycle')
    parser.add_argument('--db-path', type=Path, default=Path('data/continual_learning/continual_learning.sqlite3'))
    parser.add_argument('--review-limit', type=int, default=200)
    parser.add_argument('--review-output', type=Path, default=Path('data/continual_learning/review_queue.jsonl'))
    parser.add_argument('--export-output', type=Path, default=Path('data/continual_learning/exported_training.jsonl'))
    parser.add_argument('--min-approved-samples', type=int, default=int(os.getenv('CONTINUAL_MIN_APPROVED_SAMPLES', '500')))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    store = ContinualLearningStore(args.db_path)
    approved = store.list_annotations("validation_status IN ('approved', 'corrected', 'used_for_training')")
    subprocess.run(['python3', 'scripts/build_review_queue.py', '--limit', str(args.review_limit), '--output', str(args.review_output), '--db-path', str(args.db_path)], check=True)
    subprocess.run(['python3', 'scripts/export_continual_training_dataset.py', '--output', str(args.export_output), '--db-path', str(args.db_path), '--include-human-reviewed', '--include-france-travail-api'], check=True)
    if len(approved) < args.min_approved_samples:
        logger.info('Approved samples below threshold (%d < %d); skipping training trigger.', len(approved), args.min_approved_samples)
        return
    logger.info('Approved samples threshold reached (%d >= %d). Training can be triggered manually or by a separate GPU job.', len(approved), args.min_approved_samples)


if __name__ == '__main__':
    main()
