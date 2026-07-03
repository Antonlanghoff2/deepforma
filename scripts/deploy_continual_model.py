#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

from continual_learning.model_registry import promote_model_version
from transformers import AutoModelForTokenClassification, AutoTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('deploy_continual_model')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Deploy a continual learning model candidate atomically')
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--production-link', type=Path, default=Path('models/skill-extractor/production'))
    parser.add_argument('--service-name', default='deepforma')
    parser.add_argument('--health-url', default='http://127.0.0.1:8001/health')
    parser.add_argument('--dry-run', action='store_true')
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model_dir = args.model_dir
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    AutoModelForTokenClassification.from_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    if args.dry_run:
        logger.info('Dry-run: model validated, no deployment performed.')
        return

    promote_model_version(version_dir=model_dir, production_link=args.production_link)
    subprocess.run(['systemctl', 'restart', args.service_name], check=True)
    subprocess.run(['curl', '--fail', '--silent', '--show-error', args.health_url], check=True)
    logger.info('Deployment successful: %s', model_dir)


if __name__ == '__main__':
    main()
