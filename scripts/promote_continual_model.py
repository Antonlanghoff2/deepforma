#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

from continual_learning.model_registry import promote_model_version, registry_root, update_registry
from transformers import AutoModelForTokenClassification, AutoTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('promote_continual_model')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Promote a continual learning model candidate')
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--version', required=True)
    parser.add_argument('--previous-version', default=None)
    parser.add_argument('--git-hash', default='unknown')
    parser.add_argument('--base-model', default=None)
    parser.add_argument('--metrics-json', type=Path, default=None)
    parser.add_argument('--dataset-hashes-json', type=Path, default=None)
    parser.add_argument('--taxonomy-json', type=Path, default=None)
    parser.add_argument('--referential-json', type=Path, default=None)
    parser.add_argument('--registry-path', type=Path, default=Path('models/skill-extractor/registry.json'))
    parser.add_argument('--production-link', type=Path, default=Path('models/skill-extractor/production'))
    return parser


def _load_json(path: Path | None, default):
    if path and path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    return default


def main() -> None:
    args = build_parser().parse_args()
    model_dir = args.model_dir
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    AutoModelForTokenClassification.from_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    metrics = _load_json(args.metrics_json, {})
    dataset_hashes = _load_json(args.dataset_hashes_json, {})
    taxonomy = _load_json(args.taxonomy_json, {})
    referential = _load_json(args.referential_json, {})
    registry_root(args.registry_path.parent)
    update_registry(
        version=args.version,
        git_hash=args.git_hash,
        base_model=args.base_model,
        dataset_hashes=dataset_hashes,
        example_count=int(metrics.get('example_count', 0) or 0),
        metrics=metrics,
        taxonomy=taxonomy,
        referential=referential,
        state='production',
        previous_version=args.previous_version,
        model_dir=str(model_dir),
        registry_path=args.registry_path,
    )
    promote_model_version(version_dir=model_dir, production_link=args.production_link)
    logger.info('Model promoted: %s', model_dir)


if __name__ == '__main__':
    main()
