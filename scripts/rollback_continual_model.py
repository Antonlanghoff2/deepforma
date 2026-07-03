#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import subprocess
from pathlib import Path

from continual_learning.model_registry import load_registry, promote_model_version

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('rollback_continual_model')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Rollback the continual learning production model')
    parser.add_argument('--to-version', default=None)
    parser.add_argument('--registry-path', type=Path, default=Path('models/skill-extractor/registry.json'))
    parser.add_argument('--production-link', type=Path, default=Path('models/skill-extractor/production'))
    parser.add_argument('--service-name', default='deepforma')
    parser.add_argument('--health-url', default='http://127.0.0.1:8001/health')
    return parser


def main() -> None:
    args = build_parser().parse_args()
    registry = load_registry(args.registry_path)
    versions = registry.get('versions', [])
    target = None
    if args.to_version:
        for item in reversed(versions):
            if item.get('version') == args.to_version:
                target = item
                break
    else:
        for item in reversed(versions):
            if item.get('state') in {'production', 'candidate'}:
                target = item
                break
    if not target:
        raise SystemExit('No rollback target found.')
    promote_model_version(version_dir=target['model_dir'], production_link=args.production_link)
    subprocess.run(['systemctl', 'restart', args.service_name], check=True)
    subprocess.run(['curl', '--fail', '--silent', '--show-error', args.health_url], check=True)
    logger.info('Rollback successful: %s', target['version'])


if __name__ == '__main__':
    main()
