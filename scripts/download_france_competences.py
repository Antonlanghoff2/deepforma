#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / 'src'):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from data_sources.france_competences.client import FranceCompetencesClient


LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Télécharge les dernières ressources officielles RNCP/RS depuis data.gouv.fr.')
    parser.add_argument('--output-dir', type=Path, default=Path('data/raw/france_competences'))
    parser.add_argument('--dataset-slug', default=None)
    parser.add_argument('--include-rncp', action='store_true', default=True)
    parser.add_argument('--no-rncp', dest='include_rncp', action='store_false')
    parser.add_argument('--include-rs', action='store_true', default=True)
    parser.add_argument('--no-rs', dest='include_rs', action='store_false')
    parser.add_argument('--force', action='store_true', default=False)
    parser.add_argument('--timeout', type=int, default=None)
    parser.add_argument('--dry-run', action='store_true', default=False)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s %(message)s')
    args = build_parser().parse_args()
    client = FranceCompetencesClient(
        dataset_slug=args.dataset_slug,
        timeout=args.timeout,
        include_rncp=args.include_rncp,
        include_rs=args.include_rs,
        force_download=args.force,
    )
    metadata = client.fetch_dataset_metadata()
    resources = client.list_resources()
    selected: list[dict[str, object]] = []
    for kind, include in (('RNCP', args.include_rncp), ('RS', args.include_rs)):
        if not include:
            continue
        resource = client.select_latest_resource(resources, include_rncp=(kind == 'RNCP'), include_rs=(kind == 'RS'))
        LOGGER.info('Ressource sélectionnée pour %s: %s', kind, resource.get('title'))
        if args.dry_run:
            selected.append(
                {
                    'resource_id': resource.get('id'),
                    'resource_title': resource.get('title'),
                    'resource_url': resource.get('url'),
                    'format': resource.get('format'),
                    'checksum': (resource.get('checksum') or {}).get('value') if isinstance(resource.get('checksum'), dict) else None,
                    'size': resource.get('filesize'),
                    'source_last_modified': resource.get('last_modified'),
                    'local_path': None,
                    'verification': {'ok': True},
                }
            )
            continue
        local_path, verification = client.download_resource(resource, args.output_dir, force=args.force)
        selected.append(
            {
                'resource_id': resource.get('id'),
                'resource_title': resource.get('title'),
                'resource_url': resource.get('url'),
                'format': resource.get('format'),
                'checksum': verification.sha256,
                'size': verification.size,
                'source_last_modified': resource.get('last_modified'),
                'local_path': str(local_path),
                'verification': verification.to_dict(),
            }
        )
    manifest_path = args.output_dir / 'manifest.json'
    manifest = client.write_manifest(
        manifest_path,
        dataset_metadata=metadata,
        resources=selected,
        downloaded_at=datetime.now(timezone.utc).isoformat(),
        parser_version='1.0',
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

