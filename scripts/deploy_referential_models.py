#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Déploie les modèles référentiels vers un serveur cible')
    parser.add_argument('--source-root', type=Path, default=Path('models'))
    parser.add_argument('--target-root', type=Path, default=Path('/opt/deepforma/models'))
    parser.add_argument('--version', type=str, default='current')
    parser.add_argument('--section-model', type=str, default='referential-section-classifier')
    parser.add_argument('--ner-model', type=str, default='referential-skill-ner')
    return parser


def _rsync(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['rsync', '-a', '--delete', f'{source}/', f'{target}/'], check=True)


def _link_current(target_dir: Path, source_dir: Path) -> None:
    link = target_dir / 'current'
    if link.exists() or link.is_symlink():
        if link.is_dir() and not link.is_symlink():
            shutil.rmtree(link)
        else:
            link.unlink()
    link.symlink_to(source_dir.name)


def main() -> None:
    args = build_parser().parse_args()
    deployments = [args.section_model, args.ner_model]
    for name in deployments:
        source = args.source_root / name / args.version
        if not source.exists():
            raise FileNotFoundError(f'Modèle introuvable: {source}')
        target = args.target_root / name / args.version
        _rsync(source, target)
        _link_current(args.target_root / name, target)


if __name__ == '__main__':
    main()
