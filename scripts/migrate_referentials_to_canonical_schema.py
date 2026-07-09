from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from referentials.referential_registry import (
    count_referential_skills_detail,
    normalize_referential_payload,
)


LEGACY_CHILDREN_KEYS = (
    'subskills', 'sous_competences', 'sous_compétences',
    'detected_skills', 'competencies', 'competences', 'compétences',
    'derived_skills', 'derived_competencies',
)


def migrate_file(path: Path, *, dry_run: bool = False) -> dict[str, Any]:
    report: dict[str, Any] = {
        'file': str(path),
        'status': 'unchanged',
        'before': {},
        'after': {},
        'errors': [],
    }
    try:
        raw = json.loads(path.read_text(encoding='utf-8'))
    except Exception as exc:
        report['status'] = 'error'
        report['errors'].append(f'JSON invalide: {exc}')
        return report

    if not isinstance(raw, dict):
        report['status'] = 'error'
        report['errors'].append('Le fichier ne contient pas un objet JSON')
        return report

    before_counts = count_referential_skills_detail(raw)
    report['before'] = before_counts

    has_legacy_keys = any(
        key in raw and isinstance(raw[key], list) and raw[key]
        for key in LEGACY_CHILDREN_KEYS
    )
    has_skills = 'skills' in raw and isinstance(raw.get('skills'), list)

    if not has_legacy_keys and has_skills:
        normalized = normalize_referential_payload(raw)
        after_counts = count_referential_skills_detail(normalized)
        if after_counts == before_counts:
            report['status'] = 'already_canonical'
            report['after'] = after_counts
            return report

    normalized = normalize_referential_payload(raw)
    after_counts = count_referential_skills_detail(normalized)
    report['after'] = after_counts

    if after_counts == before_counts and has_skills:
        report['status'] = 'already_canonical'
        return report

    report['status'] = 'migrated'

    if not dry_run:
        bak_path = path.with_suffix(path.suffix + '.bak')
        if not bak_path.exists():
            shutil.copy2(path, bak_path)
        path.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding='utf-8')

    return report


def validate_file(path: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        'file': str(path),
        'valid': True,
        'errors': [],
    }
    try:
        raw = json.loads(path.read_text(encoding='utf-8'))
    except Exception as exc:
        report['valid'] = False
        report['errors'].append(f'JSON invalide: {exc}')
        return report

    if not isinstance(raw, dict):
        report['valid'] = False
        report['errors'].append('Le fichier ne contient pas un objet JSON')
        return report

    skills = raw.get('skills')
    if skills is not None and not isinstance(skills, list):
        report['valid'] = False
        report['errors'].append('skills n\'est pas une liste')

    if isinstance(skills, list):
        for i, entry in enumerate(skills):
            if not isinstance(entry, dict):
                report['valid'] = False
                report['errors'].append(f'skills[{i}] n\'est pas un objet')
            else:
                children = entry.get('children')
                if children is not None and not isinstance(children, list):
                    report['valid'] = False
                    report['errors'].append(f'skills[{i}].children n\'est pas une liste')

    metadata = raw.get('metadata', {})
    if isinstance(metadata, dict):
        declared_count = metadata.get('skills_count')
        if declared_count is not None:
            actual = count_referential_skills_detail(raw)
            if declared_count != actual.get('skills_count'):
                report['valid'] = False
                report['errors'].append(
                    f'metadata.skills_count ({declared_count}) != total récursif ({actual["skills_count"]})'
                )

    has_legacy = any(
        key in raw and isinstance(raw[key], list) and raw[key]
        for key in LEGACY_CHILDREN_KEYS
    )
    if has_legacy and isinstance(skills, list) and skills:
        report['valid'] = False
        report['errors'].append('Contient des sous-compétences dans une ancienne clé non migrée')

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description='Migrer les référentiels vers le schéma canonique')
    parser.add_argument('--input-dir', default=str(PROJECT_ROOT / 'data' / 'referentials'),
                        help='Répertoire des référentiels')
    parser.add_argument('--check-only', action='store_true',
                        help='Vérifier sans migrer')
    parser.add_argument('--dry-run', action='store_true',
                        help='Simuler la migration sans écrire')
    parser.add_argument('--output-report', default=None,
                        help='Fichier de rapport JSON')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f'Répertoire introuvable: {input_dir}')
        sys.exit(1)

    json_files: list[Path] = []
    for p in sorted(input_dir.rglob('*.json')):
        name = p.name
        if '.metadata.' in name or name == 'index.json' or name.endswith('.bak'):
            continue
        json_files.append(p)

    reports: list[dict[str, Any]] = []
    has_errors = False

    for path in json_files:
        if args.check_only:
            result = validate_file(path)
            if not result['valid']:
                has_errors = True
                print(f'INVALIDE: {path}')
                for err in result['errors']:
                    print(f'  - {err}')
            else:
                print(f'OK: {path}')
        else:
            result = migrate_file(path, dry_run=args.dry_run)
            status = result['status']
            if status == 'migrated':
                print(f'MIGRÉ: {path}')
            elif status == 'already_canonical':
                print(f'DÉJÀ CANONIQUE: {path}')
            elif status == 'error':
                has_errors = True
                print(f'ERREUR: {path}: {"; ".join(result["errors"])}')
            else:
                print(f'INCHANGÉ: {path}')
        reports.append(result)

    if args.output_report:
        output_path = Path(args.output_report)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'Rapport écrit: {output_path}')

    total = len(reports)
    if args.check_only:
        invalid = sum(1 for r in reports if not r.get('valid', True))
        print(f'\nVérification: {total} fichiers, {invalid} invalides')
        if has_errors:
            print('Exécutez: make migrate-referentials-schema')
            sys.exit(1)
    else:
        migrated = sum(1 for r in reports if r.get('status') == 'migrated')
        print(f'\nMigration: {total} fichiers, {migrated} migrés')

    sys.exit(1 if has_errors else 0)


if __name__ == '__main__':
    main()
