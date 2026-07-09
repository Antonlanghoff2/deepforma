from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from common.text import clean_text

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REFERENTIALS_DIR = PROJECT_ROOT / 'data' / 'referentials'
DEFAULT_INDEX_PATH = DEFAULT_REFERENTIALS_DIR / 'index.json'


@dataclass(frozen=True)
class ReferentialOption:
    id: str
    label: str
    type: str
    path: str | None
    record_id: str | None
    status: str
    source: str
    skill_count: int = 0
    is_selectable: bool = False
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            'id': self.id,
            'label': self.label,
            'type': self.type,
            'path': self.path,
            'record_id': self.record_id,
            'status': self.status,
            'source': self.source,
            'skill_count': self.skill_count,
            'is_selectable': self.is_selectable,
            'reason': self.reason,
        }


EXCLUDED_PATTERNS = ('.metadata.', 'index.json')


def _json_paths(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    paths = []
    for p in directory.glob('*.json'):
        name = p.name
        if any(pattern in name for pattern in EXCLUDED_PATTERNS):
            continue
        paths.append(p)
    return sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)


def _load_referential_metadata(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(payload, dict):
            return {}
        return payload
    except Exception as exc:
        LOGGER.warning('Impossible de lire le référentiel %s : %s', path, exc)
        return {}


SKILL_LIKE_KEYS = ('skills', 'competencies', 'competences', 'criteria',
                   'official_skills', 'detected_skills', 'subskills',
                   'derived_competencies', 'blocks', 'blocs')


def count_referential_skills(payload: dict[str, Any]) -> int:
    if not isinstance(payload, dict):
        return 0
    for key in SKILL_LIKE_KEYS:
        val = payload.get(key)
        if isinstance(val, list):
            count = len(val)
            if count > 0:
                return count
    return 0


def _skill_count(payload: dict[str, Any]) -> int:
    return count_referential_skills(payload)


def _build_option_from_json(path: Path) -> ReferentialOption | None:
    payload = _load_referential_metadata(path)
    if not payload:
        return ReferentialOption(
            id=path.stem,
            label=path.stem,
            type='unknown',
            path=str(path),
            record_id=None,
            status='invalid',
            source='json_file',
            skill_count=0,
            is_selectable=False,
            reason='Fichier JSON invalide',
        )

    referential_id = clean_text(payload.get('referential_id') or path.stem)
    title = (
        clean_text(payload.get('title'))
        or clean_text(payload.get('metadata', {}).get('title', ''))
        or clean_text(payload.get('document', {}).get('title', ''))
        or clean_text(payload.get('document', {}).get('file_name', ''))
        or path.stem
    )
    skill_count = _skill_count(payload)
    ref_type = clean_text(payload.get('type') or payload.get('metadata', {}).get('type', 'certification')) or 'certification'
    doc = payload.get('document', {})
    source_pdf = clean_text(doc.get('source_path', '') if isinstance(doc, dict) else '')
    record_id = clean_text(doc.get('sha256', '') if isinstance(doc, dict) else '') or None
    if not record_id and isinstance(doc, dict):
        record_id = clean_text(doc.get('id', '')) or None

    status: str
    reason: str | None = None
    is_selectable: bool = False

    if skill_count > 0:
        status = 'active'
        is_selectable = True
        reason = None
    elif not payload:
        status = 'invalid'
        is_selectable = False
        reason = 'Fichier JSON invalide'
    else:
        status = 'empty'
        is_selectable = False
        reason = 'Aucune compétence détectée'

    return ReferentialOption(
        id=referential_id,
        label=title,
        type=ref_type,
        path=str(path),
        record_id=record_id,
        status=status,
        source='imported_pdf' if 'imported' in str(path) else 'json_file',
        skill_count=skill_count,
        is_selectable=is_selectable,
        reason=reason,
    )


def _resolve_path(relative_or_absolute: str) -> Path:
    candidate = Path(relative_or_absolute)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def _has_skills_format(payload: dict[str, Any]) -> bool:
    return 'skills' in payload and isinstance(payload['skills'], list)


def _usable_for_comparison(option: ReferentialOption) -> bool:
    return option.is_selectable


def _collect_options(directory: Path, seen_ids: set[str], result: list[ReferentialOption], *, usable_only: bool) -> None:
    for path in _json_paths(directory):
        option = _build_option_from_json(path)
        if option and option.id not in seen_ids:
            if usable_only and not _usable_for_comparison(option):
                continue
            seen_ids.add(option.id)
            result.append(option)


def list_available_referentials(*, referentials_dir: str | Path | None = None, include_inactive: bool = False) -> list[ReferentialOption]:
    directory = Path(referentials_dir or DEFAULT_REFERENTIALS_DIR)
    if not directory.is_dir():
        LOGGER.warning('Répertoire des référentiels introuvable : %s', directory)
        return []

    seen_ids: set[str] = set()
    result: list[ReferentialOption] = []
    usable_only = not include_inactive

    index_path = directory / 'index.json'
    if index_path.is_file():
        try:
            entries = json.loads(index_path.read_text(encoding='utf-8'))
            if isinstance(entries, list):
                for entry in entries:
                    raw_path = entry.get('path', '')
                    if not raw_path:
                        continue
                    resolved = _resolve_path(raw_path)
                    if not resolved.is_file():
                        LOGGER.warning('Référentiel indexé introuvable : %s', resolved)
                        continue
                    option = _build_option_from_json(resolved)
                    if option and option.id not in seen_ids:
                        if usable_only and not _usable_for_comparison(option):
                            continue
                        seen_ids.add(option.id)
                        result.append(option)
        except Exception as exc:
            LOGGER.warning('Erreur de lecture de %s : %s', index_path, exc)

    _collect_options(directory, seen_ids, result, usable_only=usable_only)
    imported_dir = directory / 'imported'
    if imported_dir.is_dir():
        _collect_options(imported_dir, seen_ids, result, usable_only=usable_only)

    return result


def get_referential_option(referential_id: str, *, referentials_dir: str | Path | None = None) -> ReferentialOption | None:
    options = list_available_referentials(referentials_dir=referentials_dir, include_inactive=True)
    for option in options:
        if option.id == referential_id or option.path and Path(option.path).stem == referential_id:
            return option
        if option.record_id == referential_id:
            return option
    return None


def resolve_referential_path(referential_id: str, *, referentials_dir: str | Path | None = None) -> str | None:
    option = get_referential_option(referential_id, referentials_dir=referentials_dir)
    if option is None:
        return None
    return option.path


def convert_imported_to_skills_format(imported_payload: dict[str, Any]) -> dict[str, Any]:
    competencies = imported_payload.get('competencies', [])
    if not isinstance(competencies, list):
        return {}

    doc = imported_payload.get('document', {})
    doc_title = clean_text(doc.get('title', '') if isinstance(doc, dict) else '') or ''
    skills: list[dict[str, Any]] = []
    for comp in competencies:
        if not isinstance(comp, dict):
            continue
        code = clean_text(comp.get('code', ''))
        label = (clean_text(comp.get('official_label', ''))
                 or clean_text(comp.get('normalized_label', ''))
                 or clean_text(comp.get('label', ''))
                 or clean_text(comp.get('name', ''))
                 or clean_text(comp.get('title', ''))
                 or clean_text(comp.get('competence', ''))
                 or clean_text(comp.get('intitule', ''))
                 or clean_text(comp.get('libelle', ''))
                 or clean_text(comp.get('nom', ''))
                 or clean_text(comp.get('description', '')))
        if not label:
            continue
        normalized = clean_text(comp.get('normalized_label', ''))
        block_code = clean_text(comp.get('block_code', ''))
        activity_code = clean_text(comp.get('activity_code', ''))
        derived = comp.get('derived_skills', [])
        if isinstance(derived, list):
            technical_keywords = [clean_text(s) for s in derived if isinstance(s, str) and clean_text(s)]
        else:
            technical_keywords = []
        source_pages = comp.get('source_pages', [])
        source_page = int(source_pages[0]) if isinstance(source_pages, list) and source_pages else 0
        if not label:
            continue
        skill = {
            'id': code or f'imported_{len(skills)}',
            'block': block_code,
            'block_name': '',
            'activity': activity_code,
            'code': code,
            'label': label,
            'official_description': label,
            'normalized_label': normalized or label,
            'category': '',
            'subcategory': '',
            'technical_keywords': technical_keywords,
            'aliases': [],
            'source_page': source_page,
            'active': True,
        }
        skills.append(skill)

    blocks = imported_payload.get('blocks', [])
    block_names: dict[str, str] = {}
    if isinstance(blocks, list):
        for b in blocks:
            bc = clean_text(b.get('code', ''))
            bl = clean_text(b.get('label', ''))
            if bc and bl:
                block_names[bc] = bl
    for skill in skills:
        if skill['block'] in block_names:
            skill['block_name'] = block_names[skill['block']]

    referential_id = clean_text(imported_payload.get('referential_id')) or doc.get('sha256', '') or clean_text(doc.get('file_name', ''))
    title = clean_text(imported_payload.get('title')) or doc_title or referential_id

    return {
        'referential_id': referential_id,
        'title': title,
        'version': imported_payload.get('schema_version', '1.0'),
        'skills': skills,
        'metadata': {
            'source': 'imported_pdf',
            'source_path': clean_text(doc.get('source_path', '') if isinstance(doc, dict) else ''),
            'file_name': clean_text(doc.get('file_name', '') if isinstance(doc, dict) else ''),
            'record_id': clean_text(doc.get('sha256', '') if isinstance(doc, dict) else ''),
        },
    }


def ensure_loadable_path(option: ReferentialOption) -> str | None:
    if not option.path:
        return None
    path = Path(option.path)
    if not path.is_file():
        return None
    payload = _load_referential_metadata(path)
    if not payload:
        return None
    if _has_skills_format(payload):
        return option.path
    converted = convert_imported_to_skills_format(payload)
    if converted.get('skills'):
        converted_path = path.with_suffix('.converted.json')
        try:
            converted_path.write_text(json.dumps(converted, ensure_ascii=False, indent=2), encoding='utf-8')
            return str(converted_path)
        except Exception as exc:
            LOGGER.warning('Impossible d\'écrire la conversion %s : %s', converted_path, exc)
            return None
    if option.skill_count > 0:
        return option.path
    return None
