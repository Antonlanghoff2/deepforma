from __future__ import annotations

import hashlib
import json
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from common.text import clean_text

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REFERENTIALS_DIR = PROJECT_ROOT / 'data' / 'referentials'
DEFAULT_INDEX_PATH = DEFAULT_REFERENTIALS_DIR / 'index.json'

CANONICAL_SCHEMA_VERSION = '1.0'

CHILDREN_SOURCE_KEYS = (
    'children', 'subskills', 'sous_competences', 'sous_compétences',
    'detected_skills', 'competencies', 'competences', 'compétences',
    'derived_skills', 'derived_competencies', 'official_skills',
    'blocks', 'blocs', 'criteria',
)

SKILL_LIKE_KEYS = ('skills', 'competencies', 'competences', 'criteria',
                   'official_skills', 'detected_skills', 'subskills',
                   'derived_competencies', 'derived_skills',
                   'sous_competences', 'sous_compétences', 'compétences',
                   'blocks', 'blocs')


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
    official_skills_count: int = 0
    subskills_count: int = 0
    exploitable_skills_count: int = 0

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
            'official_skills_count': self.official_skills_count,
            'subskills_count': self.subskills_count,
            'exploitable_skills_count': self.exploitable_skills_count,
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


def _generate_id(prefix: str = 'sk', index: int = 0) -> str:
    return f'{prefix}_{uuid.uuid4().hex[:8]}_{index}'


def _extract_label(entry: Any) -> str:
    if isinstance(entry, str):
        return clean_text(entry)
    if not isinstance(entry, dict):
        return ''
    for key in ('label', 'official_label', 'normalized_label', 'name', 'title',
                'competence', 'intitule', 'libelle', 'nom', 'description',
                'canonical_label', 'surface_form', 'text', 'code'):
        val = clean_text(entry.get(key, ''))
        if val:
            return val
    return ''


def normalize_skill_entry(entry: Any, index: int = 0, parent_id: str | None = None) -> dict[str, Any] | None:
    if isinstance(entry, str):
        label = clean_text(entry)
        if not label:
            return None
        return {
            'id': _generate_id('sub', index),
            'label': label,
            'type': 'subskill',
            'description': '',
            'aliases': [],
            'category': None,
            'block': None,
            'source_page': None,
            'source_text': '',
            'confidence': 1.0,
            'status': 'pending',
            'children': [],
            'parent_id': parent_id,
        }
    if not isinstance(entry, dict):
        return None
    label = _extract_label(entry)
    if not label:
        entry_id = clean_text(entry.get('id', ''))
        if entry_id:
            label = entry_id
        else:
            return None
    entry_type = clean_text(entry.get('type', ''))
    if not entry_type:
        if parent_id is not None:
            entry_type = 'subskill'
        else:
            entry_type = 'official_skill'
    children_raw = _collect_children(entry)
    children: list[dict[str, Any]] = []
    for ci, child in enumerate(children_raw):
        normalized_child = normalize_skill_entry(child, index=ci, parent_id=None)
        if normalized_child is not None:
            normalized_child['type'] = 'subskill'
            children.append(normalized_child)
    skill_id = clean_text(entry.get('id', '')) or _generate_id('sk', index)
    source_pages = entry.get('source_pages', [])
    source_page = None
    if isinstance(source_pages, list) and source_pages:
        try:
            source_page = int(source_pages[0])
        except (TypeError, ValueError):
            source_page = None
    return {
        'id': skill_id,
        'label': label,
        'type': entry_type,
        'description': clean_text(entry.get('official_description', '') or entry.get('description', '')),
        'aliases': list(entry.get('aliases', []) or []),
        'category': entry.get('category'),
        'block': entry.get('block') or entry.get('block_code'),
        'source_page': source_page,
        'source_text': clean_text(entry.get('source_text', '')),
        'confidence': float(entry.get('confidence', 1.0)),
        'status': clean_text(entry.get('status', '')) or 'pending',
        'children': children,
        'parent_id': parent_id,
    }


def _collect_children(entry: dict[str, Any]) -> list[Any]:
    for key in CHILDREN_SOURCE_KEYS:
        val = entry.get(key)
        if isinstance(val, list) and val:
            return val
    return []


def normalize_referential_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {'schema_version': CANONICAL_SCHEMA_VERSION, 'skills': [], 'metadata': _empty_metadata()}

    raw_skills = payload.get('skills')
    skills: list[dict[str, Any]] = []

    if isinstance(raw_skills, list):
        for si, item in enumerate(raw_skills):
            normalized = normalize_skill_entry(item, index=si)
            if normalized is not None:
                skills.append(normalized)
    elif isinstance(raw_skills, dict):
        normalized = normalize_skill_entry(raw_skills, index=0)
        if normalized is not None:
            skills.append(normalized)

    if not skills:
        orphan_children: list[Any] = []
        for key in CHILDREN_SOURCE_KEYS:
            val = payload.get(key)
            if isinstance(val, list) and val:
                orphan_children.extend(val)
        if orphan_children:
            children_normalized: list[dict[str, Any]] = []
            for ci, child in enumerate(orphan_children):
                normalized_child = normalize_skill_entry(child, index=ci)
                if normalized_child is not None:
                    normalized_child['type'] = 'subskill'
                    children_normalized.append(normalized_child)
            if children_normalized:
                parent_id = _generate_id('grp', 0)
                group = {
                    'id': parent_id,
                    'label': 'Compétences extraites du référentiel',
                    'type': 'generated_group',
                    'description': '',
                    'aliases': [],
                    'category': None,
                    'block': None,
                    'source_page': None,
                    'source_text': '',
                    'confidence': 1.0,
                    'status': 'pending',
                    'children': children_normalized,
                    'parent_id': None,
                }
                skills.append(group)

    counts = _count_from_skills(skills)
    metadata = dict(payload.get('metadata', {}) or {})
    metadata.update(counts)

    result = dict(payload)
    result['schema_version'] = payload.get('schema_version') or CANONICAL_SCHEMA_VERSION
    result['skills'] = skills
    result['metadata'] = metadata
    return result


def _count_from_skills(skills: list[dict[str, Any]]) -> dict[str, int]:
    official = 0
    subskills = 0
    generated_children = 0
    for skill in skills:
        if not isinstance(skill, dict):
            continue
        stype = clean_text(skill.get('type', ''))
        label = _extract_label(skill)
        if stype == 'official_skill' and label:
            official += 1
            for child in _iter_children_recursive(skill):
                if _extract_label(child):
                    subskills += 1
        elif stype == 'generated_group':
            for child in _iter_children_recursive(skill):
                if _extract_label(child):
                    generated_children += 1
        else:
            if label:
                official += 1
            for child in _iter_children_recursive(skill):
                if _extract_label(child):
                    subskills += 1
    exploitable = official + subskills + generated_children
    return {
        'official_skills_count': official,
        'subskills_count': subskills,
        'exploitable_skills_count': exploitable,
        'skills_count': exploitable,
    }


def _iter_children_recursive(skill: dict[str, Any]):
    for child in skill.get('children', []):
        if isinstance(child, dict):
            yield child
            yield from _iter_children_recursive(child)


def _empty_metadata() -> dict[str, int]:
    return {
        'official_skills_count': 0,
        'subskills_count': 0,
        'exploitable_skills_count': 0,
        'skills_count': 0,
    }


def count_referential_skills(payload: dict[str, Any]) -> int:
    if not isinstance(payload, dict):
        return 0
    normalized = normalize_referential_payload(payload)
    return normalized.get('metadata', {}).get('exploitable_skills_count', 0)


def count_referential_skills_detail(payload: dict[str, Any]) -> dict[str, int]:
    if not isinstance(payload, dict):
        return _empty_metadata()
    normalized = normalize_referential_payload(payload)
    md = normalized.get('metadata', {})
    return {
        'official_skills_count': md.get('official_skills_count', 0),
        'subskills_count': md.get('subskills_count', 0),
        'exploitable_skills_count': md.get('exploitable_skills_count', 0),
        'skills_count': md.get('skills_count', 0),
    }


def flatten_referential_skills(payload: dict[str, Any], include_children: bool = True) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    normalized = normalize_referential_payload(payload)
    result: list[dict[str, Any]] = []
    for skill in normalized.get('skills', []):
        if not isinstance(skill, dict):
            continue
        label = _extract_label(skill)
        if not label:
            continue
        path_parts = [label]
        flat_entry = _build_flat_entry(skill, parent_id=None, path_parts=path_parts)
        result.append(flat_entry)
        if include_children:
            for child in skill.get('children', []):
                if isinstance(child, dict) and _extract_label(child):
                    child_path = [*path_parts, _extract_label(child)]
                    child_entry = _build_flat_entry(child, parent_id=skill.get('id'), path_parts=child_path)
                    result.append(child_entry)
                    if include_children:
                        _flatten_recursive(child, parent_id=child.get('id'), path_parts=child_path, result=result)
    return result


def _flatten_recursive(skill: dict[str, Any], parent_id: str | None, path_parts: list[str], result: list[dict[str, Any]]) -> None:
    for child in skill.get('children', []):
        if isinstance(child, dict) and _extract_label(child):
            child_path = [*path_parts, _extract_label(child)]
            child_entry = _build_flat_entry(child, parent_id=parent_id, path_parts=child_path)
            result.append(child_entry)
            _flatten_recursive(child, parent_id=child.get('id'), path_parts=child_path, result=result)


def _build_flat_entry(skill: dict[str, Any], parent_id: str | None, path_parts: list[str]) -> dict[str, Any]:
    return {
        'id': skill.get('id', ''),
        'label': _extract_label(skill),
        'type': skill.get('type', 'subskill'),
        'parent_id': parent_id,
        'path': ' > '.join(path_parts),
        'source_page': skill.get('source_page'),
        'block': skill.get('block'),
        'confidence': skill.get('confidence', 1.0),
        'status': skill.get('status', 'pending'),
    }


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
    counts = count_referential_skills_detail(payload)
    skill_count = counts['exploitable_skills_count']
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
        official_skills_count=counts['official_skills_count'],
        subskills_count=counts['subskills_count'],
        exploitable_skills_count=counts['exploitable_skills_count'],
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
        competencies = []

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
        children: list[dict[str, Any]] = []
        if isinstance(derived, list):
            for di, d in enumerate(derived):
                if isinstance(d, dict):
                    d_label = (clean_text(d.get('canonical_label', ''))
                               or clean_text(d.get('label', ''))
                               or clean_text(d.get('surface_form', '')))
                    if d_label:
                        children.append({
                            'id': f'{code or "imp"}_sub_{di}',
                            'label': d_label,
                            'type': 'subskill',
                            'description': '',
                            'aliases': [],
                            'category': d.get('category'),
                            'block': block_code,
                            'source_page': d.get('page_start'),
                            'source_text': clean_text(d.get('surface_form', '')),
                            'confidence': float(d.get('confidence', 1.0)),
                            'status': 'pending',
                            'children': [],
                            'parent_id': code or f'imported_{len(skills)}',
                        })
                elif isinstance(d, str):
                    d_label = clean_text(d)
                    if d_label:
                        children.append({
                            'id': f'{code or "imp"}_sub_{di}',
                            'label': d_label,
                            'type': 'subskill',
                            'description': '',
                            'aliases': [],
                            'category': None,
                            'block': block_code,
                            'source_page': None,
                            'source_text': d_label,
                            'confidence': 1.0,
                            'status': 'pending',
                            'children': [],
                            'parent_id': code or f'imported_{len(skills)}',
                        })
        technical_keywords: list[str] = []
        if isinstance(derived, list):
            technical_keywords = [clean_text(s) for s in derived if isinstance(s, str) and clean_text(s)]
        source_pages = comp.get('source_pages', [])
        source_page = int(source_pages[0]) if isinstance(source_pages, list) and source_pages else 0
        skill_id = code or f'imported_{len(skills)}'
        skill = {
            'id': skill_id,
            'label': label,
            'type': 'official_skill',
            'description': label,
            'aliases': [],
            'category': '',
            'block': block_code,
            'source_page': source_page,
            'source_text': '',
            'confidence': float(comp.get('confidence', 1.0)),
            'status': 'validated',
            'children': children,
            'parent_id': None,
            'code': code,
            'block_name': '',
            'activity': activity_code,
            'official_description': label,
            'normalized_label': normalized or label,
            'subcategory': '',
            'technical_keywords': technical_keywords,
            'active': True,
        }
        skills.append(skill)

    if not skills:
        top_derived = imported_payload.get('derived_skills', [])
        if isinstance(top_derived, list) and top_derived:
            children: list[dict[str, Any]] = []
            for di, d in enumerate(top_derived):
                if isinstance(d, dict):
                    d_label = (clean_text(d.get('canonical_label', ''))
                               or clean_text(d.get('label', ''))
                               or clean_text(d.get('surface_form', '')))
                    if d_label:
                        children.append({
                            'id': f'derived_sub_{di}',
                            'label': d_label,
                            'type': 'subskill',
                            'description': '',
                            'aliases': [],
                            'category': d.get('category'),
                            'block': None,
                            'source_page': d.get('page_start'),
                            'source_text': clean_text(d.get('surface_form', '')),
                            'confidence': float(d.get('confidence', 1.0)),
                            'status': 'pending',
                            'children': [],
                            'parent_id': 'generated_group_0',
                        })
                elif isinstance(d, str):
                    d_label = clean_text(d)
                    if d_label:
                        children.append({
                            'id': f'derived_sub_{di}',
                            'label': d_label,
                            'type': 'subskill',
                            'description': '',
                            'aliases': [],
                            'category': None,
                            'block': None,
                            'source_page': None,
                            'source_text': d_label,
                            'confidence': 1.0,
                            'status': 'pending',
                            'children': [],
                            'parent_id': 'generated_group_0',
                        })
            if children:
                group = {
                    'id': 'generated_group_0',
                    'label': 'Compétences extraites du référentiel',
                    'type': 'generated_group',
                    'description': '',
                    'aliases': [],
                    'category': None,
                    'block': None,
                    'source_page': None,
                    'source_text': '',
                    'confidence': 1.0,
                    'status': 'pending',
                    'children': children,
                    'parent_id': None,
                }
                skills.append(group)

    blocks = imported_payload.get('blocks', [])
    block_names: dict[str, str] = {}
    if isinstance(blocks, list):
        for b in blocks:
            if isinstance(b, dict):
                bc = clean_text(b.get('code', ''))
                bl = clean_text(b.get('label', ''))
                if bc and bl:
                    block_names[bc] = bl
    for skill in skills:
        block_code = skill.get('block', '')
        if block_code in block_names:
            skill['block_name'] = block_names[block_code]

    counts = _count_from_skills(skills)
    metadata = dict(_empty_metadata())
    metadata.update(counts)

    referential_id = clean_text(imported_payload.get('referential_id')) or doc.get('sha256', '') or clean_text(doc.get('file_name', ''))
    title = clean_text(imported_payload.get('title')) or doc_title or referential_id

    result = {
        'schema_version': CANONICAL_SCHEMA_VERSION,
        'referential_id': referential_id,
        'title': title,
        'version': imported_payload.get('schema_version', '1.0'),
        'skills': skills,
        'metadata': metadata,
    }
    result['metadata'].update({
        'source': 'imported_pdf',
        'source_path': clean_text(doc.get('source_path', '') if isinstance(doc, dict) else ''),
        'file_name': clean_text(doc.get('file_name', '') if isinstance(doc, dict) else ''),
        'record_id': clean_text(doc.get('sha256', '') if isinstance(doc, dict) else ''),
    })
    return result


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
    if option.exploitable_skills_count > 0 or option.skill_count > 0:
        return option.path
    return None
