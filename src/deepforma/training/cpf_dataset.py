from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from hashlib import sha1
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from common.text import clean_text, normalize_for_match, stable_hash
from deepforma.skills.normalizer import SkillTaxonomyNormalizer


DEFAULT_SPLIT_RATIOS = (0.70, 0.15, 0.15)


@dataclass(frozen=True)
class CPFTrainingExample:
    query_id: str
    query: str
    target_job: str
    required_skills: list[str]
    missing_skills: list[str]
    region_code: str | None
    department_code: str | None
    positive_uid: str
    positive_text: str
    negative_uid: str
    negative_text: str
    negative_type: str
    label_source: str
    label_confidence: float
    group_id: str
    certification_code: str | None = None
    certification_label: str | None = None
    referential_type: str | None = None
    level: str | None = None
    nsf: str | None = None
    organization: str | None = None
    siret: str | None = None
    region: str | None = None
    department: str | None = None


@dataclass(frozen=True)
class CPFTrainingSummary:
    formation_count: int
    text_count: int
    skill_count: int
    query_count: int
    positive_count: int
    negative_count: int
    heuristic_label_rate: float
    split_counts: dict[str, int]
    negative_type_counts: dict[str, int]
    certification_overlap_rate: float


@dataclass(frozen=True)
class CPFValidationResult:
    ok: bool
    errors: list[str]
    summary: CPFTrainingSummary


SKILL_NORMALIZER = SkillTaxonomyNormalizer()


def load_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(Path(path))


def ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    if hasattr(value, 'tolist') and not isinstance(value, (str, bytes, bytearray)):
        try:
            converted = value.tolist()
            if isinstance(converted, list):
                return converted
        except Exception:
            pass
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith('[') and text.endswith(']'):
            try:
                loaded = json.loads(text)
                if isinstance(loaded, list):
                    return loaded
            except Exception:
                pass
        return [part.strip() for part in text.split('|') if part.strip()]
    return [value]


def normalize_skill_labels(values: Iterable[Any], *, normalizer: SkillTaxonomyNormalizer | None = None) -> list[dict[str, Any]]:
    normalizer = normalizer or SKILL_NORMALIZER
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for value in values:
        if isinstance(value, dict):
            candidate = value.get('canonical_label') or value.get('label') or value.get('original_label')
        else:
            candidate = value
        match = normalizer.normalize(str(candidate or ''), extraction_source='cpf_dataset', confidence_floor=0.0)
        if match:
            key = match.canonical_id
            label = match.canonical_label
            aliases = match.aliases
            confidence = match.confidence
        else:
            label = clean_text(candidate)
            if not label:
                continue
            key = normalize_for_match(label)
            aliases = []
            confidence = 0.0
        if key in seen:
            continue
        seen.add(key)
        normalized.append(
            {
                'canonical_id': key,
                'canonical_label': label,
                'aliases': aliases,
                'confidence': round(float(confidence), 4),
            }
        )
    return normalized


def formation_text(row: dict[str, Any]) -> str:
    parts = [
        clean_text(row.get('title')),
        clean_text(row.get('certification_label') or row.get('certification')),
        clean_text(row.get('description')),
        clean_text(row.get('objectives')),
        clean_text(row.get('search_text')),
    ]
    return ' \n '.join(part for part in parts if part)


def derive_certification_code(row: dict[str, Any]) -> str | None:
    for key in ('certification_code', 'normalized_certification', 'certification', 'source_id'):
        value = clean_text(row.get(key))
        if value:
            return value
    return None


def derive_formation_uid(row: dict[str, Any]) -> str:
    for key in ('formation_uid', 'uid', 'id'):
        value = clean_text(row.get(key))
        if value:
            return value
    return stable_hash(row.get('title'), row.get('certification'), row.get('description'), row.get('organization'))


def build_group_id(row: dict[str, Any]) -> str:
    certification_code = clean_text(row.get('certification_code') or row.get('certification') or '')
    title = clean_text(row.get('canonical_title') or row.get('title') or '')
    text_hash = stable_hash(row.get('search_text') or formation_text(row))
    payload = '||'.join([normalize_for_match(certification_code), normalize_for_match(title), text_hash])
    return sha1(payload.encode('utf-8')).hexdigest()


def _resolve_training_field(row: dict[str, Any], keys: tuple[str, ...], *, field_name: str) -> str:
    for key in keys:
        value = clean_text(row.get(key))
        if value:
            return value
    available = sorted(key for key, value in row.items() if clean_text(value))
    expected = ', '.join(keys)
    raise ValueError(
        f"Champ obligatoire manquant pour {field_name}. Cles disponibles: {available}. Cles attendues: {expected}."
    )


def normalize_training_row(row: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise TypeError(f"Chaque ligne doit etre un dictionnaire, recu: {type(row)!r}")
    normalized = dict(row)
    anchor = _resolve_training_field(
        row,
        ('anchor_text', 'query', 'text_a', 'source_text', 'anchor'),
        field_name='anchor',
    )
    positive = _resolve_training_field(
        row,
        ('positive_text', 'candidate_text', 'text_b', 'target_text', 'positive'),
        field_name='positive',
    )
    positive_uid = _resolve_training_field(
        row,
        ('positive_uid', 'positive_id', 'candidate_uid', 'candidate_id', 'target_uid', 'target_id', 'formation_id', 'uid'),
        field_name='positive_uid',
    )
    normalized['anchor'] = anchor
    normalized['positive'] = positive
    normalized['positive_uid'] = positive_uid
    group_id = clean_text(row.get('group_id'))
    if group_id:
        normalized['group_id'] = group_id
    split = clean_text(row.get('split'))
    if split:
        normalized['split'] = split
    return normalized


def normalize_training_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [normalize_training_row(row) for row in rows]


def canonicalize_formation_row(row: dict[str, Any], *, normalizer: SkillTaxonomyNormalizer | None = None) -> dict[str, Any]:
    normalizer = normalizer or SKILL_NORMALIZER
    skills_explicit = ensure_list(row.get('skills_explicit'))
    skills_inferred = ensure_list(row.get('skills_inferred'))
    skills_normalized = ensure_list(row.get('skills_normalized'))
    skill_source = skills_normalized if len(skills_normalized) else skills_explicit if len(skills_explicit) else skills_inferred
    normalized_skills = normalize_skill_labels(skill_source, normalizer=normalizer)
    if not normalized_skills and skills_explicit:
        normalized_skills = normalize_skill_labels(skills_explicit, normalizer=normalizer)
    search_text = clean_text(row.get('search_text')) or formation_text(row)
    certification_label = clean_text(row.get('certification_label') or row.get('certification')) or None
    certification_code = derive_certification_code(row)
    formation_uid = derive_formation_uid(row)
    group_id = build_group_id(
        {
            **row,
            'formation_uid': formation_uid,
            'certification_code': certification_code,
            'title': row.get('title'),
            'search_text': search_text,
        }
    )
    return {
        'formation_uid': formation_uid,
        'title': clean_text(row.get('title')),
        'description': clean_text(row.get('description')),
        'objectives': clean_text(row.get('objectives')),
        'certification_code': certification_code,
        'certification_label': certification_label,
        'referential_type': clean_text(row.get('referential_type')) or None,
        'level': clean_text(row.get('level') or row.get('exit_level')) or None,
        'nsf': clean_text(row.get('nsf')) or None,
        'organization': clean_text(row.get('organization')) or None,
        'siret': clean_text(row.get('siret') or row.get('organization_siret')) or None,
        'region': clean_text(row.get('region')) or None,
        'region_code': clean_text(row.get('region_code')) or None,
        'department': clean_text(row.get('department')) or None,
        'department_code': clean_text(row.get('department_code')) or None,
        'distance_compatible': bool(row.get('distance_compatible')) if row.get('distance_compatible') not in (None, '') else False,
        'remote': bool(row.get('remote')) if row.get('remote') not in (None, '') else bool(row.get('distance_compatible')) if row.get('distance_compatible') not in (None, '') else False,
        'search_text': search_text,
        'skills_explicit': skills_explicit,
        'skills_inferred': skills_inferred,
        'skills_normalized': normalized_skills,
        'skills_confidence': row.get('skills_confidence') or {},
        'skills_evidence': row.get('skills_evidence') or {},
        'canonical_title': clean_text(row.get('normalized_title') or row.get('title')),
        'group_id': group_id,
    }


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open('r', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def save_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + '\n')


def split_by_group(rows: list[dict[str, Any]], *, seed: int = 42, ratios: tuple[float, float, float] = DEFAULT_SPLIT_RATIOS) -> dict[str, list[dict[str, Any]]]:
    if not rows:
        return {'train': [], 'validation': [], 'test': []}
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get('group_id') or build_group_id(row)), []).append(row)
    group_ids = list(groups)
    rng = random.Random(seed)
    rng.shuffle(group_ids)
    total = len(group_ids)
    n_train = max(1, int(round(total * ratios[0])))
    n_validation = max(1, int(round(total * ratios[1]))) if total >= 3 else max(0, total - n_train)
    if n_train + n_validation > total:
        n_validation = max(0, total - n_train)
    n_test = max(0, total - n_train - n_validation)
    if total >= 3 and n_test == 0:
        n_test = 1
        if n_train > n_validation:
            n_train = max(1, n_train - 1)
        else:
            n_validation = max(1, n_validation - 1)
    train_groups = set(group_ids[:n_train])
    validation_groups = set(group_ids[n_train:n_train + n_validation])
    test_groups = set(group_ids[n_train + n_validation:])
    return {
        'train': [row for row in rows if str(row.get('group_id') or build_group_id(row)) in train_groups],
        'validation': [row for row in rows if str(row.get('group_id') or build_group_id(row)) in validation_groups],
        'test': [row for row in rows if str(row.get('group_id') or build_group_id(row)) in test_groups],
    }


def summarize_rows(rows: list[dict[str, Any]]) -> CPFTrainingSummary:
    formation_count = len({row.get('formation_uid') for row in rows if clean_text(row.get('formation_uid'))})
    text_count = sum(1 for row in rows if clean_text(row.get('search_text')))
    skill_count = sum(1 for row in rows if ensure_list(row.get('skills_normalized')))
    query_count = len(rows)
    positive_count = sum(1 for row in rows if clean_text(row.get('positive_uid')))
    negative_count = sum(1 for row in rows if clean_text(row.get('negative_uid')))
    heuristic_label_count = sum(1 for row in rows if row.get('label_source') == 'heuristic')
    split_counts = {name: len(items) for name, items in rows_by_split(rows).items()}
    negative_type_counts: dict[str, int] = {}
    for row in rows:
        key = clean_text(row.get('negative_type')) or 'unknown'
        negative_type_counts[key] = negative_type_counts.get(key, 0) + 1
    certs_by_split = {name: {clean_text(row.get('certification_code')) for row in items if clean_text(row.get('certification_code'))} for name, items in rows_by_split(rows).items()}
    train_certs = certs_by_split.get('train', set())
    test_certs = certs_by_split.get('test', set())
    overlap = (len(train_certs & test_certs) / len(train_certs | test_certs)) if (train_certs or test_certs) else 0.0
    return CPFTrainingSummary(
        formation_count=formation_count,
        text_count=text_count,
        skill_count=skill_count,
        query_count=query_count,
        positive_count=positive_count,
        negative_count=negative_count,
        heuristic_label_rate=round(heuristic_label_count / query_count if query_count else 0.0, 4),
        split_counts=split_counts,
        negative_type_counts=negative_type_counts,
        certification_overlap_rate=round(overlap, 4),
    )


def rows_by_split(rows: list[dict[str, Any]], *, seed: int = 42, ratios: tuple[float, float, float] = DEFAULT_SPLIT_RATIOS) -> dict[str, list[dict[str, Any]]]:
    return split_by_group(rows, seed=seed, ratios=ratios)


def validate_rows(rows: list[dict[str, Any]], *, min_positives: int = 10) -> CPFValidationResult:
    errors: list[str] = []
    summary = summarize_rows(rows)
    if not rows:
        errors.append('Le dataset est vide.')
    if summary.positive_count < min_positives:
        errors.append(f'Nombre de positifs insuffisant: {summary.positive_count} < {min_positives}.')
    splits = rows_by_split(rows)
    if not splits['validation']:
        errors.append('Le split de validation est vide.')
    if summary.certification_overlap_rate > 0.0:
        errors.append('Une fuite de certifications a été détectée entre train et test.')
    return CPFValidationResult(ok=not errors, errors=errors, summary=summary)


def timestamp_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
