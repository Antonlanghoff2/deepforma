from __future__ import annotations

import csv
import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable

import requests

from common.text import clean_text, normalize_for_match

ACTIVE_ONLY_DEFAULT = os.getenv('FRANCE_COMPETENCES_ACTIVE_ONLY', 'true').lower() in {'1', 'true', 'yes', 'on'}
DATA_PATH_DEFAULT = Path(os.getenv('FRANCE_COMPETENCES_DATA_PATH', 'data/raw/france_competences'))
API_URL_DEFAULT = os.getenv('FRANCE_COMPETENCES_API_URL', '').strip()
API_TOKEN_DEFAULT = os.getenv('FRANCE_COMPETENCES_API_TOKEN', '').strip()
CACHE_DIR_DEFAULT = Path(os.getenv('FRANCE_COMPETENCES_CACHE_DIR', 'data/cache/france_competences'))


def _pick(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, '', [], {}):
            return value
    return None


def _split_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [clean_text(item) for item in value if clean_text(item)]
    text = clean_text(value)
    if not text:
        return []
    text = text.replace('\n', '|').replace(';', '|')
    return [clean_text(part) for part in text.split('|') if clean_text(part)]


def _norm_aliases(values: Iterable[Any]) -> list[str]:
    aliases: list[str] = []
    seen: set[str] = set()
    for value in values:
        alias = clean_text(value)
        key = normalize_for_match(alias)
        if not alias or not key or key in seen:
            continue
        seen.add(key)
        aliases.append(alias)
    return aliases


def _coerce_bool(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = normalize_for_match(value)
    if not text:
        return default
    if text in {'1', 'true', 'yes', 'y', 'actif', 'active'}:
        return True
    if text in {'0', 'false', 'no', 'n', 'inactif', 'inactive'}:
        return False
    return default


def _coerce_status(value: Any) -> str:
    text = normalize_for_match(value)
    if not text:
        return 'active'
    if 'inactive' in text or 'expire' in text or 'closed' in text:
        return 'inactive'
    if 'active' in text:
        return 'active'
    return clean_text(value) or 'active'


def _row_kind(path: Path, row: dict[str, Any]) -> str | None:
    name = normalize_for_match(path.name)
    keys = set(row.keys())
    if 'skill' in name or 'competence' in name or {'skill_id', 'official_label'} <= keys:
        return 'skill'
    if 'block' in name or 'bloc' in name or {'block_id', 'rncp_code'} <= keys:
        return 'block'
    if 'cert' in name or 'rncp' in name or {'type', 'level', 'valid_until'} & keys:
        return 'certification'
    if {'skill_id', 'rncp_code', 'block_id', 'official_label'} <= keys:
        return 'skill'
    if {'block_id', 'rncp_code', 'official_description'} <= keys:
        return 'block'
    if {'rncp_code', 'title', 'status'} <= keys:
        return 'certification'
    return None


@dataclass(frozen=True, slots=True)
class FranceCompetenceCertification:
    rncp_code: str
    type: str
    title: str
    status: str
    level: int | None
    valid_until: str | None
    activities: list[str]
    target_jobs: list[str]
    sectors: list[str]
    block_ids: list[str]
    source_url: str | None
    source_updated_at: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            'rncp_code': self.rncp_code,
            'type': self.type,
            'title': self.title,
            'status': self.status,
            'level': self.level,
            'valid_until': self.valid_until,
            'activities': self.activities,
            'target_jobs': self.target_jobs,
            'sectors': self.sectors,
            'block_ids': self.block_ids,
            'source_url': self.source_url,
            'source_updated_at': self.source_updated_at,
        }


@dataclass(frozen=True, slots=True)
class FranceCompetenceBlock:
    block_id: str
    rncp_code: str
    title: str
    official_description: str
    skill_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            'block_id': self.block_id,
            'rncp_code': self.rncp_code,
            'title': self.title,
            'official_description': self.official_description,
            'skill_ids': self.skill_ids,
        }


@dataclass(frozen=True, slots=True)
class FranceCompetenceSkill:
    skill_id: str
    rncp_code: str
    block_id: str
    official_label: str
    normalized_label: str
    aliases: list[str]
    source: str = 'france_competences'
    active: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            'skill_id': self.skill_id,
            'rncp_code': self.rncp_code,
            'block_id': self.block_id,
            'official_label': self.official_label,
            'normalized_label': self.normalized_label,
            'aliases': self.aliases,
            'source': self.source,
            'active': self.active,
        }


class FranceCompetencesSource(ABC):
    @abstractmethod
    def load(self, *, active_only: bool = ACTIVE_ONLY_DEFAULT) -> dict[str, list[dict[str, Any]]]:
        raise NotImplementedError


class FranceCompetencesApiClient(FranceCompetencesSource):
    def __init__(
        self,
        api_url: str | None = None,
        api_token: str | None = None,
        *,
        session: requests.Session | None = None,
        cache_dir: str | Path | None = None,
        timeout: int = 30,
        retries: int = 3,
    ) -> None:
        self.api_url = (api_url or API_URL_DEFAULT).rstrip('/')
        self.api_token = api_token or API_TOKEN_DEFAULT
        self.session = session or requests.Session()
        self.cache_dir = Path(cache_dir or CACHE_DIR_DEFAULT)
        self.timeout = timeout
        self.retries = retries

    def _cache_key(self, endpoint: str, params: dict[str, Any]) -> Path:
        payload = json.dumps({'endpoint': endpoint, 'params': params}, sort_keys=True, ensure_ascii=False)
        return self.cache_dir / f"{sha256(payload.encode('utf-8')).hexdigest()}.json"

    def _request(self, endpoint: str, params: dict[str, Any] | None = None) -> Any:
        params = params or {}
        cache_path = self._cache_key(endpoint, params)
        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding='utf-8'))
        headers = {'Accept': 'application/json'}
        if self.api_token:
            headers['Authorization'] = f'Bearer {self.api_token}'
        url = f"{self.api_url}/{endpoint.lstrip('/')}"
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                response = self.session.get(url, params=params, headers=headers, timeout=self.timeout)
                response.raise_for_status()
                payload = response.json()
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
                return payload
            except Exception as exc:
                last_error = exc
                time.sleep(min(2**attempt, 8))
        if last_error is not None:
            raise last_error
        raise RuntimeError('France Compétences API: requête impossible')

    def load(self, *, active_only: bool = ACTIVE_ONLY_DEFAULT) -> dict[str, list[dict[str, Any]]]:
        payload = {
            'certifications': self._request('certifications', {'active_only': str(active_only).lower()}),
            'blocks': self._request('blocks', {'active_only': str(active_only).lower()}),
            'skills': self._request('skills', {'active_only': str(active_only).lower()}),
        }
        result: dict[str, list[dict[str, Any]]] = {}
        for key, value in payload.items():
            if isinstance(value, list):
                result[key] = [row for row in value if isinstance(row, dict)]
            elif isinstance(value, dict):
                result[key] = [row for row in value.get('items', []) if isinstance(row, dict)]
            else:
                result[key] = []
        return result


class FranceCompetencesOpenDataImporter(FranceCompetencesSource):
    def __init__(self, data_path: str | Path | None = None, *, active_only: bool = ACTIVE_ONLY_DEFAULT) -> None:
        self.data_path = Path(data_path or DATA_PATH_DEFAULT)
        self.active_only = active_only

    @staticmethod
    def _read_records(path: Path) -> list[dict[str, Any]]:
        if path.suffix.lower() == '.jsonl':
            rows: list[dict[str, Any]] = []
            for line in path.read_text(encoding='utf-8').splitlines():
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
            return rows
        if path.suffix.lower() == '.csv':
            with path.open(encoding='utf-8', newline='') as fh:
                return list(csv.DictReader(fh))
        if path.suffix.lower() == '.json':
            payload = json.loads(path.read_text(encoding='utf-8'))
            if isinstance(payload, list):
                return [row for row in payload if isinstance(row, dict)]
            if isinstance(payload, dict):
                for key in ('items', 'results', 'data', 'certifications', 'blocks', 'skills'):
                    value = payload.get(key)
                    if isinstance(value, list):
                        return [row for row in value if isinstance(row, dict)]
                return [payload]
        return []

    @staticmethod
    def _certification_from_row(row: dict[str, Any]) -> FranceCompetenceCertification | None:
        rncp_code = clean_text(_pick(row, 'rncp_code', 'code', 'certification_code', 'reference'))
        title = clean_text(_pick(row, 'title', 'intitule', 'intitulé', 'label', 'nom'))
        if not rncp_code or not title:
            return None
        type_value = clean_text(_pick(row, 'type', 'referential_type')) or ('RNCP' if 'rncp' in normalize_for_match(rncp_code) else 'RS')
        level_raw = _pick(row, 'level', 'niveau', 'exit_level')
        try:
            level = int(level_raw) if level_raw not in (None, '') else None
        except Exception:
            level = None
        return FranceCompetenceCertification(
            rncp_code=rncp_code,
            type=type_value,
            title=title,
            status=_coerce_status(_pick(row, 'status', 'etat', 'state')),
            level=level,
            valid_until=clean_text(_pick(row, 'valid_until', 'date_fin_validite', 'expiration_date')) or None,
            activities=_split_values(_pick(row, 'activities', 'activites', 'activity', 'activities_text')),
            target_jobs=_split_values(_pick(row, 'target_jobs', 'metiers_vizes', 'metiers vises', 'jobs', 'job_titles')),
            sectors=_split_values(_pick(row, 'sectors', 'secteurs', 'sector')),
            block_ids=_split_values(_pick(row, 'block_ids', 'blocs', 'blocks')),
            source_url=clean_text(_pick(row, 'source_url', 'url', 'link')) or None,
            source_updated_at=clean_text(_pick(row, 'source_updated_at', 'updated_at', 'date_mise_a_jour')) or None,
        )

    @staticmethod
    def _block_from_row(row: dict[str, Any]) -> FranceCompetenceBlock | None:
        block_id = clean_text(_pick(row, 'block_id', 'id', 'code', 'block'))
        rncp_code = clean_text(_pick(row, 'rncp_code', 'code_certification', 'certification_code'))
        title = clean_text(_pick(row, 'title', 'intitule', 'label'))
        description = clean_text(_pick(row, 'official_description', 'description', 'text', 'content'))
        if not block_id or not rncp_code:
            return None
        return FranceCompetenceBlock(
            block_id=block_id,
            rncp_code=rncp_code,
            title=title,
            official_description=description,
            skill_ids=_split_values(_pick(row, 'skill_ids', 'skills', 'competences', 'competencies')),
        )

    @staticmethod
    def _skill_from_row(row: dict[str, Any]) -> FranceCompetenceSkill | None:
        skill_id = clean_text(_pick(row, 'skill_id', 'id', 'code'))
        rncp_code = clean_text(_pick(row, 'rncp_code', 'code_certification', 'certification_code'))
        block_id = clean_text(_pick(row, 'block_id', 'block', 'bloc'))
        official_label = clean_text(_pick(row, 'official_label', 'label', 'title', 'intitule'))
        if not skill_id or not rncp_code or not block_id or not official_label:
            return None
        active = _coerce_bool(_pick(row, 'active', 'status', 'etat'), True)
        aliases = _norm_aliases(_split_values(_pick(row, 'aliases', 'synonyms', 'alias')))
        return FranceCompetenceSkill(
            skill_id=skill_id,
            rncp_code=rncp_code,
            block_id=block_id,
            official_label=official_label,
            normalized_label=normalize_for_match(official_label),
            aliases=aliases,
            active=active,
        )

    def _iter_source_files(self) -> list[Path]:
        if self.data_path.is_file():
            return [self.data_path]
        if not self.data_path.exists():
            return []
        return sorted(path for path in self.data_path.rglob('*') if path.suffix.lower() in {'.json', '.jsonl', '.csv'})

    @lru_cache(maxsize=2)
    def load(self, *, active_only: bool = ACTIVE_ONLY_DEFAULT) -> dict[str, list[dict[str, Any]]]:
        certifications: dict[str, FranceCompetenceCertification] = {}
        blocks: dict[str, FranceCompetenceBlock] = {}
        skills: dict[str, FranceCompetenceSkill] = {}
        for path in self._iter_source_files():
            for row in self._read_records(path):
                kind = _row_kind(path, row)
                if kind == 'certification':
                    cert = self._certification_from_row(row)
                    if cert and (not active_only or cert.status == 'active'):
                        certifications[cert.rncp_code] = cert
                elif kind == 'block':
                    block = self._block_from_row(row)
                    if block:
                        blocks[block.block_id] = block
                elif kind == 'skill':
                    skill = self._skill_from_row(row)
                    if skill and (not active_only or skill.active):
                        skills[skill.skill_id] = skill
        return {
            'certifications': [item.to_dict() for item in certifications.values()],
            'blocks': [item.to_dict() for item in blocks.values()],
            'skills': [item.to_dict() for item in skills.values()],
        }
