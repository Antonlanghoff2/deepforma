from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
from difflib import SequenceMatcher

from common.text import clean_text, normalize_for_match

ROME_DATA_PATH_DEFAULT = Path('data/raw/rome')
ROME_CODE_RE = re.compile(r'^[A-Z][0-9]{4}$')


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
    text = text.replace('\n', '|').replace(';', '|').replace(',', '|')
    return [clean_text(part) for part in text.split('|') if clean_text(part)]


@dataclass(frozen=True, slots=True)
class RomeJob:
    rome_code: str
    label: str
    definition: str
    alternative_titles: list[str]
    activity_ids: list[str]
    skill_ids: list[str]
    domain: str | None = None

    @property
    def alternative_labels(self) -> list[str]:
        return self.alternative_titles

    def to_dict(self) -> dict[str, Any]:
        return {
            'rome_code': self.rome_code,
            'label': self.label,
            'definition': self.definition,
            'alternative_titles': self.alternative_titles,
            'activity_ids': self.activity_ids,
            'skill_ids': self.skill_ids,
            'domain': self.domain,
        }


@dataclass(frozen=True, slots=True)
class RomeSkill:
    rome_skill_id: str
    official_label: str
    normalized_label: str
    skill_type: str
    source: str = 'rome'

    def to_dict(self) -> dict[str, Any]:
        return {
            'rome_skill_id': self.rome_skill_id,
            'official_label': self.official_label,
            'normalized_label': self.normalized_label,
            'skill_type': self.skill_type,
            'source': self.source,
        }


@lru_cache(maxsize=8)
def _read_records(path_str: str) -> tuple[dict[str, Any], ...]:
    path = Path(path_str)
    if not path.exists():
        return tuple()
    if path.suffix.lower() == '.jsonl':
        rows: list[dict[str, Any]] = []
        for line in path.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if line:
                payload = json.loads(line)
                if isinstance(payload, dict):
                    rows.append(payload)
        return tuple(rows)
    if path.suffix.lower() == '.csv':
        with path.open(encoding='utf-8', newline='') as fh:
            return tuple(dict(row) for row in csv.DictReader(fh))
    if path.suffix.lower() == '.json':
        payload = json.loads(path.read_text(encoding='utf-8'))
        if isinstance(payload, list):
            return tuple(row for row in payload if isinstance(row, dict))
        if isinstance(payload, dict):
            for key in ('items', 'results', 'jobs', 'skills', 'job_titles', 'links'):
                value = payload.get(key)
                if isinstance(value, list):
                    return tuple(row for row in value if isinstance(row, dict))
            return (payload,)
    return tuple()


class RomeReferentialImporter:
    def __init__(self, input_path: str | Path | None = None) -> None:
        self.input_path = Path(input_path or ROME_DATA_PATH_DEFAULT)

    @staticmethod
    def _job_from_row(row: dict[str, Any]) -> RomeJob | None:
        code = clean_text(_pick(row, 'rome_code', 'code', 'id'))
        label = clean_text(_pick(row, 'label', 'title', 'intitule', 'name'))
        definition = clean_text(_pick(row, 'definition', 'description', 'text'))
        if not code or not label:
            return None
        return RomeJob(
            rome_code=code,
            label=label,
            definition=definition,
            alternative_titles=_split_values(_pick(row, 'alternative_titles', 'appellations', 'titles', 'aliases')),
            activity_ids=_split_values(_pick(row, 'activity_ids', 'activities', 'activites')),
            skill_ids=_split_values(_pick(row, 'skill_ids', 'skills', 'competences')),
            domain=clean_text(_pick(row, 'domain', 'domaine', 'sector', 'secteur')) or None,
        )

    @staticmethod
    def _skill_from_row(row: dict[str, Any]) -> RomeSkill | None:
        skill_id = clean_text(_pick(row, 'rome_skill_id', 'skill_id', 'id', 'code'))
        label = clean_text(_pick(row, 'official_label', 'label', 'name', 'title'))
        if not skill_id or not label:
            return None
        skill_type = normalize_for_match(_pick(row, 'skill_type', 'type', 'category')) or 'competence'
        if 'savoir faire' in skill_type:
            skill_type = 'savoir_faire'
        elif skill_type in {'savoir', 'connaissance'}:
            skill_type = 'savoir'
        elif skill_type not in {'savoir_faire', 'savoir', 'competence'}:
            skill_type = 'competence'
        return RomeSkill(
            rome_skill_id=skill_id,
            official_label=label,
            normalized_label=normalize_for_match(label),
            skill_type=skill_type,
        )

    @staticmethod
    def _link_from_row(row: dict[str, Any]) -> dict[str, Any] | None:
        rome_code = clean_text(_pick(row, 'rome_code', 'job_code', 'code'))
        rome_skill_id = clean_text(_pick(row, 'rome_skill_id', 'skill_id', 'skill_code'))
        if not rome_code or not rome_skill_id:
            return None
        return {
            'rome_code': rome_code,
            'rome_skill_id': rome_skill_id,
            'relation': clean_text(_pick(row, 'relation', 'type', 'link_type')) or 'official',
        }

    def load(self) -> dict[str, list[dict[str, Any]]]:
        jobs: dict[str, RomeJob] = {}
        skills: dict[str, RomeSkill] = {}
        links: list[dict[str, Any]] = []
        if self.input_path.is_file():
            files = [self.input_path]
        elif self.input_path.exists():
            files = sorted(path for path in self.input_path.rglob('*') if path.suffix.lower() in {'.json', '.jsonl', '.csv'})
        else:
            files = []
        for path in files:
            for row in _read_records(str(path)):
                name = normalize_for_match(path.name)
                if 'link' in name:
                    link = self._link_from_row(row)
                    if link:
                        links.append(link)
                    continue
                if 'skill' in name or 'competence' in name:
                    skill = self._skill_from_row(row)
                    if skill:
                        skills[skill.rome_skill_id] = skill
                    continue
                job = self._job_from_row(row)
                if job:
                    jobs[job.rome_code] = job
        return {
            'jobs': [item.to_dict() for item in jobs.values()],
            'job_titles': [{'rome_code': item.rome_code, 'title': title} for item in jobs.values() for title in item.alternative_titles],
            'skills': [item.to_dict() for item in skills.values()],
            'job_skill_links': links,
        }


@lru_cache(maxsize=4)
def _load_referential(input_path: str) -> dict[str, list[dict[str, Any]]]:
    return RomeReferentialImporter(input_path).load()


class RomeService:
    def __init__(self, input_path: str | Path | None = None) -> None:
        self.input_path = Path(input_path or ROME_DATA_PATH_DEFAULT)
        self._payload: dict[str, list[dict[str, Any]]] | None = None
        self._jobs_by_code: dict[str, RomeJob] = {}
        self._jobs: list[RomeJob] = []

    def load(self) -> dict[str, list[dict[str, Any]]]:
        if self._payload is not None:
            return self._payload
        payload = _load_referential(str(self.input_path))
        jobs = []
        jobs_by_code: dict[str, RomeJob] = {}
        for row in payload.get('jobs', []):
            if not isinstance(row, dict):
                continue
            job = RomeJob(
                rome_code=clean_text(row.get('rome_code') or ''),
                label=clean_text(row.get('label') or ''),
                definition=clean_text(row.get('definition') or ''),
                alternative_titles=_split_values(row.get('alternative_titles') or []),
                activity_ids=_split_values(row.get('activity_ids') or []),
                skill_ids=_split_values(row.get('skill_ids') or []),
                domain=clean_text(row.get('domain') or '') or None,
            )
            if job.rome_code and job.label:
                jobs.append(job)
                jobs_by_code[job.rome_code] = job
        self._payload = payload
        self._jobs = jobs
        self._jobs_by_code = jobs_by_code
        return payload

    def has_local_referential(self) -> bool:
        self.load()
        return bool(self._jobs_by_code)

    def get_all_jobs(self) -> list[RomeJob]:
        self.load()
        return list(self._jobs)

    def get(self, code: str) -> RomeJob | None:
        normalized = clean_text(code).replace(' ', '').upper()
        if not normalized:
            return None
        self.load()
        return self._jobs_by_code.get(normalized)

    def _score_job(self, query: str, job: RomeJob) -> float:
        normalized_query = normalize_for_match(query)
        if not normalized_query:
            return 0.0
        normalized_code = normalize_for_match(job.rome_code)
        if normalized_query == normalized_code:
            return 1.0
        if normalized_query in normalized_code or normalized_code in normalized_query:
            return 0.98
        haystack = ' | '.join([job.label, job.definition, *job.alternative_titles, job.domain or ''])
        normalized_haystack = normalize_for_match(haystack)
        if not normalized_haystack:
            return 0.0
        if normalized_query in normalized_haystack:
            return 0.9
        best = SequenceMatcher(None, normalized_query, normalized_haystack).ratio()
        tokens = set(normalized_query.split())
        if tokens:
            hay_tokens = set(normalized_haystack.split())
            overlap = len(tokens & hay_tokens) / len(tokens)
            best = max(best, overlap)
        return best

    def search(self, query: str, limit: int = 10) -> list[RomeJob]:
        self.load()
        query = clean_text(query)
        if not query:
            return self.get_all_jobs()[:limit]
        scored = [
            (self._score_job(query, job), job)
            for job in self._jobs
        ]
        scored = [item for item in scored if item[0] > 0]
        scored.sort(key=lambda item: (-item[0], item[1].rome_code, item[1].label))
        return [job for _, job in scored[:limit]]

    def validate(self, code: str) -> RomeJob:
        normalized = validate_rome_code(code, service=self)
        job = self.get(normalized)
        if job is None and self.has_local_referential():
            raise ValueError('Code ROME inconnu dans le référentiel chargé')
        if job is None:
            return RomeJob(
                rome_code=normalized,
                label='',
                definition='',
                alternative_titles=[],
                activity_ids=[],
                skill_ids=[],
                domain=None,
            )
        return job


@lru_cache(maxsize=1)
def get_default_rome_service() -> RomeService:
    return RomeService()


def validate_rome_code(code: str, service: RomeService | None = None) -> str:
    normalized = clean_text(code).replace(' ', '').upper()
    if not normalized:
        raise ValueError('Code ROME obligatoire')
    if not ROME_CODE_RE.match(normalized):
        raise ValueError('Format de code ROME invalide')
    active_service = service or get_default_rome_service()
    if active_service.has_local_referential() and active_service.get(normalized) is None:
        raise ValueError('Code ROME inconnu dans le référentiel chargé')
    return normalized
