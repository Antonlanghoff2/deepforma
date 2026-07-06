from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common.text import clean_text, normalize_for_match

ROME_DATA_PATH_DEFAULT = Path('data/raw/rome')


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

    def to_dict(self) -> dict[str, Any]:
        return {
            'rome_code': self.rome_code,
            'label': self.label,
            'definition': self.definition,
            'alternative_titles': self.alternative_titles,
            'activity_ids': self.activity_ids,
            'skill_ids': self.skill_ids,
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


class RomeReferentialImporter:
    def __init__(self, input_path: str | Path | None = None) -> None:
        self.input_path = Path(input_path or ROME_DATA_PATH_DEFAULT)

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
                for key in ('items', 'results', 'jobs', 'skills', 'job_titles', 'links'):
                    value = payload.get(key)
                    if isinstance(value, list):
                        return [row for row in value if isinstance(row, dict)]
                return [payload]
        return []

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
            for row in self._read_records(path):
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
