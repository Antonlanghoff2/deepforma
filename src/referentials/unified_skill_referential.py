from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from common.text import clean_text, normalize_for_match, stable_hash


@dataclass(slots=True)
class UnifiedSkillSourceLink:
    source: str
    source_id: str

    def to_dict(self) -> dict[str, Any]:
        return {'source': self.source, 'source_id': self.source_id}


@dataclass(slots=True)
class UnifiedSkill:
    canonical_skill_id: str
    canonical_label: str
    aliases: list[str] = field(default_factory=list)
    sources: list[UnifiedSkillSourceLink] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            'canonical_skill_id': self.canonical_skill_id,
            'canonical_label': self.canonical_label,
            'aliases': self.aliases,
            'sources': [item.to_dict() for item in self.sources],
        }


def canonical_skill_id(label: str) -> str:
    return stable_hash(normalize_for_match(label), length=24)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = '\n'.join(json.dumps(row, ensure_ascii=False) for row in rows)
    path.write_text(payload + ('\n' if rows else ''), encoding='utf-8')


class UnifiedSkillReferential:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path or 'data/referentials/unified/skills.jsonl')
        self._skills: list[UnifiedSkill] = []
        self._index: dict[str, UnifiedSkill] = {}

    def load(self) -> list[UnifiedSkill]:
        self._skills = []
        self._index = {}
        for row in _read_jsonl(self.path):
            skill = UnifiedSkill(
                canonical_skill_id=clean_text(row.get('canonical_skill_id') or ''),
                canonical_label=clean_text(row.get('canonical_label') or ''),
                aliases=[clean_text(alias) for alias in row.get('aliases', []) if clean_text(alias)],
                sources=[UnifiedSkillSourceLink(**item) for item in row.get('sources', []) if isinstance(item, dict) and item.get('source') and item.get('source_id')],
            )
            if not skill.canonical_skill_id or not skill.canonical_label:
                continue
            self._skills.append(skill)
            self._index[normalize_for_match(skill.canonical_label)] = skill
            for alias in skill.aliases:
                self._index[normalize_for_match(alias)] = skill
        return list(self._skills)

    def get_all_skills(self) -> list[dict[str, Any]]:
        return [item.to_dict() for item in self.load()]

    def search_exact(self, label: str) -> dict[str, Any] | None:
        if not self._skills:
            self.load()
        hit = self._index.get(normalize_for_match(label))
        return hit.to_dict() if hit else None

    def search_alias(self, label: str) -> dict[str, Any] | None:
        if not self._skills:
            self.load()
        target = normalize_for_match(label)
        for skill in self._skills:
            if target in {normalize_for_match(alias) for alias in skill.aliases}:
                return skill.to_dict()
        return None

    def search_semantic(self, label: str) -> dict[str, Any] | None:
        if not self._skills:
            self.load()
        target = normalize_for_match(label)
        best: tuple[float, UnifiedSkill | None] = (0.0, None)
        for skill in self._skills:
            score = 0.0
            for candidate in [skill.canonical_label, *skill.aliases]:
                ref = normalize_for_match(candidate)
                if not ref or not target:
                    continue
                if ref == target:
                    score = 1.0
                else:
                    overlap = len(set(ref.split()) & set(target.split())) / max(1, len(set(ref.split()) | set(target.split())))
                    prefix = 0.9 if (ref in target or target in ref) else 0.0
                    score = max(score, overlap, prefix)
            if score > best[0]:
                best = (score, skill)
        return best[1].to_dict() if best[1] and best[0] >= 0.5 else None

    def normalize_label(self, label: str | None) -> str:
        return normalize_for_match(clean_text(label))


def build_unified_skill_referential(
    *,
    france_competences: dict[str, list[dict[str, Any]]],
    rome: dict[str, list[dict[str, Any]]],
    mappings: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rncp_skills = {item['skill_id']: item for item in france_competences.get('skills', []) if item.get('skill_id')}
    rome_skills = {item['rome_skill_id']: item for item in rome.get('skills', []) if item.get('rome_skill_id')}
    rome_jobs = {item['rome_code']: item for item in rome.get('jobs', []) if item.get('rome_code')}

    grouped: dict[str, dict[str, Any]] = {}
    source_links: list[dict[str, Any]] = []

    def ensure_group(canonical_id: str, canonical_label: str) -> dict[str, Any]:
        group = grouped.setdefault(
            canonical_id,
            {
                'canonical_skill_id': canonical_id,
                'canonical_label': canonical_label,
                'aliases': [],
                'sources': [],
            },
        )
        if canonical_label and not group['canonical_label']:
            group['canonical_label'] = canonical_label
        return group

    for match in mappings:
        rncp_code = clean_text(match.get('rncp_code') or '')
        rome_code = clean_text(match.get('rome_code') or '')
        score = float(match.get('score', 0.0) or 0.0)
        if not rncp_code or not rome_code:
            continue
        canonical_id = canonical_skill_id(f'{rncp_code}|{rome_code}')
        rncp_label = next((item.get('official_label') or item.get('title') for item in rncp_skills.values() if item.get('rncp_code') == rncp_code), '')
        rome_label = rome_jobs.get(rome_code, {}).get('label', '')
        canonical_label = clean_text(rncp_label or rome_label or f'{rncp_code} / {rome_code}')
        group = ensure_group(canonical_id, canonical_label)
        group['sources'].append({'source': 'france_competences', 'source_id': rncp_code})
        group['sources'].append({'source': 'rome', 'source_id': rome_code})
        if rncp_label:
            group['aliases'].append(clean_text(rncp_label))
        if rome_label:
            group['aliases'].append(clean_text(rome_label))
        source_links.append({
            'canonical_skill_id': canonical_id,
            'canonical_label': canonical_label,
            'source': 'france_competences',
            'source_id': rncp_code,
            'linked_source': 'rome',
            'linked_source_id': rome_code,
            'score': round(score, 4),
        })

    for skill in rncp_skills.values():
        canonical_id = skill['skill_id']
        if any(link['source'] == 'france_competences' and link['source_id'] == skill['skill_id'] for link in source_links):
            continue
        label = clean_text(skill.get('official_label') or '')
        group = ensure_group(canonical_id, label)
        group['sources'].append({'source': 'france_competences', 'source_id': skill['skill_id']})
        group['aliases'].extend([clean_text(alias) for alias in skill.get('aliases', []) if clean_text(alias)])

    for skill in rome_skills.values():
        canonical_id = skill['rome_skill_id']
        if any(link['source'] == 'rome' and link['source_id'] == skill['rome_skill_id'] for link in source_links):
            continue
        label = clean_text(skill.get('official_label') or '')
        group = ensure_group(canonical_id, label)
        group['sources'].append({'source': 'rome', 'source_id': skill['rome_skill_id']})

    records: list[dict[str, Any]] = []
    for group in grouped.values():
        aliases = [alias for alias in dict.fromkeys(clean_text(alias) for alias in group['aliases'] if clean_text(alias)) if normalize_for_match(alias) != normalize_for_match(group['canonical_label'])]
        sources = [item for item in group['sources'] if item.get('source') and item.get('source_id')]
        records.append({
            'canonical_skill_id': group['canonical_skill_id'],
            'canonical_label': group['canonical_label'],
            'aliases': aliases,
            'sources': sources,
        })
    records.sort(key=lambda item: (item['canonical_label'], item['canonical_skill_id']))
    source_links.sort(key=lambda item: (item['source'], item['source_id'], item['linked_source'], item['linked_source_id']))
    return records, source_links


def write_unified_skill_referential(path: str | Path, records: list[dict[str, Any]]) -> Path:
    path = Path(path)
    _write_jsonl(path, records)
    return path
