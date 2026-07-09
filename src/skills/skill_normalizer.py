from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from common.text import clean_text, normalize_for_match


LOGGER = logging.getLogger(__name__)

DEFAULT_SKILLS_PATH = Path(__file__).resolve().parents[2] / "data" / "referentials" / "skills.json"


class SkillReferentialNotFoundError(FileNotFoundError):
    pass


@dataclass(frozen=True)
class NormalizedSkill:
    skill_id: str
    label: str
    category: str
    confidence: float
    matched_alias: str | None = None


@dataclass
class LoadReport:
    path: str
    total: int = 0
    valid: int = 0
    ignored: int = 0
    errors: list[str] = field(default_factory=list)


def normalize_skill_entry(entry: Any) -> dict | None:
    if isinstance(entry, str):
        text = clean_text(entry)
        if not text:
            LOGGER.warning('Entrée chaîne vide ignorée dans le référentiel')
            return None
        return {'label': text, 'aliases': [], 'skill_id': text}
    if isinstance(entry, dict):
        label = (clean_text(entry.get('label'))
                 or clean_text(entry.get('name'))
                 or clean_text(entry.get('skill'))
                 or clean_text(entry.get('competence'))
                 or clean_text(entry.get('title')))
        if not label:
            LOGGER.warning('Entrée dictionnaire sans label ignorée : %s', entry)
            return None
        result = dict(entry)
        result['label'] = label
        result.setdefault('aliases', [])
        if 'skill_id' not in result or not result['skill_id']:
            result['skill_id'] = label
        return result
    LOGGER.warning('Entrée de type %s ignorée dans le référentiel', type(entry).__name__)
    return None


@lru_cache(maxsize=8)
def load_referential(skills_path: str | Path | None = None) -> list[dict[str, Any]]:
    path = Path(skills_path or DEFAULT_SKILLS_PATH)
    if not path.exists():
        raise SkillReferentialNotFoundError(f'Référentiel de compétences introuvable: {path}')
    raw = json.loads(path.read_text(encoding='utf-8'))
    if isinstance(raw, dict):
        for key in ('skills', 'competencies', 'competences', 'entries', 'items', 'data'):
            entries = raw.get(key)
            if isinstance(entries, list):
                return entries
        LOGGER.warning('Aucune liste de compétences trouvée dans %s (clés: %s)', path, list(raw.keys()))
        return []
    if isinstance(raw, list):
        return raw
    LOGGER.warning('Format inattendu dans %s (type: %s)', path, type(raw).__name__)
    return []


class SkillNormalizer:
    def __init__(self, skills_path: str | Path | None = None) -> None:
        self.skills_path = Path(skills_path or DEFAULT_SKILLS_PATH)
        self.load_report: LoadReport | None = None
        self.reference: list[dict[str, Any]] = []
        self._index: list[tuple[dict[str, Any], str]] = []
        self._load()

    def _load(self) -> None:
        try:
            raw = load_referential(self.skills_path)
        except SkillReferentialNotFoundError as exc:
            LOGGER.warning('Référentiel de compétences non trouvé: %s', exc)
            self.load_report = LoadReport(path=str(self.skills_path), total=0, valid=0, ignored=0, errors=[str(exc)])
            return
        except Exception as exc:
            LOGGER.warning('Erreur de chargement du référentiel %s: %s', self.skills_path, exc)
            self.load_report = LoadReport(path=str(self.skills_path), total=0, valid=0, ignored=0, errors=[str(exc)])
            return

        report = LoadReport(path=str(self.skills_path))
        normalized: list[dict[str, Any]] = []
        for entry in raw:
            report.total += 1
            skill = normalize_skill_entry(entry)
            if skill is None:
                report.ignored += 1
                continue
            report.valid += 1
            normalized.append(skill)

        self.reference = normalized
        self._build_index()
        self.load_report = report
        if report.ignored > 0:
            LOGGER.warning('Référentiel %s: %d entrées lues, %d valides, %d ignorées',
                           self.skills_path, report.total, report.valid, report.ignored)

    def _build_index(self) -> None:
        self._index = []
        for skill in self.reference:
            self._index.append((skill, normalize_for_match(skill.get('label', ''))))
            for alias in skill.get('aliases', []) or []:
                self._index.append((skill, normalize_for_match(alias)))

    def normalize(self, candidate: str) -> tuple[str | None, float, str | None]:
        text = clean_text(candidate)
        if not text:
            return None, 0.0, None
        norm = normalize_for_match(text)
        if not norm:
            return None, 0.0, None

        best_skill: dict[str, Any] | None = None
        best_alias: str | None = None
        best_score = 0.0
        for skill, alias_norm in self._index:
            label_norm = normalize_for_match(skill.get('label', ''))
            if norm == label_norm or norm == alias_norm:
                return skill['label'], 1.0, skill.get('skill_id')
            if norm in label_norm or label_norm in norm:
                if len(norm) >= 6 and len(label_norm) >= 6:
                    score = 0.75
                    if score > best_score:
                        best_skill = skill
                        best_alias = skill.get('skill_id')
                        best_score = score
            elif alias_norm and norm in alias_norm:
                if len(norm) >= 6:
                    score = 0.65
                    if score > best_score:
                        best_skill = skill
                        best_alias = skill.get('skill_id')
                        best_score = score

        if best_skill:
            return best_skill['label'], best_score, best_alias
        return None, 0.0, None

    def normalize_many(self, candidates: Iterable[str]) -> tuple[list[dict[str, Any]], list[str]]:
        normalized: list[dict[str, Any]] = []
        unknowns: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            label, confidence, skill_id = self.normalize(candidate)
            if label:
                key = normalize_for_match(label)
                if key not in seen:
                    seen.add(key)
                    normalized.append(
                        {
                            'skill_id': skill_id,
                            'label': label,
                            'category': self.category_for_label(label),
                            'confidence': confidence,
                        }
                    )
            else:
                cleaned = clean_text(candidate)
                if cleaned:
                    unknowns.append(cleaned)
        return normalized, unknowns

    def category_for_label(self, label: str) -> str:
        for skill in self.reference:
            if skill.get('label') == label:
                return skill.get('category', 'unknown')
        return 'unknown'
