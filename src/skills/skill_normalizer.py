from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from src.common.text import clean_text, normalize_for_match


DEFAULT_SKILLS_PATH = Path(__file__).resolve().parents[2] / "data" / "referentials" / "skills.json"


@dataclass(frozen=True)
class NormalizedSkill:
    skill_id: str
    label: str
    category: str
    confidence: float
    matched_alias: str | None = None


@lru_cache(maxsize=8)
def load_referential(skills_path: str | Path | None = None) -> list[dict[str, Any]]:
    path = Path(skills_path or DEFAULT_SKILLS_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Référentiel de compétences introuvable: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


class SkillNormalizer:
    def __init__(self, skills_path: str | Path | None = None) -> None:
        self.skills_path = Path(skills_path or DEFAULT_SKILLS_PATH)
        self.reference = load_referential(self.skills_path)
        self._index: list[tuple[dict[str, Any], str]] = []
        for skill in self.reference:
            self._index.append((skill, normalize_for_match(skill.get("label", ""))))
            for alias in skill.get("aliases", []) or []:
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
            label_norm = normalize_for_match(skill.get("label", ""))
            if norm == label_norm or norm == alias_norm:
                return skill["label"], 1.0, skill["skill_id"]
            if norm in label_norm or label_norm in norm:
                if len(norm) >= 6 and len(label_norm) >= 6:
                    score = 0.75
                    if score > best_score:
                        best_skill = skill
                        best_alias = skill["skill_id"]
                        best_score = score
            elif alias_norm and norm in alias_norm:
                if len(norm) >= 6:
                    score = 0.65
                    if score > best_score:
                        best_skill = skill
                        best_alias = skill["skill_id"]
                        best_score = score

        if best_skill:
            return best_skill["label"], best_score, best_alias
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
                            "skill_id": skill_id,
                            "label": label,
                            "category": self.category_for_label(label),
                            "confidence": confidence,
                        }
                    )
            else:
                cleaned = clean_text(candidate)
                if cleaned:
                    unknowns.append(cleaned)
        return normalized, unknowns

    def category_for_label(self, label: str) -> str:
        for skill in self.reference:
            if skill.get("label") == label:
                return skill.get("category", "unknown")
        return "unknown"

