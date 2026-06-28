from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import json

from common.text import clean_text, normalize_for_match


DEFAULT_REFERENTIAL = Path(__file__).resolve().parents[3] / "data" / "referentials" / "skills.json"
HARD_NEGATIVES: dict[str, set[str]] = {
    "java": {"javascript", "js", "typescript"},
    "javascript": {"java"},
    "python": {"pythons"},
    "ia": {"machine learning", "ml", "deep learning"},
}


@dataclass(frozen=True)
class SkillMatch:
    """Correspondance canonique de compétence."""

    canonical_id: str
    canonical_label: str
    original_label: str
    aliases: list[str]
    extraction_source: str
    confidence: float


@lru_cache(maxsize=8)
def load_referential(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Charge le référentiel commun de compétences."""

    referential_path = Path(path or DEFAULT_REFERENTIAL)
    if not referential_path.exists():
        raise FileNotFoundError(f"Référentiel de compétences introuvable: {referential_path}")
    payload = json.loads(referential_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Référentiel invalide: {referential_path}")
    return payload


def _token_set(value: str) -> set[str]:
    return {token for token in normalize_for_match(value).split() if token}


class SkillTaxonomyNormalizer:
    """Normalise les labels de compétences vers une taxonomie canonique."""

    def __init__(self, referential_path: str | Path | None = None) -> None:
        self.referential_path = Path(referential_path or DEFAULT_REFERENTIAL)
        self.reference = load_referential(self.referential_path)
        self._by_norm: dict[str, dict[str, Any]] = {}
        self._alias_to_skill: dict[str, dict[str, Any]] = {}
        for skill in self.reference:
            label = clean_text(skill.get("label"))
            if not label:
                continue
            skill = dict(skill)
            skill["label"] = label
            self._by_norm[normalize_for_match(label)] = skill
            self._alias_to_skill[normalize_for_match(label)] = skill
            for alias in skill.get("aliases", []) or []:
                alias_norm = normalize_for_match(alias)
                if alias_norm:
                    self._alias_to_skill[alias_norm] = skill

    def aliases_for_label(self, canonical_label: str) -> list[str]:
        """Retourne les alias connus d'un label canonique."""

        target = clean_text(canonical_label)
        for skill in self.reference:
            if clean_text(skill.get("label")) == target:
                return [clean_text(alias) for alias in skill.get("aliases", []) or [] if clean_text(alias)]
        return []

    def _is_blocked(self, candidate_norm: str, skill_norm: str) -> bool:
        tokens = _token_set(candidate_norm)
        blocked = HARD_NEGATIVES.get(skill_norm, set())
        return any(blocked_token in tokens or blocked_token == candidate_norm for blocked_token in blocked)

    def normalize(
        self,
        candidate: str,
        *,
        extraction_source: str = "text_explicit",
        confidence_floor: float = 0.6,
    ) -> SkillMatch | None:
        """Rattache une compétence candidate à un label canonique."""

        original = clean_text(candidate)
        if not original:
            return None
        norm = normalize_for_match(original)
        if not norm:
            return None

        # Correspondance exacte d'abord.
        direct = self._alias_to_skill.get(norm)
        if direct:
            skill_norm = normalize_for_match(direct["label"])
            if not self._is_blocked(norm, skill_norm):
                return SkillMatch(
                    canonical_id=str(direct.get("skill_id") or skill_norm),
                    canonical_label=str(direct["label"]),
                    original_label=original,
                    aliases=self.aliases_for_label(str(direct["label"])),
                    extraction_source=extraction_source,
                    confidence=1.0,
                )

        # Containment prudent sur les labels longs.
        best_skill: dict[str, Any] | None = None
        best_score = 0.0
        for alias_norm, skill in self._alias_to_skill.items():
            skill_norm = normalize_for_match(skill["label"])
            if self._is_blocked(norm, skill_norm):
                continue
            if len(alias_norm) < 3 or len(norm) < 3:
                continue
            candidate_tokens = _token_set(norm)
            alias_tokens = _token_set(alias_norm)
            token_overlap = len(candidate_tokens & alias_tokens)
            if alias_norm in norm or norm in alias_norm or token_overlap:
                score = 0.75 if token_overlap else 0.68
                if alias_norm == skill_norm:
                    score = max(score, 0.8)
                if score > best_score:
                    best_skill = skill
                    best_score = score

        if best_skill and best_score >= confidence_floor:
            return SkillMatch(
                canonical_id=str(best_skill.get("skill_id") or normalize_for_match(best_skill["label"])),
                canonical_label=str(best_skill["label"]),
                original_label=original,
                aliases=self.aliases_for_label(str(best_skill["label"])),
                extraction_source=extraction_source,
                confidence=best_score,
            )
        return None

    def normalize_many(
        self,
        candidates: Iterable[str],
        *,
        extraction_source: str = "text_explicit",
        confidence_floor: float = 0.6,
    ) -> list[SkillMatch]:
        """Normalise une liste de labels candidats."""

        results: list[SkillMatch] = []
        seen: set[str] = set()
        for candidate in candidates:
            match = self.normalize(candidate, extraction_source=extraction_source, confidence_floor=confidence_floor)
            if not match:
                continue
            if match.canonical_id in seen:
                continue
            seen.add(match.canonical_id)
            results.append(match)
        return results

