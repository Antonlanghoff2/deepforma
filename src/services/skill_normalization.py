from __future__ import annotations

from functools import lru_cache
from typing import Any, Iterable

from common.text import clean_text, normalize_for_match
from skills.skill_normalizer import SkillNormalizer


@lru_cache(maxsize=1)
def get_skill_normalizer() -> SkillNormalizer:
    return SkillNormalizer()


def normalize_skill_label(label: str, *, normalizer: SkillNormalizer | None = None) -> str | None:
    cleaned = clean_text(label)
    if not cleaned:
        return None
    active_normalizer = normalizer or get_skill_normalizer()
    canonical, _, _ = active_normalizer.normalize(cleaned)
    if canonical:
        return canonical
    normalized = normalize_for_match(cleaned)
    return normalized or None


def normalize_skill_labels(labels: Iterable[Any], *, normalizer: SkillNormalizer | None = None) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for label in labels:
        if isinstance(label, dict):
            candidate = label.get("label") or label.get("canonical_label") or label.get("name")
        else:
            candidate = label
        normalized_label = normalize_skill_label(str(candidate or ""), normalizer=normalizer)
        if not normalized_label:
            continue
        key = normalize_for_match(normalized_label)
        if key and key not in seen:
            seen.add(key)
            normalized.append(normalized_label)
    return normalized


def normalize_offer_skill_labels(offer: dict[str, Any], *, normalizer: SkillNormalizer | None = None) -> list[str]:
    labels: list[str] = []
    for key in ("normalized_skills", "merged_skills", "structured_skills", "model_skills"):
        values = offer.get(key) or []
        labels.extend(normalize_skill_labels(values, normalizer=normalizer))
    seen: set[str] = set()
    deduped: list[str] = []
    for label in labels:
        key = normalize_for_match(label)
        if key and key not in seen:
            seen.add(key)
            deduped.append(label)
    return deduped
