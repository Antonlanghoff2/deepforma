from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from common.text import clean_text, normalize_for_match, split_multi_values
from skills.skill_normalizer import SkillNormalizer


DEFAULT_NORMALIZER = SkillNormalizer()


def extract_skills_from_text(text: str, normalizer: SkillNormalizer | None = None) -> list[dict[str, Any]]:
    normalizer = normalizer or DEFAULT_NORMALIZER
    text = clean_text(text)
    if not text:
        return []
    normalized_text = normalize_for_match(text)
    found: list[dict[str, Any]] = []
    for skill in normalizer.reference:
        label = skill.get("label", "")
        aliases = [label, *(skill.get("aliases", []) or [])]
        for alias in aliases:
            alias_norm = normalize_for_match(alias)
            if alias_norm and len(alias_norm) >= 4 and f" {alias_norm} " in f" {normalized_text} ":
                found.append(
                    {
                        "label": label,
                        "canonical_label": label,
                        "sources": ["text_explicit"],
                        "confidence": 0.8,
                    }
                )
                break
    return found


def merge_offer_skills(
    *,
    structured_skills: Iterable[dict[str, Any]] | None = None,
    explicit_skills: Iterable[dict[str, Any]] | None = None,
    model_skills: Iterable[dict[str, Any]] | None = None,
    rome_skills: Iterable[dict[str, Any]] | None = None,
    normalizer: SkillNormalizer | None = None,
) -> list[dict[str, Any]]:
    normalizer = normalizer or DEFAULT_NORMALIZER
    merged: dict[str, dict[str, Any]] = {}

    def add(skill: dict[str, Any], source: str, confidence_boost: float = 0.0) -> None:
        label = clean_text(skill.get("label") or skill.get("canonical_label") or "")
        if not label:
            return
        canonical_label, conf, _ = normalizer.normalize(label)
        canonical_label = canonical_label or label
        key = normalize_for_match(canonical_label)
        if key not in merged:
            merged[key] = {
                "label": label,
                "canonical_label": canonical_label,
                "sources": [],
                "confidence": 0.0,
            }
        item = merged[key]
        item["label"] = item["canonical_label"]
        if source not in item["sources"]:
            item["sources"].append(source)
        item["confidence"] = max(item["confidence"], float(skill.get("confidence", 0.0)), conf, confidence_boost)
        if source == "france_travail_structured":
            item["confidence"] = 1.0

    for skill in structured_skills or []:
        add(skill, "france_travail_structured", confidence_boost=1.0)
    for skill in explicit_skills or []:
        add(skill, "text_explicit", confidence_boost=0.8)
    for skill in model_skills or []:
        add(skill, "camembert_multilabel")
    for skill in rome_skills or []:
        add(skill, "rome_implicit", confidence_boost=0.6)

    return sorted(merged.values(), key=lambda item: (-item["confidence"], item["canonical_label"]))

