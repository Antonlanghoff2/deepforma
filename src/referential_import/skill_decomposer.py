from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from common.text import clean_text, normalize_for_match

from .models import DerivedSkill, EvaluationCriterion, OfficialCompetency


TOOL_METHOD_MAP: dict[str, tuple[str, str]] = {
    "excel": ("Tool", "Excel"),
    "crm": ("Tool", "CRM"),
    "slack": ("Tool", "Slack"),
    "google drive": ("Tool", "Google Drive"),
    "google workspace": ("Tool", "Google Workspace"),
    "teams": ("Tool", "Teams"),
    "microsoft teams": ("Tool", "Teams"),
    "lean management": ("Method", "Lean Management"),
    "pdca": ("Method", "PDCA"),
    "5s": ("Method", "5S"),
    "qqoqcp": ("Method", "QQOQCP"),
    "rfm": ("Method", "Scoring RFM"),
    "abc": ("Method", "Méthode ABC"),
    "pareto": ("Method", "Pareto"),
    "icp": ("Method", "ICP"),
    "roi": ("Knowledge", "ROI"),
    "rgpd": ("Regulatory", "RGPD"),
}

SOFT_SKILL_HINTS = {
    "coordination": "Coordination d'équipes",
    "coopération": "Coopération",
    "cooperation": "Coopération",
    "communication": "Communication",
    "organisation": "Organisation",
    "pilotage": "Pilotage",
    "analyse": "Analyse",
    "analyse critique": "Analyse critique",
    "synthèse": "Synthèse",
    "synthetise": "Synthèse",
    "autonomie": "Autonomie",
    "rigueur": "Rigueur",
}

ACTION_RE = re.compile(r"\b(?:organiser|coordonner|piloter|mettre en oeuvre|mettre en place|développer|developper|analyser|assurer|gérer|gerer|utiliser|maîtriser|maitriser)\b", re.IGNORECASE)
SPLIT_RE = re.compile(r"[\n;•·\u2022,/]")


def _add_skill(
    skills: list[DerivedSkill],
    *,
    label: str,
    canonical_label: str,
    category: str,
    source_code: str,
    source_type: str,
    surface_form: str,
    provenance: str,
    confidence: float,
    page_start: int,
    page_end: int,
    context: str,
    explicit: bool = False,
) -> None:
    key = normalize_for_match(canonical_label)
    if not key:
        return
    for item in skills:
        if normalize_for_match(item.canonical_label) == key:
            item.confidence = max(item.confidence, confidence)
            return
    skills.append(
        DerivedSkill(
            label=label,
            canonical_label=canonical_label,
            category=category,
            source_code=source_code,
            source_type=source_type,
            surface_form=surface_form,
            normalized_surface=normalize_for_match(surface_form),
            provenance=provenance,  # type: ignore[arg-type]
            confidence=confidence,
            explicit=explicit,
            page_start=page_start,
            page_end=page_end,
            context=context,
        )
    )


def _extract_terms_from_text(text: str) -> list[str]:
    parts = [clean_text(part) for part in SPLIT_RE.split(text) if clean_text(part)]
    chunks: list[str] = []
    for part in parts:
        normalized = normalize_for_match(part)
        if not normalized:
            continue
        for token, (_, canonical) in TOOL_METHOD_MAP.items():
            if token in normalized:
                chunks.append(canonical)
        for token, canonical in SOFT_SKILL_HINTS.items():
            if token in normalized:
                chunks.append(canonical)
    return chunks


def decompose_competency(
    competency: OfficialCompetency,
    criteria: list[EvaluationCriterion] | None = None,
) -> list[DerivedSkill]:
    criteria = criteria or competency.evaluation_criteria
    skills: list[DerivedSkill] = []
    texts = [competency.official_label, competency.source_text, *(criterion.criterion_label for criterion in criteria)]

    for text in texts:
        normalized = normalize_for_match(text)
        if not normalized:
            continue
        for token, (category, canonical) in TOOL_METHOD_MAP.items():
            if token in normalized:
                _add_skill(
                    skills,
                    label=canonical,
                    canonical_label=canonical,
                    category=category.lower(),
                    source_code=competency.code,
                    source_type="competency",
                    surface_form=token,
                    provenance="semantic_match",
                    confidence=0.96 if token in normalize_for_match(competency.official_label) else 0.88,
                    page_start=competency.page_start,
                    page_end=competency.page_end,
                    context=text,
                    explicit=True,
                )
        for label in _extract_terms_from_text(text):
            category = "soft_skill" if label in {"Communication", "Organisation", "Coordination d'équipes", "Coopération", "Pilotage", "Analyse", "Analyse critique", "Synthèse", "Autonomie", "Rigueur"} else "knowledge"
            if label in {"Excel", "CRM", "Slack", "Google Drive", "Google Workspace", "Teams"}:
                category = "tool"
            elif label in {"Lean Management", "PDCA", "5S", "QQOQCP", "Scoring RFM", "Méthode ABC", "Pareto", "ICP"}:
                category = "method"
            elif label == "RGPD":
                category = "regulatory"
            _add_skill(
                skills,
                label=label,
                canonical_label=label,
                category=category,
                source_code=competency.code,
                source_type="competency" if text == competency.official_label else "criterion",
                surface_form=label,
                provenance="semantic_match",
                confidence=0.82,
                page_start=competency.page_start,
                page_end=competency.page_end,
                context=text,
                explicit=False,
            )

        if ACTION_RE.search(text):
            for fragment in re.split(r"\b(?:et|ou|en|par|pour|avec)\b", text, flags=re.IGNORECASE):
                fragment = clean_text(fragment)
                if len(fragment) < 4:
                    continue
                canonical = fragment[:120]
                category = "action"
                if "management" in normalize_for_match(fragment):
                    category = "method"
                _add_skill(
                    skills,
                    label=canonical,
                    canonical_label=canonical,
                    category=category,
                    source_code=competency.code,
                    source_type="competency",
                    surface_form=fragment,
                    provenance="semantic_match",
                    confidence=0.55,
                    page_start=competency.page_start,
                    page_end=competency.page_end,
                    context=text,
                )

    return skills

