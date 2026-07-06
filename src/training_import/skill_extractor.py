from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable

from common.text import clean_text, normalize_for_match

from .models import FieldEvidence, TrainingDomain, TrainingPrerequisite, TrainingSkill, TrainingTool


ALIASES = {
    "scikit learn": "scikit-learn",
    "sklearn": "scikit-learn",
    "powerbi": "Power BI",
    "aws gcp": "AWS, GCP",
    "tensorflow 2": "TensorFlow",
    "keras": "Keras",
    "ci cd": "CI/CD",
    "ia generatives": "IA générative",
}

TOOL_HINTS = {
    "excel", "crm", "slack", "google drive", "teams", "power bi", "aws", "gcp", "docker", "kubernetes", "airflow", "mlflow", "chatgpt",
}
METHOD_HINTS = {
    "lean management", "pdca", "5s", "qqoqcp", "agile", "scrum", "ci/cd", "mlops", "prompt engineering", "rag",
}
DOMAIN_HINTS = {"data science", "machine learning", "deep learning", "ia générative", "llm", "sales", "marketing"}
SOFT_SKILL_HINTS = {"leadership", "communication", "collaboration", "coordination", "négociation", "negociation", "autonomie", "rigueur"}


def _alias(label: str) -> str:
    key = normalize_for_match(label)
    for source, target in ALIASES.items():
        if source in key:
            return target
    return clean_text(label)


def _classify(label: str) -> str:
    normalized = normalize_for_match(label)
    if any(token in normalized for token in TOOL_HINTS):
        return "tool"
    if any(token in normalized for token in METHOD_HINTS):
        return "method"
    if any(token in normalized for token in DOMAIN_HINTS):
        return "domain"
    if any(token in normalized for token in SOFT_SKILL_HINTS):
        return "soft_skill"
    return "skill"


def _split_candidates(text: str) -> list[str]:
    pieces = re.split(r"[\n;•·\u2022,/]|(?:\s+-\s+)", clean_text(text))
    return [clean_text(piece) for piece in pieces if clean_text(piece)]


def _extract_candidates(section_name: str, text: str, page: int, learning_stage: str) -> list[TrainingSkill | TrainingTool | TrainingDomain]:
    results: list[TrainingSkill | TrainingTool | TrainingDomain] = []
    seen: set[str] = set()
    for fragment in _split_candidates(text):
        canonical = _alias(fragment)
        if not canonical:
            continue
        key = normalize_for_match(canonical)
        if key in seen:
            continue
        seen.add(key)
        category = _classify(canonical)
        evidence = [FieldEvidence("skill", section_name, fragment, page, confidence=0.75, method="explicit_list")]
        if category == "tool":
            results.append(TrainingTool(canonical_name=canonical, surface_form=fragment, source_section=section_name, source_text=text, page=page, confidence=0.85, method="explicit_list", evidence=evidence))
        elif category == "domain":
            results.append(TrainingDomain(canonical_name=canonical, surface_form=fragment, source_section=section_name, source_text=text, page=page, confidence=0.78, method="explicit_list", evidence=evidence))
        else:
            results.append(TrainingSkill(canonical_name=canonical, surface_form=fragment, type=category, source_section=section_name, source_text=text, page=page, confidence=0.8, method="explicit_list", learning_stage=learning_stage, evidence=evidence))
    return results


def extract_skills(sections: dict[str, str], modules: Iterable[str], prerequisites: str = "") -> tuple[list[TrainingSkill], list[TrainingTool], list[TrainingDomain], list[TrainingPrerequisite]]:
    skills: OrderedDict[str, TrainingSkill] = OrderedDict()
    tools: OrderedDict[str, TrainingTool] = OrderedDict()
    domains: OrderedDict[str, TrainingDomain] = OrderedDict()
    prereqs: list[TrainingPrerequisite] = []

    if prerequisites:
        for fragment in _split_candidates(prerequisites):
            prereqs.append(TrainingPrerequisite(text=fragment, page=1, confidence=0.65, evidence=[FieldEvidence("prerequisite", "prerequisites", fragment, 1, confidence=0.65, method="explicit_list")]))
            canonical = _alias(fragment)
            key = normalize_for_match(canonical)
            skills.setdefault(key, TrainingSkill(canonical_name=canonical, surface_form=fragment, type="skill", source_section="prerequisites", source_text=prerequisites, page=1, confidence=0.55, method="explicit_list", learning_stage="prerequisite", evidence=[FieldEvidence("skill", "prerequisites", fragment, 1, confidence=0.55, method="explicit_list")]))

    for section_name in ("skills", "objectives", "program"):
        text = sections.get(section_name, "")
        if not text:
            continue
        for result in _extract_candidates(section_name, text, 1, "taught" if section_name != "prerequisites" else "prerequisite"):
            if isinstance(result, TrainingTool):
                tools.setdefault(normalize_for_match(result.canonical_name), result)
            elif isinstance(result, TrainingDomain):
                domains.setdefault(normalize_for_match(result.canonical_name), result)
            else:
                existing = skills.get(normalize_for_match(result.canonical_name))
                if existing:
                    existing.evidence.extend(result.evidence)
                    existing.confidence = max(existing.confidence, result.confidence)
                    if existing.learning_stage == "prerequisite" and result.learning_stage == "taught":
                        existing.learning_stage = "taught"
                else:
                    skills[normalize_for_match(result.canonical_name)] = result

    for module in modules:
        for result in _extract_candidates("program", module, 1, "taught"):
            if isinstance(result, TrainingTool):
                tools.setdefault(normalize_for_match(result.canonical_name), result)
            elif isinstance(result, TrainingDomain):
                domains.setdefault(normalize_for_match(result.canonical_name), result)
            else:
                existing = skills.get(normalize_for_match(result.canonical_name))
                if existing:
                    existing.evidence.extend(result.evidence)
                    existing.confidence = max(existing.confidence, result.confidence)
                else:
                    skills[normalize_for_match(result.canonical_name)] = result

    return list(skills.values()), list(tools.values()), list(domains.values()), prereqs
