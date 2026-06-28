from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable

import pandas as pd


@dataclass
class TerritorialSkillStats:
    territory_key: str
    offer_count: int
    skill_counts: dict[str, int]
    skill_share: dict[str, float]
    required_skills: list[str]
    desired_skills: list[str]
    contract_types: dict[str, int]
    monthly_evolution: dict[str, int]
    dominant_romes: dict[str, int]
    experience_levels: dict[str, int] = None
    diplomas: dict[str, int] = None
    telework: dict[str, int] = None
    salaries: dict[str, float] = None
    sectors: list[str] = None
    rome_labels: dict[str, str] = None

    def __post_init__(self):
        if self.experience_levels is None:
            self.experience_levels = {}
        if self.diplomas is None:
            self.diplomas = {}
        if self.telework is None:
            self.telework = {}
        if self.salaries is None:
            self.salaries = {}
        if self.sectors is None:
            self.sectors = []
        if self.rome_labels is None:
            self.rome_labels = {}


def _normalize_skill_list(values: Iterable[Any]) -> list[str]:
    cleaned: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in cleaned:
            cleaned.append(text)
    return cleaned


def compute_territorial_stats(offers: list[dict[str, Any]], territory_key: str) -> TerritorialSkillStats:
    skill_counts: Counter[str] = Counter()
    required_skills: set[str] = set()
    desired_skills: set[str] = set()
    contract_types: Counter[str] = Counter()
    monthly_evolution: Counter[str] = Counter()
    dominant_romes: Counter[str] = Counter()
    rome_labels: dict[str, str] = {}
    experience_levels: Counter[str] = Counter()
    diplomas: Counter[str] = Counter()
    telework: Counter[str] = Counter()
    salaries_list: list[float] = []
    sectors: set[str] = set()

    for offer in offers:
        skills = _normalize_skill_list(offer.get("normalized_skills", []))
        for skill in skills:
            skill_counts[skill] += 1

        for skill in offer.get("structured_skills", []):
            label = str(skill.get("label", "")).strip()
            if not label:
                continue
            requirement = str(skill.get("requirement", "") or "").lower()
            if "requ" in requirement or "oblig" in requirement or "must" in requirement:
                required_skills.add(label)
            else:
                desired_skills.add(label)

        contract = offer.get("contract_type") or offer.get("contract_label")
        if contract:
            contract_types[str(contract)] += 1

        rome_code = offer.get("rome_code")
        if rome_code:
            code_str = str(rome_code)
            dominant_romes[code_str] += 1
            rome_label = offer.get("rome_label") or offer.get("rome_appellation")
            if rome_label and code_str not in rome_labels:
                rome_labels[code_str] = str(rome_label)

        creation_date = offer.get("creation_date")
        if creation_date:
            try:
                dt = datetime.fromisoformat(str(creation_date).replace("Z", "+00:00"))
                monthly_evolution[dt.strftime("%Y-%m")] += 1
            except ValueError:
                pass

        exp_level = offer.get("experience_level") or offer.get("experience")
        if exp_level:
            experience_levels[str(exp_level)] += 1

        diploma = offer.get("diploma") or offer.get("diplome") or offer.get("education_level")
        if diploma:
            diplomas[str(diploma)] += 1

        telework_val = offer.get("telework") or offer.get("teleworking")
        if telework_val:
            telework[str(telework_val)] += 1

        salary = offer.get("salary") or offer.get("salaire")
        if salary:
            try:
                salaries_list.append(float(salary))
            except (TypeError, ValueError):
                pass

        sector = offer.get("sector") or offer.get("secteur")
        if sector:
            sectors.add(str(sector))

    total_offers = len(offers)
    skill_share = {
        skill: round(count / total_offers * 100, 2) if total_offers else 0.0
        for skill, count in skill_counts.items()
    }

    salary_stats = None
    if salaries_list:
        salary_stats = {
            "min": round(min(salaries_list), 2),
            "max": round(max(salaries_list), 2),
            "mean": round(sum(salaries_list) / len(salaries_list), 2),
            "count": len(salaries_list),
        }

    return TerritorialSkillStats(
        territory_key=territory_key,
        offer_count=total_offers,
        skill_counts=dict(skill_counts),
        skill_share=skill_share,
        required_skills=sorted(required_skills),
        desired_skills=sorted(desired_skills),
        contract_types=dict(contract_types),
        monthly_evolution=dict(monthly_evolution),
        dominant_romes=dict(dominant_romes),
        experience_levels=dict(experience_levels),
        diplomas=dict(diplomas),
        telework=dict(telework),
        salaries=salary_stats,
        sectors=sorted(sectors),
        rome_labels=rome_labels,
    )


def stats_to_dataframe(stats: TerritorialSkillStats) -> pd.DataFrame:
    rows = []
    for skill, count in sorted(stats.skill_counts.items(), key=lambda item: (-item[1], item[0])):
        rows.append(
            {
                "territory_key": stats.territory_key,
                "skill": skill,
                "count": count,
                "share_percent": stats.skill_share.get(skill, 0.0),
                "offer_count": stats.offer_count,
            }
        )
    return pd.DataFrame(rows)
