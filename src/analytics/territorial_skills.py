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
        if offer.get("rome_code"):
            dominant_romes[str(offer["rome_code"])] += 1
        creation_date = offer.get("creation_date")
        if creation_date:
            try:
                dt = datetime.fromisoformat(str(creation_date).replace("Z", "+00:00"))
                monthly_evolution[dt.strftime("%Y-%m")] += 1
            except ValueError:
                pass

    total_offers = len(offers)
    skill_share = {
        skill: round(count / total_offers * 100, 2) if total_offers else 0.0
        for skill, count in skill_counts.items()
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

