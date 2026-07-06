from __future__ import annotations

from typing import Sequence

from common.text import normalize_for_match
from domain.models import JobOffer, MarketAnalysis, Recommendation, Skill, Territory
from services.recommendation_service import RecommendationService


def _skill_labels(skills: Sequence[Skill]) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()
    for skill in skills:
        label = skill.normalized_name or skill.name
        key = normalize_for_match(label)
        if not key or key in seen:
            continue
        seen.add(key)
        labels.append(label)
    return labels


def _offer_to_payload(offer: JobOffer) -> dict[str, object]:
    return {
        "title": offer.title,
        "description": offer.description,
        "rome_code": offer.rome_code,
        "location": offer.location,
        "normalized_skills": [
            {"canonical_label": skill.normalized_name or skill.name, "label": skill.name}
            for skill in offer.skills
        ],
    }


def analyze_market_fit(
    profile_skills: list[Skill],
    territory: Territory,
    offers: list[JobOffer],
    certifications: list[object] | None = None,
    trainings: list[object] | None = None,
) -> MarketAnalysis:
    """Analyse minimale du marché à partir du service de recommandation existant.

    Cette façade sert de point de convergence futur sans casser les implémentations historiques.
    """

    service = RecommendationService()
    formation_skills = _skill_labels(profile_skills)
    payload_offers = [_offer_to_payload(offer) for offer in offers]
    report = service.compare(formation_skills, payload_offers)

    common_skills = list(report.covered_skills)
    missing_skills = [item.label for item in report.missing_priority_skills]
    top_demanded_skills = [item.label for item in report.market_skills[:10]]
    recommendations: list[Recommendation] = [
        Recommendation(
            target_id=offer.offer_id,
            title=offer.title,
            score=float(index + 1),
            reasons=[],
            explanation=offer.description[:160] if offer.description else None,
            source="job_offer",
            metadata={"rome_code": offer.rome_code, "location": offer.location},
        )
        for index, offer in enumerate(offers[:5])
    ]

    return MarketAnalysis(
        profile_skills=profile_skills,
        territory=territory,
        offer_count=report.offer_count,
        coverage_score=float(report.coverage_score),
        common_skills=common_skills,
        missing_skills=missing_skills,
        top_demanded_skills=top_demanded_skills,
        recommendations=recommendations,
    )
