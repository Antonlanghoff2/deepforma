from __future__ import annotations

from domain.models import JobOffer, Skill, Territory
from services.market_analysis import analyze_market_fit


def test_domain_models_are_instantiable() -> None:
    skill = Skill(name="Python", normalized_name="python", category="tool", source="manual", confidence=0.9)
    offer = JobOffer(
        offer_id="offer-1",
        title="Data Scientist",
        description="Analyse de données et machine learning.",
        skills=[skill],
        rome_code="M1805",
        location="Paris",
    )
    territory = Territory(code="75", label="Paris", department_code="75", region_code="11", remote_allowed=True)

    assert offer.skills[0].name == "Python"
    assert territory.department_code == "75"


def test_analyze_market_fit_uses_existing_recommendation_service() -> None:
    profile_skills = [Skill(name="Python"), Skill(name="SQL")]
    territory = Territory(code="75", label="Paris", department_code="75", region_code="11", remote_allowed=True)
    offers = [
        JobOffer(
            offer_id="offer-1",
            title="Data Scientist",
            description="Recherche Python et SQL pour la modélisation.",
            skills=[Skill(name="Python"), Skill(name="SQL")],
            rome_code="M1805",
            location="Paris",
        )
    ]

    result = analyze_market_fit(profile_skills, territory, offers)

    assert result.offer_count == 1
    assert result.coverage_score >= 0
    assert "Python" in result.common_skills or "python" in result.common_skills
