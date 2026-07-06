from __future__ import annotations

from services.market_context import build_market_context
from services.recommendation_service import RecommendationService


class SkillExtractionItem:
    def __init__(self, label: str) -> None:
        self.normalized_label = label


class SkillExtractionResult:
    def __init__(self) -> None:
        self.skills = [SkillExtractionItem('Python')]
        self.tools = [SkillExtractionItem('SQL')]


def test_build_market_context_returns_recommendation_and_analysis() -> None:
    service = RecommendationService()
    context = build_market_context(
        skill_extraction=SkillExtractionResult(),
        normalized_offers=[
            {
                'title': 'Data Scientist',
                'description': 'Python SQL',
                'normalized_skills': [{'canonical_label': 'Python'}, {'canonical_label': 'SQL'}],
                'rome_code': 'M1805',
                'location': 'Paris',
            }
        ],
        departement='75',
        recommendation_service=service,
    )

    assert 'recommendation' in context
    assert 'market_analysis' in context
    assert context['recommendation'].offer_count == 1
    assert context['market_analysis'].offer_count == 1
    assert context['market_offers_preview']
