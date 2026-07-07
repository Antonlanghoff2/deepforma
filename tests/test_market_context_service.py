from __future__ import annotations

from services.market_context import assert_offers_match_rome, build_market_context, filter_offers_by_exact_rome, UnexpectedRomeOfferError
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


def test_filter_offers_by_exact_rome_accepts_only_exact_code():
    accepted, rejected = filter_offers_by_exact_rome(
        [
            {'offer_id': '1', 'romeCode': 'M1805'},
            {'offer_id': '2', 'romeCode': 'M1802'},
            {'offer_id': '3'},
        ],
        ' m1805 ',
    )
    assert [offer['offer_id'] for offer in accepted] == ['1']
    assert [offer['reason'] for offer in rejected] == ['ROME_MISMATCH', 'MISSING_ROME_CODE']


def test_filter_offers_by_exact_rome_rejects_invalid_code():
    accepted, rejected = filter_offers_by_exact_rome(
        [{'offer_id': '1', 'romeCode': 'M18A5'}],
        'M1805',
    )
    assert not accepted
    assert rejected[0]['reason'] == 'INVALID_ROME_CODE'


def test_assert_offers_match_rome_raises_on_mismatch():
    with pytest.raises(UnexpectedRomeOfferError):
        assert_offers_match_rome(
            [type('Offer', (), {'offer_id': '1', 'rome_code': 'M1802'})()],
            'M1805',
        )
