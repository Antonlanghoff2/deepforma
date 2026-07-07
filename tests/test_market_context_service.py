from __future__ import annotations

from services.market_context import assert_offers_match_rome, build_market_context, fetch_offers_by_rome_codes, filter_offers_by_exact_rome, merge_offers_by_id, MultiRomeSearchResult, RomeSearchStats, UnexpectedRomeOfferError
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


def test_merge_offers_by_id_merges_duplicate_ids() -> None:
    merged = merge_offers_by_id({
        'M1805': [
            {'offer_id': '1', 'title': 'A', 'skills': [{'name': 'Python'}], 'matched_requested_rome_codes': ['M1805']},
        ],
        'M1802': [
            {'offer_id': '1', 'title': 'A', 'skills': [{'name': 'Python'}, {'name': 'SQL'}], 'matched_requested_rome_codes': ['M1802']},
        ],
    })
    assert len(merged) == 1
    assert merged[0]['matched_requested_rome_codes'] == ['M1805', 'M1802']
    assert len(merged[0]['skills']) == 3 or len(merged[0]['skills']) == 2


def test_fetch_offers_by_rome_codes_merges_results(monkeypatch) -> None:
    from domain.models import Territory
    from services.market_context import OfferCollection

    def fake_fetch(client, rome_code, territory, **kwargs):
        if rome_code == 'M1805':
            return OfferCollection([{'offer_id': '1', 'title': 'A', 'rome_code': 'M1805', 'matched_requested_rome_codes': ['M1805']}], audit={'raw_count': 1, 'accepted_count': 1, 'rejected_count': 0, 'rejections': []})
        return OfferCollection([{'offer_id': '2', 'title': 'B', 'rome_code': 'M1802', 'matched_requested_rome_codes': ['M1802']}], audit={'raw_count': 1, 'accepted_count': 1, 'rejected_count': 0, 'rejections': []})

    monkeypatch.setattr('services.market_context.fetch_offers_by_rome', fake_fetch)
    result = fetch_offers_by_rome_codes(['M1805', 'M1802'], Territory(code='75', label='Paris', type='departement', department_code='75'))
    assert result.requested_rome_codes == ['M1805', 'M1802']
    assert {offer['offer_id'] for offer in result.offers} == {'1', '2'}
    assert len(result.stats_by_rome) == 2
