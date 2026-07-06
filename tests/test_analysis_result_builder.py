from __future__ import annotations

from dataclasses import dataclass

from models.analysis_result import OpenExtractedSkill, Recommendation, SkillExtractionInfo
from services.analysis_result_builder import build_analysis_result


@dataclass
class TerritorialStats:
    skill_counts: dict[str, int]
    offer_count: int
    contract_types: dict[str, int]


def test_build_analysis_result_builds_summary_and_comparison() -> None:
    analysis = {
        'binary': {'is_ia': True, 'predicted_class': 1, 'probability_ia': 0.91, 'probability_non_ia': 0.09},
        'skills': {
            'predictions': [
                {'label': 'Machine Learning', 'probability': 0.9},
                {'label': 'Python', 'probability': 0.1},
            ],
            'score_std': 0.2,
            'score_max': 0.9,
            'score_mean': 0.5,
            'score_min': 0.1,
            'family_groups': [],
        },
        'checkpoint_audit': {'strict_load_success': True, 'appears_random_init': False, 'classifier_params': {}},
        'inference_time_ms': 12.3,
    }
    skill_extraction = SkillExtractionInfo(
        status='success',
        skills=[OpenExtractedSkill(normalized_label='Python', confidence=0.95)],
        tools=[OpenExtractedSkill(normalized_label='SQL', confidence=0.91)],
    )
    recommendation = type('RecommendationReport', (), {
        'market_skills': [type('MarketSkillSummary', (), {'label': 'Python', 'offer_count': 3, 'share_percent': 60.0})()],
        'covered_skills': ['Python'],
        'missing_priority_skills': [],
        'offer_count': 1,
        'matched_market_offers': 1,
    })()
    territorial_stats = TerritorialStats(skill_counts={'Python': 3}, offer_count=1, contract_types={'CDI': 1})

    result = build_analysis_result(
        analysis=analysis,
        normalized_offers=[{'title': 'Data Scientist'}],
        recommendation=recommendation,
        territorial_stats=territorial_stats,
        departement='75',
        threshold=0.35,
        skill_extraction=skill_extraction,
    )

    assert result.comparison_available is True
    assert result.recommendations_available is True
    assert result.summary['total_offers_analyzed'] == 1
    assert result.summary['skill_extraction_status'] == 'success'
    assert result.model_metadata.validation_status == 'entraîné (non validé)'
    assert result.territorial_market.offer_count == 1
