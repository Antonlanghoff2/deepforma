from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class AIRecommendationSourceScore:
    source: str
    score: float
    details: dict[str, Any] = field(default_factory=dict)


DEFAULT_WEIGHTS = {
    'manually_validated_rule': 1.0,
    'exact_rule': 1.0,
    'lexical_rule': 0.92,
    'semantic_rule': 0.88,
    'multilabel_model': 0.70,
}


def fuse_ai_recommendation_scores(
    scores: list[AIRecommendationSourceScore],
    *,
    model_score_std: float | None = None,
    model_mean_score: float | None = None,
    model_non_discriminant: bool = False,
) -> dict[str, Any]:
    weights = dict(DEFAULT_WEIGHTS)
    if model_non_discriminant or (model_score_std is not None and model_score_std < 0.03) or (model_mean_score is not None and 0.45 <= model_mean_score <= 0.55):
        weights['multilabel_model'] = 0.0
    weighted_total = 0.0
    weight_sum = 0.0
    contributions: list[dict[str, Any]] = []
    for item in scores:
        weight = weights.get(item.source, 0.5)
        if item.source == 'semantic_rule':
            weight = min(1.0, max(0.0, item.score))
        if item.source == 'multilabel_model' and weights.get('multilabel_model', 0.0) == 0.0:
            weight = 0.0
        weighted_total += item.score * weight
        weight_sum += weight
        contributions.append({'source': item.source, 'score': round(item.score, 4), 'weight': round(weight, 4)})
    final_score = round(weighted_total / weight_sum, 4) if weight_sum else 0.0
    return {'score': final_score, 'contributions': contributions, 'weights': weights}
