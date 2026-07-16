from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import math

import numpy as np

from common.text import clean_text, normalize_for_match

from ._common import safe_divide, to_jsonable


@dataclass(slots=True)
class RecommendationCase:
    case_id: str
    profile: str
    owned_skills: list[str]
    missing_skills: list[str]
    target_job: str
    rome_codes: list[str]
    territory: str
    relevant_formations: dict[str, int]


@dataclass(slots=True)
class RecommendationEvaluationReport:
    metrics: dict[str, float]
    per_case: list[dict[str, Any]]
    catalog_coverage: float
    diversity: float
    candidate_count_mean: float | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


def _item_id(item: Any) -> str:
    if isinstance(item, dict):
        return clean_text(item.get("formation_id") or item.get("id") or item.get("uid") or "")
    return clean_text(item)


def _item_text(item: dict[str, Any]) -> str:
    parts = [item.get("title") or "", item.get("skills") or item.get("skill_labels") or ""]
    tokens: list[str] = []
    for part in parts:
        if isinstance(part, list):
            tokens.extend(clean_text(value) for value in part if clean_text(value))
        else:
            text = clean_text(part)
            if text:
                tokens.append(text)
    return " ".join(tokens)


def _ideal_dcg(relevances: list[int], *, k: int) -> float:
    ordered = sorted(relevances, reverse=True)[:k]
    dcg = 0.0
    for index, rel in enumerate(ordered, start=1):
        dcg += ((2 ** rel) - 1) / math.log2(index + 1)
    return dcg


def _dcg(relevances: list[int], *, k: int) -> float:
    dcg = 0.0
    for index, rel in enumerate(relevances[:k], start=1):
        dcg += ((2 ** rel) - 1) / math.log2(index + 1)
    return dcg


def _pairwise_diversity(items: list[dict[str, Any]], catalog: dict[str, dict[str, Any]] | None) -> float:
    if len(items) < 2:
        return 0.0
    similarities: list[float] = []
    for left_index in range(len(items)):
        for right_index in range(left_index + 1, len(items)):
            left_id = _item_id(items[left_index])
            right_id = _item_id(items[right_index])
            left_meta = catalog.get(left_id, {}) if catalog else {}
            right_meta = catalog.get(right_id, {}) if catalog else {}
            left_text = _item_text(left_meta)
            right_text = _item_text(right_meta)
            left_tokens = set(normalize_for_match(left_text).split())
            right_tokens = set(normalize_for_match(right_text).split())
            if not left_tokens and not right_tokens:
                continue
            intersection = len(left_tokens & right_tokens)
            union = len(left_tokens | right_tokens) or 1
            similarities.append(intersection / union)
    if not similarities:
        return 0.0
    return float(1.0 - np.mean(similarities))


def evaluate_recommendation(
    cases: list[RecommendationCase],
    rankings: dict[str, list[dict[str, Any]]],
    *,
    catalog: dict[str, dict[str, Any]] | None = None,
    k_values: tuple[int, ...] = (1, 3, 5, 10),
) -> RecommendationEvaluationReport:
    case_rows: list[dict[str, Any]] = []
    per_k_precision = {k: [] for k in k_values}
    per_k_recall = {k: [] for k in k_values}
    per_k_hit = {k: [] for k in k_values}
    per_k_ndcg = {5: [], 10: []}
    mrrs: list[float] = []
    aps: list[float] = []
    recommended_ids: set[str] = set()
    diversity_scores: list[float] = []
    candidate_counts: list[int] = []
    warnings: list[str] = []

    for case in cases:
        ranked_items = rankings.get(case.case_id, [])
        ranked_ids = [_item_id(item) for item in ranked_items if _item_id(item)]
        candidate_counts.append(len(ranked_ids))
        relevant = {formation_id: int(relevance) for formation_id, relevance in case.relevant_formations.items() if int(relevance) > 0}
        relevant_ids = set(relevant)
        if not relevant_ids:
            warnings.append(f"Cas {case.case_id}: aucune formation pertinente annotée.")
            continue
        ranked_relevances = [int(relevant.get(item_id, 0)) for item_id in ranked_ids]
        first_hit_rank = next((index for index, item_id in enumerate(ranked_ids, start=1) if item_id in relevant_ids), None)
        mrrs.append(1.0 / first_hit_rank if first_hit_rank else 0.0)
        precisions_at_hits: list[float] = []
        cumulative_relevant = 0
        for index, rel in enumerate(ranked_relevances, start=1):
            if rel > 0:
                cumulative_relevant += 1
                precisions_at_hits.append(cumulative_relevant / index)
        aps.append(sum(precisions_at_hits) / len(relevant_ids) if relevant_ids else 0.0)

        for k in k_values:
            topk = ranked_ids[:k]
            recommended_ids.update(topk)
            hits = sum(1 for item_id in topk if item_id in relevant_ids)
            per_k_precision[k].append(safe_divide(hits, k))
            per_k_recall[k].append(safe_divide(hits, len(relevant_ids)))
            per_k_hit[k].append(1.0 if hits > 0 else 0.0)
        for k in (5, 10):
            if k in per_k_ndcg:
                gains = [relevant.get(item_id, 0) for item_id in ranked_ids]
                dcg = _dcg(gains, k=k)
                idcg = _ideal_dcg(list(relevant.values()), k=k)
                per_k_ndcg[k].append(safe_divide(dcg, idcg))

        diversity_scores.append(_pairwise_diversity(ranked_items[:10], catalog))
        case_rows.append(
            {
                "case_id": case.case_id,
                "first_hit_rank": first_hit_rank,
                "relevant_count": len(relevant_ids),
                "ranked_count": len(ranked_ids),
            }
        )

    metrics: dict[str, float] = {}
    for k in k_values:
        metrics[f"precision_at_{k}"] = float(np.mean(per_k_precision[k])) if per_k_precision[k] else 0.0
        metrics[f"recall_at_{k}"] = float(np.mean(per_k_recall[k])) if per_k_recall[k] else 0.0
        metrics[f"hit_rate_at_{k}"] = float(np.mean(per_k_hit[k])) if per_k_hit[k] else 0.0
    metrics["mrr"] = float(np.mean(mrrs)) if mrrs else 0.0
    metrics["map"] = float(np.mean(aps)) if aps else 0.0
    metrics["ndcg_at_5"] = float(np.mean(per_k_ndcg[5])) if per_k_ndcg[5] else 0.0
    metrics["ndcg_at_10"] = float(np.mean(per_k_ndcg[10])) if per_k_ndcg[10] else 0.0
    catalog_size = len(catalog or {}) or len(recommended_ids) or 1
    catalog_coverage = safe_divide(len(recommended_ids), catalog_size)
    diversity = float(np.mean(diversity_scores)) if diversity_scores else 0.0
    candidate_count_mean = float(np.mean(candidate_counts)) if candidate_counts else None

    return RecommendationEvaluationReport(
        metrics=metrics,
        per_case=case_rows,
        catalog_coverage=catalog_coverage,
        diversity=diversity,
        candidate_count_mean=candidate_count_mean,
        warnings=warnings,
    )
