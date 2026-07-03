from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable

from common.text import clean_text, normalize_for_match, stable_hash


DEFAULT_WEIGHTS = {
    "low_confidence": 1.0,
    "small_margin": 1.0,
    "unknown_reference": 1.0,
    "model_france_travail_disagreement": 1.0,
    "ambiguous_skill": 1.0,
    "embedding_distance": 1.0,
    "new_skill_frequency": 1.0,
    "new_family": 1.0,
    "new_territory": 1.0,
    "high_confidence_control": 0.2,
}


@dataclass(frozen=True)
class ReviewCandidate:
    offer_row_id: int
    offer_id: str
    content_version: str
    title: str
    description_original: str
    territory: str | None
    job_family: str | None
    model_version: str | None
    confidence: float
    margin: float
    unknown_reference_count: int
    disagreement_count: int
    ambiguous_skill_count: int
    embedding_distance: float
    new_skill_frequency: float
    is_new_family: bool
    is_new_territory: bool
    control_sample: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


class ReviewQueueSelector:
    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.weights = dict(DEFAULT_WEIGHTS)
        if weights:
            self.weights.update(weights)

    def score_candidate(self, candidate: ReviewCandidate) -> float:
        features = {
            "low_confidence": max(0.0, 1.0 - candidate.confidence),
            "small_margin": max(0.0, 1.0 - candidate.margin),
            "unknown_reference": min(1.0, candidate.unknown_reference_count),
            "model_france_travail_disagreement": min(1.0, candidate.disagreement_count),
            "ambiguous_skill": min(1.0, candidate.ambiguous_skill_count),
            "embedding_distance": min(1.0, candidate.embedding_distance),
            "new_skill_frequency": min(1.0, candidate.new_skill_frequency),
            "new_family": 1.0 if candidate.is_new_family else 0.0,
            "new_territory": 1.0 if candidate.is_new_territory else 0.0,
            "high_confidence_control": 1.0 if candidate.control_sample else 0.0,
        }
        score = 0.0
        for key, value in features.items():
            score += self.weights.get(key, 0.0) * value
        return float(score)

    @staticmethod
    def _bucket_key(candidate: ReviewCandidate) -> tuple[str, str, str]:
        territory = candidate.territory or "unknown-territory"
        family = candidate.job_family or "unknown-family"
        skill_bucket = "control" if candidate.control_sample else "review"
        return territory, family, skill_bucket

    def select(
        self,
        candidates: Iterable[ReviewCandidate],
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        scored: list[dict[str, Any]] = []
        for candidate in candidates:
            item = {
                "candidate": candidate,
                "score": self.score_candidate(candidate),
            }
            scored.append(item)

        if not scored:
            return []

        scored.sort(key=lambda item: (-item["score"], item["candidate"].offer_row_id))

        bucket_counts: Counter[tuple[str, str, str]] = Counter()
        family_counts: Counter[str] = Counter()
        territory_counts: Counter[str] = Counter()
        output: list[dict[str, Any]] = []
        fairness_limit = max(1, limit // 10)

        for item in scored:
            candidate = item["candidate"]
            bucket = self._bucket_key(candidate)
            if bucket_counts[bucket] >= fairness_limit:
                continue
            if family_counts[candidate.job_family or "unknown-family"] >= fairness_limit:
                continue
            if territory_counts[candidate.territory or "unknown-territory"] >= fairness_limit:
                continue
            bucket_counts[bucket] += 1
            family_counts[candidate.job_family or "unknown-family"] += 1
            territory_counts[candidate.territory or "unknown-territory"] += 1
            output.append(
                {
                    "offer_row_id": candidate.offer_row_id,
                    "offer_id": candidate.offer_id,
                    "content_version": candidate.content_version,
                    "title": candidate.title,
                    "description_original": candidate.description_original,
                    "territory": candidate.territory,
                    "job_family": candidate.job_family,
                    "model_version": candidate.model_version,
                    "confidence": candidate.confidence,
                    "margin": candidate.margin,
                    "unknown_reference_count": candidate.unknown_reference_count,
                    "disagreement_count": candidate.disagreement_count,
                    "ambiguous_skill_count": candidate.ambiguous_skill_count,
                    "embedding_distance": candidate.embedding_distance,
                    "new_skill_frequency": candidate.new_skill_frequency,
                    "is_new_family": candidate.is_new_family,
                    "is_new_territory": candidate.is_new_territory,
                    "control_sample": candidate.control_sample,
                    "score": round(float(item["score"]), 6),
                    "metadata": candidate.metadata,
                }
            )
            if len(output) >= limit:
                break

        return output


def build_review_candidates_from_offers(
    offers: list[dict[str, Any]],
    *,
    recent_skill_counts: dict[str, int] | None = None,
    historical_skill_counts: dict[str, int] | None = None,
    seen_families: set[str] | None = None,
    seen_territories: set[str] | None = None,
) -> list[ReviewCandidate]:
    def _load_list(value: Any) -> list[dict[str, Any]]:
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if isinstance(value, str) and value.strip():
            try:
                parsed = json.loads(value)
                return [item for item in parsed if isinstance(item, dict)] if isinstance(parsed, list) else []
            except Exception:
                return []
        return []

    recent_skill_counts = recent_skill_counts or {}
    historical_skill_counts = historical_skill_counts or {}
    seen_families = seen_families or set()
    seen_territories = seen_territories or set()

    candidates: list[ReviewCandidate] = []
    for offer in offers:
        predicted_skills = _load_list(offer.get("predicted_skills") or offer.get("predicted_skills_json"))
        confidence_values = [float(item.get("confidence", 0.0)) for item in predicted_skills if isinstance(item, dict)]
        confidence = min(confidence_values) if confidence_values else 0.0
        ordered = sorted(confidence_values, reverse=True)
        margin = (ordered[0] - ordered[1]) if len(ordered) >= 2 else (ordered[0] if ordered else 0.0)
        unknown_reference_count = sum(1 for item in predicted_skills if item.get("source") == "unknown_reference")
        disagreement_count = sum(1 for item in predicted_skills if item.get("source") == "france_travail_disagreement")
        ambiguous_skill_count = sum(1 for item in predicted_skills if item.get("ambiguous"))
        detected_forms = _load_list(offer.get("detected_forms") or offer.get("detected_forms_json"))
        embedding_distance = _safe_float(offer.get("embedding_distance"), 0.0)
        territory = offer.get("territory")
        job_family = offer.get("job_family")
        recent_count = sum(recent_skill_counts.get(normalize_for_match(item.get("canonical_name") or item.get("label") or ""), 0) for item in predicted_skills if isinstance(item, dict))
        historical_count = sum(historical_skill_counts.get(normalize_for_match(item.get("canonical_name") or item.get("label") or ""), 0) for item in predicted_skills if isinstance(item, dict))
        freq = (recent_count / max(historical_count, 1)) if recent_count else 0.0
        candidates.append(
            ReviewCandidate(
                offer_row_id=int(offer["id"]),
                offer_id=str(offer["offer_id"]),
                content_version=str(offer["content_version"]),
                title=clean_text(offer.get("title")),
                description_original=clean_text(offer.get("description_original")),
                territory=clean_text(territory) or None,
                job_family=clean_text(job_family) or None,
                model_version=offer.get("model_version"),
                confidence=confidence,
                margin=margin,
                unknown_reference_count=unknown_reference_count,
                disagreement_count=disagreement_count,
                ambiguous_skill_count=ambiguous_skill_count,
                embedding_distance=embedding_distance,
                new_skill_frequency=float(freq),
                is_new_family=(clean_text(job_family) not in seen_families) if job_family else False,
                is_new_territory=(clean_text(territory) not in seen_territories) if territory else False,
                control_sample=confidence >= 0.85 and len(detected_forms) > 0,
                metadata={
                    "structured_count": len(_load_list(offer.get("structured_skills") or offer.get("structured_skills_json"))),
                    "predicted_count": len(predicted_skills),
                },
            )
        )
    return candidates

