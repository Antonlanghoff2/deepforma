from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

from common.text import clean_text, normalize_for_match
from skills.skill_normalizer import SkillNormalizer
from services.skill_normalization import (
    get_skill_normalizer,
    normalize_offer_skill_labels,
    normalize_skill_label,
    normalize_skill_labels,
)


@dataclass(frozen=True)
class MarketSkillSummary:
    label: str
    offer_count: int
    share_percent: float


@dataclass(frozen=True)
class RecommendationReport:
    formation_skills: list[str]
    market_skills: list[MarketSkillSummary]
    covered_skills: list[str]
    missing_priority_skills: list[MarketSkillSummary]
    coverage_score: float
    offer_count: int
    matched_market_offers: int


class RecommendationService:
    def __init__(self, normalizer: SkillNormalizer | None = None) -> None:
        self.normalizer = normalizer or get_skill_normalizer()

    def normalize_label(self, label: str) -> str | None:
        return normalize_skill_label(label, normalizer=self.normalizer)

    def normalize_labels(self, labels: Iterable[Any]) -> list[str]:
        return normalize_skill_labels(labels, normalizer=self.normalizer)

    def _offer_skill_labels(self, offer: dict[str, Any]) -> list[str]:
        return normalize_offer_skill_labels(offer, normalizer=self.normalizer)

    def summarize_market(self, offers: list[dict[str, Any]]) -> tuple[int, dict[str, int], dict[str, float]]:
        skill_offer_counts: Counter[str] = Counter()
        offer_count = len(offers)
        for offer in offers:
            labels = self._offer_skill_labels(offer)
            for label in set(labels):
                skill_offer_counts[label] += 1
        skill_share = {
            label: round((count / offer_count) * 100, 2) if offer_count else 0.0
            for label, count in skill_offer_counts.items()
        }
        return offer_count, dict(skill_offer_counts), skill_share

    def compare(self, formation_skills: Iterable[Any], offers: list[dict[str, Any]]) -> RecommendationReport:
        formation = self.normalize_labels(formation_skills)
        formation_keys = {normalize_for_match(label) for label in formation}

        offer_count, skill_counts, skill_share = self.summarize_market(offers)
        market_skills = [
            MarketSkillSummary(label=label, offer_count=count, share_percent=skill_share.get(label, 0.0))
            for label, count in sorted(skill_counts.items(), key=lambda item: (-item[1], item[0]))
        ]

        covered_skills = [label for label in formation if normalize_for_match(label) in {normalize_for_match(item.label) for item in market_skills}]
        missing_priority_skills = [
            item for item in market_skills
            if normalize_for_match(item.label) not in formation_keys
        ]

        total_weight = sum(item.offer_count for item in market_skills)
        matched_weight = sum(
            item.offer_count for item in market_skills
            if normalize_for_match(item.label) in formation_keys
        )
        coverage_score = round((matched_weight / total_weight) * 100, 2) if total_weight else 0.0

        return RecommendationReport(
            formation_skills=formation,
            market_skills=market_skills,
            covered_skills=covered_skills,
            missing_priority_skills=missing_priority_skills,
            coverage_score=coverage_score,
            offer_count=offer_count,
            matched_market_offers=matched_weight,
        )
