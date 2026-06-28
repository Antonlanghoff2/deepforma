from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

from common.text import clean_text, normalize_for_match
from skills.skill_normalizer import SkillNormalizer


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
        self.normalizer = normalizer or SkillNormalizer()

    def normalize_label(self, label: str) -> str | None:
        cleaned = clean_text(label)
        if not cleaned:
            return None
        canonical, _, _ = self.normalizer.normalize(cleaned)
        if canonical:
            return canonical
        normalized = normalize_for_match(cleaned)
        return normalized or None

    def normalize_labels(self, labels: Iterable[Any]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for label in labels:
            if isinstance(label, dict):
                candidate = label.get('label') or label.get('canonical_label') or label.get('name')
            else:
                candidate = label
            normalized_label = self.normalize_label(str(candidate or ''))
            if not normalized_label:
                continue
            key = normalize_for_match(normalized_label)
            if key and key not in seen:
                seen.add(key)
                normalized.append(normalized_label)
        return normalized

    def _offer_skill_labels(self, offer: dict[str, Any]) -> list[str]:
        labels: list[str] = []
        for key in ('normalized_skills', 'merged_skills', 'structured_skills', 'model_skills'):
            values = offer.get(key) or []
            for item in values:
                if isinstance(item, dict):
                    candidate = item.get('canonical_label') or item.get('label')
                else:
                    candidate = item
                normalized = self.normalize_label(str(candidate or ''))
                if normalized:
                    labels.append(normalized)
        seen: set[str] = set()
        deduped: list[str] = []
        for label in labels:
            key = normalize_for_match(label)
            if key and key not in seen:
                seen.add(key)
                deduped.append(label)
        return deduped

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
