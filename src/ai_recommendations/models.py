from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class AIRecommendationRuleCategory:
    label: str
    score: float
    method: str
    status: str


@dataclass(frozen=True, slots=True)
class AIRecommendationRule:
    id: str
    keyword: str
    normalized_keyword: str
    categories: list[AIRecommendationRuleCategory] = field(default_factory=list)
    recommendation: str = ''
    match_type: str = 'hybrid'
    priority: int = 50
    enabled: bool = True
    is_default: bool = False
    source: str = 'dataset_recommandations_IA_complet.csv'
    source_line: int = 0
    review_status: str = 'to_review'
    metadata: dict[str, Any] = field(default_factory=dict)
    aliases: list[str] = field(default_factory=list)
    aliases_normalized: list[str] = field(default_factory=list)


@dataclass(slots=True)
class AIRecommendationImportReport:
    total_lines: int = 0
    imported_lines: int = 0
    review_lines: int = 0
    duplicate_lines: int = 0
    anomalies: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class AIRecommendationRuleMatchEvidence:
    text: str
    rule_id: str
    similarity: float


@dataclass(frozen=True, slots=True)
class AIRecommendationRuleMatch:
    rule_id: str
    keyword: str
    recommendation: str
    score: float
    match_method: str
    matched_text: str
    categories: list[dict[str, Any]] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    evidence: list[AIRecommendationRuleMatchEvidence] = field(default_factory=list)
    status: str = 'reliable'


@dataclass(frozen=True, slots=True)
class AIRecommendationSearchResult:
    rule: AIRecommendationRule
    score: float
    method: str
    matched_text: str
    evidence: list[AIRecommendationRuleMatchEvidence] = field(default_factory=list)
    status: str = 'reliable'


@dataclass(frozen=True, slots=True)
class AIRecommendationCategoryMapping:
    rule_id: str
    keyword: str
    categories: list[AIRecommendationRuleCategory] = field(default_factory=list)
    source: str = 'manual'
    status: str = 'to_review'
