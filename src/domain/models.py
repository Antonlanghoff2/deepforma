from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True, slots=True)
class IARecommendation:
    recommendation_id: str
    keyword: str
    keyword_normalized: str
    recommendation: str
    source_file: str | None = None
    is_default: bool = False
    is_active: bool = True
    category: str | None = None
    created_at: str | None = None


@dataclass(frozen=True, slots=True)
class IARecommendationMatch:
    skill_original: str
    skill_normalized: str
    matched_keyword: str
    recommendation: str
    score: float
    match_method: str
    confidence_label: str


@dataclass(frozen=True, slots=True)
class Skill:
    name: str
    normalized_name: str | None = None
    category: str | None = None
    source: str | None = None
    confidence: float | None = None


@dataclass(frozen=True, slots=True)
class RomeOccupation:
    code: str
    label: str
    alternative_labels: list[str] = field(default_factory=list)
    domain: str | None = None


@dataclass(frozen=True, slots=True)
class JobOffer:
    offer_id: str
    title: str
    description: str
    skills: list[Skill] = field(default_factory=list)
    rome_code: str | None = None
    rome_label: str | None = None
    matched_requested_rome_codes: list[str] = field(default_factory=list)
    location: str | None = None


@dataclass(frozen=True, slots=True)
class Certification:
    code: str
    title: str
    skills: list[Skill] = field(default_factory=list)
    rome_codes: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class Training:
    training_id: str
    title: str
    skills: list[Skill] = field(default_factory=list)
    rome_codes: list[str] = field(default_factory=list)
    territory: str | None = None


@dataclass(frozen=True, slots=True)
class Territory:
    code: str | None = None
    label: str | None = None
    type: str | None = None
    radius_km: int | None = None
    department_code: str | None = None
    region_code: str | None = None
    remote_allowed: bool = True


@dataclass(frozen=True, slots=True)
class MarketTarget:
    rome_code: str
    rome_label: str | None = None
    territory: Territory | None = None
    contract_types: list[str] = field(default_factory=list)
    period: str | None = None


@dataclass(frozen=True, slots=True)
class PdfAnalysis:
    analysis_id: str
    document_title: str
    certification_title: str | None = None
    blocks: list[Any] = field(default_factory=list)
    skills: list[Skill] = field(default_factory=list)
    selected_rome_occupations: list[RomeOccupation] = field(default_factory=list)
    territory_code: str | None = None
    territory_label: str | None = None
    analysis_status: str = "PDF_ANALYZED"
    market_search_status: str = "WAITING_FOR_ROME"
    selected_rome_code: str | None = None
    selected_rome_label: str | None = None


@dataclass(frozen=True, slots=True)
class MatchResult:
    score: float
    common_skills: list[Skill] = field(default_factory=list)
    missing_skills: list[Skill] = field(default_factory=list)
    matched_offers: int = 0
    explanation: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Recommendation:
    target_id: str
    title: str
    score: float
    reasons: list[str] = field(default_factory=list)
    explanation: str | None = None
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MarketAnalysis:
    profile_skills: list[Skill] = field(default_factory=list)
    territory: Territory | None = None
    offer_count: int = 0
    coverage_score: float = 0.0
    common_skills: list[str] = field(default_factory=list)
    missing_skills: list[str] = field(default_factory=list)
    top_demanded_skills: list[str] = field(default_factory=list)
    recommendations: list[Recommendation] = field(default_factory=list)
