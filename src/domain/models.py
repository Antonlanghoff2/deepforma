from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Skill:
    name: str
    normalized_name: str | None = None
    category: str | None = None
    source: str | None = None
    confidence: float | None = None


@dataclass(frozen=True, slots=True)
class JobOffer:
    offer_id: str
    title: str
    description: str
    skills: list[Skill] = field(default_factory=list)
    rome_code: str | None = None
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
    department_code: str | None = None
    region_code: str | None = None
    remote_allowed: bool = True


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
