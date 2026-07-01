from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any


@dataclass
class OpenExtractedSkill:
    source_label: str = ""
    normalized_label: str = ""
    type: str = "technical_skill"
    source_text: str = ""
    start: int = 0
    end: int = 0
    confidence: float = 0.0
    method: str = "rule"
    referential_id: str | None = None
    referential_source: str | None = None
    ia_categories: list[str] = field(default_factory=list)


@dataclass
class SkillExtractionInfo:
    status: str = "success"  # success|partial|failed
    skills: list[OpenExtractedSkill] = field(default_factory=list)
    tools: list[OpenExtractedSkill] = field(default_factory=list)
    knowledge_items: list[OpenExtractedSkill] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class IAClassificationInfo:
    status: str = "unavailable"  # success|unreliable|unavailable
    categories: list[dict[str, Any]] = field(default_factory=list)
    families: list[dict[str, Any]] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    score_min: float = 0.0
    score_max: float = 0.0
    score_mean: float = 0.0
    score_std: float = 0.0
    discriminating: bool = False
    warnings: list[str] = field(default_factory=list)
    threshold_applied: float = 0.35


@dataclass
class SkillInfo:
    id: str | None = None
    label: str = ""
    score_brut: float = 0.0
    score_calibre: float | None = None
    niveau_confiance: str = "faible"
    statut: str = "mentionné"
    presence: str = "indeterminate"
    passage_source: str = ""
    methode_detection: str = "camembert_multilabel"
    synonymes: list[str] = field(default_factory=list)
    seuil_applique: float = 0.35


@dataclass
class ClassificationInfo:
    is_ia: bool = False
    predicted_class: int = 0
    probability_ia: float = 0.0
    probability_non_ia: float = 0.0
    state: str = "incertaine"
    state_description: str = ""
    gap: float = 0.0


@dataclass
class MarketSkillInfo:
    label: str = ""
    offer_count: int = 0
    share_percent: float = 0.0


@dataclass
class TerritorialMarketInfo:
    territory: str = ""
    period: str = ""
    offer_count: int = 0
    exploitable_offers: int = 0
    dominant_jobs: list[dict[str, Any]] = field(default_factory=list)
    dominant_sectors: list[str] = field(default_factory=list)
    top_skills: list[MarketSkillInfo] = field(default_factory=list)
    emergent_skills: list[str] = field(default_factory=list)
    declining_skills: list[str] = field(default_factory=list)
    experience_levels: dict[str, int] = field(default_factory=dict)
    diplomas: dict[str, int] = field(default_factory=dict)
    contract_types: dict[str, int] = field(default_factory=dict)
    telework: dict[str, int] = field(default_factory=dict)
    salaries: dict[str, float] | None = None
    statistical_robustness: str = "faible"
    alert: str = ""


@dataclass
class MarketComparisonItem:
    skill: str = ""
    in_formation: bool = False
    detection_confidence: float = 0.0
    frequency_in_offers: float = 0.0
    offer_count: int = 0
    coverage_level: str = ""
    priority: str = ""
    example_offers: list[str] = field(default_factory=list)


@dataclass
class Recommendation:
    type: str = ""
    skill: str = ""
    justification: str = ""
    impact_estime: str = ""
    offer_count: int = 0
    offer_percent: float = 0.0
    priorite: str = ""
    niveau_confiance: str = ""


@dataclass
class QualityInfo:
    model_loaded: bool = False
    skills_discriminating: bool = False
    score_min: float = 0.0
    score_max: float = 0.0
    score_mean: float = 0.0
    score_std: float = 0.0
    offers_sufficient: bool = False
    warnings: list[str] = field(default_factory=list)


@dataclass
class ModelMetadata:
    binary_model: str = ""
    multilabel_model: str = ""
    model_name: str = "Classifieur IA"
    taxonomy_version: str = ""
    validation_status: str = "non validé"
    binary_checkpoint: str = ""
    multilabel_checkpoint: str = ""
    device: str = ""
    max_length: int = 512
    num_labels: int = 18
    labels: list[str] = field(default_factory=list)
    thresholds: dict[str, Any] = field(default_factory=dict)
    inference_time_ms: float = 0.0
    classifier_weight_stats: dict[str, float] | None = None


@dataclass
class CheckpointAuditInfo:
    config_present: bool = False
    weights_present: bool = False
    weights_size_bytes: int = 0
    architecture_declared: str = ""
    num_labels_declared: int = 0
    num_labels_effective: int = 0
    problem_type: str = ""
    id2label_count: int = 0
    label2id_count: int = 0
    strict_load_success: bool = False
    missing_keys: list[str] = field(default_factory=list)
    unexpected_keys: list[str] = field(default_factory=list)
    ignored_keys: list[str] = field(default_factory=list)
    appears_random_init: bool = True
    body_params_match_base: bool | str = True
    parameter_errors: list[str] = field(default_factory=list)
    classifier_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisResult:
    formation_analysis_status: str = "unreliable"
    comparison_available: bool = False
    recommendations_available: bool = False
    skills_presence: str = "indeterminate"
    blocking_reasons: list[str] = field(default_factory=list)

    summary: dict[str, Any] = field(default_factory=dict)
    classification: ClassificationInfo = field(default_factory=ClassificationInfo)
    detected_skills: list[SkillInfo] = field(default_factory=list)
    implicit_skills: list[SkillInfo] = field(default_factory=list)
    rejected_skills: list[SkillInfo] = field(default_factory=list)
    low_confidence_skills: list[SkillInfo] = field(default_factory=list)
    indeterminate_skills: list[SkillInfo] = field(default_factory=list)
    territorial_market: TerritorialMarketInfo = field(default_factory=TerritorialMarketInfo)
    formation_market_comparison: list[MarketComparisonItem] = field(default_factory=list)
    comparison_categories: dict[str, list[MarketComparisonItem]] = field(default_factory=dict)
    global_score: dict[str, Any] = field(default_factory=dict)
    missing_skills: list[MarketSkillInfo] = field(default_factory=list)
    recommendations: list[Recommendation] = field(default_factory=list)
    quality: QualityInfo = field(default_factory=QualityInfo)
    model_metadata: ModelMetadata = field(default_factory=ModelMetadata)
    checkpoint_audit: CheckpointAuditInfo = field(default_factory=CheckpointAuditInfo)
    skill_extraction: SkillExtractionInfo = field(default_factory=SkillExtractionInfo)
    ia_classification: IAClassificationInfo = field(default_factory=IAClassificationInfo)

    def to_dict(self) -> dict[str, Any]:
        return {
            "formation_analysis_status": self.formation_analysis_status,
            "comparison_available": self.comparison_available,
            "recommendations_available": self.recommendations_available,
            "skills_presence": self.skills_presence,
            "blocking_reasons": self.blocking_reasons,
            "summary": self.summary,
            "classification": asdict(self.classification),
            "detected_skills": [asdict(s) for s in self.detected_skills],
            "implicit_skills": [asdict(s) for s in self.implicit_skills],
            "rejected_skills": [asdict(s) for s in self.rejected_skills],
            "low_confidence_skills": [asdict(s) for s in self.low_confidence_skills],
            "indeterminate_skills": [asdict(s) for s in self.indeterminate_skills],
            "territorial_market": asdict(self.territorial_market) if self.territorial_market else {},
            "formation_market_comparison": [asdict(c) for c in self.formation_market_comparison],
            "comparison_categories": {k: [asdict(c) for c in v] for k, v in self.comparison_categories.items()},
            "global_score": self.global_score,
            "missing_skills": [asdict(s) for s in self.missing_skills],
            "recommendations": [asdict(r) for r in self.recommendations],
            "quality": asdict(self.quality),
            "model_metadata": asdict(self.model_metadata),
            "checkpoint_audit": asdict(self.checkpoint_audit),
            "skill_extraction": asdict(self.skill_extraction),
            "ia_classification": asdict(self.ia_classification),
            "ia_families": self.ia_classification.families,
        }

    def to_json(self, indent: int = 2) -> str:
        import json
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
