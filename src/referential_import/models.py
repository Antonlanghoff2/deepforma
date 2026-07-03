from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


ReviewStatus = Literal["pending", "approved", "corrected", "rejected", "excluded", "used_for_training"]
LabelProvenance = Literal[
    "human_review",
    "france_travail_api",
    "exact_reference_match",
    "semantic_match",
    "model_prediction",
    "imported_gold_dataset",
]
IssueSeverity = Literal["info", "warning", "error"]


def _to_jsonable(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    return value


@dataclass(slots=True)
class Serializable:
    def to_dict(self) -> dict[str, Any]:
        return {key: _to_jsonable(value) for key, value in asdict(self).items()}


@dataclass(slots=True)
class ReferentialDocument(Serializable):
    id: str
    source_path: str
    file_name: str
    sha256: str
    schema_version: str = "1.0"
    importer_version: str = "0.1.0"
    page_count: int = 0
    title: str = ""
    source_type: str = "pdf"
    text_extraction_method: str = "pdftotext"
    review_status: ReviewStatus = "pending"
    collected_at: str = ""
    validated_at: str = ""
    validated_by: str = ""
    notes: str = ""


@dataclass(slots=True)
class ReferentialBlock(Serializable):
    code: str
    label: str
    page_start: int
    page_end: int
    confidence: float = 0.0
    text: str = ""
    review_status: ReviewStatus = "pending"
    source_pages: list[int] = field(default_factory=list)


@dataclass(slots=True)
class ReferentialActivity(Serializable):
    code: str
    block_code: str
    label: str
    page_start: int
    page_end: int
    confidence: float = 0.0
    text: str = ""
    review_status: ReviewStatus = "pending"
    source_pages: list[int] = field(default_factory=list)


@dataclass(slots=True)
class EvaluationCriterion(Serializable):
    code: str
    competency_code: str
    criterion_label: str
    normalized_label: str
    page_start: int
    page_end: int
    confidence: float = 0.0
    review_status: ReviewStatus = "pending"
    source_pages: list[int] = field(default_factory=list)
    source_text: str = ""
    provenance: LabelProvenance = "human_review"


@dataclass(slots=True)
class DerivedSkill(Serializable):
    label: str
    canonical_label: str
    category: str
    source_code: str
    source_type: str
    surface_form: str
    normalized_surface: str
    provenance: LabelProvenance = "semantic_match"
    confidence: float = 0.0
    explicit: bool = False
    page_start: int = 0
    page_end: int = 0
    context: str = ""


@dataclass(slots=True)
class OfficialCompetency(Serializable):
    code: str
    official_label: str
    normalized_label: str
    block_code: str
    activity_code: str
    page_start: int
    page_end: int
    derived_skills: list[DerivedSkill] = field(default_factory=list)
    tools_methods: list[dict[str, Any]] = field(default_factory=list)
    knowledge_items: list[dict[str, Any]] = field(default_factory=list)
    evaluation_criteria: list[EvaluationCriterion] = field(default_factory=list)
    confidence: float = 0.0
    review_status: ReviewStatus = "pending"
    source_pages: list[int] = field(default_factory=list)
    source_text: str = ""
    provenance: LabelProvenance = "human_review"


@dataclass(slots=True)
class ImportIssue(Serializable):
    severity: IssueSeverity
    code: str
    message: str
    page: int | None = None
    entity_code: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ImportReport(Serializable):
    schema_version: str
    importer_version: str
    document_id: str
    source_hash: str
    pages: int
    blocks: int
    activities: int
    competencies: int
    criteria: int
    derived_skills: int
    tools_methods: int
    errors: list[ImportIssue] = field(default_factory=list)
    warnings: list[ImportIssue] = field(default_factory=list)
    review_items: list[dict[str, Any]] = field(default_factory=list)
    score_global: float = 0.0
    coverage_score: float = 0.0
    duplicate_document: bool = False
    extraction_mode: str = "layout"
    notes: str = ""

