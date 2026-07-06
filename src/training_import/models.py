from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


ConfidenceProfile = Literal["low", "medium", "high"]
LayoutProfile = Literal["linear", "table", "numbered_sections", "step_guide", "business_sheet", "mixed"]
ReviewStatus = Literal["pending", "approved", "corrected", "rejected"]
SkillType = Literal["skill", "tool", "method", "knowledge", "domain", "soft_skill"]


def _jsonable(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


@dataclass(slots=True)
class Serializable:
    def to_dict(self) -> dict[str, Any]:
        return {key: _jsonable(value) for key, value in asdict(self).items()}


@dataclass(slots=True)
class FieldEvidence(Serializable):
    field_name: str
    source_section: str
    value_text: str
    page: int
    bbox: tuple[float, float, float, float] | None = None
    confidence: float = 0.0
    method: str = "explicit"


@dataclass(slots=True)
class ImportWarning(Serializable):
    code: str
    message: str
    page: int | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TrainingProvider(Serializable):
    name: str = ""
    canonical_name: str = ""
    source_text: str = ""
    page: int | None = None
    confidence: float = 0.0


@dataclass(slots=True)
class TrainingCertification(Serializable):
    label: str = ""
    code: str = ""
    cpf: str = "unknown"
    source_text: str = ""
    page: int | None = None
    confidence: float = 0.0


@dataclass(slots=True)
class TrainingObjective(Serializable):
    text: str = ""
    page: int = 0
    confidence: float = 0.0
    source_section: str = "objectives"
    evidence: list[FieldEvidence] = field(default_factory=list)


@dataclass(slots=True)
class TrainingModule(Serializable):
    title: str = ""
    code: str = ""
    content: str = ""
    page_start: int = 0
    page_end: int = 0
    source_section: str = "program"
    confidence: float = 0.0
    evidence: list[FieldEvidence] = field(default_factory=list)


@dataclass(slots=True)
class TrainingSkill(Serializable):
    canonical_name: str = ""
    surface_form: str = ""
    type: SkillType = "skill"
    source_section: str = "program"
    source_text: str = ""
    page: int = 0
    confidence: float = 0.0
    method: str = "rule"
    learning_stage: str = "taught"
    evidence: list[FieldEvidence] = field(default_factory=list)


@dataclass(slots=True)
class TrainingTool(Serializable):
    canonical_name: str = ""
    surface_form: str = ""
    source_section: str = "program"
    source_text: str = ""
    page: int = 0
    confidence: float = 0.0
    method: str = "rule"
    evidence: list[FieldEvidence] = field(default_factory=list)


@dataclass(slots=True)
class TrainingDomain(Serializable):
    canonical_name: str = ""
    surface_form: str = ""
    source_section: str = "program"
    source_text: str = ""
    page: int = 0
    confidence: float = 0.0
    method: str = "rule"
    evidence: list[FieldEvidence] = field(default_factory=list)


@dataclass(slots=True)
class TrainingPrerequisite(Serializable):
    text: str = ""
    page: int = 0
    confidence: float = 0.0
    evidence: list[FieldEvidence] = field(default_factory=list)


@dataclass(slots=True)
class TrainingProgram(Serializable):
    title: str = ""
    reference: str = ""
    duration_text: str = ""
    duration_hours: float | None = None
    duration_days: float | None = None
    duration_weeks: float | None = None
    duration_months: float | None = None
    level: str = ""
    format: str = ""
    price_text: str = ""
    price_amount: float | None = None
    price_kind: str = ""
    cpf: str = "unknown"
    public: str = ""
    prerequisites_text: str = ""
    objectives_text: str = ""
    pages_source: list[int] = field(default_factory=list)
    confidence: float = 0.0
    extraction_method: str = ""
    modules: list[TrainingModule] = field(default_factory=list)
    objectives: list[TrainingObjective] = field(default_factory=list)
    skills: list[TrainingSkill] = field(default_factory=list)
    tools: list[TrainingTool] = field(default_factory=list)
    domains: list[TrainingDomain] = field(default_factory=list)
    prerequisites: list[TrainingPrerequisite] = field(default_factory=list)
    certification: TrainingCertification = field(default_factory=TrainingCertification)
    evidence: list[FieldEvidence] = field(default_factory=list)


@dataclass(slots=True)
class TrainingDocument(Serializable):
    id: str
    source_path: str
    file_name: str
    sha256: str
    page_count: int
    schema_version: str = "1.0"
    importer_version: str = "0.1.0"
    title: str = ""
    provider: TrainingProvider = field(default_factory=TrainingProvider)
    program: TrainingProgram = field(default_factory=TrainingProgram)
    layout_profile: str = "linear"
    layout_confidence: float = 0.0
    review_required: bool = True
    confidence: float = 0.0
    extraction_method: str = ""
    source_pages: list[int] = field(default_factory=list)
    warnings: list[ImportWarning] = field(default_factory=list)


@dataclass(slots=True)
class TrainingImportReport(Serializable):
    schema_version: str
    importer_version: str
    document_id: str
    source_hash: str
    pages: int
    layout_profile: str
    layout_confidence: float
    title_found: bool
    provider_found: bool
    reference_found: bool
    duration_found: bool
    level_found: bool
    format_found: bool
    price_found: bool
    cpf_found: str
    certification_found: bool
    public_found: bool
    prerequisites_found: bool
    objectives_count: int
    modules_count: int
    skills_count: int
    tools_count: int
    domains_count: int
    warnings: list[ImportWarning] = field(default_factory=list)
    errors: list[ImportWarning] = field(default_factory=list)
    review_required: bool = True
    confidence: float = 0.0
    extraction_method: str = ""
