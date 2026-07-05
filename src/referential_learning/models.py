from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

SectionLabel = Literal[
    'TITLE', 'PROVIDER', 'REFERENCE', 'DURATION', 'LEVEL', 'FORMAT', 'PRICE', 'CPF',
    'CERTIFICATION', 'PUBLIC', 'PREREQUISITES', 'OBJECTIVES', 'PROGRAM', 'MODULE',
    'SKILLS', 'TOOLS', 'DOMAINS', 'FOOTER', 'OTHER',
]

EntityLabel = Literal[
    'SKILL', 'SOFT_SKILL', 'TOOL', 'METHOD', 'KNOWLEDGE', 'DOMAIN', 'DEGREE',
    'CERTIFICATION', 'DURATION', 'PRICE', 'REFERENCE', 'PROVIDER', 'OTHER',
]

def _jsonable(value: Any) -> Any:
    if hasattr(value, 'to_dict'):
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
class PdfBlock(Serializable):
    block_id: str
    page: int
    order: int
    text: str
    bbox: tuple[float, float, float, float] | None = None
    font_size: float | None = None
    font_name: str = ''
    bold: bool = False
    block_type: str = 'text'
    line_count: int = 0
    source_file: str = ''
    document_id: str = ''

@dataclass(slots=True)
class PdfPage(Serializable):
    number: int
    width: float | None = None
    height: float | None = None
    text: str = ''
    blocks: list[PdfBlock] = field(default_factory=list)
    text_length: int = 0
    density: float = 0.0
    has_text_layer: bool = True

@dataclass(slots=True)
class PdfDocument(Serializable):
    document_id: str
    source_file: str
    path: str
    sha256: str
    file_size: int
    page_count: int
    extraction_method: str
    pages: list[PdfPage] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    needs_ocr: bool = False

@dataclass(slots=True)
class AnnotationBlock(Serializable):
    block_id: str
    page: int
    bbox: list[float] | None
    text: str
    predicted_section: SectionLabel
    confidence: float
    approved_section: SectionLabel | None = None
    status: str = 'pending'
    order: int = 0
    source_file: str = ''
    document_id: str = ''

@dataclass(slots=True)
class AnnotationEntity(Serializable):
    entity_id: str
    start: int
    end: int
    text: str
    predicted_label: EntityLabel
    approved_label: EntityLabel | None = None
    canonical_name: str = ''
    confidence: float = 0.0
    page: int = 0
    block_id: str = ''
    source_file: str = ''
    document_id: str = ''
    status: str = 'pending'
    referential_id: str | None = None
    evidence: str = ''

@dataclass(slots=True)
class AnnotationPage(Serializable):
    number: int
    text: str
    blocks: list[AnnotationBlock] = field(default_factory=list)
    entities: list[AnnotationEntity] = field(default_factory=list)

@dataclass(slots=True)
class AnnotationDocument(Serializable):
    document_id: str
    source_file: str
    sha256: str
    page_count: int
    pages: list[AnnotationPage] = field(default_factory=list)
    blocks: list[AnnotationBlock] = field(default_factory=list)
    entities: list[AnnotationEntity] = field(default_factory=list)
    status: str = 'pending'
    needs_ocr: bool = False
    notes: str = ''
    metadata: dict[str, Any] = field(default_factory=dict)
