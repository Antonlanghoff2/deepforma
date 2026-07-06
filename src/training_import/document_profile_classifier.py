from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .layout_analyzer import analyze_layout
from .models import LayoutProfile
from .pdf_document_loader import PdfDocument


@dataclass(slots=True)
class ProfileResult:
    layout_profile: LayoutProfile
    confidence: float
    evidence: list[dict[str, Any]] = field(default_factory=list)


def classify_document_profile(document: PdfDocument) -> ProfileResult:
    layout = analyze_layout(document)
    return ProfileResult(
        layout_profile=layout.layout_profile,
        confidence=layout.confidence,
        evidence=[{"label": item.label, "score": item.score, "details": item.details} for item in layout.evidence],
    )
