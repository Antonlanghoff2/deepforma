from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from common.text import normalize_for_match

from .pdf_document_loader import PdfDocument, PdfTextBlock
from .models import LayoutProfile


@dataclass(slots=True)
class LayoutEvidence:
    label: str
    score: float
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LayoutAnalysis:
    layout_profile: LayoutProfile
    confidence: float
    evidence: list[LayoutEvidence] = field(default_factory=list)


def _block_x(block: PdfTextBlock) -> float | None:
    if not block.bbox:
        return None
    x0, _, x1, _ = block.bbox
    return (x0 + x1) / 2.0


def analyze_layout(document: PdfDocument) -> LayoutAnalysis:
    pages = document.pages
    blocks = [block for page in pages for block in page.blocks]
    if not blocks:
        return LayoutAnalysis("linear", 0.2, [LayoutEvidence("empty_document", 0.2)])

    texts = [normalize_for_match(block.text) for block in blocks]
    block_count = len(blocks)
    x_values = [value for value in (_block_x(block) for block in blocks) if value is not None]
    x_groups = len({round(value / 80.0) for value in x_values}) if x_values else 0
    table_markers = sum(1 for text in texts if any(marker in text for marker in ("module", "contenu", "etape", "étape", "thème", "theme")))
    numbered_markers = sum(1 for text in texts if text[:1].isdigit() or text.startswith(("a", "b", "c", "1 ", "2 ", "3 ")))
    step_markers = sum(1 for text in texts if any(marker in text for marker in ("étape", "etape", "phase", "sprint", "semaine")))
    business_markers = sum(1 for text in texts if any(marker in text for marker in ("organisme", "prix", "cpf", "niveau", "public", "pré-requis", "prerequis")))

    evidence: list[LayoutEvidence] = []
    score_by_profile = {
        "table": min(0.95, 0.2 + table_markers * 0.15 + min(x_groups, 4) * 0.12),
        "numbered_sections": min(0.95, 0.2 + numbered_markers * 0.08),
        "step_guide": min(0.95, 0.2 + step_markers * 0.12),
        "business_sheet": min(0.95, 0.25 + business_markers * 0.15),
        "linear": 0.35,
    }
    profile = max(score_by_profile, key=score_by_profile.get)
    confidence = score_by_profile[profile]
    if table_markers:
        evidence.append(LayoutEvidence("table_markers", score_by_profile["table"], {"markers": table_markers, "groups": x_groups}))
    if numbered_markers:
        evidence.append(LayoutEvidence("numbered_sections_markers", score_by_profile["numbered_sections"], {"markers": numbered_markers}))
    if step_markers:
        evidence.append(LayoutEvidence("step_markers", score_by_profile["step_guide"], {"markers": step_markers}))
    if business_markers:
        evidence.append(LayoutEvidence("business_markers", score_by_profile["business_sheet"], {"markers": business_markers}))
    if len({block.page_number for block in blocks}) > 1 and profile == "linear":
        evidence.append(LayoutEvidence("multi_page_linear", 0.3, {"pages": len({block.page_number for block in blocks})}))
    return LayoutAnalysis(profile, confidence, evidence)
