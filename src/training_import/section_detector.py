from __future__ import annotations

from dataclasses import dataclass, field

from common.text import clean_text, normalize_for_match

from .pdf_document_loader import PdfDocument


SECTION_VARIANTS = {
    "objectives": [
        "objectifs",
        "objectifs pedagogiques",
        "objectifs d apprentissage",
        "enjeux business",
        "a l issue de cette formation",
        "vous serez capable de",
    ],
    "program": [
        "programme",
        "programme detaille",
        "contenu",
        "contenu du programme",
        "contenu de la formation",
        "deroule de la formation",
        "le deroule de la formation",
    ],
    "skills": [
        "competences",
        "competences visees",
        "competences acquises",
        "competences developpees",
        "competences que vous developperez",
        "competences acquises a l issue de la formation",
    ],
    "prerequisites": ["prerequis", "pre requis", "conditions d acces"],
}


@dataclass(slots=True)
class DetectedSection:
    key: str
    title: str
    normalized_title: str
    page_start: int
    page_end: int
    content: str = ""
    evidence: list[str] = field(default_factory=list)


def _normalize_title(value: str) -> str:
    return normalize_for_match(value)


def _match_section(line: str) -> str | None:
    normalized = _normalize_title(line)
    for key, variants in SECTION_VARIANTS.items():
        for variant in variants:
            if normalized.startswith(_normalize_title(variant)):
                return key
    return None


def detect_sections(document: PdfDocument) -> list[DetectedSection]:
    sections: list[DetectedSection] = []
    current: DetectedSection | None = None
    for page in document.pages:
        lines = [clean_text(line) for line in page.text.splitlines() if clean_text(line)]
        for line in lines:
            section_key = _match_section(line)
            if section_key:
                if current is not None:
                    sections.append(current)
                current = DetectedSection(
                    key=section_key,
                    title=line,
                    normalized_title=_normalize_title(line),
                    page_start=page.number,
                    page_end=page.number,
                    evidence=[f"page:{page.number}"],
                )
                continue
            if current is not None:
                current.content = clean_text(f"{current.content}\n{line}")
                current.page_end = page.number
    if current is not None:
        sections.append(current)
    return sections
