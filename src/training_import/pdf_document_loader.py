from __future__ import annotations

import logging
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from common.text import clean_text


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class PdfTextBlock:
    text: str
    bbox: tuple[float, float, float, float] | None = None
    page_number: int = 0
    line_number: int = 0
    order: int = 0
    font_size: float | None = None
    font_name: str = ""
    bold: bool = False
    block_type: str = "text"


@dataclass(slots=True)
class PdfPage:
    number: int
    width: float | None = None
    height: float | None = None
    text: str = ""
    blocks: list[PdfTextBlock] = field(default_factory=list)


@dataclass(slots=True)
class PdfDocument:
    path: Path
    pages: list[PdfPage]
    extraction_method: str
    warnings: list[str] = field(default_factory=list)


def _load_fitz() -> Any | None:
    try:
        import fitz  # type: ignore
    except Exception:
        return None
    return fitz


def _extract_with_fitz(path: Path) -> PdfDocument:
    fitz = _load_fitz()
    if fitz is None:
        raise ImportError("PyMuPDF indisponible")
    doc = fitz.open(path)
    pages: list[PdfPage] = []
    for page_index, page in enumerate(doc, start=1):
        blocks: list[PdfTextBlock] = []
        raw = page.get_text("dict")
        order = 0
        for block in raw.get("blocks", []):
            if block.get("type", 0) != 0:
                continue
            bbox = tuple(float(item) for item in block.get("bbox", (0, 0, 0, 0)))
            lines: list[str] = []
            font_name = ""
            font_size = None
            bold = False
            line_no = 0
            for line in block.get("lines", []):
                line_no += 1
                words: list[str] = []
                for span in line.get("spans", []):
                    text = clean_text(span.get("text"))
                    if not text:
                        continue
                    words.append(text)
                    font_name = font_name or clean_text(span.get("font"))
                    try:
                        font_size = max(font_size or 0.0, float(span.get("size", 0.0)))
                    except Exception:
                        pass
                    bold = bold or "bold" in clean_text(span.get("font")).lower()
                if words:
                    lines.append(" ".join(words))
            text = clean_text("\n".join(lines))
            if not text:
                continue
            blocks.append(
                PdfTextBlock(
                    text=text,
                    bbox=bbox,
                    page_number=page_index,
                    line_number=line_no,
                    order=order,
                    font_size=font_size,
                    font_name=font_name,
                    bold=bold,
                    block_type="text",
                )
            )
            order += 1
        page_text = "\n".join(block.text for block in blocks if block.text)
        pages.append(PdfPage(number=page_index, width=float(page.rect.width), height=float(page.rect.height), text=page_text, blocks=blocks))
    return PdfDocument(path=path, pages=pages, extraction_method="pymupdf")


def _extract_with_pdftotext(path: Path) -> PdfDocument:
    result = subprocess.run(["pdftotext", "-layout", "-enc", "UTF-8", str(path), "-"], check=True, capture_output=True, text=True)
    page_texts = result.stdout.split("\f") if result.stdout else [""]
    pages: list[PdfPage] = []
    for index, page_text in enumerate(page_texts, start=1):
        cleaned = clean_text(page_text)
        blocks = [PdfTextBlock(text=line, page_number=index, line_number=idx, order=idx) for idx, line in enumerate(cleaned.splitlines(), start=1) if clean_text(line)]
        pages.append(PdfPage(number=index, text=cleaned, blocks=blocks))
    return PdfDocument(path=path, pages=pages, extraction_method="pdftotext-layout")


def load_pdf_document(path: str | Path, *, prefer_geometry: bool = True) -> PdfDocument:
    pdf_path = Path(path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF introuvable: {pdf_path}")
    warnings: list[str] = []
    if prefer_geometry:
        try:
            return _extract_with_fitz(pdf_path)
        except Exception as exc:
            warnings.append(f"PyMuPDF indisponible ou échec d'extraction: {exc}")
            LOGGER.info("Fallback pdftotext pour %s: %s", pdf_path, exc)
    document = _extract_with_pdftotext(pdf_path)
    document.warnings.extend(warnings)
    return document
