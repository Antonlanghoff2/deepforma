from __future__ import annotations

import logging
import subprocess
import tempfile
import xml.etree.ElementTree as ET
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
    font_name: str | None = None
    bold: bool | None = None
    italic: bool | None = None


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


def _run_pdftotext(args: list[str]) -> str:
    result = subprocess.run(args, check=True, capture_output=True, text=True)
    return result.stdout


def _strip_ns(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[1]
    return tag


def _parse_bbox_output(xml_payload: str, path: Path) -> PdfDocument:
    root = ET.fromstring(xml_payload)
    pages: list[PdfPage] = []
    for page_index, page_el in enumerate([el for el in root.iter() if _strip_ns(el.tag) == "page"], start=1):
        try:
            width = float(page_el.attrib.get("width", "0") or 0.0)
            height = float(page_el.attrib.get("height", "0") or 0.0)
        except ValueError:
            width = height = None
        blocks: list[PdfTextBlock] = []
        block_order = 0
        for block_el in [el for el in page_el.iter() if _strip_ns(el.tag) == "block"]:
            text_parts: list[str] = []
            line_number = 0
            for line_el in [el for el in block_el.iter() if _strip_ns(el.tag) == "line"]:
                line_number += 1
                words = [clean_text(word_el.text) for word_el in line_el.iter() if _strip_ns(word_el.tag) == "word" and clean_text(word_el.text)]
                if words:
                    text_parts.append(" ".join(words))
            text = clean_text("\n".join(text_parts))
            if not text:
                continue
            bbox = None
            bbox_raw = block_el.attrib.get("bbox")
            if bbox_raw:
                try:
                    x0, y0, x1, y1 = [float(item) for item in bbox_raw.split()]
                    bbox = (x0, y0, x1, y1)
                except Exception:
                    bbox = None
            blocks.append(PdfTextBlock(text=text, bbox=bbox, page_number=page_index, line_number=line_number, order=block_order))
            block_order += 1
        page_text = "\n".join(block.text for block in blocks if block.text)
        pages.append(PdfPage(number=page_index, width=width, height=height, text=page_text, blocks=blocks))
    return PdfDocument(path=path, pages=pages, extraction_method="pdftotext-bbox-layout")


def _extract_bbox_layout(path: Path) -> PdfDocument:
    with tempfile.NamedTemporaryFile(suffix=".html", delete=True) as tmp:
        args = ["pdftotext", "-bbox-layout", "-enc", "UTF-8", str(path), tmp.name]
        subprocess.run(args, check=True, capture_output=True, text=True)
        xml_payload = Path(tmp.name).read_text(encoding="utf-8", errors="ignore")
    if not xml_payload.strip():
        raise RuntimeError("pdftotext n'a produit aucun contenu bbox.")
    return _parse_bbox_output(xml_payload, path)


def _extract_layout_text(path: Path) -> PdfDocument:
    output = _run_pdftotext(["pdftotext", "-layout", "-enc", "UTF-8", str(path), "-"])
    page_texts = output.split("\f") if output else [""]
    pages = []
    for index, text in enumerate(page_texts, start=1):
        cleaned = clean_text(text)
        pages.append(PdfPage(number=index, text=cleaned, blocks=[]))
    return PdfDocument(path=path, pages=pages, extraction_method="pdftotext-layout")


def load_pdf_document(path: str | Path, *, prefer_geometry: bool = True) -> PdfDocument:
    pdf_path = Path(path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF introuvable: {pdf_path}")
    warnings: list[str] = []
    if prefer_geometry:
        try:
            document = _extract_bbox_layout(pdf_path)
            return document
        except Exception as exc:
            warnings.append(f"Extraction géométrique indisponible: {exc}")
            LOGGER.info("Fallback texte brut pour %s: %s", pdf_path, exc)
    document = _extract_layout_text(pdf_path)
    document.warnings.extend(warnings)
    return document

