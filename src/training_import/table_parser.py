from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from common.text import clean_text, normalize_for_match

from .pdf_document_loader import PdfDocument, PdfTextBlock


@dataclass(slots=True)
class ParsedTableRow:
    page: int
    cells: list[str] = field(default_factory=list)
    source_blocks: list[dict[str, Any]] = field(default_factory=list)


def _x_center(block: PdfTextBlock) -> float | None:
    if not block.bbox:
        return None
    x0, _, x1, _ = block.bbox
    return (x0 + x1) / 2.0


def _cluster(values: list[float]) -> list[float]:
    if not values:
        return []
    values = sorted(values)
    clusters = [[values[0]]]
    for value in values[1:]:
        if abs(value - clusters[-1][-1]) <= 120:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    return [sum(cluster) / len(cluster) for cluster in clusters]


def parse_tables(document: PdfDocument) -> list[ParsedTableRow]:
    rows: list[ParsedTableRow] = []
    for page in document.pages:
        if not page.blocks:
            continue
        x_centers = [value for value in (_x_center(block) for block in page.blocks) if value is not None]
        column_centers = _cluster(x_centers)
        if len(column_centers) < 2:
            for block in page.blocks:
                text = clean_text(block.text)
                if "|" in text:
                    rows.append(ParsedTableRow(page=page.number, cells=[clean_text(item) for item in text.split("|")], source_blocks=[{"text": text, "bbox": block.bbox}]))
            continue
        sorted_blocks = sorted(page.blocks, key=lambda item: ((item.bbox[1] if item.bbox else 0.0), (item.bbox[0] if item.bbox else 0.0), item.order))
        current: list[str] = []
        current_y = None
        source_blocks: list[dict[str, Any]] = []
        for block in sorted_blocks:
            text = clean_text(block.text)
            if not text:
                continue
            y = block.bbox[1] if block.bbox else 0.0
            if current_y is None or abs(y - current_y) <= 18:
                current.append(text)
                source_blocks.append({"text": text, "bbox": block.bbox})
                current_y = y if current_y is None else min(current_y, y)
            else:
                rows.append(ParsedTableRow(page=page.number, cells=current, source_blocks=source_blocks))
                current = [text]
                source_blocks = [{"text": text, "bbox": block.bbox}]
                current_y = y
        if current:
            rows.append(ParsedTableRow(page=page.number, cells=current, source_blocks=source_blocks))
    return rows
