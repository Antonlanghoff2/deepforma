from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from common.text import clean_text, normalize_for_match

from .pdf_loader import PdfDocument, PdfPage, PdfTextBlock


HEADER_LABELS = {
    "activity": "Référentiel d'activités",
    "competency": "Référentiel de compétences",
    "assessment": "Modalités d'évaluation",
    "criteria": "Critères d'évaluation",
}


@dataclass(slots=True)
class ExtractedCell:
    column_name: str
    text: str
    page_number: int
    bbox: tuple[float, float, float, float] | None = None
    order: int = 0


@dataclass(slots=True)
class ExtractedTablePage:
    page_number: int
    columns: dict[str, list[ExtractedCell]] = field(default_factory=dict)
    column_bounds: list[tuple[float, float]] = field(default_factory=list)
    layout_quality: float = 0.0
    header_detected: bool = False


def _block_x_center(block: PdfTextBlock) -> float | None:
    if not block.bbox:
        return None
    x0, _, x1, _ = block.bbox
    return (x0 + x1) / 2.0


def _cluster_positions(values: list[float], *, max_clusters: int = 4) -> list[float]:
    if not values:
        return []
    ordered = sorted(values)
    if len(ordered) <= max_clusters:
        return ordered
    gaps = [(right - left, idx) for idx, (left, right) in enumerate(zip(ordered, ordered[1:]))]
    gap_threshold = max(35.0, (ordered[-1] - ordered[0]) * 0.12)
    split_points = [idx for gap, idx in gaps if gap >= gap_threshold]
    if not split_points:
        return [sum(ordered) / len(ordered)]
    clusters: list[list[float]] = []
    start = 0
    for split in split_points[: max_clusters - 1]:
        clusters.append(ordered[start : split + 1])
        start = split + 1
    clusters.append(ordered[start:])
    return [sum(cluster) / len(cluster) for cluster in clusters if cluster]


def _detect_header_columns(page: PdfPage) -> list[tuple[str, float]] | None:
    candidates: list[tuple[str, float]] = []
    for block in page.blocks:
        text = normalize_for_match(block.text)
        if not text:
            continue
        if any(normalize_for_match(label) in text for label in HEADER_LABELS.values()):
            x_center = _block_x_center(block)
            if x_center is not None:
                candidates.append((block.text, x_center))
    if len(candidates) >= 2:
        items = sorted(candidates, key=lambda item: item[1])
        return items
    return None


def _assign_column_name(index: int, count: int) -> str:
    order = ["activity", "competency", "assessment", "criteria"]
    if count >= 4 and index < 4:
        return order[index]
    if index == 0:
        return "activity"
    if index == 1:
        return "competency"
    if index == 2:
        return "assessment"
    return "criteria"


def detect_tables(document: PdfDocument) -> list[ExtractedTablePage]:
    pages: list[ExtractedTablePage] = []
    for page in document.pages:
        if not page.blocks:
            pages.append(
                ExtractedTablePage(
                    page_number=page.number,
                    columns={"full_text": [ExtractedCell("full_text", page.text, page.number, None, 0)]},
                    layout_quality=0.1,
                    header_detected=False,
                )
            )
            continue

        header_columns = _detect_header_columns(page)
        x_centers = [value for value in (_block_x_center(block) for block in page.blocks) if value is not None]
        center_groups = _cluster_positions(x_centers, max_clusters=4)
        if header_columns:
            center_groups = [value for _, value in header_columns]
        bounds: list[tuple[float, float]] = []
        if center_groups:
            sorted_centers = sorted(center_groups)
            if len(sorted_centers) == 1:
                bounds = [(sorted_centers[0] - 1_000, sorted_centers[0] + 1_000)]
            else:
                edges = [float("-inf")]
                for left, right in zip(sorted_centers, sorted_centers[1:]):
                    edges.append((left + right) / 2.0)
                edges.append(float("inf"))
                bounds = list(zip(edges[:-1], edges[1:], strict=False))
        else:
            bounds = [(float("-inf"), float("inf"))]

        columns: dict[str, list[ExtractedCell]] = {}
        for block in sorted(page.blocks, key=lambda item: ((item.bbox[1] if item.bbox else 0.0), (item.bbox[0] if item.bbox else 0.0), item.order)):
            x_center = _block_x_center(block)
            column_index = 0
            if x_center is not None and bounds:
                for idx, (lower, upper) in enumerate(bounds):
                    if lower <= x_center < upper:
                        column_index = idx
                        break
                else:
                    column_index = min(range(len(bounds)), key=lambda idx: abs(x_center - sum(bounds[idx]) / 2.0))
            column_name = _assign_column_name(column_index, len(bounds))
            columns.setdefault(column_name, []).append(
                ExtractedCell(
                    column_name=column_name,
                    text=clean_text(block.text),
                    page_number=page.number,
                    bbox=block.bbox,
                    order=block.order,
                )
            )
        if page.text and "full_text" not in columns:
            columns["full_text"] = [ExtractedCell("full_text", page.text, page.number, None, len(columns.get("full_text", [])))]
        pages.append(
            ExtractedTablePage(
                page_number=page.number,
                columns=columns,
                column_bounds=[(lower, upper) for lower, upper in bounds],
                layout_quality=0.85 if header_columns else 0.65,
                header_detected=bool(header_columns),
            )
        )
    return pages

