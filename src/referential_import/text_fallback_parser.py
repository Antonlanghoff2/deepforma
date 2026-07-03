from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from common.text import clean_text, normalize_for_match


BLOCK_RE = re.compile(r"\bBloc\s+(\d+)\b", re.IGNORECASE)
ACTIVITY_RE = re.compile(r"\bActivit[eé]\s+(\d+)\b", re.IGNORECASE)
ACTIVITY_CODE_RE = re.compile(r"\bA(\d+\.\d+)\b", re.IGNORECASE)
COMPETENCY_RE = re.compile(r"\bC(\d+\.\d+)(?!\.\d)\b", re.IGNORECASE)
CRITERION_RE = re.compile(r"\bCE(\d+\.\d+\.\d+)\b", re.IGNORECASE)
HEADER_RE = re.compile(r"\b(r[eé]f[eé]rentiel|modalit[eé]s?|crit[eè]res?)\b", re.IGNORECASE)

SPLIT_BOUNDARY_RE = re.compile(r"(?=\b(?:Bloc\s+\d+|Activit[eé]\s+\d+|A\d+\.\d+|C\d+\.\d+(?!\.\d)|CE\d+\.\d+\.\d+)\b)", re.IGNORECASE)

@dataclass(slots=True)
class FallbackLine:
    page_number: int
    line_number: int
    text: str


def _segment_page_text(page_text: str) -> list[str]:
    text = clean_text(page_text)
    if not text:
        return []
    if "\n" in text:
        chunks = [clean_text(line) for line in text.splitlines() if clean_text(line)]
        if chunks:
            return chunks
    segments = [clean_text(chunk) for chunk in SPLIT_BOUNDARY_RE.split(text) if clean_text(chunk)]
    if segments:
        return segments
    return [text]


def iter_fallback_lines(pages: list[str]) -> list[FallbackLine]:
    lines: list[FallbackLine] = []
    for page_number, page_text in enumerate(pages, start=1):
        for line_number, raw_line in enumerate(_segment_page_text(page_text), start=1):
            text = clean_text(raw_line)
            if text:
                lines.append(FallbackLine(page_number=page_number, line_number=line_number, text=text))
    return lines


def strip_known_codes(text: str) -> str:
    text = BLOCK_RE.sub("", text)
    text = ACTIVITY_RE.sub("", text)
    text = ACTIVITY_CODE_RE.sub("", text)
    text = COMPETENCY_RE.sub("", text)
    text = CRITERION_RE.sub("", text)
    text = HEADER_RE.sub("", text)
    return clean_text(text)


def line_contains_modality(text: str) -> bool:
    normalized = normalize_for_match(text)
    return bool(
        "modalite evaluation" in normalized
        or "modalites evaluation" in normalized
        or "conditions evaluation" in normalized
        or "modalite" in normalized and "evaluation" in normalized
    )

