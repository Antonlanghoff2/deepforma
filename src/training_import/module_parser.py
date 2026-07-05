from __future__ import annotations

import re
from typing import Iterable

from common.text import clean_text, normalize_for_match

from .models import FieldEvidence, TrainingModule


MODULE_RE = re.compile(r"^(?:module|étape|etape|phase|s(?:e)?maine)\s*(\d+(?:-\d+)?)?\s*[:\-]?\s*(.*)$", re.IGNORECASE)
NUMBERED_RE = re.compile(r"^(?:\d+|[A-Z]\d+|§\s*\d+)\s*[.)-]?\s*(.*)$")


def _strip_numbering(line: str) -> str:
    match = NUMBERED_RE.match(line)
    if match:
        return clean_text(match.group(1) or "")
    return clean_text(line)


def parse_modules(sections: Iterable[tuple[str, str, int]]) -> list[TrainingModule]:
    modules: list[TrainingModule] = []
    current: TrainingModule | None = None
    for title, content, page in sections:
        lines = [clean_text(line) for line in f"{title}\n{content}".splitlines() if clean_text(line)]
        for line in lines:
            match = MODULE_RE.match(line)
            if match:
                if current is not None:
                    modules.append(current)
                code = clean_text(match.group(1) or f"{len(modules) + 1}")
                label = clean_text(match.group(2) or line)
                current = TrainingModule(
                    title=label or line,
                    code=f"Module {code}" if not normalize_for_match(code).startswith("module") else code,
                    content="",
                    page_start=page,
                    page_end=page,
                    confidence=0.8,
                    evidence=[FieldEvidence("module", "program", line, page, confidence=0.8, method="rule")],
                )
                continue
            if current is None and NUMBERED_RE.match(line):
                current = TrainingModule(
                    title=clean_text(NUMBERED_RE.sub(r"\\1", line)) or line,
                    code=f"Module {len(modules) + 1}",
                    content="",
                    page_start=page,
                    page_end=page,
                    confidence=0.5,
                    evidence=[FieldEvidence("module", "program", line, page, confidence=0.5, method="rule")],
                )
                continue
            if current is not None:
                current.content = clean_text(f"{current.content}\n{line}")
                current.page_end = page
        if current is not None and current.content:
            pass
    if current is not None:
        modules.append(current)
    return modules
