from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from common.text import clean_text, normalize_for_match

from .models import DerivedSkill, EvaluationCriterion, OfficialCompetency
from .text_fallback_parser import COMPETENCY_RE, CRITERION_RE, line_contains_modality, strip_known_codes


@dataclass(slots=True)
class CompetencyContext:
    block_code: str = ""
    activity_code: str = ""
    competency_code: str = ""
    current_competency: OfficialCompetency | None = None
    current_criterion: EvaluationCriterion | None = None


def competency_code_to_block_activity(code: str) -> tuple[str, str]:
    digits = code.replace("C", "").strip()
    parts = digits.split(".")
    if len(parts) >= 2:
        block_code = f"BLOC_{parts[0]}"
        activity_code = f"A{parts[0]}.{parts[1]}"
        return block_code, activity_code
    return "", ""


def create_competency(code: str, label: str, page_number: int, context: CompetencyContext, confidence: float = 0.92) -> OfficialCompetency:
    block_code, activity_code = competency_code_to_block_activity(code)
    if context.block_code:
        block_code = context.block_code or block_code
    if context.activity_code:
        activity_code = context.activity_code or activity_code
    return OfficialCompetency(
        code=code,
        official_label=clean_text(label),
        normalized_label=normalize_for_match(label),
        block_code=block_code,
        activity_code=activity_code,
        page_start=page_number,
        page_end=page_number,
        confidence=confidence,
        source_pages=[page_number],
        source_text=clean_text(label),
    )


def append_competency_text(competency: OfficialCompetency, text: str) -> None:
    text = clean_text(text)
    if not text:
        return
    if competency.official_label:
        competency.official_label = clean_text(f"{competency.official_label} {text}")
    else:
        competency.official_label = text
    competency.normalized_label = normalize_for_match(competency.official_label)
    competency.source_text = clean_text(f"{competency.source_text} {text}")


def create_criterion(code: str, competency_code: str, label: str, page_number: int, confidence: float = 0.88) -> EvaluationCriterion:
    return EvaluationCriterion(
        code=code,
        competency_code=competency_code,
        criterion_label=clean_text(label),
        normalized_label=normalize_for_match(label),
        page_start=page_number,
        page_end=page_number,
        confidence=confidence,
        source_pages=[page_number],
        source_text=clean_text(label),
    )


def append_criterion_text(criterion: EvaluationCriterion, text: str) -> None:
    text = clean_text(text)
    if not text:
        return
    if criterion.criterion_label:
        criterion.criterion_label = clean_text(f"{criterion.criterion_label} {text}")
    else:
        criterion.criterion_label = text
    criterion.normalized_label = normalize_for_match(criterion.criterion_label)
    criterion.source_text = clean_text(f"{criterion.source_text} {text}")


def parse_competency_and_criteria_lines(
    lines: list[tuple[int, str]],
    *,
    context: CompetencyContext | None = None,
    page_number: int,
) -> tuple[list[OfficialCompetency], list[EvaluationCriterion], CompetencyContext]:
    context = context or CompetencyContext()
    competencies: list[OfficialCompetency] = []
    criteria: list[EvaluationCriterion] = []

    for _, raw_line in lines:
        line = clean_text(raw_line)
        if not line or line_contains_modality(line):
            continue
        competency_match = COMPETENCY_RE.search(line)
        criterion_match = CRITERION_RE.search(line)
        if competency_match and not criterion_match:
            code = f"C{competency_match.group(1)}"
            label = strip_known_codes(line[competency_match.end():] or line)
            if context.current_competency is not None:
                competencies.append(context.current_competency)
            context.current_competency = create_competency(code, label, page_number, context)
            context.competency_code = code
            context.current_criterion = None
            continue
        if criterion_match:
            code = f"CE{criterion_match.group(1)}"
            label = strip_known_codes(line[criterion_match.end():] or line)
            competency_code = context.competency_code or f"C{criterion_match.group(1).rsplit('.', 1)[0]}"
            criterion = create_criterion(code, competency_code, label, page_number)
            criteria.append(criterion)
            if context.current_competency is not None and context.current_competency.code == competency_code:
                context.current_competency.evaluation_criteria.append(criterion)
                context.current_criterion = criterion
            else:
                context.current_criterion = criterion
            continue
        if context.current_criterion is not None:
            append_criterion_text(context.current_criterion, line)
            continue
        if context.current_competency is not None:
            append_competency_text(context.current_competency, strip_known_codes(line))

    if context.current_competency is not None:
        competencies.append(context.current_competency)
        context.current_competency = None
    return competencies, criteria, context

