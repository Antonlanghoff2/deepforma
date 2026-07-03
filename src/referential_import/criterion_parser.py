from __future__ import annotations

from typing import Any

from common.text import clean_text, normalize_for_match

from .models import EvaluationCriterion


def attach_criteria_to_competencies(criteria: list[EvaluationCriterion], competencies: list[Any]) -> list[str]:
    competency_index = {getattr(item, "code", ""): item for item in competencies}
    issues: list[str] = []
    for criterion in criteria:
        target = competency_index.get(criterion.competency_code)
        if target is None:
            issues.append(f"Critère orphelin: {criterion.code} -> {criterion.competency_code}")
            continue
        target.evaluation_criteria.append(criterion)
        if criterion.page_start < target.page_start:
            target.page_start = criterion.page_start
        if criterion.page_end > target.page_end:
            target.page_end = criterion.page_end
    return issues

