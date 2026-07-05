from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from common.text import clean_text, normalize_for_match

from .models import ImportIssue, ImportReport, OfficialCompetency


def _issue(severity: str, code: str, message: str, *, page: int | None = None, entity_code: str | None = None, **details: Any) -> ImportIssue:
    return ImportIssue(severity=severity, code=code, message=message, page=page, entity_code=entity_code, details=dict(details))


def validate_import(
    *,
    document_id: str,
    source_hash: str,
    pages: int,
    blocks: list[Any],
    activities: list[Any],
    competencies: list[OfficialCompetency],
    criteria: list[Any],
    derived_skills: list[Any],
    tools_methods: list[Any],
    importer_version: str,
    duplicate_document: bool,
    extraction_mode: str,
    notes: str = "",
) -> ImportReport:
    errors: list[ImportIssue] = []
    warnings: list[ImportIssue] = []
    review_items: list[dict[str, Any]] = []

    if pages <= 0:
        errors.append(_issue("error", "no_pages", "Aucune page détectée."))
    if not blocks:
        warnings.append(_issue("warning", "no_block", "Aucun bloc structuré détecté; le document reste exploitable en revue."))
        review_items.append({
            "type": "missing_blocks",
            "message": "Aucun bloc structuré détecté.",
        })
    if not competencies:
        warnings.append(_issue("warning", "no_competency", "Aucune compétence officielle détectée; revue humaine requise."))
        review_items.append({
            "type": "missing_competencies",
            "message": "Aucune compétence officielle détectée.",
        })

    block_codes = [getattr(item, "code", "") for item in blocks]
    if len(block_codes) != len(set(block_codes)):
        warnings.append(_issue("warning", "duplicate_block_code", "Un code bloc est dupliqué."))

    activity_codes = [getattr(item, "code", "") for item in activities]
    if len(activity_codes) != len(set(activity_codes)):
        warnings.append(_issue("warning", "duplicate_activity_code", "Un code activité est dupliqué."))

    competency_codes = [item.code for item in competencies]
    if len(competency_codes) != len(set(competency_codes)):
        errors.append(_issue("error", "duplicate_competency_code", "Un code compétence est dupliqué."))

    criterion_codes = [getattr(item, "code", "") for item in criteria]
    if len(criterion_codes) != len(set(criterion_codes)):
        errors.append(_issue("error", "duplicate_criterion_code", "Un code critère est dupliqué."))

    block_numbers: dict[str, str] = {}
    for competency in competencies:
        if competency.code.startswith("C"):
            expected_block = competency.code[1:].split(".", 1)[0]
            actual_block = competency.block_code.replace("BLOC_", "")
            if actual_block and expected_block and expected_block != actual_block:
                warnings.append(
                    _issue(
                        "warning",
                        "block_mismatch",
                        "Le rattachement bloc/compétence semble incohérent.",
                        entity_code=competency.code,
                        expected_block=expected_block,
                        actual_block=actual_block,
                    )
                )
        if competency.page_end < competency.page_start:
            errors.append(_issue("error", "page_order", "Page de fin antérieure à la page de début.", entity_code=competency.code))
        normalized = normalize_for_match(competency.official_label)
        if "modalite" in normalized and "evaluation" in normalized:
            errors.append(_issue("error", "modality_in_competency", "Une modalité d'évaluation est incluse dans le texte officiel.", entity_code=competency.code))
        if "ce" in normalized and len(normalized.split()) > 1:
            warnings.append(_issue("warning", "criterion_in_competency", "Un code critère apparaît dans le texte d'une compétence.", entity_code=competency.code))
        if len(clean_text(competency.official_label)) < 8:
            warnings.append(_issue("warning", "short_competency", "La compétence officielle est très courte.", entity_code=competency.code))
        if len(clean_text(competency.official_label)) > 2000:
            warnings.append(_issue("warning", "long_competency", "La compétence officielle est très longue.", entity_code=competency.code))
        if not competency.evaluation_criteria:
            warnings.append(_issue("warning", "competency_without_criteria", "Aucun critère rattaché à la compétence.", entity_code=competency.code))
        for criterion in competency.evaluation_criteria:
            if criterion.competency_code != competency.code:
                warnings.append(
                    _issue(
                        "warning",
                        "criterion_link_mismatch",
                        "Le critère est associé à un code incompatible.",
                        entity_code=criterion.code,
                        expected_competency=competency.code,
                        actual_competency=criterion.competency_code,
                    )
                )

    for criterion in criteria:
        if criterion.page_end < criterion.page_start:
            errors.append(_issue("error", "criterion_page_order", "Page de fin antérieure à la page de début.", entity_code=criterion.code))

    coverage_score = 0.0
    if competencies:
        with_criteria = sum(1 for comp in competencies if comp.evaluation_criteria)
        coverage_score = with_criteria / len(competencies)
        if coverage_score < 0.5:
            warnings.append(_issue("warning", "low_coverage", "Le score de couverture des critères est faible.", coverage=coverage_score))

    for competency in competencies:
        if not competency.evaluation_criteria:
            review_items.append(
                {
                    "type": "competency_without_criteria",
                    "code": competency.code,
                    "message": "Compétence sans critère associé.",
                }
            )
        if competency.provenance != "human_review":
            review_items.append(
                {
                    "type": "provenance_check",
                    "code": competency.code,
                    "message": f"Provenance non humaine: {competency.provenance}",
                }
            )

    score_global = max(0.0, min(1.0, 0.4 + (coverage_score * 0.6) - (0.1 if warnings else 0.0) - (0.2 if errors else 0.0)))
    if errors:
        status = "failed"
    elif warnings:
        status = "review_required"
    else:
        status = "success"
    report = ImportReport(
        schema_version="1.0",
        importer_version=importer_version,
        document_id=document_id,
        source_hash=source_hash,
        pages=pages,
        blocks=len(blocks),
        activities=len(activities),
        competencies=len(competencies),
        criteria=len(criteria),
        derived_skills=len(derived_skills),
        tools_methods=len(tools_methods),
        errors=errors,
        warnings=warnings,
        review_items=review_items,
        score_global=round(score_global, 4),
        coverage_score=round(coverage_score, 4),
        duplicate_document=duplicate_document,
        extraction_mode=extraction_mode,
        notes=notes,
        status=status,
    )
    return report

