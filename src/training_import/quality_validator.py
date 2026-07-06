from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from common.text import clean_text, normalize_for_match

from .models import ImportWarning, TrainingDocument, TrainingImportReport, TrainingProgram


def _warn(code: str, message: str, **details: Any) -> ImportWarning:
    return ImportWarning(code=code, message=message, details=dict(details))


def validate_training_import(
    *,
    document: TrainingDocument,
    program: TrainingProgram,
    layout_profile: str,
    layout_confidence: float,
    warnings: list[ImportWarning],
    errors: list[ImportWarning],
    importer_version: str,
) -> TrainingImportReport:
    title_found = bool(clean_text(program.title))
    provider_found = bool(clean_text(document.provider.name))
    reference_found = bool(clean_text(program.reference))
    duration_found = bool(clean_text(program.duration_text))
    level_found = bool(clean_text(program.level))
    format_found = bool(clean_text(program.format))
    price_found = bool(clean_text(program.price_text))
    certification_found = bool(clean_text(program.certification.label or program.certification.code))
    public_found = bool(clean_text(program.public))
    prerequisites_found = bool(clean_text(program.prerequisites_text))
    objectives_count = len(program.objectives)
    modules_count = len(program.modules)
    skills_count = len(program.skills)
    tools_count = len(program.tools)
    domains_count = len(program.domains)

    if not title_found:
        errors.append(_warn("missing_title", "Document sans titre."))
    if not modules_count and not objectives_count and not skills_count:
        errors.append(_warn("empty_program", "Aucun programme détecté."))
    if not skills_count:
        warnings.append(_warn("no_skills", "Aucune compétence détectée."))
    if modules_count and any(not clean_text(module.content) for module in program.modules):
        warnings.append(_warn("module_without_content", "Au moins un module est vide."))
    if program.duration_text and "mois" in normalize_for_match(program.duration_text) and program.duration_hours is not None:
        warnings.append(_warn("duration_converted", "Durée en mois convertie en heures sans convention explicite."))
    if "gratuit" in normalize_for_match(program.price_text) and program.price_amount is not None:
        warnings.append(_warn("price_ambiguous", "Prix ambigu pour un document gratuit."))
    if program.cpf == "unknown":
        warnings.append(_warn("cpf_unknown", "Statut CPF inconnu."))

    confidence = max(0.0, min(1.0, 0.2 + 0.1 * min(modules_count, 5) + 0.1 * min(skills_count + tools_count, 8) + 0.1 * layout_confidence - 0.05 * len(errors)))
    review_required = bool(errors or warnings or layout_profile == "mixed" or layout_confidence < 0.6)
    return TrainingImportReport(
        schema_version="1.0",
        importer_version=importer_version,
        document_id=document.id,
        source_hash=document.sha256,
        pages=document.page_count,
        layout_profile=layout_profile,
        layout_confidence=layout_confidence,
        title_found=title_found,
        provider_found=provider_found,
        reference_found=reference_found,
        duration_found=duration_found,
        level_found=level_found,
        format_found=format_found,
        price_found=price_found,
        cpf_found=program.cpf,
        certification_found=certification_found,
        public_found=public_found,
        prerequisites_found=prerequisites_found,
        objectives_count=objectives_count,
        modules_count=modules_count,
        skills_count=skills_count,
        tools_count=tools_count,
        domains_count=domains_count,
        warnings=warnings,
        errors=errors,
        review_required=review_required,
        confidence=round(confidence, 4),
        extraction_method=document.extraction_method,
    )
