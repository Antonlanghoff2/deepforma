from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common.text import clean_text

from .document_profile_classifier import classify_document_profile
from .exporters import to_export_dict
from .field_extractor import extract_fields
from .models import FieldEvidence, ImportWarning, TrainingDocument, TrainingImportReport, TrainingObjective, TrainingProvider
from .module_parser import parse_modules
from .pdf_document_loader import load_pdf_document
from .quality_validator import validate_training_import
from .section_detector import detect_sections
from .skill_extractor import extract_skills


IMPORTER_VERSION = "0.1.0"


def _document_id(path: Path, sha256: str) -> str:
    return hashlib.sha256(f"{path.name}|{sha256}".encode("utf-8")).hexdigest()[:24]


class TrainingImportService:
    def __init__(self, *, importer_version: str = IMPORTER_VERSION) -> None:
        self.importer_version = importer_version

    def analyze(self, input_path: str | Path) -> dict[str, Any]:
        pdf_path = Path(input_path)
        raw_hash = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
        document_loader = load_pdf_document(pdf_path)
        profile = classify_document_profile(document_loader)
        sections = detect_sections(document_loader)
        provider, program = extract_fields(document_loader, sections)

        objective_text = clean_text(program.objectives_text)
        program.objectives = [
            TrainingObjective(text=line, page=1, confidence=0.7, source_section="objectives", evidence=[FieldEvidence("objective", "objectives", line, 1, confidence=0.7, method="rule")])
            for line in [clean_text(line) for line in objective_text.splitlines() if clean_text(line)]
        ]

        module_pairs = [(section.title, section.content, section.page_start) for section in sections if section.key == "program"]
        program.modules = parse_modules(module_pairs)
        skills, tools, domains, prerequisites = extract_skills(
            {section.key: section.content for section in sections},
            [module.content for module in program.modules],
            prerequisites=program.prerequisites_text,
        )
        program.skills = skills
        program.tools = tools
        program.domains = domains
        program.prerequisites = prerequisites
        document = TrainingDocument(
            id=_document_id(pdf_path, raw_hash),
            source_path=str(pdf_path),
            file_name=pdf_path.name,
            sha256=raw_hash,
            page_count=len(document_loader.pages),
            title=program.title,
            provider=provider,
            program=program,
        )
        document.layout_profile = profile.layout_profile
        document.layout_confidence = profile.confidence
        document.review_required = True
        document.confidence = round(max(0.0, min(1.0, 0.25 + profile.confidence * 0.5 + 0.05 * (len(program.skills) + len(program.tools)))), 4)
        document.extraction_method = document_loader.extraction_method
        document.source_pages = [page.number for page in document_loader.pages]
        document.warnings = [ImportWarning(code="layout_profile", message=f"Profil détecté: {profile.layout_profile}", details={"evidence": profile.evidence})]

        warnings: list[ImportWarning] = []
        errors: list[ImportWarning] = []
        report = validate_training_import(
            document=document,
            program=program,
            layout_profile=profile.layout_profile,
            layout_confidence=profile.confidence,
            warnings=warnings,
            errors=errors,
            importer_version=self.importer_version,
        )
        document.review_required = report.review_required
        export = to_export_dict(document)
        return {
            "document": document,
            "profile": profile,
            "sections": sections,
            "program": program,
            "report": report,
            "warnings": warnings,
            "errors": errors,
            "export": export,
            "source_document": document_loader,
        }

    def analyze_many(self, paths: list[str | Path]) -> list[dict[str, Any]]:
        return [self.analyze(path) for path in paths]
