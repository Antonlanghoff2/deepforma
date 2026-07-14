from __future__ import annotations

import hashlib
import logging
import os
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common.text import clean_text, normalize_for_match

from .competency_parser import CompetencyContext, parse_competency_and_criteria_lines
from .models import DerivedSkill, EvaluationCriterion, ImportIssue, ImportReport, OfficialCompetency, ReferentialActivity, ReferentialBlock, ReferentialDocument
from .title_extractor import extract_referential_title
from .pdf_loader import load_pdf_document
from .skill_decomposer import decompose_competency
from .store import ReferentialImportStore
from .table_extractor import detect_tables
from .text_fallback_parser import iter_fallback_lines, strip_known_codes
from .validators import validate_import


IMPORTER_VERSION = "0.1.0"
LOGGER = logging.getLogger(__name__)


def _block_from_dict(payload: dict[str, Any]) -> ReferentialBlock:
    return ReferentialBlock(**payload)


def _activity_from_dict(payload: dict[str, Any]) -> ReferentialActivity:
    return ReferentialActivity(**payload)


def _criterion_from_dict(payload: dict[str, Any]) -> EvaluationCriterion:
    return EvaluationCriterion(**payload)


def _skill_from_dict(payload: dict[str, Any]) -> DerivedSkill:
    return DerivedSkill(**payload)


def _competency_from_dict(payload: dict[str, Any]) -> OfficialCompetency:
    data = dict(payload)
    data.setdefault('evaluation_criteria', [])
    data.setdefault('derived_skills', [])
    data.setdefault('tools_methods', [])
    data.setdefault('knowledge_items', [])
    data.setdefault('source_pages', [])
    data.setdefault('source_text', '')
    data.setdefault('provenance', 'human_review')
    data.setdefault('review_status', 'pending')
    data['evaluation_criteria'] = [_criterion_from_dict(item) for item in data.get('evaluation_criteria', [])]
    data['derived_skills'] = [_skill_from_dict(item) for item in data.get('derived_skills', [])]
    return OfficialCompetency(**data)


def _report_from_dict(payload: dict[str, Any]) -> ImportReport:
    data = dict(payload)
    data['errors'] = [ImportIssue(**item) for item in data.get('errors', [])]
    data['warnings'] = [ImportIssue(**item) for item in data.get('warnings', [])]
    return ImportReport(**data)


def analysis_from_export(export: dict[str, Any]) -> dict[str, Any]:
    document = ReferentialDocument(**export['document'])
    blocks = [_block_from_dict(item) for item in export.get('blocks', [])]
    activities = [_activity_from_dict(item) for item in export.get('activities', [])]
    competencies = [_competency_from_dict(item) for item in export.get('competencies', [])]
    criteria = [_criterion_from_dict(item) for item in export.get('criteria', [])]
    derived_skills = [_skill_from_dict(item) for item in export.get('derived_skills', [])]
    tools_methods = [_skill_from_dict(item) for item in export.get('tools_methods', [])]
    report = _report_from_dict(export['report'])
    title_extraction = export.get('title_extraction')
    return {
        'document': document,
        'blocks': blocks,
        'activities': activities,
        'competencies': competencies,
        'criteria': criteria,
        'derived_skills': derived_skills,
        'tools_methods': tools_methods,
        'report': report,
        'export': export,
        'title_extraction': title_extraction,
        'duplicate_document': report.duplicate_document,
    }


def build_export_payload(analysis: dict[str, Any]) -> dict[str, Any]:
    document: ReferentialDocument = analysis['document']
    blocks = analysis.get('blocks', [])
    activities = analysis.get('activities', [])
    competencies = analysis.get('competencies', [])
    criteria = analysis.get('criteria', [])
    derived_skills = analysis.get('derived_skills', [])
    tools_methods = analysis.get('tools_methods', [])
    report: ImportReport = analysis['report']
    title_extraction = analysis.get('title_extraction')
    return {
        'schema_version': '1.0',
        'document': document.to_dict(),
        'blocks': [item.to_dict() for item in blocks],
        'activities': [item.to_dict() for item in activities],
        'competencies': [item.to_dict() for item in competencies],
        'criteria': [item.to_dict() for item in criteria],
        'derived_skills': [item.to_dict() for item in derived_skills],
        'tools_methods': [item.to_dict() for item in tools_methods],
        'report': report.to_dict(),
        'title_extraction': title_extraction.to_dict() if hasattr(title_extraction, 'to_dict') else title_extraction,
        'warnings': [item.to_dict() for item in report.warnings],
        'errors': [item.to_dict() for item in report.errors],
    }




def _maybe_apply_referential_ml_enrichment(analysis: dict[str, Any]) -> dict[str, Any]:
    enabled = os.getenv('REFERENTIAL_ML_IMPORT_ENABLED', 'false').lower() in {'1', 'true', 'yes', 'on'}
    if not enabled:
        return analysis
    try:
        from referential_learning.pdf_loader import load_pdf_document as ml_load_pdf_document
        from referential_learning.pipeline import build_annotation_document, enrich_with_ml_predictions

        pdf_path = Path(analysis['document'].source_path)
        ml_document = ml_load_pdf_document(pdf_path)
        enriched = enrich_with_ml_predictions(build_annotation_document(ml_document))
        analysis['ml_enrichment'] = enriched.to_dict()
    except Exception as exc:
        analysis['ml_enrichment_error'] = str(exc)
    return analysis


def _infer_document_title_from_pages(pages: list[Any]) -> str:
    title_re = re.compile(r"^[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9'’&(),./ -]{8,}$")
    excluded_prefixes = (
        "bloc ",
        "activite ",
        "activité ",
        "a1.",
        "c1.",
        "ce1.",
    )
    excluded_markers = (
        "référentiel d'activités",
        "referentiel d'activites",
        "référentiel de compétences",
        "referentiel de competences",
        "modalités d’évaluation",
        "modalites d evaluation",
        "critères d’évaluation",
        "criteres d evaluation",
    )
    marker_patterns = [re.escape(marker) for marker in excluded_markers]
    marker_split_re = re.compile(r"(?:" + "|".join(marker_patterns) + r")", flags=re.IGNORECASE)
    fallback_candidate = ""

    for page in pages[:2]:
        raw_text = getattr(page, "text", "") or ""
        if not raw_text:
            continue
        lines = [clean_text(line) for line in raw_text.splitlines() if clean_text(line)]
        if len(lines) >= 2:
            first, second = lines[0], lines[1]
            second_normalized = normalize_for_match(second)
            if len(first) <= 40 and len(second.split()) >= 2 and not any(
                second_normalized.startswith(prefix)
                for prefix in ("referentiel", "bloc", "activite", "a1", "c1", "ce1")
            ):
                return second
        for line in lines:
            normalized = line.lower()
            if any(normalized.startswith(prefix) for prefix in excluded_prefixes):
                continue
            if any(marker in normalized for marker in excluded_markers):
                parts = marker_split_re.split(line, maxsplit=1)
                prefix = clean_text(parts[0])
                suffix = clean_text(parts[1]) if len(parts) > 1 else ""
                for candidate in (prefix, suffix):
                    normalized_candidate = clean_text(candidate)
                    if normalized_candidate and len(normalized_candidate.split()) >= 2 and title_re.match(normalized_candidate):
                        return normalized_candidate
                continue
            if line.isupper() and len(line) >= 10 and title_re.match(line):
                return line
            if title_re.match(line) and len(line.split()) >= 2:
                fallback_candidate = line
    return fallback_candidate


def _extract_document_metadata(pages: list[Any]) -> dict[str, Any]:
    raw_page_texts = [getattr(page, "text", "") or "" for page in pages if clean_text(getattr(page, "text", "") or "")]
    full_text = "\n".join(clean_text(text) for text in raw_page_texts if clean_text(text))
    first_page_lines = [clean_text(line) for line in (raw_page_texts[0].splitlines() if raw_page_texts else []) if clean_text(line)]

    provider = ""
    title = _infer_document_title_from_pages(pages)
    if len(first_page_lines) >= 2:
        provider_candidate = first_page_lines[0]
        title_candidate = first_page_lines[1]
        if len(provider_candidate) <= 40 and len(title_candidate.split()) >= 2:
            provider = provider_candidate
            if not title:
                title = title_candidate
    if not title and first_page_lines:
        title = first_page_lines[0]

    reference = ""
    reference_match = re.search(r"\bRéférence\s*(?:[:\-–—])\s*([^\n\r]+)", full_text, flags=re.IGNORECASE)
    if reference_match:
        reference = clean_text(reference_match.group(1)).split(" ")[0]

    duration_hours = None
    duration_match = re.search(r"\bDurée\s*(?:[:\-–—])\s*(\d+)\s*heures?", full_text, flags=re.IGNORECASE)
    if duration_match:
        try:
            duration_hours = int(duration_match.group(1))
        except Exception:
            duration_hours = None

    cpf_eligible: bool | None = None
    cpf_match = re.search(r"\b(?:Éligible\s+CPF|CPF)\s*(?:[:\-–—])\s*(oui|non)", full_text, flags=re.IGNORECASE)
    if cpf_match:
        cpf_eligible = clean_text(cpf_match.group(1)).lower() == "oui"

    level = ""
    level_match = re.search(r"\bNiveau\s*(?:[:\-–—])\s*([^\n\r]+)", full_text, flags=re.IGNORECASE)
    if level_match:
        level = clean_text(level_match.group(1))

    format_value = ""
    format_match = re.search(r"\bFormat\s*(?:[:\-–—])\s*([^\n\r]+)", full_text, flags=re.IGNORECASE)
    if format_match:
        format_value = clean_text(format_match.group(1))

    return {
        "provider": provider,
        "title": title,
        "reference": reference,
        "duration_hours": duration_hours,
        "cpf_eligible": cpf_eligible,
        "level": level,
        "format": format_value,
        "full_text": full_text,
    }


def _semantic_derived_skills_from_document(document_loader: Any) -> tuple[list[DerivedSkill], dict[str, Any]]:
    try:
        from referential_learning.pipeline import build_annotation_document
    except Exception as exc:
        LOGGER.warning("Semantic referential fallback unavailable: %s", exc)
        return [], {"error": str(exc)}

    try:
        from referential_learning.pdf_loader import load_pdf_document as semantic_load_pdf_document
        semantic_document = semantic_load_pdf_document(document_loader.path)
        annotation = build_annotation_document(semantic_document)
    except Exception as exc:
        LOGGER.warning("Semantic referential annotation failed: %s", exc)
        return [], {"error": str(exc)}

    semantic_skills: list[DerivedSkill] = []
    seen: set[tuple[str, str, int]] = set()
    label_to_category = {
        "SKILL": "skill",
        "SOFT_SKILL": "soft_skill",
        "TOOL": "tool",
        "METHOD": "method",
        "KNOWLEDGE": "knowledge",
        "DOMAIN": "domain",
    }
    for entity in annotation.entities:
        category = label_to_category.get(entity.predicted_label)
        if not category:
            continue
        canonical = clean_text(entity.canonical_name or entity.text)
        if not canonical:
            continue
        page_number = int(entity.page or 1)
        key = (normalize_for_match(canonical), category, page_number)
        if key in seen:
            continue
        seen.add(key)
        semantic_skills.append(
            DerivedSkill(
                label=clean_text(entity.text or canonical),
                canonical_label=canonical,
                category=category,
                source_code=f"SEMANTIC:{page_number}",
                source_type="semantic_entity",
                surface_form=clean_text(entity.text or canonical),
                normalized_surface=normalize_for_match(entity.text or canonical),
                provenance="semantic_match",
                confidence=float(entity.confidence or 0.0),
                explicit=bool(entity.predicted_label in {"SKILL", "TOOL"}),
                page_start=page_number,
                page_end=page_number,
                context=clean_text(entity.evidence or "referential_learning"),
            )
        )
    return semantic_skills, annotation.to_dict()


class ReferentialImportService:
    def __init__(self, *, store: ReferentialImportStore | None = None, output_dir: str | Path | None = None) -> None:
        self.store = store or ReferentialImportStore()
        self.output_dir = Path(output_dir or "data/referentials/imported")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _document_id(source_path: Path, sha256: str) -> str:
        return hashlib.sha256(f"{source_path.name}|{sha256}".encode("utf-8")).hexdigest()[:24]

    @staticmethod
    def _code_or_fallback(prefix: str, text: str, fallback_index: int) -> str:
        match = normalize_for_match(text).split()
        if prefix == "BLOC" and match:
            digits = next((part for part in match if part.isdigit()), str(fallback_index))
            return f"BLOC_{digits}"
        if prefix == "A":
            for token in text.split():
                token = token.strip()
                if token.startswith("A") and "." in token:
                    return token.rstrip(":;.,")
        return f"{prefix}_{fallback_index}"

    @staticmethod
    def _normalize_title_candidate(text: str) -> str:
        candidate = clean_text(text)
        if not candidate:
            return ""
        marker_re = re.compile(
            r"(?:référentiel|referentiel|modalités d['’]?évaluation|modalites d evaluation|critères d['’]?évaluation|criteres d evaluation)",
            flags=re.IGNORECASE,
        )
        match = marker_re.search(candidate)
        if match:
            candidate = candidate[: match.start()].strip()
        candidate = re.sub(
            r"^(?:référentiel|referentiel)(?:\s+(?:d['’]?activit(?:é|e)s?|de\s+compétences?|de\s+competences?|d['’]?évaluation|d['’]?evaluation))?(?:\s*[-:–—]\s*)?",
            "",
            candidate,
            flags=re.IGNORECASE,
        ).strip()
        candidate = re.sub(r"^(?:référentiel|referentiel)\s+", "", candidate, flags=re.IGNORECASE).strip()
        candidate = candidate.strip(" -:–—")
        if not candidate:
            return ""
        if candidate.isupper() or sum(1 for char in candidate if char.isupper()) >= max(2, len(candidate) // 2):
            candidate = candidate.lower()
            candidate = candidate[:1].upper() + candidate[1:] if candidate else candidate
        return candidate

    @staticmethod
    def _infer_document_title(pages: list[Any]) -> str:
        return _infer_document_title_from_pages(pages)

    def analyze(self, input_path: str | Path) -> dict[str, Any]:
        pdf_path = Path(input_path)
        LOGGER.info("[referential-import] loading pdf=%s", pdf_path)
        document_loader = load_pdf_document(pdf_path)
        page_count = len(document_loader.pages)
        total_chars = sum(len(clean_text(getattr(page, "text", "") or "")) for page in document_loader.pages)
        total_blocks = sum(len(getattr(page, "blocks", []) or []) for page in document_loader.pages)
        LOGGER.info("[referential-import] loaded pages=%s chars=%s blocks=%s method=%s", page_count, total_chars, total_blocks, document_loader.extraction_method)
        raw_hash = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
        document_id = self._document_id(pdf_path, raw_hash)
        duplicate_document = self.store.has_document(raw_hash, IMPORTER_VERSION)
        title_extraction = extract_referential_title(document_loader, file_name=pdf_path.name)
        metadata = _extract_document_metadata(document_loader.pages)
        priority_title = (
            title_extraction.certification_title
            or title_extraction.target_job_title
            or title_extraction.document_title
            or title_extraction.title
        )
        inferred_title = priority_title or clean_text(pdf_path.stem) or pdf_path.name
        if priority_title == title_extraction.certification_title and priority_title:
            inferred_title_type = "certification_title"
        elif priority_title == title_extraction.target_job_title and priority_title:
            inferred_title_type = "target_job_title"
        elif priority_title == title_extraction.document_title and priority_title:
            inferred_title_type = "document_title"
        elif priority_title == title_extraction.title and priority_title:
            inferred_title_type = title_extraction.title_type or "document_title"
        else:
            inferred_title_type = "filename" if inferred_title == clean_text(pdf_path.stem) or inferred_title == pdf_path.name else title_extraction.title_type or "filename"
        table_pages = detect_tables(document_loader)
        LOGGER.info("[referential-import] table pages=%s", len(table_pages))

        blocks: list[ReferentialBlock] = []
        activities: list[ReferentialActivity] = []
        competencies: list[OfficialCompetency] = []
        criteria: list[Any] = []
        derived_skills: list[DerivedSkill] = []
        issues: list[ImportIssue] = []
        context = CompetencyContext()
        block_index = 0

        for table_page in table_pages:
            activity_cells = table_page.columns.get("activity", [])
            for cell in activity_cells:
                text = clean_text(cell.text)
                if not text:
                    continue
                normalized = normalize_for_match(text)
                if normalized.startswith("bloc "):
                    block_index += 1
                    block_code = self._code_or_fallback("BLOC", text, block_index)
                    blocks.append(
                        ReferentialBlock(
                            code=block_code,
                            label=strip_known_codes(text),
                            page_start=cell.page_number,
                            page_end=cell.page_number,
                            confidence=0.90 if table_page.header_detected else 0.70,
                            text=text,
                            source_pages=[cell.page_number],
                        )
                    )
                    context.block_code = block_code
                    continue
                if normalized.startswith("activite ") or normalized.startswith("activité "):
                    activity_code = self._code_or_fallback("A", text, len(activities) + 1)
                    activities.append(
                        ReferentialActivity(
                            code=activity_code,
                            block_code=context.block_code,
                            label=strip_known_codes(text),
                            page_start=cell.page_number,
                            page_end=cell.page_number,
                            confidence=0.86,
                            text=text,
                            source_pages=[cell.page_number],
                        )
                    )
                    context.activity_code = activity_code
                    continue
                if text.startswith("A") and "." in text:
                    activity_code = text.split()[0].rstrip(":;.,")
                    activities.append(
                        ReferentialActivity(
                            code=activity_code,
                            block_code=context.block_code,
                            label=strip_known_codes(text),
                            page_start=cell.page_number,
                            page_end=cell.page_number,
                            confidence=0.82,
                            text=text,
                            source_pages=[cell.page_number],
                        )
                    )
                    context.activity_code = activity_code

            comp_lines = [(cell.page_number, cell.text) for cell in table_page.columns.get("competency", [])]
            page_comps: list[OfficialCompetency] = []
            page_criteria: list[EvaluationCriterion] = []
            if comp_lines:
                page_comps, page_criteria, context = parse_competency_and_criteria_lines(comp_lines, context=context, page_number=table_page.page_number)
                competencies.extend(page_comps)
                criteria.extend(page_criteria)

            crit_lines = [(cell.page_number, cell.text) for cell in table_page.columns.get("criteria", [])]
            if crit_lines:
                _, crit_from_columns, context = parse_competency_and_criteria_lines(crit_lines, context=context, page_number=table_page.page_number)
                criteria.extend(crit_from_columns)
                page_criteria.extend(crit_from_columns)

            page_full_text = [cell.text for cell in table_page.columns.get("full_text", [])]
            page_text_blob = "\n".join(page_full_text)
            normalized_blob = normalize_for_match(page_text_blob)
            should_fallback = (
                not page_comps
                or not page_criteria
                or not table_page.header_detected
                or (table_page.layout_quality < 0.7)
                or bool(normalized_blob and any(token in normalized_blob for token in ["bloc", "activite", "competence", "critere"]))
            )
            if should_fallback and page_full_text:
                fallback_lines = iter_fallback_lines(page_full_text)
                line_pairs = [(line.line_number, line.text) for line in fallback_lines]
                fallback_comps, fallback_criteria, context = parse_competency_and_criteria_lines(line_pairs, context=context, page_number=table_page.page_number)
                competencies.extend(fallback_comps)
                criteria.extend(fallback_criteria)

        if context.current_competency is not None:
            competencies.append(context.current_competency)
            context.current_competency = None

        block_map: dict[str, ReferentialBlock] = {}
        for block in blocks:
            block_map.setdefault(block.code, block)
        blocks = list(block_map.values())

        activity_map: dict[str, ReferentialActivity] = {}
        for activity in activities:
            activity_map.setdefault(activity.code, activity)
        activities = list(activity_map.values())

        competency_map: dict[str, OfficialCompetency] = {}
        for competency in competencies:
            existing = competency_map.get(competency.code)
            if existing is None:
                competency_map[competency.code] = competency
                continue
            if len(clean_text(competency.official_label)) > len(clean_text(existing.official_label)):
                existing.official_label = competency.official_label
                existing.normalized_label = competency.normalized_label
                existing.source_text = competency.source_text
            existing.evaluation_criteria.extend(competency.evaluation_criteria)
        competencies = list(competency_map.values())

        criterion_map: dict[str, Any] = {}
        for criterion in criteria:
            criterion_map.setdefault(criterion.code, criterion)
        criteria = list(criterion_map.values())

        for competency in competencies:
            competency.derived_skills = decompose_competency(competency, competency.evaluation_criteria)
            derived_skills.extend(competency.derived_skills)

        semantic_annotation: dict[str, Any] | None = None
        if not competencies:
            LOGGER.info("[referential-import] no codified competencies detected; running semantic fallback")
            semantic_skills, semantic_annotation = _semantic_derived_skills_from_document(document_loader)
            if semantic_skills:
                derived_skills.extend(semantic_skills)
                LOGGER.info("[referential-import] semantic fallback extracted skills=%s", len(semantic_skills))
            else:
                LOGGER.info("[referential-import] semantic fallback produced no skills")

        tools_methods = [item for item in derived_skills if item.category in {"tool", "method", "regulatory"}]
        report = validate_import(
            document_id=document_id,
            source_hash=raw_hash,
            pages=len(document_loader.pages),
            blocks=blocks,
            activities=activities,
            competencies=competencies,
            criteria=criteria,
            derived_skills=derived_skills,
            tools_methods=tools_methods,
            importer_version=IMPORTER_VERSION,
            duplicate_document=duplicate_document,
            extraction_mode=document_loader.extraction_method,
            notes="",
        )
        document = ReferentialDocument(
            id=document_id,
            source_path=str(pdf_path),
            file_name=pdf_path.name,
            sha256=raw_hash,
            schema_version="1.0",
            importer_version=IMPORTER_VERSION,
            page_count=len(document_loader.pages),
            title=inferred_title,
            document_title=title_extraction.document_title or inferred_title,
            certification_title=title_extraction.certification_title,
            target_job_title=title_extraction.target_job_title,
            rncp_code=title_extraction.rncp_code,
            title_type=inferred_title_type,
            title_confidence=title_extraction.confidence if inferred_title_type != "filename" else 0.35,
            title_source_page=title_extraction.source_page,
            title_source_text=title_extraction.source_text or "",
            title_candidates=[candidate.to_dict() for candidate in title_extraction.candidates],
            title_warnings=list(title_extraction.warnings),
            provider=metadata.get("provider", ""),
            reference=metadata.get("reference", ""),
            duration_hours=metadata.get("duration_hours"),
            cpf_eligible=metadata.get("cpf_eligible"),
            source_type="pdf",
            text_extraction_method=document_loader.extraction_method,
            review_status="pending",
            collected_at=datetime.now(timezone.utc).isoformat(),
            notes="",
        )

        export_payload = {
            "schema_version": "1.0",
            "document": document.to_dict(),
            "title_extraction": title_extraction.to_dict(),
            "blocks": [item.to_dict() for item in blocks],
            "competencies": [item.to_dict() for item in competencies],
            "criteria": [item.to_dict() for item in criteria],
            "derived_skills": [item.to_dict() for item in derived_skills],
            "tools_methods": [item.to_dict() for item in tools_methods],
            "report": report.to_dict(),
            "warnings": [item.to_dict() for item in report.warnings],
            "errors": [item.to_dict() for item in report.errors],
        }
        analysis = {
            "document": document,
            "title_extraction": title_extraction,
            "blocks": blocks,
            "activities": activities,
            "competencies": competencies,
            "criteria": criteria,
            "derived_skills": derived_skills,
            "tools_methods": tools_methods,
            "report": report,
            "export": export_payload,
            "duplicate_document": duplicate_document,
            "source_document": document_loader,
            "metadata": metadata,
            "semantic_annotation": semantic_annotation,
        }
        if title_extraction.warnings:
            report.warnings.extend(
                [ImportIssue(severity='warning', code='title_extraction', message=warning) for warning in title_extraction.warnings]
            )
        analysis["export"] = build_export_payload(analysis)
        LOGGER.info(
            "[referential-import] validation status=%s blocks=%s activities=%s competencies=%s criteria=%s derived=%s",
            report.status,
            report.blocks,
            report.activities,
            report.competencies,
            report.criteria,
            report.derived_skills,
        )
        return _maybe_apply_referential_ml_enrichment(analysis)

    def approve(self, analysis: dict[str, Any], *, validated_by: str = "human_review") -> Path:
        document: ReferentialDocument = analysis["document"]
        report: ImportReport = analysis["report"]
        blocks = analysis["blocks"]
        activities = analysis["activities"]
        competencies = analysis["competencies"]
        criteria = analysis["criteria"]
        derived_skills = analysis["derived_skills"]
        issues = [*report.errors, *report.warnings]
        document.review_status = "approved"
        document.validated_at = datetime.now(timezone.utc).isoformat()
        document.validated_by = validated_by
        self.store.save_import(
            document,
            report,
            blocks,
            activities,
            competencies,
            criteria,
            derived_skills,
            issues,
            review_status="approved",
        )
        output_path = self.output_dir / f"{document.file_name}.json"
        export_payload = analysis.get("export") or build_export_payload(analysis)
        try:
            from referentials.referential_registry import convert_imported_to_skills_format, normalize_referential_payload
            canonical = convert_imported_to_skills_format(export_payload)
            if canonical.get("skills"):
                export_payload["skills"] = canonical["skills"]
                export_payload["metadata"] = {**export_payload.get("metadata", {}), **canonical.get("metadata", {})}
                export_payload["referential_id"] = canonical.get("referential_id", export_payload.get("referential_id", ""))
                export_payload["title"] = canonical.get("title", export_payload.get("title", ""))
        except Exception as exc:
            LOGGER.warning("Impossible de normaliser vers le schéma canonique: %s", exc)
        output_path.write_text(json.dumps(export_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        
        # Générer automatiquement les candidats d'annotation
        try:
            from scripts.generate_annotation_candidates_from_imported import generate_annotation_candidates_from_imported
            generate_annotation_candidates_from_imported(self.output_dir)
            LOGGER.info("Candidats d'annotation générés pour %s", document.file_name)
        except Exception as exc:
            LOGGER.warning("Impossible de générer les candidats d'annotation: %s", exc)
        
        return output_path
