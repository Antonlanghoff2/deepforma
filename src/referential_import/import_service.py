from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common.text import clean_text, normalize_for_match

from .competency_parser import CompetencyContext, parse_competency_and_criteria_lines
from .models import DerivedSkill, EvaluationCriterion, ImportIssue, ImportReport, OfficialCompetency, ReferentialActivity, ReferentialBlock, ReferentialDocument
from .pdf_loader import load_pdf_document
from .skill_decomposer import decompose_competency
from .store import ReferentialImportStore
from .table_extractor import detect_tables
from .text_fallback_parser import iter_fallback_lines, strip_known_codes
from .validators import validate_import


IMPORTER_VERSION = "0.1.0"


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
        'warnings': [item.to_dict() for item in report.warnings],
        'errors': [item.to_dict() for item in report.errors],
    }


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
        for page in pages[:2]:
            page_text = clean_text(getattr(page, "text", "") or "")
            if not page_text:
                continue
            for raw_line in page_text.splitlines():
                line = clean_text(raw_line)
                if not line:
                    continue
                normalized = line.lower()
                if any(normalized.startswith(prefix) for prefix in excluded_prefixes):
                    continue
                if any(marker in normalized for marker in excluded_markers):
                    parts = marker_split_re.split(line, maxsplit=1)
                    prefix = clean_text(parts[0])
                    suffix = clean_text(parts[1]) if len(parts) > 1 else ""
                    for candidate in (prefix, suffix):
                        normalized_candidate = ReferentialImportService._normalize_title_candidate(candidate)
                        if normalized_candidate and len(normalized_candidate.split()) >= 2 and title_re.match(normalized_candidate):
                            return normalized_candidate
                    continue
                normalized_line = ReferentialImportService._normalize_title_candidate(line)
                if line.isupper() and len(line) >= 10 and normalized_line:
                    return normalized_line
                if title_re.match(normalized_line) and len(normalized_line.split()) >= 2:
                    return normalized_line
        return ""

    def analyze(self, input_path: str | Path) -> dict[str, Any]:
        pdf_path = Path(input_path)
        document_loader = load_pdf_document(pdf_path)
        raw_hash = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
        document_id = self._document_id(pdf_path, raw_hash)
        duplicate_document = self.store.has_document(raw_hash, IMPORTER_VERSION)
        table_pages = detect_tables(document_loader)
        inferred_title = self._infer_document_title(document_loader.pages)

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
            source_type="pdf",
            text_extraction_method=document_loader.extraction_method,
            review_status="pending",
            collected_at=datetime.now(timezone.utc).isoformat(),
            notes="",
        )

        export_payload = {
            "schema_version": "1.0",
            "document": document.to_dict(),
            "blocks": [item.to_dict() for item in blocks],
            "competencies": [item.to_dict() for item in competencies],
            "criteria": [item.to_dict() for item in criteria],
            "derived_skills": [item.to_dict() for item in derived_skills],
            "tools_methods": [item.to_dict() for item in tools_methods],
            "report": report.to_dict(),
            "warnings": [item.to_dict() for item in report.warnings],
            "errors": [item.to_dict() for item in report.errors],
        }
        return {
            "document": document,
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
        }

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
        output_path.write_text(json.dumps(analysis["export"], ensure_ascii=False, indent=2), encoding="utf-8")
        return output_path
