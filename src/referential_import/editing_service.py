from __future__ import annotations

import logging
from typing import Any

from common.text import clean_text, normalize_for_match

from .models import DerivedSkill, OfficialCompetency

REJECTED_STATUSES = {"rejected", "excluded"}
ACTIVE_STATUSES = {"pending", "approved", "corrected"}

LOGGER = logging.getLogger(__name__)


class ReferentialEditingService:
    def apply_edits(self, analysis: dict[str, Any], form: Any) -> dict[str, Any]:
        title = clean_text(form.get("validated_title") or "")
        if title:
            analysis["document"].title = title
        note = clean_text(form.get("validation_note") or "")
        if note:
            analysis["document"].notes = note

        original_competencies = list(analysis.get("competencies", []))
        anchor = original_competencies[0] if original_competencies else None
        updated_competencies: list[Any] = []

        for competency in original_competencies:
            label_key = f"competency_label__{competency.code}"
            status_key = f"competency_status__{competency.code}"
            remove_key = f"remove_competency__{competency.code}"

            if str(form.get(remove_key) or "").strip() in {"1", "true", "yes", "on"}:
                competency.review_status = "rejected"
                updated_competencies.append(competency)
                continue

            if label_key in form:
                corrected = clean_text(form.get(label_key))
                if corrected:
                    competency.official_label = corrected
                    competency.normalized_label = corrected
            if status_key in form:
                status = clean_text(form.get(status_key))
                if status:
                    competency.review_status = status
            updated_competencies.append(competency)

        manual_labels = self._parse_new_competencies(form)
        for index, label in enumerate(manual_labels, start=1):
            code = f"MANUAL_{index}"
            if any(
                normalize_for_match(getattr(item, "official_label", "") or "")
                == normalize_for_match(label)
                for item in updated_competencies
            ):
                continue
            updated_competencies.append(
                OfficialCompetency(
                    code=code,
                    official_label=label,
                    normalized_label=label,
                    block_code=getattr(anchor, "block_code", "MANUAL") if anchor else "MANUAL",
                    activity_code=getattr(anchor, "activity_code", "MANUAL") if anchor else "MANUAL",
                    page_start=getattr(anchor, "page_start", 1) if anchor else 1,
                    page_end=getattr(anchor, "page_end", 1) if anchor else 1,
                    confidence=1.0,
                    review_status="approved",
                    source_pages=list(getattr(anchor, "source_pages", []) or [1]) if anchor else [1],
                    source_text=label,
                    provenance="human_review",
                )
            )

        updated_derived_skills: list[Any] = []
        for index, skill in enumerate(list(analysis.get("derived_skills", []))):
            remove_key = f"remove_derived_skill__{index}"
            if str(form.get(remove_key) or "").strip() in {"1", "true", "yes", "on"}:
                continue
            label_key = f"derived_skill_label__{index}"
            canonical_key = f"derived_skill_canonical__{index}"
            category_key = f"derived_skill_category__{index}"
            if label_key in form:
                label = clean_text(form.get(label_key))
                if label:
                    skill.label = label
                    skill.surface_form = label
            if canonical_key in form:
                canonical = clean_text(form.get(canonical_key))
                if canonical:
                    skill.canonical_label = canonical
            if category_key in form:
                category = clean_text(form.get(category_key))
                if category in {"skill", "method", "tool", "domain", "soft_skill", "knowledge", "regulatory", "action"}:
                    skill.category = category
            skill.normalized_surface = normalize_for_match(skill.surface_form or skill.label)
            updated_derived_skills.append(skill)

        manual_skills = self._parse_new_derived_skills(form)
        seen_manual: set[str] = set()
        for skill in manual_skills:
            key = normalize_for_match(skill.canonical_label or skill.label)
            if key in seen_manual:
                continue
            seen_manual.add(key)
            updated_derived_skills.append(skill)

        kept_codes = {competency.code for competency in updated_competencies}
        analysis["competencies"] = updated_competencies
        analysis["criteria"] = [
            c for c in analysis.get("criteria", []) if c.competency_code in kept_codes
        ]
        analysis["derived_skills"] = [
            s for s in updated_derived_skills
            if getattr(s, "source_code", "") not in kept_codes
        ]
        analysis["tools_methods"] = [
            s for s in analysis["derived_skills"]
            if getattr(s, "category", "") in {"tool", "method", "regulatory"}
        ]
        if analysis.get("report") is not None:
            analysis["report"].competencies = len(analysis.get("competencies", []))
            analysis["report"].criteria = len(analysis.get("criteria", []))
            analysis["report"].derived_skills = len(analysis.get("derived_skills", []))
            analysis["report"].tools_methods = len(analysis.get("tools_methods", []))
        for competency in analysis.get("competencies", []):
            competency.evaluation_criteria = [
                c for c in analysis.get("criteria", [])
                if c.competency_code == competency.code
            ]

        self.apply_criterion_edits(analysis, form)
        return analysis

    def reject_skill(self, analysis: dict[str, Any], skill_code: str) -> dict[str, Any]:
        for competency in analysis.get("competencies", []):
            if competency.code == skill_code:
                competency.review_status = "rejected"
                break
        return analysis

    def restore_skill(self, analysis: dict[str, Any], skill_code: str) -> dict[str, Any]:
        for competency in analysis.get("competencies", []):
            if competency.code == skill_code:
                competency.review_status = "pending"
                break
        return analysis

    def add_skill(self, analysis: dict[str, Any], label: str) -> dict[str, Any]:
        label = clean_text(label)
        if not label:
            return analysis
        competencies = analysis.get("competencies", [])
        anchor = competencies[0] if competencies else None
        if any(
            normalize_for_match(getattr(c, "official_label", "") or "") == normalize_for_match(label)
            for c in competencies
        ):
            return analysis
        existing_codes = {c.code for c in competencies}
        index = 1
        while f"MANUAL_{index}" in existing_codes:
            index += 1
        code = f"MANUAL_{index}"
        competencies.append(
            OfficialCompetency(
                code=code,
                official_label=label,
                normalized_label=label,
                block_code=getattr(anchor, "block_code", "MANUAL") if anchor else "MANUAL",
                activity_code=getattr(anchor, "activity_code", "MANUAL") if anchor else "MANUAL",
                page_start=getattr(anchor, "page_start", 1) if anchor else 1,
                page_end=getattr(anchor, "page_end", 1) if anchor else 1,
                confidence=1.0,
                review_status="approved",
                source_pages=list(getattr(anchor, "source_pages", []) or [1]) if anchor else [1],
                source_text=label,
                provenance="human_review",
            )
        )
        return analysis

    def apply_criterion_edits(self, analysis: dict[str, Any], form: Any) -> dict[str, Any]:
        for criterion in list(analysis.get("criteria", [])):
            label_key = f"criterion_label__{criterion.code}"
            status_key = f"criterion_status__{criterion.code}"
            if label_key in form:
                corrected = clean_text(form.get(label_key))
                if corrected:
                    criterion.criterion_label = corrected
                    criterion.normalized_label = corrected
            if status_key in form:
                status = clean_text(form.get(status_key))
                if status:
                    criterion.review_status = status
        return analysis

    def reject_derived_skill(self, analysis: dict[str, Any], skill_index: int) -> dict[str, Any]:
        derived = analysis.get("derived_skills", [])
        if 0 <= skill_index < len(derived):
            derived[skill_index].review_status = "rejected"
        return analysis

    def restore_derived_skill(self, analysis: dict[str, Any], skill_index: int) -> dict[str, Any]:
        derived = analysis.get("derived_skills", [])
        if 0 <= skill_index < len(derived):
            derived[skill_index].review_status = "pending"
        return analysis

    def add_derived_skill(self, analysis: dict[str, Any], label: str, category: str = "skill", canonical: str = "") -> dict[str, Any]:
        label = clean_text(label)
        if not label:
            return analysis
        canonical = clean_text(canonical) or label
        if category not in {"skill", "method", "tool", "domain", "soft_skill", "knowledge", "regulatory", "action"}:
            category = "skill"
        derived = analysis.get("derived_skills", [])
        if any(
            normalize_for_match(getattr(s, "canonical_label", "") or getattr(s, "label", "") or "")
            == normalize_for_match(canonical)
            for s in derived
        ):
            return analysis
        skill = DerivedSkill(
            label=label,
            canonical_label=canonical,
            category=category,
            source_code=f"MANUAL_SKILL_{len(derived) + 1}",
            source_type="human_review",
            surface_form=label,
            normalized_surface=normalize_for_match(label),
            provenance="human_review",
            confidence=1.0,
            explicit=True,
            page_start=1,
            page_end=1,
            context=label,
        )
        derived.append(skill)
        return analysis

    def get_active_competencies(self, analysis: dict[str, Any]) -> list[OfficialCompetency]:
        return [
            c for c in analysis.get("competencies", [])
            if c.review_status not in REJECTED_STATUSES
        ]

    def get_inactive_competencies(self, analysis: dict[str, Any]) -> list[OfficialCompetency]:
        return [
            c for c in analysis.get("competencies", [])
            if c.review_status in REJECTED_STATUSES
        ]

    def is_editable_state(self, state: dict[str, Any]) -> bool:
        return state.get("analysis_status") in {
            "PDF_ANALYZED", "COMPETENCIES_EDITED",
        }

    @staticmethod
    def _parse_new_competencies(form: Any) -> list[str]:
        labels = []
        seen: set[str] = set()
        for raw_line in clean_text(form.get("new_competency_labels") or "").splitlines():
            label = clean_text(raw_line)
            if not label:
                continue
            key = normalize_for_match(label)
            if key in seen:
                continue
            seen.add(key)
            labels.append(label)
        return labels

    @staticmethod
    def _parse_new_derived_skills(form: Any) -> list[DerivedSkill]:
        skills: list[DerivedSkill] = []
        for raw_line in clean_text(form.get("new_derived_skill_labels") or "").splitlines():
            label, category, canonical = _parse_manual_detected_skill_line(raw_line)
            if not label:
                continue
            skills.append(
                DerivedSkill(
                    label=label,
                    canonical_label=canonical or label,
                    category=category,
                    source_code=f"MANUAL_SKILL_{len(skills) + 1}",
                    source_type="human_review",
                    surface_form=label,
                    normalized_surface=normalize_for_match(label),
                    provenance="human_review",
                    confidence=1.0,
                    explicit=True,
                    page_start=1,
                    page_end=1,
                    context=label,
                )
            )
        return skills


def _parse_manual_detected_skill_line(raw: str) -> tuple[str, str, str]:
    parts = [p.strip() for p in raw.split("|")]
    label = clean_text(parts[0]) if len(parts) > 0 else ""
    category = clean_text(parts[1]) if len(parts) > 1 else "skill"
    canonical = clean_text(parts[2]) if len(parts) > 2 else label
    if category not in {"skill", "method", "tool", "domain", "soft_skill", "knowledge", "regulatory", "action"}:
        category = "skill"
    return label, category, canonical
