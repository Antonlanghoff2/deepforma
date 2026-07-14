from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from statistics import median
from typing import Any

from common.text import clean_text, normalize_for_match

from .models import Serializable
from .pdf_loader import PdfDocument, PdfPage, PdfTextBlock


LOGGER = logging.getLogger(__name__)

GENERIC_DOCUMENT_TITLES = {
    "référentiel de certification",
    "référentiel de compétences",
    "référentiel d'activités",
    "référentiel d'évaluation",
    "fiche rncp",
    "certification professionnelle",
    "sommaire",
    "annexe",
}

LABEL_GROUPS: dict[str, tuple[str, ...]] = {
    "certification_title": (
        "intitule de la certification",
        "intitule du titre",
        "intitule du metier",
        "intitule de l emploi",
        "nom de la certification",
        "titre de la certification",
        "certification professionnelle",
        "titre professionnel",
    ),
    "target_job_title": (
        "intitule du métier",
        "intitule du metier",
        "intitule de l'emploi",
        "intitule de l emploi",
        "emploi visé",
        "emploi vise",
        "métier visé",
        "metier vise",
    ),
}

RNCP_RE = re.compile(r"\brncp\s*\d+\b", flags=re.IGNORECASE)
URL_RE = re.compile(r"https?://|www\.", flags=re.IGNORECASE)
DATE_RE = re.compile(
    r"\b(?:19|20)\d{2}\b|\b(?:janvier|février|fevrier|mars|avril|mai|juin|juillet|août|aout|septembre|octobre|novembre|décembre|decembre)\b",
    flags=re.IGNORECASE,
)
PAGE_RE = re.compile(r"^\s*(?:page\s*)?\d+\s*$", flags=re.IGNORECASE)
SECTION_RE = re.compile(r"^(?:bloc|activit[ée]|activite|a\d+(?:\s+\d+)?|c\d+(?:\s+\d+)?|ce\d+(?:\s+\d+)?)", flags=re.IGNORECASE)


@dataclass(slots=True)
class TitleCandidate(Serializable):
    text: str
    score: float
    page: int | None
    source_text: str | None
    reasons: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ExtractedReferentialTitle(Serializable):
    title: str | None
    title_type: str
    confidence: float
    source_page: int | None
    source_text: str | None
    candidates: list[TitleCandidate] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    document_title: str = ""
    certification_title: str = ""
    target_job_title: str = ""
    rncp_code: str = ""


@dataclass(slots=True)
class _Line:
    text: str
    page: int
    index: int
    block_index: int
    bbox: tuple[float, float, float, float] | None
    font_size: float | None
    bold: bool | None
    page_width: float | None
    page_height: float | None
    source_text: str


def normalize_title_text(text: str) -> str:
    cleaned = clean_text(text)
    if not cleaned:
        return ""
    parts: list[str] = []
    for raw_line in cleaned.replace("•", " ").replace("·", " ").splitlines():
        line = clean_text(raw_line)
        if not line:
            continue
        line = re.sub(r"^[\-–—*•·]+\s*", "", line)
        line = re.sub(r"^[:;]+\s*", "", line)
        line = re.sub(r"\s+[:;]+\s*$", "", line)
        if not line:
            continue
        if PAGE_RE.match(line):
            continue
        if normalize_for_match(line) in {"page", "p"}:
            continue
        parts.append(line)
    candidate = clean_text(" ".join(parts))
    candidate = re.sub(r"\s+", " ", candidate).strip(" -:–—\t")
    if len(candidate) > 250:
        candidate = candidate[:250].rstrip()
    return candidate


def _normalize_filename_title(file_name: str) -> str:
    stem = clean_text(file_name)
    if not stem:
        return ""
    stem = re.sub(r"\.[A-Za-z0-9]{1,5}$", "", stem)
    stem = stem.replace("_", " ").replace("-", " ")
    stem = re.sub(r"^tmp[0-9a-z]{4,}$", "", stem, flags=re.IGNORECASE).strip()
    return normalize_title_text(stem)


def _flatten_lines(document: PdfDocument) -> list[_Line]:
    lines: list[_Line] = []
    for page in document.pages:
        if page.blocks:
            for block_index, block in enumerate(page.blocks):
                block_lines = [clean_text(line) for line in clean_text(block.text).splitlines() if clean_text(line)]
                if not block_lines:
                    continue
                for line_index, line in enumerate(block_lines):
                    lines.append(
                        _Line(
                            text=line,
                            page=page.number,
                            index=line_index,
                            block_index=block_index,
                            bbox=block.bbox,
                            font_size=block.font_size,
                            bold=block.bold,
                            page_width=page.width,
                            page_height=page.height,
                            source_text=clean_text(block.text),
                        )
                    )
        elif page.text:
            for line_index, raw_line in enumerate(page.text.splitlines()):
                line = clean_text(raw_line)
                if not line:
                    continue
                lines.append(
                    _Line(
                        text=line,
                        page=page.number,
                        index=line_index,
                        block_index=line_index,
                        bbox=None,
                        font_size=None,
                        bold=None,
                        page_width=page.width,
                        page_height=page.height,
                        source_text=line,
                    )
                )
    return lines


def _is_generic(text: str) -> bool:
    normalized = normalize_for_match(text)
    if not normalized:
        return True
    if normalized in {normalize_for_match(item) for item in GENERIC_DOCUMENT_TITLES}:
        return True
    if normalized.startswith(("referentiel de certification", "referentiel de competences", "fiche rncp", "sommaire", "annexe")):
        return True
    return False


def _is_section_like(text: str) -> bool:
    normalized = normalize_for_match(text)
    if not normalized:
        return True
    if SECTION_RE.match(normalized):
        return True
    if URL_RE.search(text):
        return True
    if PAGE_RE.match(text):
        return True
    if len(normalized.split()) > 24:
        return True
    if re.search(r"\b\d+\s*(?:heures?|jours?|mois?)\b", normalized):
        return True
    metadata_markers = (
        "duree",
        "prix",
        "reference",
        "niveau",
        "version",
        "date",
        "tarif",
        "ttc",
        "euros",
        "euro",
        "organisme",
        "certificateur",
    )
    if any(marker in normalized for marker in metadata_markers):
        return True
    return False


def _looks_like_title(text: str) -> bool:
    normalized = normalize_for_match(text)
    words = normalized.split()
    return 2 <= len(words) <= 18 and not _is_section_like(text)


def _page_font_median(lines: list[_Line], page: int) -> float | None:
    values = [line.font_size for line in lines if line.page == page and line.font_size is not None]
    if not values:
        return None
    return float(median(values))


def _center_bonus(line: _Line) -> tuple[float, bool]:
    if not line.bbox or not line.page_width:
        return 0.0, False
    x0, _, x1, _ = line.bbox
    ratio = ((x0 + x1) / 2.0) / line.page_width if line.page_width else 0.5
    if 0.35 <= ratio <= 0.65:
        return 0.04, True
    return 0.0, False


def _collect_following_title(lines: list[_Line], start_index: int) -> tuple[str, str | None, int | None]:
    fragments: list[str] = []
    source_lines: list[str] = []
    source_page: int | None = None
    for offset in range(1, 4):
        idx = start_index + offset
        if idx >= len(lines):
            break
        candidate = lines[idx]
        if _is_section_like(candidate.text) or _is_generic(candidate.text):
            break
        fragment = normalize_title_text(candidate.text)
        if not fragment:
            continue
        fragments.append(fragment)
        source_lines.append(candidate.text)
        source_page = candidate.page
        if len(normalize_for_match(" ".join(fragments)).split()) >= 18:
            break
    if not fragments:
        return "", None, None
    return normalize_title_text(" ".join(fragments)), "\n".join(source_lines), source_page


def _extract_after_label(text: str, label: str) -> str:
    normalized = normalize_for_match(text)
    label_norm = normalize_for_match(label)
    if not normalized.startswith(label_norm):
        return ""
    if ":" in text:
        prefix, suffix = text.split(":", 1)
        if normalize_for_match(prefix).startswith(label_norm):
            return normalize_title_text(suffix)
    if " - " in text:
        prefix, suffix = text.split(" - ", 1)
        if normalize_for_match(prefix).startswith(label_norm):
            return normalize_title_text(suffix)
    return ""


def _label_type(line: _Line) -> tuple[str | None, str | None]:
    normalized = normalize_for_match(line.text)
    for label_type, labels in LABEL_GROUPS.items():
        for label in labels:
            if normalized.startswith(normalize_for_match(label)):
                return label_type, label
    return None, None


GENERIC_MARKER_RE = re.compile(
    r"(?:référentiel d['’]?activit(?:é|e)s?|referentiel d['’]?activites?|référentiel de compétences?|referentiel de competences?|référentiel de certification|referentiel de certification|modalités d['’]?évaluation|modalites d['’]?evaluation|critères d['’]?évaluation|criteres d['’]?evaluation)",
    flags=re.IGNORECASE,
)


def _split_generic_marker(text: str) -> str | None:
    match = GENERIC_MARKER_RE.search(text)
    if not match:
        return None
    prefix = normalize_title_text(text[: match.start()])
    suffix = normalize_title_text(text[match.end() :])
    return prefix or suffix or None


def _score_candidate(
    line: _Line,
    text: str,
    *,
    source_kind: str,
    explicit_label_type: str | None,
    explicit_label: str | None,
    rncp_code: str | None,
    page_median: float | None,
    page_line_count: int,
) -> tuple[float, list[str]]:
    if source_kind == "filename":
        return 0.35, ["filename_fallback"]

    score = 0.18
    reasons: list[str] = []
    if source_kind == "metadata":
        score += 0.08
        reasons.append("metadata")
    if source_kind == "visual":
        score += 0.12
        reasons.append("visual")
    if explicit_label_type:
        score += 0.56
        reasons.append(f"explicit_label={explicit_label_type}")
        if explicit_label:
            reasons.append(f"label={explicit_label}")
    if line.page == 1:
        score += 0.08
        reasons.append("first_page")
    if line.index <= max(2, page_line_count // 3):
        score += 0.05
        reasons.append("first_third")
    if line.font_size is not None and page_median is not None and line.font_size > page_median:
        score += min(0.10, 0.03 * max(line.font_size - page_median, 1.0))
        reasons.append(f"font_size>{page_median:.1f}")
    if line.bold:
        score += 0.05
        reasons.append("bold")
    center_bonus, is_centered = _center_bonus(line)
    if is_centered:
        score += center_bonus
        reasons.append("centered")
    if rncp_code:
        score += 0.12
        reasons.append(f"rncp={rncp_code}")
    words = len(normalize_for_match(text).split())
    if 2 <= words <= 18:
        score += 0.05
        reasons.append("word_count_ok")
    if _looks_like_title(text):
        score += 0.05
        reasons.append("title_like")
    if _is_generic(text):
        score -= 0.8
        reasons.append("generic_penalty")
    if _is_section_like(text):
        score -= 0.45
        reasons.append("section_penalty")
    if URL_RE.search(text):
        score -= 0.4
        reasons.append("url_penalty")
    if DATE_RE.search(text):
        score -= 0.25
        reasons.append("date_penalty")
    if words > 18:
        score -= 0.35
        reasons.append("long_sentence_penalty")
    if text.endswith('.') and words > 6:
        score -= 0.30
        reasons.append("sentence_penalty")
    return max(0.0, min(1.0, round(score, 4))), reasons


def _make_candidate(text: str, *, page: int | None, source_text: str | None, score: float, reasons: list[str]) -> TitleCandidate:
    return TitleCandidate(
        text=normalize_title_text(text),
        score=round(score, 4),
        page=page,
        source_text=normalize_title_text(source_text or "") or source_text,
        reasons=reasons,
    )


def extract_referential_title(document: PdfDocument, *, file_name: str | None = None, document_title: str | None = None) -> ExtractedReferentialTitle:
    lines = _flatten_lines(document)
    filename_title = _normalize_filename_title(file_name or document.path.name)
    filename_candidate = None
    if filename_title:
        filename_candidate = _make_candidate(
            filename_title,
            page=None,
            source_text=file_name or document.path.name,
            score=0.35 if not _is_generic(filename_title) else 0.25,
            reasons=["filename_fallback"] if not _is_generic(filename_title) else ["filename_generic_fallback"],
        )

    if not lines:
        warnings = ["Aucun libellé explicite trouvé."]
        if filename_candidate is not None:
            warnings.append("Résultat issu du nom du fichier.")
            return ExtractedReferentialTitle(
                title=filename_candidate.text,
                title_type="filename",
                confidence=filename_candidate.score,
                source_page=filename_candidate.page,
                source_text=filename_candidate.source_text,
                candidates=[filename_candidate],
                warnings=warnings,
                document_title=filename_candidate.text,
            )
        warnings.append("Aucun intitulé exploitable n'a été trouvé.")
        return ExtractedReferentialTitle(
            title=None,
            title_type="unknown",
            confidence=0.0,
            source_page=None,
            source_text=None,
            candidates=[],
            warnings=warnings,
            document_title="",
        )

    page_counts: dict[int, int] = {}
    page_rncp: dict[int, str] = {}
    for line in lines:
        page_counts[line.page] = page_counts.get(line.page, 0) + 1
        if line.page not in page_rncp:
            match = RNCP_RE.search(line.text)
            if match:
                page_rncp[line.page] = match.group(0).upper().replace(" ", "")

    candidates: list[TitleCandidate] = []
    certification_title = ""
    target_job_title = ""
    best_non_generic = ""
    best_non_generic_score = -1.0
    best_non_generic_page: int | None = None
    best_non_generic_source: str | None = None

    for index, line in enumerate(lines):
        label_type, label = _label_type(line)
        if label_type:
            extracted = _extract_after_label(line.text, label or "")
            source_text = line.text
            source_page = line.page
            if not extracted:
                extracted, source_text, source_page = _collect_following_title(lines, index)
            if extracted:
                candidate_text = normalize_title_text(extracted)
                if candidate_text:
                    score, reasons = _score_candidate(
                        line,
                        candidate_text,
                        source_kind="visual" if line.font_size is not None or line.bold else "line",
                        explicit_label_type=label_type,
                        explicit_label=label,
                        rncp_code=page_rncp.get(line.page),
                        page_median=_page_font_median(lines, line.page),
                        page_line_count=page_counts.get(line.page, 0),
                    )
                    candidate = TitleCandidate(
                        text=candidate_text,
                        score=score,
                        page=source_page,
                        source_text=normalize_title_text(source_text or line.source_text) or source_text or line.source_text,
                        reasons=["following_lines" if not _extract_after_label(line.text, label or "") else "same_line"] + reasons,
                    )
                    LOGGER.info('[referential-import] title candidate="%s" score=%.2f page=%s reasons=%s', candidate.text[:120], candidate.score, candidate.page, candidate.reasons)
                    candidates.append(candidate)
                    if label_type == "certification_title" and not certification_title:
                        certification_title = candidate.text
                    if label_type == "target_job_title" and not target_job_title:
                        target_job_title = candidate.text
                    if not _is_generic(candidate.text) and candidate.score >= 0.25 and not best_non_generic:
                        best_non_generic = candidate.text
                        best_non_generic_page = candidate.page
                        best_non_generic_source = candidate.source_text
            continue

        marker_candidate = _split_generic_marker(line.text)
        if marker_candidate and _looks_like_title(marker_candidate) and not _is_generic(marker_candidate):
            source_kind = "visual" if line.page == 1 and line.index <= 4 else "line"
            score, reasons = _score_candidate(
                line,
                marker_candidate,
                source_kind=source_kind,
                explicit_label_type=None,
                explicit_label=None,
                rncp_code=page_rncp.get(line.page),
                page_median=_page_font_median(lines, line.page),
                page_line_count=page_counts.get(line.page, 0),
            )
            if score >= 0.25:
                candidate = TitleCandidate(
                    text=marker_candidate,
                    score=score,
                    page=line.page,
                    source_text=normalize_title_text(line.source_text) or line.source_text,
                    reasons=["generic_marker_prefix"] + reasons,
                )
                LOGGER.info('[referential-import] title candidate="%s" score=%.2f page=%s reasons=%s', candidate.text[:120], candidate.score, candidate.page, candidate.reasons)
                candidates.append(candidate)
                if not _is_generic(candidate.text) and score >= 0.25 and not best_non_generic:
                    best_non_generic = candidate.text
                    best_non_generic_page = candidate.page
                    best_non_generic_source = candidate.source_text
            continue

        if _is_generic(line.text) or _is_section_like(line.text):
            continue

        source_kind = "visual" if line.page == 1 and line.index <= 4 else "line"
        if source_kind == "line" and line.page > 1 and line.index > 2 and line.font_size is None and not line.bold:
            continue

        candidate_text = normalize_title_text(line.text)
        if not candidate_text:
            continue
        score, reasons = _score_candidate(
            line,
            candidate_text,
            source_kind=source_kind,
            explicit_label_type=None,
            explicit_label=None,
            rncp_code=page_rncp.get(line.page),
            page_median=_page_font_median(lines, line.page),
            page_line_count=page_counts.get(line.page, 0),
        )
        if score < 0.25:
            continue
        candidate = TitleCandidate(
            text=candidate_text,
            score=score,
            page=line.page,
            source_text=normalize_title_text(line.source_text) or line.source_text,
            reasons=reasons,
        )
        LOGGER.info('[referential-import] title candidate="%s" score=%.2f page=%s reasons=%s', candidate.text[:120], candidate.score, candidate.page, candidate.reasons)
        candidates.append(candidate)
        if not _is_generic(candidate.text) and score >= 0.25 and score > best_non_generic_score:
            best_non_generic = candidate.text
            best_non_generic_score = score
            best_non_generic_page = candidate.page
            best_non_generic_source = candidate.source_text

    if document_title:
        normalized_document_title = normalize_title_text(document_title)
        if normalized_document_title and not _is_generic(normalized_document_title):
            candidate = _make_candidate(
                normalized_document_title,
                page=1,
                source_text=document_title,
                score=0.45,
                reasons=["metadata"],
            )
            candidates.append(candidate)
            if candidate.score > best_non_generic_score:
                best_non_generic = candidate.text
                best_non_generic_score = candidate.score
                best_non_generic_page = candidate.page
                best_non_generic_source = candidate.source_text

    selected_candidate: TitleCandidate | None = None
    title_type = "unknown"
    warnings: list[str] = []

    sorted_candidates = sorted(candidates, key=lambda item: (-item.score, item.page or 9999, len(item.text)))
    if sorted_candidates:
        selected_candidate = sorted_candidates[0]
        if len(sorted_candidates) > 1 and (selected_candidate.score - sorted_candidates[1].score) < 0.08:
            warnings.append("Plusieurs candidats ont des scores proches.")
        if _is_generic(selected_candidate.text):
            warnings.append("L'intitulé retenu semble générique.")
            selected_candidate = filename_candidate or selected_candidate
            title_type = "filename" if filename_candidate is not None else "document_title"
        elif selected_candidate.text == certification_title:
            title_type = "certification_title"
        elif selected_candidate.text == target_job_title:
            title_type = "target_job_title"
        else:
            title_type = "document_title"
    elif filename_candidate is not None:
        selected_candidate = filename_candidate
        title_type = "filename"

    if selected_candidate is None:
        warnings.append("Aucun intitulé exploitable n'a été trouvé.")
        return ExtractedReferentialTitle(
            title=None,
            title_type="unknown",
            confidence=0.0,
            source_page=None,
            source_text=None,
            candidates=[],
            warnings=warnings,
            document_title=best_non_generic,
            certification_title=certification_title,
            target_job_title=target_job_title,
            rncp_code=next(iter(page_rncp.values()), ""),
        )

    if not any("explicit_label=" in reason for candidate in candidates for reason in candidate.reasons):
        warnings.append("Aucun libellé explicite trouvé.")
    if selected_candidate == filename_candidate:
        warnings.append("Résultat issu du nom du fichier.")
    if selected_candidate.score < 0.70:
        warnings.append("Confiance inférieure à 0.70.")
    if _is_generic(selected_candidate.text):
        warnings.append("Le résultat semble générique.")

    title_candidates = sorted(candidates, key=lambda item: (-item.score, item.page or 9999, len(item.text)))
    if filename_candidate is not None and all(normalize_for_match(candidate.text) != normalize_for_match(filename_candidate.text) for candidate in title_candidates):
        title_candidates.append(filename_candidate)
        title_candidates = sorted(title_candidates, key=lambda item: (-item.score, item.page or 9999, len(item.text)))

    LOGGER.info('[referential-import] selected title="%s" type=%s confidence=%.2f page=%s', selected_candidate.text[:120], title_type, selected_candidate.score, selected_candidate.page)
    return ExtractedReferentialTitle(
        title=selected_candidate.text,
        title_type=title_type,
        confidence=selected_candidate.score,
        source_page=selected_candidate.page,
        source_text=selected_candidate.source_text,
        candidates=title_candidates[:5],
        warnings=warnings,
        document_title=best_non_generic or selected_candidate.text,
        certification_title=certification_title,
        target_job_title=target_job_title,
        rncp_code=next(iter(page_rncp.values()), ""),
    )
