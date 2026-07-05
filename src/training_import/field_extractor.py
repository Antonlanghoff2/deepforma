from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from common.text import clean_text, normalize_for_match

from .models import FieldEvidence, TrainingCertification, TrainingProgram, TrainingProvider
from .pdf_document_loader import PdfDocument
from .section_detector import DetectedSection


TITLE_HINTS = (
    "intitule",
    "titre",
    "programme",
    "formation",
)
REFERENCE_RE = re.compile(r"\b(?:RNCP|RS|REF|REF\.|réf\.?|reference|référence)\s*[:#-]?\s*([A-Z0-9_.-]{3,})", re.IGNORECASE)
DURATION_RE = re.compile(r"(?P<value>\d+(?:[.,]\d+)?)\s*(?P<unit>heures?|h|jours?|j|semaines?|mois)\b", re.IGNORECASE)
PRICE_RE = re.compile(r"\b(?:(?P<amount>\d[\d\s]*(?:[,.]\d+)?)\s*(?:€|eur|euros?)|gratuite?|gratuit|sur devis|pris en charge|à partir de)\b", re.IGNORECASE)
LEVEL_RE = re.compile(r"\b(?:niveau\s*[0-9ivx]+|bac\+?\d|cap|bep|licence|master)\b", re.IGNORECASE)
FORMAT_RE = re.compile(r"\b(?:présentiel|presentiel|distanciel|à distance|a distance|hybride|alternance|full[- ]?time|part[- ]?time)\b", re.IGNORECASE)
CPF_RE = re.compile(r"\bcpf\b", re.IGNORECASE)


def _first_match(pattern: re.Pattern[str], text: str) -> re.Match[str] | None:
    return pattern.search(text)


def _looks_like_title(line: str) -> bool:
    normalized = normalize_for_match(line)
    if not normalized:
        return False
    if any(marker in normalized for marker in ("referentiel", "activites", "competences", "modalites", "evaluation")):
        return False
    if any(marker in normalized for marker in ("programme", "contenu", "objectifs", "prerequis", "prérequis")):
        return False
    words = normalized.split()
    if len(words) < 2 or len(words) > 14:
        return False
    return True


def _extract_title_from_line(line: str) -> str:
    cleaned = clean_text(line)
    if not cleaned:
        return ""
    if ":" in cleaned:
        head, tail = cleaned.split(":", 1)
        head_norm = normalize_for_match(head)
        if any(marker in head_norm for marker in TITLE_HINTS):
            return clean_text(tail)
    patterns = [
        re.compile(r"(?:intitul[eé]|titre)\s*(?:du\s*programme|de\s*la\s*formation|détecté|detecte)?\s*[:\-]?\s*(.+)$", re.IGNORECASE),
        re.compile(r"(?:formation|programme)\s*[:\-]\s*(.+)$", re.IGNORECASE),
    ]
    for pattern in patterns:
        match = pattern.search(cleaned)
        if match:
            value = clean_text(match.group(1))
            if value:
                return value
    if _looks_like_title(cleaned):
        return cleaned
    return ""


def _resolve_title(document: PdfDocument, sections: list[DetectedSection]) -> tuple[str, list[FieldEvidence]]:
    evidence: list[FieldEvidence] = []
    for page in document.pages[:2]:
        for line in page.text.splitlines():
            title = _extract_title_from_line(line)
            if title:
                evidence.append(FieldEvidence("title", "header", title, page.number, confidence=0.88, method="explicit"))
                return title, evidence
    for page in document.pages[:2]:
        for line in page.text.splitlines():
            cleaned = clean_text(line)
            if _looks_like_title(cleaned):
                evidence.append(FieldEvidence("title", "header", cleaned, page.number, confidence=0.55, method="filename_or_header"))
                return cleaned, evidence
    fallback = document.path.stem.replace("_", " ").replace("-", " ").strip()
    evidence.append(FieldEvidence("title", "filename", fallback, 1, confidence=0.2, method="filename"))
    return fallback, evidence


def _find_in_text(pattern: re.Pattern[str], texts: list[tuple[int, str]], field_name: str) -> tuple[str, list[FieldEvidence]]:
    for page, text in texts:
        match = _first_match(pattern, text)
        if match:
            value = clean_text(match.group(0))
            return value, [FieldEvidence(field_name, "header", value, page, confidence=0.85, method="explicit")]
    return "", []


def extract_fields(document: PdfDocument, sections: list[DetectedSection]) -> tuple[TrainingProvider, TrainingProgram]:
    texts = [(page.number, page.text) for page in document.pages]
    title, title_evidence = _resolve_title(document, sections)
    reference, ref_evidence = _find_in_text(REFERENCE_RE, texts, "reference")
    duration_text, duration_evidence = _find_in_text(DURATION_RE, texts, "duration")
    level, level_evidence = _find_in_text(LEVEL_RE, texts, "level")
    format_text, format_evidence = _find_in_text(FORMAT_RE, texts, "format")
    price_text, price_evidence = _find_in_text(PRICE_RE, texts, "price")
    cpf = "unknown"
    cpf_evidence: list[FieldEvidence] = []
    for page, text in texts:
        if CPF_RE.search(text):
            cpf = "true"
            cpf_evidence.append(FieldEvidence("cpf", "header", "CPF", page, confidence=0.95, method="explicit"))
            break

    provider = TrainingProvider()
    for page, text in texts[:2]:
        lines = [clean_text(line) for line in text.splitlines() if clean_text(line)]
        for idx, line in enumerate(lines[:8]):
            normalized_line = normalize_for_match(line)
            if any(marker in normalized_line for marker in ("organisme", "provider", "éditeur", "editeur")):
                value = line.split(':', 1)[-1].strip() if ':' in line else (lines[idx + 1] if idx + 1 < len(lines) else line)
                provider = TrainingProvider(name=value, canonical_name=value, source_text=line, page=page, confidence=0.8)
                break
        if provider.name:
            break

    certification = TrainingCertification(label=reference, code=reference, cpf=cpf, source_text=reference, page=ref_evidence[0].page if ref_evidence else None, confidence=0.8 if reference else 0.0)

    program = TrainingProgram(
        title=title,
        reference=reference,
        duration_text=duration_text,
        level=level,
        format=format_text,
        price_text=price_text,
        cpf=cpf,
        pages_source=[page.number for page in document.pages],
        confidence=0.5,
        extraction_method=document.extraction_method,
        objectives_text="
".join(section.content for section in sections if section.key == "objectives"),
        prerequisites_text="
".join(section.content for section in sections if section.key == "prerequisites"),
        certification=certification,
        evidence=[*title_evidence, *ref_evidence, *duration_evidence, *level_evidence, *format_evidence, *price_evidence, *cpf_evidence],
    )
    return provider, program
