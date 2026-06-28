from __future__ import annotations

import hashlib
import html
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from common.text import clean_text, normalize_for_match, stable_hash
from deepforma.cpf.columns import ColumnDetection


CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")


def strip_html(value: Any) -> str:
    """Supprime le HTML en conservant le texte lisible."""

    text = clean_text(value)
    if not text:
        return ""
    text = html.unescape(str(value))
    text = re.sub(r"<\s*br\s*/?\s*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</p\s*>", "\n", text, flags=re.IGNORECASE)
    text = TAG_RE.sub(" ", text)
    return normalize_spaces(text)


def normalize_spaces(value: Any) -> str:
    """Normalise les espaces et caractères de contrôle."""

    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = CONTROL_CHARS_RE.sub(" ", text)
    text = text.replace("\u200b", " ").replace("\ufeff", " ")
    text = WHITESPACE_RE.sub(" ", text).strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def clean_scalar(value: Any) -> str:
    """Nettoie une cellule texte en conservant les accents."""

    if value is None:
        return ""
    text = html.unescape(str(value))
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"<\s*br\s*/?\s*>", "\n", text, flags=re.IGNORECASE)
    text = TAG_RE.sub(" ", text)
    text = CONTROL_CHARS_RE.sub(" ", text)
    text = text.replace("\u200b", " ").replace("\ufeff", " ")
    text = WHITESPACE_RE.sub(" ", text).strip()
    if text.lower() in {"", "nan", "none", "null"}:
        return ""
    return text


def normalize_siret(value: Any) -> str | None:
    """Normalise un SIRET à 14 chiffres."""

    text = clean_scalar(value)
    digits = re.sub(r"\D", "", text)
    if len(digits) != 14:
        return None
    return digits


def normalize_department_code(value: Any) -> str | None:
    """Normalise un code départemental."""

    text = clean_scalar(value).upper().replace(" ", "")
    if not text:
        return None
    if text in {"2A", "2B"}:
        return text
    digits = re.sub(r"\D", "", text)
    if not digits:
        return None
    if len(digits) == 1:
        digits = digits.zfill(2)
    if len(digits) > 3:
        digits = digits[:3]
    return digits.zfill(2)


def normalize_region_code(value: Any) -> str | None:
    """Normalise un code région."""

    text = clean_scalar(value).upper().replace(" ", "")
    if not text:
        return None
    if text in {"2A", "2B"}:
        return text
    digits = re.sub(r"\D", "", text)
    if not digits:
        return None
    if len(digits) <= 2:
        return digits.zfill(2)
    return digits[:3]


def normalize_referential_type(value: Any) -> str | None:
    """Réduit le type de référentiel à RNCP, RS ou nul."""

    text = normalize_for_match(value)
    if not text:
        return None
    if "rncp" in text:
        return "RNCP"
    if text == "rs" or " rs " in f" {text} " or "registre specifique" in text or "repertoire specifique" in text:
        return "RS"
    return None


def build_search_text(row: dict[str, Any]) -> str:
    """Construit le texte canonique de recherche."""

    fields = [
        row.get("title"),
        row.get("certification"),
        row.get("description"),
        row.get("objectives"),
        row.get("nsf"),
        row.get("exit_level"),
        row.get("referential_type"),
    ]
    parts: list[str] = []
    for value in fields:
        text = clean_scalar(value)
        if text:
            parts.append(text)
    return " \n ".join(parts)


def build_formation_uid(row: dict[str, Any]) -> str:
    """Construit un identifiant stable pour une formation."""

    return stable_hash(
        row.get("source_id"),
        row.get("certification"),
        row.get("organization_siret"),
        row.get("title"),
        row.get("department_code"),
        length=24,
    )


def row_hash(row: dict[str, Any]) -> str:
    """Hash de ligne utile pour l'update incrémental."""

    payload = "|".join(
        clean_scalar(row.get(key))
        for key in ["title", "certification", "organization", "organization_siret", "department_code", "region_code", "search_text"]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalize_row(
    raw_row: dict[str, Any],
    detection: ColumnDetection,
) -> dict[str, Any]:
    """Normalise une ligne brute du catalogue CPF."""

    field = lambda key: raw_row.get(detection.resolved.get(key) or "")  # noqa: E731
    source_id = clean_scalar(field("id"))
    title = clean_scalar(field("title"))
    certification = clean_scalar(field("certification"))
    description = strip_html(field("description"))
    objectives = clean_scalar(field("objectives"))
    exit_level = clean_scalar(field("exit_level"))
    nsf = clean_scalar(field("nsf"))
    organization = clean_scalar(field("organization"))
    organization_siret = normalize_siret(field("organization_siret"))
    region = clean_scalar(field("region"))
    department = clean_scalar(field("department"))
    region_code = normalize_region_code(field("region_code"))
    department_code = normalize_department_code(field("department_code"))
    referential_type = normalize_referential_type(field("referential_type") or certification)
    search_row = {
        "title": title,
        "certification": certification,
        "description": description,
        "objectives": objectives,
        "nsf": nsf,
        "exit_level": exit_level,
        "referential_type": referential_type,
    }
    search_text = build_search_text(search_row)
    normalized_title = normalize_for_match(title)
    normalized_certification = normalize_for_match(certification)
    normalized_organization = normalize_for_match(organization)
    normalized_description = normalize_for_match(description)
    formation_uid = build_formation_uid(
        {
            "source_id": source_id,
            "certification": certification,
            "organization_siret": organization_siret,
            "title": normalized_title,
            "department_code": department_code,
        }
    )
    return {
        "source_id": source_id or None,
        "title": title or None,
        "certification": certification or None,
        "description": description or None,
        "objectives": objectives or None,
        "exit_level": exit_level or None,
        "nsf": nsf or None,
        "organization": organization or None,
        "organization_siret": organization_siret,
        "region": region or None,
        "department": department or None,
        "region_code": region_code,
        "department_code": department_code,
        "referential_type": referential_type,
        "search_text": search_text or None,
        "normalized_title": normalized_title or None,
        "normalized_certification": normalized_certification or None,
        "normalized_organization": normalized_organization or None,
        "normalized_description": normalized_description or None,
        "formation_uid": formation_uid,
    }


@dataclass
class DeduplicationStats:
    """Statistiques de déduplication."""

    seen: int = 0
    kept: int = 0
    exact_duplicates: int = 0
    near_duplicates: int = 0


class CPFDeduper:
    """Déduplication exacte puis quasi-identique par territoire."""

    def __init__(self, similarity_threshold: float = 0.96) -> None:
        self.similarity_threshold = similarity_threshold
        self.seen_uids: set[str] = set()
        self.buckets: dict[tuple[str | None, str | None, str | None], list[dict[str, Any]]] = {}
        self.stats = DeduplicationStats()

    @staticmethod
    def _similarity(left: str, right: str) -> float:
        try:
            from rapidfuzz.fuzz import ratio

            return float(ratio(left, right)) / 100.0
        except Exception:
            import difflib

            return difflib.SequenceMatcher(None, left, right).ratio()

    def is_duplicate(self, row: dict[str, Any]) -> bool:
        """Indique si une ligne doit être écartée."""

        self.stats.seen += 1
        uid = row.get("formation_uid")
        if uid in self.seen_uids:
            self.stats.exact_duplicates += 1
            return True

        bucket_key = (
            row.get("normalized_certification"),
            row.get("normalized_organization"),
            row.get("department_code") or row.get("region_code"),
        )
        title = row.get("normalized_title") or ""
        for candidate in self.buckets.get(bucket_key, []):
            if candidate.get("department_code") and row.get("department_code") and candidate["department_code"] != row.get("department_code"):
                continue
            if candidate.get("region_code") and row.get("region_code") and candidate["region_code"] != row.get("region_code"):
                continue
            if self._similarity(candidate.get("normalized_title") or "", title) >= self.similarity_threshold:
                self.stats.near_duplicates += 1
                return True
        return False

    def register(self, row: dict[str, Any]) -> None:
        """Enregistre une ligne conservée dans les structures de déduplication."""

        uid = row.get("formation_uid")
        if uid:
            self.seen_uids.add(str(uid))
        bucket_key = (
            row.get("normalized_certification"),
            row.get("normalized_organization"),
            row.get("department_code") or row.get("region_code"),
        )
        self.buckets.setdefault(bucket_key, []).append(row)
        self.stats.kept += 1

