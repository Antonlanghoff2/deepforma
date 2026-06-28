from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from common.text import clean_text, normalize_for_match


DEFAULT_COLUMNS_CONFIG = Path(__file__).resolve().parents[3] / "config" / "cpf_columns.yaml"


def _normalize_alias(value: Any) -> str:
    text = clean_text(value)
    if not text:
        return ""
    return normalize_for_match(text)


@dataclass(frozen=True)
class ColumnDetection:
    """Résultat de détection de colonnes CPF."""

    resolved: dict[str, str | None]
    candidates: dict[str, list[str]]


def load_column_aliases(config_path: str | Path | None = None) -> dict[str, list[str]]:
    """Charge la configuration d'alias des colonnes CPF."""

    path = Path(config_path or DEFAULT_COLUMNS_CONFIG)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Configuration CPF invalide: {path}")
    aliases: dict[str, list[str]] = {}
    for canonical, values in payload.items():
        if not isinstance(values, list):
            continue
        aliases[str(canonical)] = [str(value) for value in values if clean_text(value)]
    return aliases


def detect_columns(columns: list[str], alias_map: dict[str, list[str]]) -> ColumnDetection:
    """Associe les colonnes réelles aux champs canoniques."""

    normalized_columns = {normalize_for_match(col): col for col in columns}
    lower_columns = {str(col).strip().lower(): col for col in columns}

    resolved: dict[str, str | None] = {}
    candidates: dict[str, list[str]] = {}

    for canonical, aliases in alias_map.items():
        ranked: list[tuple[int, str]] = []
        for alias in aliases:
            alias_norm = _normalize_alias(alias)
            if not alias_norm:
                continue
            for col_norm, original in normalized_columns.items():
                score = 0
                if col_norm == alias_norm:
                    score = 100
                elif alias_norm in col_norm or col_norm in alias_norm:
                    score = 90 - abs(len(col_norm) - len(alias_norm))
                elif col_norm.replace(" ", "") == alias_norm.replace(" ", ""):
                    score = 85
                if score:
                    ranked.append((score, original))
        if not ranked:
            # compatibilité casse brute
            for alias in aliases:
                alias_key = str(alias).strip().lower()
                if alias_key in lower_columns:
                    ranked.append((100, lower_columns[alias_key]))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        resolved[canonical] = ranked[0][1] if ranked else None
        candidates[canonical] = [item[1] for item in ranked[:5]]

    return ColumnDetection(resolved=resolved, candidates=candidates)


def anonymize_value(value: Any) -> str:
    """Anonymise une valeur pour un rapport de schéma."""

    text = clean_text(value)
    if not text:
        return ""
    if len(text) <= 4:
        return "***"
    if "@" in text:
        return "[email]"
    if text.replace(" ", "").isdigit():
        return f"{text[:2]}***{text[-2:]}"
    return f"{text[:3]}***{text[-2:]}"

