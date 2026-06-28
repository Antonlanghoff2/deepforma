from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from common.text import clean_text
from deepforma.cpf.columns import ColumnDetection, anonymize_value, detect_columns, load_column_aliases
from deepforma.cpf.io import detect_text_format, json_dump


DATE_PATTERNS = [
    re.compile(r"^\d{4}-\d{2}-\d{2}$"),
    re.compile(r"^\d{2}/\d{2}/\d{4}$"),
    re.compile(r"^\d{4}/\d{2}/\d{2}$"),
]


@dataclass(frozen=True)
class ColumnStats:
    """Statistiques simples pour une colonne."""

    name: str
    missing_rate: float
    estimated_type: str
    examples: list[str]


def _estimate_type(values: list[Any]) -> str:
    """Estime le type d'une colonne à partir d'un échantillon."""

    if not values:
        return "unknown"
    cleaned = [clean_text(value) for value in values if clean_text(value)]
    if not cleaned:
        return "all_missing"
    numeric = 0
    integer = 0
    date_like = 0
    for value in cleaned:
        if any(pattern.match(value) for pattern in DATE_PATTERNS):
            date_like += 1
        if re.fullmatch(r"[+-]?\d+", value):
            integer += 1
            numeric += 1
        elif re.fullmatch(r"[+-]?\d+(?:[.,]\d+)?", value):
            numeric += 1
    total = len(cleaned)
    if date_like / total >= 0.6:
        return "date"
    if integer / total >= 0.8:
        return "integer"
    if numeric / total >= 0.8:
        return "numeric"
    return "text"


def _sample_examples(values: list[Any], limit: int = 3) -> list[str]:
    """Retourne quelques exemples anonymisés."""

    examples: list[str] = []
    for value in values:
        cleaned = clean_text(value)
        if not cleaned:
            continue
        examples.append(anonymize_value(cleaned))
        if len(examples) >= limit:
            break
    return examples


def inspect_catalog(
    csv_path: str | Path,
    config_path: str | Path | None = None,
    *,
    chunksize: int = 50_000,
    sample_limit: int = 5_000,
) -> dict[str, Any]:
    """Analyse un catalogue CPF sans charger l'intégralité du fichier en mémoire."""

    path = Path(csv_path)
    fmt = detect_text_format(path)
    alias_map = load_column_aliases(config_path)

    total_rows = 0
    missing_counts: Counter[str] = Counter()
    sample_values: dict[str, list[Any]] = defaultdict(list)
    columns: list[str] | None = None

    for chunk_index, chunk in enumerate(
        pd.read_csv(path, sep=fmt.separator, encoding=fmt.encoding, chunksize=chunksize, low_memory=False)
    ):
        if columns is None:
            columns = list(chunk.columns)
        total_rows += len(chunk)
        for col in chunk.columns:
            series = chunk[col]
            missing_counts[col] += int(series.isna().sum() + series.astype(str).isin({"", "nan", "None", "null"}).sum())
            if len(sample_values[col]) < sample_limit:
                sample_values[col].extend(series.dropna().tolist()[: max(0, sample_limit - len(sample_values[col]))])
        if total_rows >= sample_limit * 4:
            break

    columns = columns or []
    detection: ColumnDetection = detect_columns(columns, alias_map)
    column_stats: list[ColumnStats] = []
    for col in columns:
        non_missing = max(total_rows - missing_counts.get(col, 0), 0)
        missing_rate = round((missing_counts.get(col, 0) / total_rows) if total_rows else 0.0, 4)
        values = sample_values.get(col, [])
        column_stats.append(
            ColumnStats(
                name=col,
                missing_rate=missing_rate,
                estimated_type=_estimate_type(values),
                examples=_sample_examples(values),
            )
        )

    report = {
        "path": str(path),
        "encoding": fmt.encoding,
        "separator": fmt.separator,
        "row_sampled": total_rows,
        "column_count": len(columns),
        "columns": columns,
        "missing_rates": {stat.name: stat.missing_rate for stat in column_stats},
        "estimated_types": {stat.name: stat.estimated_type for stat in column_stats},
        "examples": {stat.name: stat.examples for stat in column_stats},
        "resolved_columns": detection.resolved,
        "candidate_columns": detection.candidates,
        "alias_map": alias_map,
    }
    return report


def save_schema_report(report: dict[str, Any], output_path: str | Path) -> None:
    """Écrit le rapport de schéma en JSON."""

    json_dump(Path(output_path), report)

