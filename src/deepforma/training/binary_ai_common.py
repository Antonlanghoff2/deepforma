from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from common.text import clean_text


DEFAULT_LOW_CONFIDENCE_MAX_PROBABILITY = 0.60
DEFAULT_REVIEW_LOW_THRESHOLD = 0.35
DEFAULT_REVIEW_HIGH_THRESHOLD = 0.65


def positive_class_index(classes: Iterable[Any]) -> int:
    class_list = list(classes)
    if 1 not in class_list:
        raise ValueError(f"Classe positive absente: {class_list}")
    return class_list.index(1)


def positive_class_probability(probabilities: Any, classes: Iterable[Any]) -> np.ndarray:
    matrix = np.asarray(probabilities, dtype=float)
    if matrix.ndim == 1:
        return matrix.reshape(-1)
    index = positive_class_index(classes)
    if matrix.shape[1] <= index:
        raise ValueError(
            f"Sortie de probabilite invalide: colonne {index} absente pour une matrice de forme {matrix.shape}"
        )
    return np.asarray(matrix[:, index], dtype=float).reshape(-1)


@dataclass(frozen=True, slots=True)
class TextSufficiencyResult:
    sufficient: bool
    reason: str | None
    cleaned_text: str
    token_count: int
    char_count: int


_SIGNIFICANT_RE = re.compile(r"[A-Za-zÀ-ÿ0-9]")
_TOKEN_RE = re.compile(r"\b[\wÀ-ÿ']+\b", re.UNICODE)


def assess_text_sufficiency(
    text: str,
    *,
    min_token_count: int = 1,
    min_char_count: int = 3,
    max_symbol_ratio: float = 0.65,
) -> TextSufficiencyResult:
    cleaned = clean_text(text)
    if not cleaned:
        return TextSufficiencyResult(False, "texte_insuffisant", "", 0, 0)

    tokens = _TOKEN_RE.findall(cleaned)
    token_count = len(tokens)
    compact = re.sub(r"\s+", "", cleaned)
    significant_chars = _SIGNIFICANT_RE.findall(cleaned)
    char_count = len(significant_chars)
    symbol_ratio = 0.0 if not compact else 1.0 - (char_count / len(compact))

    if char_count < min_char_count:
        return TextSufficiencyResult(False, "texte_insuffisant", cleaned, token_count, char_count)
    if token_count < min_token_count:
        return TextSufficiencyResult(False, "texte_insuffisant", cleaned, token_count, char_count)
    if len(compact) <= 2:
        return TextSufficiencyResult(False, "texte_insuffisant", cleaned, token_count, char_count)
    if symbol_ratio > max_symbol_ratio:
        return TextSufficiencyResult(False, "texte_insuffisant", cleaned, token_count, char_count)

    if token_count == 1:
        token = tokens[0]
        token_lower = token.lower()
        if len(token) <= 2:
            return TextSufficiencyResult(False, "texte_insuffisant", cleaned, token_count, char_count)
        if len(token) >= 8 and not re.search(r"[aeiouyàâäéèêëîïôöùûü]", token_lower):
            return TextSufficiencyResult(False, "texte_insuffisant", cleaned, token_count, char_count)
        if len(set(token_lower)) <= 2 and len(token) >= 6:
            return TextSufficiencyResult(False, "texte_insuffisant", cleaned, token_count, char_count)

    return TextSufficiencyResult(True, None, cleaned, token_count, char_count)


def classify_binary_probability(
    probability_ia: float | None,
    *,
    low_threshold: float = DEFAULT_REVIEW_LOW_THRESHOLD,
    high_threshold: float = DEFAULT_REVIEW_HIGH_THRESHOLD,
    low_confidence_max_probability: float = DEFAULT_LOW_CONFIDENCE_MAX_PROBABILITY,
) -> dict[str, Any]:
    if probability_ia is None:
        return {
            "label": "indetermine",
            "reason": "texte_insuffisant",
            "requires_review": True,
            "status": "unavailable",
            "warning": "Texte insuffisant ou prédiction indisponible.",
        }

    probability_ia = float(probability_ia)
    probability_non_ia = float(max(0.0, 1.0 - probability_ia))
    confidence = max(probability_ia, probability_non_ia)
    if probability_ia >= high_threshold:
        label = "IA"
        reason = None
        requires_review = False
        status = "ok" if confidence >= low_confidence_max_probability else "low_confidence"
    elif probability_ia <= low_threshold:
        label = "non-IA"
        reason = None
        requires_review = False
        status = "ok" if confidence >= low_confidence_max_probability else "low_confidence"
    else:
        label = "indetermine"
        reason = "zone_incertaine"
        requires_review = True
        status = "low_confidence"

    warning = None
    if status == "low_confidence":
        warning = "Scores proches de 0,5 : classification peu fiable."

    return {
        "label": label,
        "reason": reason,
        "requires_review": requires_review,
        "status": status,
        "warning": warning,
        "probability_ia": probability_ia,
        "probability_non_ia": probability_non_ia,
        "confidence": confidence,
        "threshold_low": float(low_threshold),
        "threshold_high": float(high_threshold),
        "low_confidence_max_probability": float(low_confidence_max_probability),
    }
