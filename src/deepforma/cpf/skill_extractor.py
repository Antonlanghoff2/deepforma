from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import Any, Iterable

from common.text import clean_text, normalize_for_match
from deepforma.skills.normalizer import SkillMatch, SkillTaxonomyNormalizer


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+|•|·|\u2022")


@dataclass(frozen=True)
class SkillExtractionResult:
    """Résultat d'extraction de compétences pour une formation."""

    skills_explicit: list[dict[str, Any]]
    skills_inferred: list[dict[str, Any]]
    skills_normalized: list[dict[str, Any]]
    skills_confidence: dict[str, float]
    skills_evidence: dict[str, list[dict[str, str]]]


def _sentences_from_text(text: str) -> list[str]:
    sentences: list[str] = []
    for part in SENTENCE_SPLIT_RE.split(text or ""):
        cleaned = clean_text(part)
        if cleaned:
            sentences.append(cleaned)
    return sentences


def _best_evidence(text: str, alias: str) -> str:
    normalized_text = normalize_for_match(text)
    normalized_alias = normalize_for_match(alias)
    if not normalized_text or not normalized_alias:
        return clean_text(text)[:240]
    index = normalized_text.find(normalized_alias)
    if index >= 0:
        return clean_text(text)[:240]
    return clean_text(text)[:240]


class CPFSkillExtractor:
    """Extrait et normalise les compétences d'une formation CPF."""

    def __init__(
        self,
        normalizer: SkillTaxonomyNormalizer | None = None,
        *,
        confidence_threshold: float = 0.65,
    ) -> None:
        self.normalizer = normalizer or SkillTaxonomyNormalizer()
        self.confidence_threshold = confidence_threshold

    def _collect_text_fields(self, row: dict[str, Any]) -> dict[str, str]:
        return {
            "title": clean_text(row.get("title")),
            "certification": clean_text(row.get("certification")),
            "description": clean_text(row.get("description")),
            "objectives": clean_text(row.get("objectives")),
            "nsf": clean_text(row.get("nsf")),
        }

    def _alias_matches(self, text: str, field: str) -> list[SkillMatch]:
        matches: list[SkillMatch] = []
        normalized_text = normalize_for_match(text)
        for skill in self.normalizer.reference:
            label = clean_text(skill.get("label"))
            aliases = [label, *(skill.get("aliases", []) or [])]
            for alias in aliases:
                alias_norm = normalize_for_match(alias)
                if not alias_norm or len(alias_norm) < 3:
                    continue
                if f" {alias_norm} " in f" {normalized_text} " or normalized_text.startswith(alias_norm) or normalized_text.endswith(alias_norm):
                    match = self.normalizer.normalize(
                        alias,
                        extraction_source=f"{field}:alias",
                        confidence_floor=self.confidence_threshold,
                    )
                    if match:
                        matches.append(match)
                    break
        return matches

    def _semantic_matches(self, sentences: Iterable[str]) -> list[SkillMatch]:
        matches: list[SkillMatch] = []
        for sentence in sentences:
            sentence_norm = normalize_for_match(sentence)
            if len(sentence_norm) < 12:
                continue
            for skill in self.normalizer.reference:
                label = clean_text(skill.get("label"))
                label_norm = normalize_for_match(label)
                if not label_norm or len(label_norm) < 3:
                    continue
                ratio = difflib.SequenceMatcher(None, sentence_norm, label_norm).ratio()
                if ratio < 0.86:
                    continue
                match = self.normalizer.normalize(
                    label,
                    extraction_source="semantic",
                    confidence_floor=self.confidence_threshold,
                )
                if match:
                    matches.append(
                        SkillMatch(
                            canonical_id=match.canonical_id,
                            canonical_label=match.canonical_label,
                            original_label=sentence,
                            aliases=match.aliases,
                            extraction_source="semantic",
                            confidence=max(match.confidence, ratio),
                        )
                    )
        return matches

    def extract(self, row: dict[str, Any]) -> SkillExtractionResult:
        """Extrait les compétences d'une ligne normalisée."""

        fields = self._collect_text_fields(row)
        explicit: list[SkillMatch] = []
        evidence: dict[str, list[dict[str, str]]] = {}

        for field_name, text in fields.items():
            if not text:
                continue
            field_matches = self._alias_matches(text, field_name)
            explicit.extend(field_matches)
            for match in field_matches:
                evidence.setdefault(match.canonical_id, []).append(
                    {
                        "field": field_name,
                        "evidence": _best_evidence(text, match.original_label),
                        "source": match.extraction_source,
                    }
                )

        sentences = []
        for text in fields.values():
            sentences.extend(_sentences_from_text(text))
        inferred = self._semantic_matches(sentences)

        normalized: dict[str, dict[str, Any]] = {}
        confidences: dict[str, float] = {}
        for match in explicit + inferred:
            current = normalized.setdefault(
                match.canonical_id,
                {
                    "canonical_id": match.canonical_id,
                    "canonical_label": match.canonical_label,
                    "original_label": match.original_label,
                    "aliases": match.aliases,
                    "extraction_source": match.extraction_source,
                    "confidence": 0.0,
                },
            )
            current["confidence"] = max(float(current["confidence"]), float(match.confidence))
            current["original_label"] = match.original_label
            current["extraction_source"] = match.extraction_source
            confidences[match.canonical_id] = max(confidences.get(match.canonical_id, 0.0), float(match.confidence))

        explicit_payload = [
            {
                "canonical_id": match.canonical_id,
                "canonical_label": match.canonical_label,
                "original_label": match.original_label,
                "aliases": match.aliases,
                "extraction_source": match.extraction_source,
                "confidence": round(match.confidence, 4),
            }
            for match in explicit
        ]
        inferred_payload = [
            {
                "canonical_id": match.canonical_id,
                "canonical_label": match.canonical_label,
                "original_label": match.original_label,
                "aliases": match.aliases,
                "extraction_source": match.extraction_source,
                "confidence": round(match.confidence, 4),
            }
            for match in inferred
        ]
        normalized_payload = [
            {
                **item,
                "confidence": round(float(item["confidence"]), 4),
            }
            for item in sorted(normalized.values(), key=lambda item: (-float(item["confidence"]), item["canonical_label"]))
        ]
        return SkillExtractionResult(
            skills_explicit=explicit_payload,
            skills_inferred=inferred_payload,
            skills_normalized=normalized_payload,
            skills_confidence={key: round(value, 4) for key, value in confidences.items()},
            skills_evidence=evidence,
        )

