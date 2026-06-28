from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Protocol

import numpy as np
import pandas as pd

from common.text import clean_text, normalize_for_match
from deepforma.cpf.embeddings import NumpyVectorIndex, VectorIndexBackend, normalize_vectors
from deepforma.skills.normalizer import SkillTaxonomyNormalizer


class SearchableIndex(Protocol):
    """Protocole minimal pour interroger un index de formations."""

    def search(self, vector: np.ndarray, top_k: int = 10) -> list[tuple[str, float]]:
        ...


@dataclass(frozen=True)
class RecommenderWeights:
    """Pondérations du score final."""

    skill_coverage: float = 40.0
    semantic_similarity: float = 30.0
    territory: float = 15.0
    certification_level: float = 10.0
    data_quality: float = 5.0


@dataclass(frozen=True)
class RecommenderConfig:
    """Configuration du moteur de recommandation."""

    limit: int = 10
    candidates_multiplier: int = 6
    max_rerank: int = 100
    similarity_threshold: float = 0.78
    duplicate_similarity_threshold: float = 0.93
    weights: RecommenderWeights = field(default_factory=RecommenderWeights)


def _normalize_skill_list(values: Iterable[Any], normalizer: SkillTaxonomyNormalizer) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for value in values:
        label = clean_text(value.get("canonical_label") if isinstance(value, dict) else value)
        if not label:
            continue
        match = normalizer.normalize(label, extraction_source="user_input", confidence_floor=0.0)
        canonical_id = match.canonical_id if match else normalize_for_match(label)
        canonical_label = match.canonical_label if match else label
        if canonical_id in seen:
            continue
        seen.add(canonical_id)
        normalized.append(
            {
                "canonical_id": canonical_id,
                "canonical_label": canonical_label,
                "original_label": label,
                "confidence": 1.0 if match else 0.5,
            }
        )
    return normalized


def _skill_set(items: Iterable[dict[str, Any]]) -> set[str]:
    return {normalize_for_match(item.get("canonical_label") or item.get("original_label") or "") for item in items if item}


def _sentence_similarity(left: str, right: str) -> float:
    return difflib.SequenceMatcher(None, normalize_for_match(left), normalize_for_match(right)).ratio()


def _territory_score(candidate: dict[str, Any], region_code: str | None, department_code: str | None, remote_allowed: bool) -> float:
    candidate_region = clean_text(candidate.get("region_code")) or None
    candidate_department = clean_text(candidate.get("department_code")) or None
    if department_code and candidate_department and normalize_for_match(department_code) == normalize_for_match(candidate_department):
        return 1.0
    if region_code and candidate_region and normalize_for_match(region_code) == normalize_for_match(candidate_region):
        return 0.8
    if remote_allowed and candidate.get("remote_allowed") not in {False, "false", "False", 0}:
        return 0.7
    if remote_allowed:
        return 0.45
    return 0.0


def _level_score(candidate: dict[str, Any], required_level: str | None) -> float:
    if not required_level:
        return 0.5
    candidate_level = clean_text(candidate.get("exit_level") or candidate.get("level"))
    if not candidate_level:
        return 0.25
    required = normalize_for_match(required_level)
    candidate_norm = normalize_for_match(candidate_level)
    if required and required == candidate_norm:
        return 1.0
    if required and required in candidate_norm:
        return 0.8
    if candidate_norm and required and candidate_norm in required:
        return 0.7
    return 0.3


def _quality_score(candidate: dict[str, Any]) -> float:
    fields = [
        candidate.get("title"),
        candidate.get("certification"),
        candidate.get("organization"),
        candidate.get("search_text"),
        candidate.get("skills_normalized"),
    ]
    present = sum(1 for value in fields if clean_text(value) or value)
    return present / len(fields) if fields else 0.0


def _coverage_components(candidate_skills: set[str], missing_skills: list[dict[str, Any]], desired_skills: list[dict[str, Any]]) -> tuple[float, list[str], list[str]]:
    missing_labels = [item["canonical_label"] for item in missing_skills]
    desired_labels = [item["canonical_label"] for item in desired_skills]
    covered = [label for label in missing_labels if normalize_for_match(label) in candidate_skills]
    uncovered = [label for label in missing_labels if normalize_for_match(label) not in candidate_skills]
    desired_bonus = sum(1 for label in desired_labels if normalize_for_match(label) in candidate_skills)
    if not missing_labels:
        return (0.0 if not desired_labels else min(1.0, desired_bonus / len(desired_labels))), covered, uncovered
    return covered, uncovered


class TrainingRecommender:
    """Moteur de recommandation CPF explicable et déterministe."""

    def __init__(
        self,
        metadata: pd.DataFrame | list[dict[str, Any]],
        index: SearchableIndex | VectorIndexBackend | None = None,
        *,
        normalizer: SkillTaxonomyNormalizer | None = None,
        config: RecommenderConfig | None = None,
    ) -> None:
        self.normalizer = normalizer or SkillTaxonomyNormalizer()
        self.config = config or RecommenderConfig()
        self.metadata = self._metadata_to_dataframe(metadata)
        if "skills_normalized" not in self.metadata.columns:
            self.metadata["skills_normalized"] = [[] for _ in range(len(self.metadata))]
        self.metadata["skills_normalized"] = self.metadata["skills_normalized"].apply(self._normalize_metadata_skills)
        self._vocab: list[str] = sorted(
            {
                token
                for text in self.metadata["search_text"].fillna("").astype(str)
                for token in normalize_for_match(text).split()
                if token
            }
        )
        self.index = index or self._build_fallback_index()
        self._metadata_by_id = {
            str(row["formation_uid"]): row.to_dict()
            for _, row in self.metadata.iterrows()
            if clean_text(row.get("formation_uid"))
        }

    def _metadata_to_dataframe(self, metadata: pd.DataFrame | list[dict[str, Any]]) -> pd.DataFrame:
        frame = metadata.copy() if isinstance(metadata, pd.DataFrame) else pd.DataFrame(metadata)
        if "formation_uid" not in frame.columns:
            raise KeyError("La colonne formation_uid est requise pour la recommandation.")
        if "search_text" not in frame.columns:
            if "title" in frame.columns:
                frame["search_text"] = frame["title"].astype(str)
            else:
                frame["search_text"] = ""
        return frame.fillna("")

    def _normalize_metadata_skills(self, value: Any) -> list[str]:
        if isinstance(value, list):
            labels = []
            for item in value:
                if isinstance(item, dict):
                    candidate = item.get("canonical_label") or item.get("label") or item.get("original_label")
                else:
                    candidate = item
                normalized = self.normalizer.normalize(str(candidate or ""), extraction_source="metadata", confidence_floor=0.0)
                if normalized:
                    labels.append(normalized.canonical_label)
                else:
                    labels.append(clean_text(candidate))
            return [label for label in dict.fromkeys(labels) if label]
        if isinstance(value, str) and value.strip():
            labels = [part.strip() for part in value.split("|") if part.strip()]
            return [label for label in dict.fromkeys(labels) if label]
        return []

    def _build_fallback_index(self) -> NumpyVectorIndex:
        texts = self.metadata["search_text"].fillna("").astype(str).tolist()
        vocab = self._vocab
        if not vocab:
            return NumpyVectorIndex()
        vectors = []
        for text in texts:
            tokens = set(normalize_for_match(text).split())
            vec = np.array([1.0 if token in tokens else 0.0 for token in vocab], dtype=np.float32)
            vectors.append(vec)
        index = NumpyVectorIndex()
        index.add(np.vstack(vectors), self.metadata["formation_uid"].astype(str).tolist())
        return index

    def build_query_text(
        self,
        *,
        target_job: str,
        missing_skills: list[dict[str, Any]],
        desired_skills: list[dict[str, Any]],
        user_skills: list[dict[str, Any]],
    ) -> str:
        """Construit une requête sémantique pour l'index."""

        parts = [clean_text(target_job)]
        for item in missing_skills:
            parts.append(clean_text(item.get("canonical_label")))
        for item in desired_skills:
            parts.append(clean_text(item.get("canonical_label")))
        for item in user_skills:
            parts.append(clean_text(item.get("canonical_label")))
        return " | ".join(part for part in parts if part)

    def _get_query_vector(self, query_text: str) -> np.ndarray:
        vocab = self._vocab
        if not vocab:
            return np.zeros((1, 1), dtype=np.float32)
        tokens = set(normalize_for_match(query_text).split())
        vector = np.array([[1.0 if token in tokens else 0.0 for token in vocab]], dtype=np.float32)
        return normalize_vectors(vector)

    def _candidate_row(self, formation_uid: str) -> dict[str, Any] | None:
        return self._metadata_by_id.get(formation_uid)

    def _semantic_score(self, candidate: dict[str, Any], query_text: str, index_score: float) -> float:
        base = max(index_score, _sentence_similarity(query_text, candidate.get("search_text") or ""))
        return min(1.0, base)

    def _dedupe_similar(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        for row in rows:
            duplicate = False
            for kept in selected:
                same_cert = normalize_for_match(row.get("certification")) == normalize_for_match(kept.get("certification"))
                same_org = normalize_for_match(row.get("organization")) == normalize_for_match(kept.get("organization"))
                title_sim = _sentence_similarity(row.get("title") or "", kept.get("title") or "")
                if same_cert and same_org and title_sim >= self.config.duplicate_similarity_threshold:
                    duplicate = True
                    break
            if not duplicate:
                selected.append(row)
        return selected

    def recommend(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        """Retourne les meilleures formations pour une requête utilisateur."""

        target_job = clean_text(payload.get("target_job"))
        user_skills = _normalize_skill_list(payload.get("user_skills", []), self.normalizer)
        missing_skills = _normalize_skill_list(payload.get("missing_skills", []), self.normalizer)
        desired_skills = _normalize_skill_list(payload.get("desired_skills", []), self.normalizer)
        region_code = clean_text(payload.get("region_code")) or None
        department_code = clean_text(payload.get("department_code")) or None
        remote_allowed = bool(payload.get("remote_allowed", True))
        required_level = clean_text(payload.get("required_level")) or None
        limit = int(payload.get("limit") or self.config.limit)

        query_text = self.build_query_text(
            target_job=target_job,
            missing_skills=missing_skills,
            desired_skills=desired_skills,
            user_skills=user_skills,
        )
        query_vector = self._get_query_vector(query_text)
        candidate_count = max(limit, self.config.max_rerank)
        initial_hits = self.index.search(query_vector, top_k=min(len(self.metadata), candidate_count * self.config.candidates_multiplier))

        scored: list[dict[str, Any]] = []
        for formation_uid, semantic_raw_score in initial_hits:
            candidate = self._candidate_row(str(formation_uid))
            if not candidate:
                continue
            candidate_department = clean_text(candidate.get("department_code")) or None
            candidate_region = clean_text(candidate.get("region_code")) or None
            territory_match = False
            if department_code and candidate_department:
                territory_match = normalize_for_match(department_code) == normalize_for_match(candidate_department)
            elif region_code and candidate_region:
                territory_match = normalize_for_match(region_code) == normalize_for_match(candidate_region)
            elif remote_allowed:
                territory_match = True
            if not territory_match and not remote_allowed:
                continue
            candidate_skills = {normalize_for_match(skill) for skill in candidate.get("skills_normalized", []) if skill}
            covered = [item["canonical_label"] for item in missing_skills if normalize_for_match(item["canonical_label"]) in candidate_skills]
            uncovered = [item["canonical_label"] for item in missing_skills if normalize_for_match(item["canonical_label"]) not in candidate_skills]
            if not covered and missing_skills:
                semantic_score = min(1.0, float(semantic_raw_score)) * 0.4
            else:
                semantic_score = self._semantic_score(candidate, query_text, float(semantic_raw_score))
            coverage_ratio = (len(covered) / len(missing_skills)) if missing_skills else 0.0
            territory_ratio = _territory_score(candidate, region_code, department_code, remote_allowed)
            level_ratio = _level_score(candidate, required_level)
            quality_ratio = _quality_score(candidate)
            global_score = (
                self.config.weights.skill_coverage * coverage_ratio
                + self.config.weights.semantic_similarity * semantic_score
                + self.config.weights.territory * territory_ratio
                + self.config.weights.certification_level * level_ratio
                + self.config.weights.data_quality * quality_ratio
            )
            if missing_skills and not covered:
                global_score = min(global_score, 45.0)
            explanation_parts = []
            if covered:
                explanation_parts.append("compétences couvertes: " + ", ".join(covered))
            if uncovered and not covered:
                explanation_parts.append("aucune compétence manquante couverte")
            explanation_parts.append(f"similarité métier: {semantic_score:.2f}")
            explanation_parts.append(f"territoire: {territory_ratio:.2f}")
            scored.append(
                {
                    "formation_uid": str(candidate.get("formation_uid")),
                    "title": candidate.get("title") or "",
                    "organization": candidate.get("organization") or "",
                    "certification": candidate.get("certification") or "",
                    "referential_type": candidate.get("referential_type") or None,
                    "region": candidate.get("region") or candidate_region or None,
                    "department": candidate.get("department") or candidate_department or None,
                    "global_score": round(float(global_score), 2),
                    "semantic_score": round(float(semantic_score * 100), 2),
                    "skill_coverage_score": round(float(coverage_ratio * 100), 2),
                    "territory_score": round(float(territory_ratio * 100), 2),
                    "covered_skills": covered,
                    "uncovered_skills": uncovered,
                    "explanation": " | ".join(explanation_parts),
                    "source": "Mon Compte Formation",
                    "_diversity_key": (
                        normalize_for_match(candidate.get("organization")),
                        normalize_for_match(candidate.get("certification")),
                    ),
                }
            )

        scored.sort(
            key=lambda item: (
                -item["global_score"],
                -item["skill_coverage_score"],
                -item["semantic_score"],
                -item["territory_score"],
                item["title"],
            )
        )
        deduped = self._dedupe_similar(scored)
        for item in deduped:
            item.pop("_diversity_key", None)
        return deduped[:limit]

