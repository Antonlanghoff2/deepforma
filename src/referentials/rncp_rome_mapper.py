from __future__ import annotations

import os
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from common.text import clean_text, normalize_for_match, stable_hash
from .france_competences import FranceCompetenceCertification
from .rome_referential import RomeJob

AUTO_VALIDATE_THRESHOLD = float(os.getenv('RNCP_ROME_AUTO_VALIDATE_THRESHOLD', '0.82'))
REVIEW_THRESHOLD = float(os.getenv('RNCP_ROME_REVIEW_THRESHOLD', '0.65'))
EMBEDDING_MODEL = os.getenv('RNCP_ROME_EMBEDDING_MODEL', '').strip() or None


def _token_set(value: str) -> set[str]:
    return {token for token in normalize_for_match(value).split() if token}


def _jaccard(left: str, right: str) -> float:
    lset = _token_set(left)
    rset = _token_set(right)
    if not lset or not rset:
        return 0.0
    return len(lset & rset) / len(lset | rset)


def _sequence(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, normalize_for_match(left), normalize_for_match(right)).ratio()


def _best_similarity(left_items: list[str], right_items: list[str]) -> tuple[float, list[dict[str, str]]]:
    best = 0.0
    evidence: list[dict[str, str]] = []
    for left in left_items:
        for right in right_items:
            score = max(_sequence(left, right), _jaccard(left, right))
            if score > best:
                best = score
                evidence = [{'left': clean_text(left), 'right': clean_text(right)}]
            elif score == best and score > 0:
                evidence.append({'left': clean_text(left), 'right': clean_text(right)})
    return best, evidence


def _skill_similarity(cert_skills: list[str], rome_skills: list[str]) -> tuple[float, list[dict[str, str]]]:
    if not cert_skills or not rome_skills:
        return 0.0, []
    return _best_similarity(cert_skills, rome_skills)


@dataclass(slots=True)
class RNCPRomeMatch:
    rncp_code: str
    rome_code: str
    score: float
    match_method: str
    validated: bool
    evidence: dict[str, list[dict[str, str]]]

    def to_dict(self) -> dict[str, Any]:
        return {
            'rncp_code': self.rncp_code,
            'rome_code': self.rome_code,
            'score': round(float(self.score), 4),
            'match_method': self.match_method,
            'validated': self.validated,
            'evidence': self.evidence,
        }


class RNCPRomeMapper:
    def __init__(self, *, auto_validate_threshold: float = AUTO_VALIDATE_THRESHOLD, review_threshold: float = REVIEW_THRESHOLD, embedding_model: str | None = EMBEDDING_MODEL) -> None:
        self.auto_validate_threshold = auto_validate_threshold
        self.review_threshold = review_threshold
        self.embedding_model = embedding_model or None

    def _semantic_similarity(self, left: str, right: str) -> float:
        if self.embedding_model:
            model_path = Path(self.embedding_model)
            if model_path.exists() and model_path.is_dir():
                try:
                    from sentence_transformers import SentenceTransformer  # type: ignore

                    model = SentenceTransformer(str(model_path))
                    vectors = model.encode([left, right], normalize_embeddings=True)
                    return float(vectors[0] @ vectors[1])
                except Exception:
                    pass
        return max(_sequence(left, right), _jaccard(left, right))

    def score(
        self,
        certification: FranceCompetenceCertification,
        rome_job: RomeJob,
        *,
        cert_skill_labels: list[str] | None = None,
        rome_skill_labels: list[str] | None = None,
        official: bool = False,
    ) -> RNCPRomeMatch:
        title_similarity, title_evidence = _best_similarity(
            [certification.title, *certification.target_jobs],
            [rome_job.label, *rome_job.alternative_titles, rome_job.definition],
        )
        activity_similarity, activity_evidence = _best_similarity(
            certification.activities + certification.target_jobs,
            [rome_job.definition, *rome_job.activity_ids, *rome_job.alternative_titles],
        )
        skill_similarity, skill_evidence = _skill_similarity(cert_skill_labels or certification.activities, rome_skill_labels or rome_job.skill_ids)
        if official:
            score = 1.0
            method = 'official'
        else:
            score = round(title_similarity * 0.25 + skill_similarity * 0.60 + activity_similarity * 0.15, 4)
            if skill_similarity >= 0.75 and title_similarity >= 0.5:
                method = 'hybrid'
            elif skill_similarity >= title_similarity and skill_similarity >= activity_similarity:
                method = 'skills'
            elif title_similarity >= activity_similarity:
                method = 'title'
            else:
                method = 'semantic'
        validated = score >= self.auto_validate_threshold
        evidence = {
            'title_matches': title_evidence,
            'skill_matches': skill_evidence,
            'activity_matches': activity_evidence,
        }
        return RNCPRomeMatch(certification.rncp_code, rome_job.rome_code, score, method, validated, evidence)

    def map_many(self, certifications: list[FranceCompetenceCertification], rome_jobs: list[RomeJob]) -> list[RNCPRomeMatch]:
        results: list[RNCPRomeMatch] = []
        for cert in certifications:
            for job in rome_jobs:
                match = self.score(cert, job, cert_skill_labels=cert.activities, rome_skill_labels=job.skill_ids)
                if match.score >= self.review_threshold:
                    results.append(match)
        results.sort(key=lambda item: (-item.score, item.rncp_code, item.rome_code))
        return results

    @staticmethod
    def canonical_group_id(match: RNCPRomeMatch) -> str:
        return stable_hash(match.rncp_code, match.rome_code, length=24)
