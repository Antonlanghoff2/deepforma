from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import difflib
import re

import numpy as np

from common.text import clean_text, normalize_for_match


DEFAULT_GOLD_DATASET_PATH = Path("data/training/skill_extraction/gold.jsonl")


@dataclass(slots=True)
class SkillExtractionDocument:
    document_id: str
    gold_skills: list[str]
    predicted_skills: list[str]


@dataclass(slots=True)
class SkillExtractionScores:
    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int


@dataclass(slots=True)
class SkillExtractionEvaluationReport:
    status: str
    gold_dataset_path: str | None
    document_count: int
    evaluated_document_count: int
    gold_skill_count: int
    predicted_skill_count: int
    exact_match: SkillExtractionScores | None
    normalized_match: SkillExtractionScores | None
    semantic_match: SkillExtractionScores | None
    mean_false_positives_per_document: float | None
    mean_missing_skills_per_document: float | None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "gold_dataset_path": self.gold_dataset_path,
            "document_count": self.document_count,
            "evaluated_document_count": self.evaluated_document_count,
            "gold_skill_count": self.gold_skill_count,
            "predicted_skill_count": self.predicted_skill_count,
            "exact_match": self.exact_match.__dict__ if self.exact_match else None,
            "normalized_match": self.normalized_match.__dict__ if self.normalized_match else None,
            "semantic_match": self.semantic_match.__dict__ if self.semantic_match else None,
            "mean_false_positives_per_document": self.mean_false_positives_per_document,
            "mean_missing_skills_per_document": self.mean_missing_skills_per_document,
            "warnings": self.warnings,
        }


def _simple_plural_normalize(text: str) -> str:
    tokens = []
    for token in normalize_for_match(text).split():
        if len(token) > 3 and token.isalpha() and not token.isupper() and not token.endswith("ss"):
            if token.endswith("es"):
                token = token[:-2]
            elif token.endswith("s"):
                token = token[:-1]
        tokens.append(token)
    return " ".join(tokens).strip()


def normalize_skill_text(text: str) -> str:
    cleaned = clean_text(text)
    if not cleaned:
        return ""
    cleaned = _simple_plural_normalize(cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _extract_skill_list(doc: dict[str, Any], *, preferred_keys: tuple[str, ...]) -> list[str]:
    for key in preferred_keys:
        value = doc.get(key)
        if isinstance(value, list):
            result = []
            for item in value:
                if isinstance(item, dict):
                    label = item.get("label") or item.get("canonical_label") or item.get("name")
                else:
                    label = item
                label = clean_text(label)
                if label:
                    result.append(label)
            if result:
                return result
        elif isinstance(value, str):
            parts = [clean_text(part) for part in re.split(r"[|;,\n•·]+", value) if clean_text(part)]
            if parts:
                return parts
    return []


def _load_encoder(embedding_model: str | Path | None):
    if not embedding_model:
        return None
    model_path = Path(embedding_model)
    if not model_path.exists():
        return None
    try:
        from deepforma.cpf.embeddings import build_encoder

        return build_encoder(str(model_path))
    except Exception:
        return None


def _greedy_semantic_matches(gold: list[str], predicted: list[str], encoder: Any | None, *, threshold: float) -> int:
    if not gold or not predicted:
        return 0
    if encoder is None:
        gold_norm = [normalize_skill_text(item) for item in gold]
        pred_norm = [normalize_skill_text(item) for item in predicted]
        matches = 0
        used: set[int] = set()
        for prediction in pred_norm:
            best_index = None
            best_score = 0.0
            for index, gold_item in enumerate(gold_norm):
                if index in used or not gold_item:
                    continue
                score = difflib.SequenceMatcher(None, prediction, gold_item).ratio()
                if score > best_score:
                    best_score = score
                    best_index = index
            if best_index is not None and best_score >= threshold:
                used.add(best_index)
                matches += 1
        return matches

    try:
        gold_vectors = encoder.encode(gold, normalize_embeddings=True, convert_to_numpy=True)
        pred_vectors = encoder.encode(predicted, normalize_embeddings=True, convert_to_numpy=True)
    except Exception:
        return 0
    similarity = np.matmul(np.asarray(pred_vectors, dtype=float), np.asarray(gold_vectors, dtype=float).T)
    used_gold: set[int] = set()
    matches = 0
    for pred_index in np.argsort(-similarity.max(axis=1)):
        row = similarity[pred_index]
        best_index = None
        best_score = 0.0
        for gold_index, score in enumerate(row):
            if gold_index in used_gold:
                continue
            if score > best_score:
                best_score = float(score)
                best_index = gold_index
        if best_index is not None and best_score >= threshold:
            used_gold.add(best_index)
            matches += 1
    return matches


def _scores_from_counts(tp: int, fp: int, fn: int) -> SkillExtractionScores:
    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    f1 = float((2 * precision * recall) / (precision + recall)) if (precision + recall) else 0.0
    return SkillExtractionScores(
        precision=precision,
        recall=recall,
        f1=f1,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
    )


def evaluate_skill_extraction(
    documents: list[dict[str, Any]],
    *,
    embedding_model: str | Path | None = None,
    semantic_threshold: float = 0.75,
    gold_dataset_path: str | Path | None = DEFAULT_GOLD_DATASET_PATH,
) -> SkillExtractionEvaluationReport:
    encoder = _load_encoder(embedding_model)
    warnings: list[str] = []
    if embedding_model and encoder is None:
        warnings.append("Modèle d'embeddings non disponible localement; correspondance sémantique repliée sur une similarité lexicale.")

    exact_tp = exact_fp = exact_fn = 0
    norm_tp = norm_fp = norm_fn = 0
    sem_tp = sem_fp = sem_fn = 0
    fp_per_doc: list[int] = []
    fn_per_doc: list[int] = []
    evaluated_documents = 0
    gold_skill_count = 0
    predicted_skill_count = 0

    for document in documents:
        gold = _extract_skill_list(document, preferred_keys=("gold_skills", "skills", "annotations", "reference_skills"))
        predicted = _extract_skill_list(document, preferred_keys=("predicted_skills", "extracted_skills", "predictions", "skills_predicted"))
        if not gold:
            continue
        evaluated_documents += 1
        gold_skill_count += len(gold)
        predicted_skill_count += len(predicted)

        gold_exact = [clean_text(item) for item in gold if clean_text(item)]
        pred_exact = [clean_text(item) for item in predicted if clean_text(item)]
        exact_matches = len(set(gold_exact) & set(pred_exact))
        exact_tp += exact_matches
        exact_fp += max(0, len(set(pred_exact) - set(gold_exact)))
        exact_fn += max(0, len(set(gold_exact) - set(pred_exact)))

        gold_norm = [normalize_skill_text(item) for item in gold if normalize_skill_text(item)]
        pred_norm = [normalize_skill_text(item) for item in predicted if normalize_skill_text(item)]
        norm_matches = len(set(gold_norm) & set(pred_norm))
        norm_tp += norm_matches
        norm_fp += max(0, len(set(pred_norm) - set(gold_norm)))
        norm_fn += max(0, len(set(gold_norm) - set(pred_norm)))

        sem_matches = _greedy_semantic_matches(gold, predicted, encoder, threshold=semantic_threshold)
        sem_tp += sem_matches
        sem_fp += max(0, len(predicted) - sem_matches)
        sem_fn += max(0, len(gold) - sem_matches)

        fp_per_doc.append(max(0, len(predicted) - sem_matches))
        fn_per_doc.append(max(0, len(gold) - sem_matches))

    if evaluated_documents == 0:
        expected_path = str(Path(gold_dataset_path)) if gold_dataset_path else "data/training/skill_extraction/gold.jsonl"
        warnings.append(f"Aucun gold dataset exploitable trouvé. Chemin attendu: {expected_path}.")
        return SkillExtractionEvaluationReport(
            status="not_evaluated",
            gold_dataset_path=expected_path,
            document_count=len(documents),
            evaluated_document_count=0,
            gold_skill_count=0,
            predicted_skill_count=0,
            exact_match=None,
            normalized_match=None,
            semantic_match=None,
            mean_false_positives_per_document=None,
            mean_missing_skills_per_document=None,
            warnings=warnings,
        )

    return SkillExtractionEvaluationReport(
        status="evaluated",
        gold_dataset_path=str(Path(gold_dataset_path)) if gold_dataset_path else None,
        document_count=len(documents),
        evaluated_document_count=evaluated_documents,
        gold_skill_count=gold_skill_count,
        predicted_skill_count=predicted_skill_count,
        exact_match=_scores_from_counts(exact_tp, exact_fp, exact_fn),
        normalized_match=_scores_from_counts(norm_tp, norm_fp, norm_fn),
        semantic_match=_scores_from_counts(sem_tp, sem_fp, sem_fn),
        mean_false_positives_per_document=float(np.mean(fp_per_doc)) if fp_per_doc else 0.0,
        mean_missing_skills_per_document=float(np.mean(fn_per_doc)) if fn_per_doc else 0.0,
        warnings=warnings,
    )
