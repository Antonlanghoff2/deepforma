from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    hamming_loss,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    f1_score,
)

from ._common import safe_divide, to_jsonable


@dataclass(slots=True)
class MultilabelLabelMetrics:
    label: str
    support_positive: int
    support_negative: int
    predicted_positive: int
    predicted_negative: int
    precision: float
    recall: float
    f1: float
    average_precision: float | None
    prevalence: float


@dataclass(slots=True)
class MultilabelEvaluationReport:
    labels: list[str]
    thresholds: dict[str, float]
    metrics: dict[str, float]
    per_label: list[MultilabelLabelMetrics]
    score_distribution: dict[str, Any]
    label_score_distribution: list[dict[str, Any]]
    top1_top2_gap_mean: float
    label_cardinality_true: float
    label_cardinality_predicted: float
    mean_labels_per_example: float
    positive_label_counts: dict[str, int]
    negative_label_counts: dict[str, int]
    labels_never_predicted: list[str]
    labels_always_predicted: list[str]
    labels_without_positive_examples: list[str]
    labels_with_zero_f1: list[str]
    precision_at_k: dict[str, float]
    recall_at_k: dict[str, float]
    exact_match_ratio: float
    warnings: list[str] = field(default_factory=list)
    non_discriminant: bool = False
    non_discriminant_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


def _as_2d(y_true: np.ndarray | list[list[int]] | list[list[bool]] | list[list[float]]) -> np.ndarray:
    values = np.asarray(y_true)
    if values.ndim != 2:
        raise ValueError("y_true et y_score doivent être des matrices 2D.")
    return values


def _topk_metrics(y_true: np.ndarray, y_score: np.ndarray, *, k: int) -> tuple[float, float]:
    precisions: list[float] = []
    recalls: list[float] = []
    for row_true, row_score in zip(y_true, y_score):
        topk = np.argsort(-row_score)[:k]
        hits = int(row_true[topk].sum())
        positives = int(row_true.sum())
        precisions.append(safe_divide(hits, k))
        recalls.append(0.0 if positives == 0 else safe_divide(hits, positives))
    return float(np.mean(precisions) if precisions else 0.0), float(np.mean(recalls) if recalls else 0.0)


def _score_stats(scores: np.ndarray) -> dict[str, float]:
    flat = scores.reshape(-1)
    return {
        "min": float(np.min(flat)) if flat.size else 0.0,
        "max": float(np.max(flat)) if flat.size else 0.0,
        "mean": float(np.mean(flat)) if flat.size else 0.0,
        "median": float(np.median(flat)) if flat.size else 0.0,
        "std": float(np.std(flat)) if flat.size else 0.0,
    }


def _detect_non_discriminant(y_true: np.ndarray, y_score: np.ndarray, *, average_precision_micro: float | None, prevalence: float) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    flat = y_score.reshape(-1)
    score_std = float(np.std(flat)) if flat.size else 0.0
    distance_from_half = float(np.mean(np.abs(flat - 0.5))) if flat.size else 0.0
    score_range = float(np.max(flat) - np.min(flat)) if flat.size else 0.0
    if score_std < 0.02:
        reasons.append(f"variance trop faible ({score_std:.4f})")
    if distance_from_half < 0.05:
        reasons.append(f"scores trop proches de 0.5 (distance moyenne={distance_from_half:.4f})")
    if score_range < 0.10:
        reasons.append(f"probabilités presque constantes (range={score_range:.4f})")
    if average_precision_micro is not None and abs(average_precision_micro - prevalence) < 0.02:
        reasons.append(
            "PR-AUC proche de la prévalence "
            f"({average_precision_micro:.4f} vs {prevalence:.4f})"
        )
    return (len(reasons) >= 2, reasons)


def evaluate_multilabel_classification(
    y_true: np.ndarray | list[list[int]] | list[list[bool]] | list[list[float]],
    y_score: np.ndarray | list[list[float]],
    labels: list[str],
    *,
    thresholds: dict[str, float] | None = None,
    top_k_values: tuple[int, ...] = (1, 3, 5),
) -> MultilabelEvaluationReport:
    y_true_arr = _as_2d(y_true).astype(int)
    y_score_arr = _as_2d(y_score).astype(float)
    if y_true_arr.shape != y_score_arr.shape:
        raise ValueError("y_true et y_score doivent avoir la même forme.")
    if y_true_arr.shape[1] != len(labels):
        raise ValueError("Le nombre de labels ne correspond pas à la matrice.")

    thresholds = thresholds or {label: 0.5 for label in labels}
    threshold_values = np.array([float(thresholds.get(label, 0.5)) for label in labels], dtype=float)
    y_pred = (y_score_arr >= threshold_values.reshape(1, -1)).astype(int)

    metrics = {
        "micro_precision": float(precision_score(y_true_arr, y_pred, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(y_true_arr, y_pred, average="micro", zero_division=0)),
        "micro_f1": float(f1_score(y_true_arr, y_pred, average="micro", zero_division=0)),
        "macro_precision": float(precision_score(y_true_arr, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true_arr, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true_arr, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true_arr, y_pred, average="weighted", zero_division=0)),
        "samples_f1": float(f1_score(y_true_arr, y_pred, average="binary", zero_division=0)) if y_true_arr.shape[1] == 1 else float(f1_score(y_true_arr, y_pred, average="samples", zero_division=0)),
        "hamming_loss": float(hamming_loss(y_true_arr, y_pred)),
        "exact_match_ratio": float(np.mean(np.all(y_true_arr == y_pred, axis=1))) if len(y_true_arr) else 0.0,
    }
    try:
        metrics["average_precision_micro"] = float(average_precision_score(y_true_arr, y_score_arr, average="micro"))
    except Exception:
        metrics["average_precision_micro"] = 0.0

    valid_macro_aps: list[float] = []
    per_label: list[MultilabelLabelMetrics] = []
    positive_counts: dict[str, int] = {}
    negative_counts: dict[str, int] = {}

    for index, label in enumerate(labels):
        truth = y_true_arr[:, index].astype(int)
        pred = y_pred[:, index].astype(int)
        score = y_score_arr[:, index].astype(float)
        support_pos = int(truth.sum())
        support_neg = int(len(truth) - support_pos)
        predicted_pos = int(pred.sum())
        predicted_neg = int(len(pred) - predicted_pos)
        avg_prec: float | None
        if support_pos == 0 or support_pos == len(truth):
            avg_prec = None
        else:
            try:
                avg_prec = float(average_precision_score(truth, score))
                valid_macro_aps.append(avg_prec)
            except Exception:
                avg_prec = None
        per_label.append(
            MultilabelLabelMetrics(
                label=label,
                support_positive=support_pos,
                support_negative=support_neg,
                predicted_positive=predicted_pos,
                predicted_negative=predicted_neg,
                precision=float(precision_score(truth, pred, zero_division=0)),
                recall=float(recall_score(truth, pred, zero_division=0)),
                f1=float(f1_score(truth, pred, zero_division=0)),
                average_precision=avg_prec,
                prevalence=float(safe_divide(support_pos, len(truth))),
            )
        )
        positive_counts[label] = support_pos
        negative_counts[label] = support_neg

    metrics["average_precision_macro"] = float(np.mean(valid_macro_aps)) if valid_macro_aps else 0.0

    precision_at_k: dict[str, float] = {}
    recall_at_k: dict[str, float] = {}
    for k in top_k_values:
        precision_at_k[str(k)], recall_at_k[str(k)] = _topk_metrics(y_true_arr, y_score_arr, k=k)

    score_distribution = _score_stats(y_score_arr)
    label_score_distribution = []
    for index, label in enumerate(labels):
        stats = _score_stats(y_score_arr[:, index : index + 1])
        stats["label"] = label
        label_score_distribution.append(stats)

    label_cardinality_true = float(np.mean(y_true_arr.sum(axis=1))) if len(y_true_arr) else 0.0
    label_cardinality_predicted = float(np.mean(y_pred.sum(axis=1))) if len(y_pred) else 0.0

    top1_top2_gap = []
    for row in y_score_arr:
        ordered = np.sort(row)[::-1]
        if len(ordered) >= 2:
            top1_top2_gap.append(float(ordered[0] - ordered[1]))
    top1_top2_gap_mean = float(np.mean(top1_top2_gap)) if top1_top2_gap else 0.0

    prevalence = float(np.mean(y_true_arr)) if y_true_arr.size else 0.0
    is_non_discriminant, reasons = _detect_non_discriminant(
        y_true_arr,
        y_score_arr,
        average_precision_micro=metrics.get("average_precision_micro"),
        prevalence=prevalence,
    )
    labels_never_predicted = [label for label, row in zip(labels, per_label) if row.predicted_positive == 0]
    labels_always_predicted = [label for label, row in zip(labels, per_label) if row.predicted_negative == 0]
    labels_without_positive_examples = [label for label, row in zip(labels, per_label) if row.support_positive == 0]
    labels_with_zero_f1 = [label for label, row in zip(labels, per_label) if row.f1 == 0.0]
    warnings: list[str] = []
    if is_non_discriminant:
        warnings.append("Le modèle semble inexploitable: " + "; ".join(reasons))
    if labels_never_predicted:
        warnings.append("Labels jamais prédits: " + ", ".join(labels_never_predicted[:10]))
    if labels_always_predicted:
        warnings.append("Labels toujours prédits: " + ", ".join(labels_always_predicted[:10]))
    if labels_without_positive_examples:
        warnings.append("Labels sans exemple positif: " + ", ".join(labels_without_positive_examples[:10]))
    if labels_with_zero_f1:
        warnings.append("Labels avec F1 nul: " + ", ".join(labels_with_zero_f1[:10]))
    if metrics["average_precision_micro"] < prevalence + 0.01:
        warnings.append(
            f"PR-AUC micro proche de la prévalence ({metrics['average_precision_micro']:.4f} vs {prevalence:.4f})."
        )
    if metrics["exact_match_ratio"] == 0.0 and metrics["micro_f1"] == 0.0:
        warnings.append("Aucun recouvrement exact ou partiel détecté.")

    return MultilabelEvaluationReport(
        labels=labels,
        thresholds={label: float(thresholds.get(label, 0.5)) for label in labels},
        metrics=metrics,
        per_label=per_label,
        score_distribution=score_distribution,
        label_score_distribution=label_score_distribution,
        top1_top2_gap_mean=top1_top2_gap_mean,
        label_cardinality_true=label_cardinality_true,
        label_cardinality_predicted=label_cardinality_predicted,
        mean_labels_per_example=label_cardinality_true,
        positive_label_counts=positive_counts,
        negative_label_counts=negative_counts,
        labels_never_predicted=labels_never_predicted,
        labels_always_predicted=labels_always_predicted,
        labels_without_positive_examples=labels_without_positive_examples,
        labels_with_zero_f1=labels_with_zero_f1,
        precision_at_k=precision_at_k,
        recall_at_k=recall_at_k,
        exact_match_ratio=float(metrics["exact_match_ratio"]),
        warnings=warnings,
        non_discriminant=is_non_discriminant,
        non_discriminant_reasons=reasons,
    )
