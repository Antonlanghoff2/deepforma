from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from ._common import to_jsonable


@dataclass(slots=True)
class BinaryClassMetrics:
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    f1: float
    macro_f1: float
    mcc: float
    roc_auc: float | None
    pr_auc: float | None


@dataclass(slots=True)
class BinaryClassReport:
    label_names: tuple[str, str] = ("non_ia", "ia")
    positive_label_name: str = "ia"
    negative_label_name: str = "non_ia"
    threshold: float = 0.5
    total_examples: int = 0
    real_positive_count: int = 0
    real_negative_count: int = 0
    predicted_positive_count: int = 0
    predicted_negative_count: int = 0
    positive_prevalence: float = 0.0
    predicted_positive_rate: float = 0.0
    metrics: BinaryClassMetrics | None = None
    per_class: dict[str, dict[str, float]] = field(default_factory=dict)
    confusion_matrix: list[list[int]] = field(default_factory=list)
    confusion_counts: dict[str, int] = field(default_factory=dict)
    probability_stats: dict[str, float] = field(default_factory=dict)
    roc_curve: dict[str, list[float]] = field(default_factory=dict)
    pr_curve: dict[str, list[float]] = field(default_factory=dict)
    support: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    alerts: list[str] = field(default_factory=list)
    threshold_optimization: dict[str, Any] = field(default_factory=dict)
    label_convention: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


def _class_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: tuple[str, str]) -> dict[str, dict[str, float]]:
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    return {
        labels[0]: {
            "precision": float(precision[0]),
            "recall": float(recall[0]),
            "f1": float(f1[0]),
            "support": int(support[0]),
        },
        labels[1]: {
            "precision": float(precision[1]),
            "recall": float(recall[1]),
            "f1": float(f1[1]),
            "support": int(support[1]),
        },
    }


def _probability_stats(scores: np.ndarray) -> dict[str, float]:
    flat = np.asarray(scores, dtype=float).reshape(-1)
    if flat.size == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0, "std": 0.0}
    return {
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "mean": float(np.mean(flat)),
        "median": float(np.median(flat)),
        "std": float(np.std(flat)),
    }


def _alerts(
    *,
    metrics: BinaryClassMetrics,
    total_examples: int,
    real_positive_count: int,
    predicted_positive_count: int,
    predicted_negative_count: int,
    threshold: float,
) -> list[str]:
    alerts: list[str] = []
    majority_count = max(real_positive_count, total_examples - real_positive_count)
    majority_accuracy = float(majority_count / total_examples) if total_examples else 0.0
    if predicted_positive_count == 0:
        alerts.append("Le modèle prédit uniquement la classe non-IA. L'accuracy est trompeuse car elle correspond à la classe majoritaire.")
    if predicted_negative_count == 0:
        alerts.append("Le modèle prédit uniquement la classe IA.")
    if metrics.recall == 0.0:
        alerts.append("Le rappel positif est nul.")
    if abs(metrics.balanced_accuracy - 0.5) < 0.02:
        alerts.append("La balanced accuracy est proche de 0.5.")
    if metrics.mcc == 0.0:
        alerts.append("Le MCC est nul.")
    if total_examples and abs(metrics.accuracy - majority_accuracy) < 0.03:
        alerts.append("L'accuracy correspond approximativement à la classe majoritaire.")
    if threshold != 0.5:
        alerts.append(f"Seuil de décision optimisé à {threshold:.4f}.")
    return alerts


def evaluate_binary_classification(
    y_true: np.ndarray | list[int] | list[bool],
    y_score: np.ndarray | list[float],
    *,
    threshold: float = 0.5,
    label_names: tuple[str, str] = ("non_ia", "ia"),
) -> BinaryClassReport:
    y_true_arr = np.asarray(y_true, dtype=int).reshape(-1)
    y_score_arr = np.asarray(y_score, dtype=float).reshape(-1)
    if y_true_arr.shape[0] != y_score_arr.shape[0]:
        raise ValueError("y_true et y_score doivent avoir la même longueur.")

    y_pred = (y_score_arr >= threshold).astype(int)
    metrics = BinaryClassMetrics(
        accuracy=float(accuracy_score(y_true_arr, y_pred)),
        balanced_accuracy=float(balanced_accuracy_score(y_true_arr, y_pred)),
        precision=float(precision_score(y_true_arr, y_pred, zero_division=0)),
        recall=float(recall_score(y_true_arr, y_pred, zero_division=0)),
        f1=float(f1_score(y_true_arr, y_pred, zero_division=0)),
        macro_f1=float(f1_score(y_true_arr, y_pred, average="macro", zero_division=0)),
        mcc=float(matthews_corrcoef(y_true_arr, y_pred)),
        roc_auc=None,
        pr_auc=None,
    )
    try:
        metrics.roc_auc = float(roc_auc_score(y_true_arr, y_score_arr))
    except Exception:
        metrics.roc_auc = None
    try:
        metrics.pr_auc = float(average_precision_score(y_true_arr, y_score_arr))
    except Exception:
        metrics.pr_auc = None

    class_metrics = _class_metrics(y_true_arr, y_pred, label_names)
    conf = confusion_matrix(y_true_arr, y_pred, labels=[0, 1]).tolist()
    total_examples = int(y_true_arr.shape[0])
    real_positive_count = int((y_true_arr == 1).sum())
    real_negative_count = int((y_true_arr == 0).sum())
    predicted_positive_count = int((y_pred == 1).sum())
    predicted_negative_count = int((y_pred == 0).sum())
    support = {
        label_names[0]: real_negative_count,
        label_names[1]: real_positive_count,
    }
    confusion_counts = {
        "tn": int(conf[0][0]),
        "fp": int(conf[0][1]),
        "fn": int(conf[1][0]),
        "tp": int(conf[1][1]),
    }
    probability_stats = _probability_stats(y_score_arr)
    warnings: list[str] = []
    if metrics.roc_auc is None:
        warnings.append("ROC-AUC indisponible: une seule classe est présente dans les données.")
    if metrics.pr_auc is None:
        warnings.append("PR-AUC indisponible: une seule classe est présente dans les données.")

    alerts = _alerts(
        metrics=metrics,
        total_examples=total_examples,
        real_positive_count=real_positive_count,
        predicted_positive_count=predicted_positive_count,
        predicted_negative_count=predicted_negative_count,
        threshold=threshold,
    )

    roc_payload: dict[str, list[float]] = {}
    pr_payload: dict[str, list[float]] = {}
    try:
        fpr, tpr, thresholds = roc_curve(y_true_arr, y_score_arr)
        roc_payload = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": thresholds.tolist()}
    except Exception:
        pass
    try:
        precision, recall, thresholds = precision_recall_curve(y_true_arr, y_score_arr)
        pr_payload = {"precision": precision.tolist(), "recall": recall.tolist(), "thresholds": thresholds.tolist()}
    except Exception:
        pass

    return BinaryClassReport(
        label_names=label_names,
        positive_label_name=label_names[1],
        negative_label_name=label_names[0],
        threshold=threshold,
        total_examples=total_examples,
        real_positive_count=real_positive_count,
        real_negative_count=real_negative_count,
        predicted_positive_count=predicted_positive_count,
        predicted_negative_count=predicted_negative_count,
        positive_prevalence=float(real_positive_count / total_examples) if total_examples else 0.0,
        predicted_positive_rate=float(predicted_positive_count / total_examples) if total_examples else 0.0,
        metrics=metrics,
        per_class=class_metrics,
        confusion_matrix=conf,
        confusion_counts=confusion_counts,
        probability_stats=probability_stats,
        roc_curve=roc_payload,
        pr_curve=pr_payload,
        support=support,
        warnings=warnings,
        alerts=alerts,
        label_convention={
            "negative": label_names[0],
            "positive": label_names[1],
            "positive_index": 1,
        },
    )


def predict_from_scores(y_score: np.ndarray | list[float], threshold: float = 0.5) -> np.ndarray:
    scores = np.asarray(y_score, dtype=float).reshape(-1)
    return (scores >= threshold).astype(int)
