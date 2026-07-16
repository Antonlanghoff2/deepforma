from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Iterable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        value = float(value)
    except Exception:
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return float(value)


def _array(values: Iterable[Any]) -> np.ndarray:
    return np.asarray(list(values), dtype=float)


def _as_int_array(values: Iterable[Any]) -> np.ndarray:
    return np.asarray(list(values), dtype=int)


def _to_python(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {key: _to_python(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_to_python(item) for item in obj]
    return obj


def _probability_stats(scores: np.ndarray) -> dict[str, float | None]:
    if scores.size == 0:
        return {key: None for key in ["min", "max", "mean", "median", "std"]}
    return {
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "mean": float(np.mean(scores)),
        "median": float(np.median(scores)),
        "std": float(np.std(scores, ddof=0)),
    }


def _distribution(values: np.ndarray) -> dict[str, int]:
    if values.size == 0:
        return {"0": 0, "1": 0}
    unique, counts = np.unique(values.astype(int), return_counts=True)
    mapping = {str(int(label)): int(count) for label, count in zip(unique, counts)}
    mapping.setdefault("0", 0)
    mapping.setdefault("1", 0)
    return mapping


def _safe_metric(func, *args, default: float | None = None, **kwargs) -> float | None:
    try:
        return float(func(*args, **kwargs))
    except Exception:
        return default


@dataclass(frozen=True, slots=True)
class ThresholdOptimizationResult:
    threshold: float
    mode: str
    baseline_threshold: float
    baseline_metrics: dict[str, float | None]
    optimized_metrics: dict[str, float | None]
    candidates: list[dict[str, float | None]]


@dataclass(frozen=True, slots=True)
class BinaryClassificationReport:
    model_name: str | None
    threshold: float
    total_examples: int
    real_positive_count: int
    real_negative_count: int
    predicted_positive_count: int
    predicted_negative_count: int
    positive_rate: float
    predicted_positive_rate: float
    confusion_counts: dict[str, int]
    metrics: dict[str, float | None]
    per_class: dict[str, dict[str, float | None]]
    classification_report: dict[str, Any]
    probability_stats: dict[str, float | None]
    probability_distribution: dict[str, int]
    roc_curve: dict[str, list[float]]
    precision_recall_curve: dict[str, list[float]]
    threshold_curve: list[dict[str, float | None]]
    calibration_curve: dict[str, list[float]]
    inference_time_ms: float | None = None
    model_size_bytes: int | None = None
    alerts: list[str] = None  # type: ignore[assignment]
    generated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _to_python(asdict(self))


def _build_alerts(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    metrics: dict[str, float | None],
    threshold: float,
) -> list[str]:
    alerts: list[str] = []
    predicted_classes = set(y_pred.tolist())
    if predicted_classes == {0}:
        alerts.append("Le modèle prédit uniquement la classe non-IA. L'accuracy est trompeuse car elle correspond à la classe majoritaire.")
    if predicted_classes == {1}:
        alerts.append("Le modèle prédit uniquement la classe IA.")
    if metrics.get("recall_ia") == 0:
        alerts.append("Le rappel positif est nul.")
    if metrics.get("balanced_accuracy") is not None and abs(metrics["balanced_accuracy"] - 0.5) <= 0.05:
        alerts.append("La balanced accuracy est proche de 0.5.")
    if metrics.get("mcc") == 0:
        alerts.append("Le MCC est nul.")
    prevalence = float(np.mean(y_true)) if y_true.size else 0.0
    if metrics.get("accuracy") is not None and abs(metrics["accuracy"] - max(prevalence, 1.0 - prevalence)) <= 0.05:
        alerts.append("L'accuracy correspond approximativement à la classe majoritaire.")
    if np.allclose(y_score, 0.5, atol=0.05) or float(np.std(y_score)) < 0.02:
        alerts.append("Les probabilités sont quasi constantes autour de 0.5.")
    if threshold is not None and abs(threshold - 0.5) < 1e-6:
        alerts.append("Seuil de décision par défaut 0.5 utilisé.")
    return alerts


def _threshold_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, float | None]:
    y_pred = (y_score >= threshold).astype(int)
    metrics = {
        "threshold": float(threshold),
        "accuracy": _safe_metric(accuracy_score, y_true, y_pred),
        "balanced_accuracy": _safe_metric(balanced_accuracy_score, y_true, y_pred),
        "precision_ia": _safe_metric(precision_score, y_true, y_pred, zero_division=0),
        "recall_ia": _safe_metric(recall_score, y_true, y_pred, zero_division=0),
        "f1_ia": _safe_metric(f1_score, y_true, y_pred, zero_division=0),
        "precision_non_ia": _safe_metric(precision_score, 1 - y_true, 1 - y_pred, zero_division=0),
        "recall_non_ia": _safe_metric(recall_score, 1 - y_true, 1 - y_pred, zero_division=0),
        "f1_non_ia": _safe_metric(f1_score, 1 - y_true, 1 - y_pred, zero_division=0),
        "f1_macro": _safe_metric(f1_score, y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": _safe_metric(f1_score, y_true, y_pred, average="weighted", zero_division=0),
        "mcc": _safe_metric(matthews_corrcoef, y_true, y_pred),
        "cohen_kappa": _safe_metric(cohen_kappa_score, y_true, y_pred),
    }
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = _safe_metric(roc_auc_score, y_true, y_score)
        metrics["pr_auc"] = _safe_metric(average_precision_score, y_true, y_score)
        metrics["log_loss"] = _safe_metric(log_loss, y_true, np.vstack([1 - y_score, y_score]).T, labels=[0, 1])
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None
        metrics["log_loss"] = None
    metrics["brier_score"] = _safe_metric(brier_score_loss, y_true, y_score)
    return metrics


def optimize_binary_threshold(
    y_true: Iterable[Any],
    y_score: Iterable[Any],
    *,
    mode: str = "maximize_f1",
    min_recall: float | None = None,
    min_precision: float | None = None,
) -> ThresholdOptimizationResult:
    y_true_arr = _as_int_array(y_true)
    y_score_arr = _array(y_score)
    candidates = sorted(set(float(x) for x in np.concatenate([y_score_arr, np.array([0.0, 0.5, 1.0])])))
    candidate_reports: list[dict[str, float | None]] = []
    baseline_metrics = _threshold_metrics(y_true_arr, y_score_arr, 0.5)
    best_threshold = 0.5
    best_metrics = baseline_metrics
    best_score = -1e9

    for threshold in candidates:
        metrics = _threshold_metrics(y_true_arr, y_score_arr, threshold)
        candidate_reports.append(metrics)
        score = metrics.get("f1_ia") or 0.0
        if mode == "maximize_macro_f1":
            score = metrics.get("f1_macro") or 0.0
        elif mode == "youden_j":
            score = float((metrics.get("recall_ia") or 0.0) - (1.0 - (metrics.get("recall_non_ia") or 0.0)))
        elif mode == "recall_min":
            if min_recall is not None and (metrics.get("recall_ia") or 0.0) < min_recall:
                continue
        elif mode == "precision_min":
            if min_precision is not None and (metrics.get("precision_ia") or 0.0) < min_precision:
                continue
        elif mode != "maximize_f1":
            raise ValueError(f"Mode de seuil inconnu: {mode}")
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
            best_metrics = metrics

    return ThresholdOptimizationResult(
        threshold=float(best_threshold),
        mode=mode,
        baseline_threshold=0.5,
        baseline_metrics=baseline_metrics,
        optimized_metrics=best_metrics,
        candidates=candidate_reports,
    )


def evaluate_binary_classification(
    y_true: Iterable[Any],
    y_score: Iterable[Any],
    *,
    threshold: float | None = None,
    model_name: str | None = None,
    inference_time_ms: float | None = None,
    model_size_bytes: int | None = None,
) -> BinaryClassificationReport:
    y_true_arr = _as_int_array(y_true)
    y_score_arr = _array(y_score)
    if y_true_arr.size != y_score_arr.size:
        raise ValueError("y_true et y_score doivent avoir la meme longueur")
    chosen_threshold = 0.5 if threshold is None else float(threshold)
    y_pred = (y_score_arr >= chosen_threshold).astype(int)
    metrics = _threshold_metrics(y_true_arr, y_score_arr, chosen_threshold)
    metrics["total_examples"] = int(y_true_arr.size)
    metrics["real_positive_count"] = int(y_true_arr.sum())
    metrics["real_negative_count"] = int(y_true_arr.size - y_true_arr.sum())
    metrics["predicted_positive_count"] = int(y_pred.sum())
    metrics["predicted_negative_count"] = int(y_pred.size - y_pred.sum())
    confusion = confusion_matrix(y_true_arr, y_pred, labels=[0, 1])
    prob_stats = _probability_stats(y_score_arr)
    precision, recall, thresholds = precision_recall_curve(y_true_arr, y_score_arr) if len(np.unique(y_true_arr)) > 1 else (np.array([]), np.array([]), np.array([]))
    fpr, tpr, roc_thresholds = roc_curve(y_true_arr, y_score_arr) if len(np.unique(y_true_arr)) > 1 else (np.array([]), np.array([]), np.array([]))

    threshold_curve: list[dict[str, float | None]] = []
    for candidate_threshold in sorted(set(float(x) for x in np.concatenate([y_score_arr, np.array([0.0, 0.5, 1.0])]))):
        threshold_curve.append(_threshold_metrics(y_true_arr, y_score_arr, candidate_threshold))

    # Simple calibration points from quantiles.
    if y_score_arr.size:
        quantiles = np.linspace(0.0, 1.0, 6)
        bins = np.quantile(y_score_arr, quantiles)
        bin_ids = np.digitize(y_score_arr, bins[1:-1], right=True)
        calibration_x = []
        calibration_y = []
        for bin_id in sorted(set(bin_ids.tolist())):
            mask = bin_ids == bin_id
            if not mask.any():
                continue
            calibration_x.append(float(np.mean(y_score_arr[mask])))
            calibration_y.append(float(np.mean(y_true_arr[mask])))
    else:
        calibration_x = []
        calibration_y = []

    report = BinaryClassificationReport(
        model_name=model_name,
        threshold=chosen_threshold,
        total_examples=int(y_true_arr.size),
        real_positive_count=int(y_true_arr.sum()),
        real_negative_count=int(y_true_arr.size - y_true_arr.sum()),
        predicted_positive_count=int(y_pred.sum()),
        predicted_negative_count=int(y_pred.size - y_pred.sum()),
        positive_rate=float(np.mean(y_true_arr)) if y_true_arr.size else 0.0,
        predicted_positive_rate=float(np.mean(y_pred)) if y_pred.size else 0.0,
        confusion_counts={
            "tn": int(confusion[0, 0]) if confusion.size else 0,
            "fp": int(confusion[0, 1]) if confusion.size else 0,
            "fn": int(confusion[1, 0]) if confusion.size else 0,
            "tp": int(confusion[1, 1]) if confusion.size else 0,
        },
        metrics=metrics,
        per_class={
            "non_ia": {
                "precision": metrics.get("precision_non_ia"),
                "recall": metrics.get("recall_non_ia"),
                "f1": metrics.get("f1_non_ia"),
            },
            "ia": {
                "precision": metrics.get("precision_ia"),
                "recall": metrics.get("recall_ia"),
                "f1": metrics.get("f1_ia"),
            },
        },
        classification_report=classification_report(y_true_arr, y_pred, labels=[0, 1], target_names=["non_ia", "ia"], output_dict=True, zero_division=0),
        probability_stats=prob_stats,
        probability_distribution={str(k): int(v) for k, v in zip(*np.unique(np.round(y_score_arr, 4), return_counts=True))} if y_score_arr.size else {},
        roc_curve={"fpr": fpr.tolist() if len(fpr) else [], "tpr": tpr.tolist() if len(tpr) else [], "thresholds": roc_thresholds.tolist() if len(roc_thresholds) else []},
        precision_recall_curve={"precision": precision.tolist() if len(precision) else [], "recall": recall.tolist() if len(recall) else [], "thresholds": thresholds.tolist() if len(thresholds) else []},
        threshold_curve=threshold_curve,
        calibration_curve={"predicted": calibration_x, "observed": calibration_y},
        inference_time_ms=inference_time_ms,
        model_size_bytes=model_size_bytes,
        alerts=[],
        generated_at=datetime.now(timezone.utc).isoformat(),
    )
    alerts = _build_alerts(y_true=y_true_arr, y_pred=y_pred, y_score=y_score_arr, metrics=metrics, threshold=chosen_threshold)
    object.__setattr__(report, "alerts", alerts)
    return report


def save_thresholds_json(
    path: str | Path,
    thresholds: dict[str, float] | ThresholdOptimizationResult,
    *,
    model_name: str,
    version: str,
    metric: str,
) -> Path:
    payload = {
        "model_name": model_name,
        "version": version,
        "date": datetime.now(timezone.utc).isoformat(),
        "metric_optimized": metric,
    }
    if isinstance(thresholds, ThresholdOptimizationResult):
        payload["thresholds"] = {"global": float(thresholds.threshold)}
        payload["baseline_threshold"] = float(thresholds.baseline_threshold)
        payload["baseline_metrics"] = thresholds.baseline_metrics
        payload["optimized_metrics"] = thresholds.optimized_metrics
    else:
        payload["thresholds"] = {key: float(value) for key, value in thresholds.items()}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_python(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    return path

