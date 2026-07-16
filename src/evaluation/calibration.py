from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import json
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from ._common import dump_json


@dataclass(slots=True)
class ThresholdOptimizationResult:
    model_name: str
    version: str
    generated_at: str
    metric: str
    thresholds: dict[str, float]
    per_label_scores: dict[str, dict[str, float]]
    baseline_metrics: dict[str, float] = field(default_factory=dict)
    optimized_metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "version": self.version,
            "generated_at": self.generated_at,
            "metric": self.metric,
            "thresholds": self.thresholds,
            "per_label_scores": self.per_label_scores,
            "baseline_metrics": self.baseline_metrics,
            "optimized_metrics": self.optimized_metrics,
        }


@dataclass(slots=True)
class BinaryThresholdOptimizationResult:
    model_name: str
    version: str
    generated_at: str
    metric: str
    threshold: float
    baseline_metrics: dict[str, float]
    optimized_metrics: dict[str, float]
    candidate_count: int
    baseline_threshold: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "version": self.version,
            "generated_at": self.generated_at,
            "metric": self.metric,
            "threshold": self.threshold,
            "baseline_threshold": self.baseline_threshold,
            "candidate_count": self.candidate_count,
            "baseline_metrics": self.baseline_metrics,
            "optimized_metrics": self.optimized_metrics,
        }


def _threshold_grid(scores: np.ndarray, *, step: float = 0.01) -> np.ndarray:
    unique = np.unique(np.concatenate([scores.reshape(-1), np.array([0.0, 0.5, 1.0], dtype=float)]))
    if unique.size <= 200:
        return unique
    grid = np.round(np.arange(0.0, 1.0 + step, step), 4)
    return grid


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, zero_division=0, average="binary")
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0
    youden_j = float(recall + specificity - 1.0)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, zero_division=0, average="macro")
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "macro_f1": float(macro_f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "youden_j": youden_j,
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def _select_threshold(
    truth: np.ndarray,
    score: np.ndarray,
    grid: np.ndarray,
    *,
    mode: str,
    min_precision: float | None = None,
    min_recall: float | None = None,
) -> tuple[float, dict[str, float]]:
    best_threshold = 0.5
    best_score = -1.0
    best_metrics: dict[str, float] = {}

    for threshold in grid:
        pred = (score >= threshold).astype(int)
        metrics = _binary_metrics(truth, pred)
        if mode == "maximize_f1":
            target = metrics["f1"]
        elif mode == "maximize_macro_f1":
            target = metrics["macro_f1"]
        elif mode == "maximize_youden_j":
            target = metrics["youden_j"]
        elif mode == "min_precision":
            if min_precision is not None and metrics["precision"] < min_precision:
                continue
            target = metrics["recall"]
        elif mode == "min_recall":
            if min_recall is not None and metrics["recall"] < min_recall:
                continue
            target = metrics["precision"]
        else:
            raise ValueError(f"Mode inconnu: {mode}")

        is_better = target > best_score
        if not is_better and target == best_score:
            is_better = threshold < best_threshold
        if is_better:
            best_threshold = float(threshold)
            best_score = float(target)
            best_metrics = metrics

    return best_threshold, best_metrics


def optimize_binary_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    mode: str = "maximize_f1",
    min_precision: float | None = None,
    min_recall: float | None = None,
    step: float = 0.01,
) -> BinaryThresholdOptimizationResult:
    y_true_arr = np.asarray(y_true, dtype=int).reshape(-1)
    y_score_arr = np.asarray(y_score, dtype=float).reshape(-1)
    if y_true_arr.shape != y_score_arr.shape:
        raise ValueError("y_true et y_score doivent avoir la même longueur.")
    baseline_pred = (y_score_arr >= 0.5).astype(int)
    baseline_metrics = _binary_metrics(y_true_arr, baseline_pred)
    grid = _threshold_grid(y_score_arr, step=step)
    threshold, optimized_metrics = _select_threshold(
        y_true_arr,
        y_score_arr,
        grid,
        mode=mode,
        min_precision=min_precision,
        min_recall=min_recall,
    )
    return BinaryThresholdOptimizationResult(
        model_name="",
        version="",
        generated_at=datetime.now(timezone.utc).isoformat(),
        metric=mode,
        threshold=threshold,
        baseline_metrics=baseline_metrics,
        optimized_metrics=optimized_metrics,
        candidate_count=int(grid.size),
    )


def optimize_thresholds(
    y_true: np.ndarray,
    y_score: np.ndarray,
    labels: list[str],
    *,
    mode: str = "maximize_f1",
    min_precision: float | None = None,
    min_recall: float | None = None,
    step: float = 0.01,
) -> ThresholdOptimizationResult:
    y_true_arr = np.asarray(y_true, dtype=int)
    y_score_arr = np.asarray(y_score, dtype=float)
    if y_true_arr.shape != y_score_arr.shape:
        raise ValueError("y_true et y_score doivent avoir la même forme.")
    if y_true_arr.shape[1] != len(labels):
        raise ValueError("Le nombre de labels ne correspond pas.")

    thresholds: dict[str, float] = {}
    per_label_scores: dict[str, dict[str, float]] = {}
    grid_cache = [_threshold_grid(y_score_arr[:, index], step=step) for index in range(len(labels))]

    for index, label in enumerate(labels):
        truth = y_true_arr[:, index]
        score = y_score_arr[:, index]
        grid = grid_cache[index]
        positives = int(truth.sum())
        negatives = int(len(truth) - positives)
        if positives == 0:
            best_threshold = 1.0
            best_metrics = _binary_metrics(truth, (score >= best_threshold).astype(int))
            thresholds[label] = best_threshold
            per_label_scores[label] = {
                "best_threshold": best_threshold,
                "best_target_score": 0.0,
                **best_metrics,
            }
            continue
        if negatives == 0:
            best_threshold = 0.0
            best_metrics = _binary_metrics(truth, (score >= best_threshold).astype(int))
            thresholds[label] = best_threshold
            per_label_scores[label] = {
                "best_threshold": best_threshold,
                "best_target_score": 0.0,
                **best_metrics,
            }
            continue

        best_threshold = 0.5
        best_score = -1.0
        best_metrics: dict[str, float] = {}

        for threshold in grid:
            pred = (score >= threshold).astype(int)
            metrics = _binary_metrics(truth, pred)
            if mode == "maximize_f1":
                target = metrics["f1"]
            elif mode == "maximize_macro_f1":
                target = metrics["macro_f1"]
            elif mode == "maximize_youden_j":
                target = metrics["youden_j"]
            elif mode == "min_precision":
                if min_precision is not None and metrics["precision"] < min_precision:
                    continue
                target = metrics["recall"]
            elif mode == "min_recall":
                if min_recall is not None and metrics["recall"] < min_recall:
                    continue
                target = metrics["precision"]
            else:
                raise ValueError(f"Mode inconnu: {mode}")

            is_better = target > best_score
            if not is_better and target == best_score:
                is_better = threshold < best_threshold
            if is_better:
                best_threshold = float(threshold)
                best_score = float(target)
                best_metrics = metrics

        thresholds[label] = float(best_threshold)
        per_label_scores[label] = {
            "best_threshold": float(best_threshold),
            "best_target_score": float(best_score),
            **best_metrics,
        }

    return ThresholdOptimizationResult(
        model_name="",
        version="",
        generated_at=datetime.now(timezone.utc).isoformat(),
        metric=mode,
        thresholds=thresholds,
        per_label_scores=per_label_scores,
    )


def save_binary_threshold_json(
    output_path: str | Path,
    result: BinaryThresholdOptimizationResult,
    *,
    model_name: str,
    version: str,
    metric: str,
) -> Path:
    path = Path(output_path)
    payload = {
        **result.to_dict(),
        "model_name": model_name,
        "version": version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metric": metric,
    }
    dump_json(path, payload)
    return path


def save_thresholds_json(
    output_path: str | Path,
    result: ThresholdOptimizationResult,
    *,
    model_name: str,
    version: str,
    metric: str,
) -> Path:
    path = Path(output_path)
    payload = {
        **result.to_dict(),
        "model_name": model_name,
        "version": version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metric": metric,
    }
    dump_json(path, payload)
    return path
