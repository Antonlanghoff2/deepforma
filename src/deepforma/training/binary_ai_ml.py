from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from common.text import clean_text
from deepforma.evaluation.binary_classification_metrics import (
    BinaryClassificationReport,
    ThresholdOptimizationResult,
    evaluate_binary_classification,
    optimize_binary_threshold,
    save_thresholds_json,
)


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BinaryAITrainingConfig:
    seed: int = 42
    classifier: str = "logistic"
    use_char_features: bool = True
    word_ngram_min: int = 1
    word_ngram_max: int = 2
    char_ngram_min: int = 3
    char_ngram_max: int = 5
    word_min_df: int = 2
    char_min_df: int = 2
    word_max_features: int | None = 50_000
    char_max_features: int | None = 50_000
    c_values: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
    class_weight: str = "balanced"
    max_iter: int = 2000
    cv_splits: int = 5
    threshold_mode: str = "maximize_f1"
    min_recall: float | None = None
    min_precision: float | None = None


@dataclass(frozen=True, slots=True)
class BinaryAIModelArtifacts:
    model_dir: str
    model_name: str
    model_version: str
    threshold: float
    train_report: dict[str, Any]
    validation_report: dict[str, Any]
    test_report: dict[str, Any]
    best_params: dict[str, Any]
    threshold_result: dict[str, Any]
    explainability: dict[str, str]


def _now_version() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _resolve_text(value: Any) -> str:
    text = clean_text(value)
    return text if text else ""


def _ensure_text_column(frame: pd.DataFrame) -> pd.Series:
    if "text" in frame.columns:
        return frame["text"].fillna("").astype(str)
    if "texte_modele" in frame.columns:
        return frame["texte_modele"].fillna("").astype(str)
    if "intitule" in frame.columns or "titre" in frame.columns:
        parts = []
        for _, row in frame.iterrows():
            fields = [row.get("intitule"), row.get("titre"), row.get("description"), row.get("objectifs"), row.get("programme")]
            parts.append(" \n ".join(part for part in (_resolve_text(value) for value in fields) if part))
        return pd.Series(parts, index=frame.index)
    raise ValueError("Le dataset ne contient ni colonne 'text' ni colonne exploitable pour construire le texte.")


def _build_feature_union(config: BinaryAITrainingConfig) -> FeatureUnion:
    transformers = [
        (
            "word",
            TfidfVectorizer(
                analyzer="word",
                ngram_range=(config.word_ngram_min, config.word_ngram_max),
                min_df=config.word_min_df,
                max_features=config.word_max_features,
                sublinear_tf=True,
                strip_accents="unicode",
                lowercase=True,
            ),
        )
    ]
    if config.use_char_features:
        transformers.append(
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(config.char_ngram_min, config.char_ngram_max),
                    min_df=config.char_min_df,
                    max_features=config.char_max_features,
                    sublinear_tf=True,
                    strip_accents="unicode",
                    lowercase=True,
                ),
            )
        )
    return FeatureUnion(transformers)


def _build_classifier(config: BinaryAITrainingConfig, *, c_value: float | None = None):
    if config.classifier == "logistic":
        return LogisticRegression(
            C=float(c_value or 1.0),
            class_weight=config.class_weight,
            max_iter=config.max_iter,
            random_state=config.seed,
            solver="liblinear",
        )
    if config.classifier == "sgd":
        return SGDClassifier(
            loss="log_loss",
            class_weight=config.class_weight,
            random_state=config.seed,
            max_iter=config.max_iter,
            tol=1e-3,
        )
    if config.classifier == "linearsvc":
        base = LinearSVC(class_weight=config.class_weight, random_state=config.seed)
        return CalibratedClassifierCV(base, method="sigmoid", cv=3)
    raise ValueError(f"Classifieur inconnu: {config.classifier}")


def build_pipeline(config: BinaryAITrainingConfig, *, c_value: float | None = None) -> Pipeline:
    return Pipeline(
        [
            ("features", _build_feature_union(config)),
            ("classifier", _build_classifier(config, c_value=c_value)),
        ]
    )


def _group_cv_splits(frame: pd.DataFrame, *, seed: int, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    groups = frame["group_id"].astype(str).to_numpy()
    y = frame["is_ai"].astype(int).to_numpy()
    unique_groups = pd.Index(groups).nunique()
    if unique_groups < 2:
        return []
    splits = min(n_splits, unique_groups)
    if splits < 2:
        return []
    splitter = GroupKFold(n_splits=splits)
    dummy = np.zeros((len(frame), 1))
    return list(splitter.split(dummy, y, groups))


def _candidate_grid(config: BinaryAITrainingConfig) -> list[dict[str, Any]]:
    grid: list[dict[str, Any]] = []
    for use_char in (False, True):
        if use_char and not config.use_char_features:
            continue
        for word_min_df in (1, config.word_min_df, max(1, config.word_min_df + 1)):
            for word_max_features in (20_000, 50_000, config.word_max_features):
                for c_value in config.c_values:
                    grid.append(
                        {
                            "use_char_features": use_char,
                            "word_min_df": int(word_min_df),
                            "word_max_features": None if word_max_features is None else int(word_max_features),
                            "c_value": float(c_value),
                        }
                    )
    unique = []
    seen = set()
    for candidate in grid:
        key = tuple(sorted(candidate.items()))
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def _with_overrides(config: BinaryAITrainingConfig, overrides: dict[str, Any]) -> BinaryAITrainingConfig:
    payload = asdict(config)
    payload.update(overrides)
    return BinaryAITrainingConfig(**payload)


def search_best_configuration(
    train_frame: pd.DataFrame,
    *,
    config: BinaryAITrainingConfig | None = None,
) -> dict[str, Any]:
    config = config or BinaryAITrainingConfig()
    cv_splits = _group_cv_splits(train_frame, seed=config.seed, n_splits=config.cv_splits)
    if not cv_splits:
        LOGGER.warning("Pas assez de groupes pour une CV groupee; utilisation du dataset complet comme fallback.")
        base_config = config
        pipeline = build_pipeline(base_config, c_value=1.0)
        pipeline.fit(_ensure_text_column(train_frame), train_frame["is_ai"].astype(int))
        proba = pipeline.predict_proba(_ensure_text_column(train_frame))[:, 1]
        threshold_result = optimize_binary_threshold(train_frame["is_ai"], proba, mode=config.threshold_mode, min_recall=config.min_recall, min_precision=config.min_precision)
        return {
            "config": base_config,
            "pipeline": pipeline,
            "mean_cv_f1": float(f1_score(train_frame["is_ai"], (proba >= threshold_result.threshold).astype(int), zero_division=0)),
            "threshold_result": threshold_result,
            "best_params": {"c_value": 1.0, "use_char_features": base_config.use_char_features, "word_min_df": base_config.word_min_df, "word_max_features": base_config.word_max_features},
        }

    candidates = _candidate_grid(config)
    texts = _ensure_text_column(train_frame)
    y = train_frame["is_ai"].astype(int).to_numpy()
    groups = train_frame["group_id"].astype(str).to_numpy()
    best: dict[str, Any] | None = None

    for candidate in candidates:
        candidate_config = _with_overrides(
            config,
            {
                "use_char_features": candidate["use_char_features"],
                "word_min_df": candidate["word_min_df"],
                "word_max_features": candidate["word_max_features"],
            },
        )
        fold_scores: list[float] = []
        failed = False
        for train_idx, valid_idx in cv_splits:
            try:
                pipeline = build_pipeline(candidate_config, c_value=candidate["c_value"])
                pipeline.fit(texts.iloc[train_idx], y[train_idx])
                score = pipeline.predict_proba(texts.iloc[valid_idx])[:, 1]
                threshold = optimize_binary_threshold(
                    y[valid_idx],
                    score,
                    mode=config.threshold_mode,
                    min_recall=config.min_recall,
                    min_precision=config.min_precision,
                ).threshold
                pred = (score >= threshold).astype(int)
                fold_scores.append(float(f1_score(y[valid_idx], pred, zero_division=0)))
            except Exception as exc:
                LOGGER.debug("Configuration ignoree (%s): %s", candidate, exc)
                failed = True
                break
        if failed:
            continue
        mean_f1 = float(np.mean(fold_scores)) if fold_scores else 0.0
        if best is None or mean_f1 > best["mean_cv_f1"]:
            best = {
                "config": candidate_config,
                "mean_cv_f1": mean_f1,
                "best_params": candidate,
            }

    if best is None:
        LOGGER.warning("Aucune configuration de CV n'a fonctionne; fallback sur la configuration par defaut.")
        best = {
            "config": config,
            "mean_cv_f1": 0.0,
            "best_params": {"c_value": 1.0, "use_char_features": config.use_char_features, "word_min_df": config.word_min_df, "word_max_features": config.word_max_features},
        }
    final_pipeline = build_pipeline(best["config"], c_value=best["best_params"]["c_value"])
    final_pipeline.fit(texts, y)
    proba = final_pipeline.predict_proba(texts)[:, 1]
    threshold_result = optimize_binary_threshold(
        y,
        proba,
        mode=config.threshold_mode,
        min_recall=config.min_recall,
        min_precision=config.min_precision,
    )
    best["pipeline"] = final_pipeline
    best["threshold_result"] = threshold_result
    return best


def _top_feature_coefficients(pipeline: Pipeline, *, top_n: int = 50) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_union: FeatureUnion = pipeline.named_steps["features"]
    classifier = pipeline.named_steps["classifier"]
    if hasattr(classifier, "base_estimator"):
        estimator = classifier.base_estimator
    else:
        estimator = classifier
    if not hasattr(estimator, "coef_"):
        empty = pd.DataFrame(columns=["feature", "weight"])
        return empty, empty
    feature_names = feature_union.get_feature_names_out()
    coefficients = np.asarray(estimator.coef_).ravel()
    order = np.argsort(coefficients)
    negative = pd.DataFrame({"feature": feature_names[order[:top_n]], "weight": coefficients[order[:top_n]]})
    positive = pd.DataFrame({"feature": feature_names[order[::-1][:top_n]], "weight": coefficients[order[::-1][:top_n]]})
    return positive, negative


def _save_explainability(output_dir: Path, pipeline: Pipeline) -> dict[str, str]:
    positive, negative = _top_feature_coefficients(pipeline)
    positive_path = output_dir / "ml_top_positive_features.csv"
    negative_path = output_dir / "ml_top_negative_features.csv"
    positive.to_csv(positive_path, index=False, encoding="utf-8")
    negative.to_csv(negative_path, index=False, encoding="utf-8")
    return {"positive_features": str(positive_path), "negative_features": str(negative_path)}


def fit_binary_ai_ml(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    *,
    output_dir: str | Path,
    config: BinaryAITrainingConfig | None = None,
) -> BinaryAIModelArtifacts:
    config = config or BinaryAITrainingConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    search = search_best_configuration(train_frame, config=config)
    pipeline: Pipeline = search["pipeline"]
    train_elapsed = (time.perf_counter() - start) * 1000.0

    def _predict(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        texts = _ensure_text_column(frame)
        probabilities = pipeline.predict_proba(texts)[:, 1]
        predictions = (probabilities >= search["threshold_result"].threshold).astype(int)
        return probabilities, predictions

    train_scores, _ = _predict(train_frame)
    validation_scores, _ = _predict(validation_frame)
    test_scores, _ = _predict(test_frame)

    train_report = evaluate_binary_classification(
        train_frame["is_ai"],
        train_scores,
        threshold=search["threshold_result"].threshold,
        model_name="binary_ai_ml",
        inference_time_ms=None,
        model_size_bytes=None,
    )
    validation_report = evaluate_binary_classification(
        validation_frame["is_ai"],
        validation_scores,
        threshold=search["threshold_result"].threshold,
        model_name="binary_ai_ml",
    )
    test_report = evaluate_binary_classification(
        test_frame["is_ai"],
        test_scores,
        threshold=search["threshold_result"].threshold,
        model_name="binary_ai_ml",
    )

    pipeline_path = output_dir / "pipeline.joblib"
    vectorizer_path = output_dir / "vectorizer.joblib"
    classifier_path = output_dir / "classifier.joblib"
    joblib.dump(pipeline, pipeline_path)
    joblib.dump(pipeline.named_steps["features"], vectorizer_path)
    joblib.dump(pipeline.named_steps["classifier"], classifier_path)

    thresholds_path = save_thresholds_json(
        output_dir / "thresholds.json",
        search["threshold_result"],
        model_name="binary_ai_ml",
        version=_now_version(),
        metric=config.threshold_mode,
    )

    explainability = _save_explainability(output_dir, pipeline)
    model_size_bytes = sum(path.stat().st_size for path in [pipeline_path, vectorizer_path, classifier_path, thresholds_path])
    metadata = {
        "model_name": "binary_ai_ml",
        "model_version": _now_version(),
        "seed": config.seed,
        "classifier": config.classifier,
        "best_params": search["best_params"],
        "threshold": float(search["threshold_result"].threshold),
        "threshold_mode": config.threshold_mode,
        "threshold_result": asdict(search["threshold_result"]),
        "train_rows": int(len(train_frame)),
        "validation_rows": int(len(validation_frame)),
        "test_rows": int(len(test_frame)),
        "train_group_count": int(train_frame["group_id"].nunique()),
        "validation_group_count": int(validation_frame["group_id"].nunique()),
        "test_group_count": int(test_frame["group_id"].nunique()),
        "pretrained_model": False,
        "pretrained_embeddings": False,
        "random_initialization": True,
        "model_size_bytes": int(model_size_bytes),
        "train_report": train_report.to_dict(),
        "validation_report": validation_report.to_dict(),
        "test_report": test_report.to_dict(),
        "paths": {
            "pipeline": str(pipeline_path),
            "vectorizer": str(vectorizer_path),
            "classifier": str(classifier_path),
            "thresholds": str(thresholds_path),
            **explainability,
        },
        "training_time_ms": train_elapsed,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    return BinaryAIModelArtifacts(
        model_dir=str(output_dir),
        model_name="binary_ai_ml",
        model_version=metadata["model_version"],
        threshold=float(search["threshold_result"].threshold),
        train_report=train_report.to_dict(),
        validation_report=validation_report.to_dict(),
        test_report=test_report.to_dict(),
        best_params=search["best_params"],
        threshold_result=asdict(search["threshold_result"]),
        explainability=explainability,
    )


def load_binary_ai_ml(model_dir: str | Path) -> tuple[Pipeline, dict[str, Any]]:
    model_dir = Path(model_dir)
    pipeline = joblib.load(model_dir / "pipeline.joblib")
    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    return pipeline, metadata


def predict_binary_ai_ml(
    text: str,
    *,
    model_dir: str | Path,
) -> dict[str, Any]:
    pipeline, metadata = load_binary_ai_ml(model_dir)
    start = time.perf_counter()
    text = clean_text(text)
    probability = float(pipeline.predict_proba([text])[:, 1][0])
    threshold = float(metadata.get("threshold", 0.5))
    label = "IA" if probability >= threshold else "non-IA"
    latency_ms = (time.perf_counter() - start) * 1000.0
    return {
        "label": label,
        "probability_ai": probability,
        "threshold": threshold,
        "model_name": metadata.get("model_name", "binary_ai_ml"),
        "model_version": metadata.get("model_version", ""),
        "pretrained": False,
        "latency_ms": latency_ms,
    }

