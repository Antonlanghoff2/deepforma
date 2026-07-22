from __future__ import annotations

import json
import math
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.model_selection import StratifiedKFold, train_test_split

from deepforma.evaluation.binary_classification_metrics import evaluate_binary_classification


SEED = 42
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def package_versions(packages: Iterable[str]) -> dict[str, str | None]:
    output: dict[str, str | None] = {}
    for name in packages:
        try:
            output[name] = package_version(name)
        except PackageNotFoundError:
            output[name] = None
    return output


def find_repo_root(start: str | Path | None = None) -> Path:
    current = Path(start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / ".git").exists() and (candidate / "data" / "raw").exists():
            return candidate
    raise FileNotFoundError("Impossible de localiser la racine du dépôt.")


def read_dataset(dataset_path: str | Path) -> pd.DataFrame:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier Excel introuvable: {path}")
    try:
        frame = pd.read_excel(path, sheet_name=0, engine="openpyxl")
    except Exception as exc:
        raise RuntimeError(f"Erreur lors du chargement du classeur Excel {path}: {exc}") from exc
    return frame


def normalize_binary_labels(frame: pd.DataFrame, label_column: str) -> pd.Series:
    values = frame[label_column]
    if values.isna().any():
        raise ValueError(f"La colonne cible '{label_column}' contient des valeurs manquantes.")
    try:
        normalized = pd.to_numeric(values, errors="raise").astype(int)
    except Exception as exc:
        raise ValueError(f"Impossible de convertir '{label_column}' en entier binaire.") from exc
    uniques = sorted(normalized.unique().tolist())
    if uniques != [0, 1]:
        raise ValueError(f"Labels invalides détectés dans '{label_column}': {uniques}. Seules les valeurs 0 et 1 sont autorisées.")
    return normalized


def audit_dataset(frame: pd.DataFrame, *, text_column: str, label_column: str) -> dict[str, Any]:
    labels = normalize_binary_labels(frame, label_column)
    text = frame[text_column].astype(str).fillna("")
    text_clean = text.astype(str).str.strip()
    char_length = text_clean.str.len()
    word_length = text_clean.str.split().str.len()

    duplicates_full = frame.duplicated().sum()
    duplicate_texts = frame.duplicated(subset=[text_column]).sum()
    contradictory = frame.groupby(text_column)[label_column].nunique(dropna=False)
    contradictory = contradictory[contradictory > 1]

    missing_by_column = frame.isna().sum().astype(int).to_dict()
    label_distribution = labels.value_counts().sort_index().astype(int).to_dict()

    short_thresholds = {
        "len_chars_lt_5": int((char_length < 5).sum()),
        "len_words_lt_3": int((word_length < 3).sum()),
        "len_chars_le_10": int((char_length <= 10).sum()),
        "len_words_le_3": int((word_length <= 3).sum()),
    }

    return {
        "n_rows": int(len(frame)),
        "n_columns": int(frame.shape[1]),
        "columns": list(frame.columns),
        "text_column": text_column,
        "label_column": label_column,
        "label_values": sorted(labels.unique().tolist()),
        "label_distribution": label_distribution,
        "label_distribution_ratio": {str(k): float(v / len(frame)) for k, v in label_distribution.items()},
        "missing_by_column": missing_by_column,
        "exact_duplicates": int(duplicates_full),
        "duplicate_text_values": int(duplicate_texts),
        "contradictory_labels_by_text": int(len(contradictory)),
        "short_text_counts": short_thresholds,
        "length_stats_chars": {
            "min": float(char_length.min()),
            "mean": float(char_length.mean()),
            "median": float(char_length.median()),
            "max": float(char_length.max()),
            "p05": float(char_length.quantile(0.05)),
            "p10": float(char_length.quantile(0.10)),
            "p25": float(char_length.quantile(0.25)),
            "p75": float(char_length.quantile(0.75)),
            "p90": float(char_length.quantile(0.90)),
            "p95": float(char_length.quantile(0.95)),
        },
        "length_stats_words": {
            "min": float(word_length.min()),
            "mean": float(word_length.mean()),
            "median": float(word_length.median()),
            "max": float(word_length.max()),
            "p05": float(word_length.quantile(0.05)),
            "p10": float(word_length.quantile(0.10)),
            "p25": float(word_length.quantile(0.25)),
            "p75": float(word_length.quantile(0.75)),
            "p90": float(word_length.quantile(0.90)),
            "p95": float(word_length.quantile(0.95)),
        },
    }


def enrich_with_ids(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    enriched = enriched.reset_index(drop=False).rename(columns={"index": "source_index"})
    enriched["record_id"] = enriched["source_index"].map(lambda value: f"row_{int(value):04d}")
    return enriched


def create_stratified_splits(
    frame: pd.DataFrame,
    *,
    label_column: str,
    seed: int = SEED,
) -> pd.DataFrame:
    working = frame.copy()
    train_frame, temp_frame = train_test_split(
        working,
        test_size=TEST_FRACTION + VALIDATION_FRACTION,
        random_state=seed,
        stratify=working[label_column],
    )
    relative_validation = VALIDATION_FRACTION / (VALIDATION_FRACTION + TEST_FRACTION)
    validation_frame, test_frame = train_test_split(
        temp_frame,
        test_size=1 - relative_validation,
        random_state=seed,
        stratify=temp_frame[label_column],
    )
    train_frame = train_frame.copy()
    validation_frame = validation_frame.copy()
    test_frame = test_frame.copy()
    train_frame["split"] = "train"
    validation_frame["split"] = "validation"
    test_frame["split"] = "test"
    split_frame = pd.concat([train_frame, validation_frame, test_frame], ignore_index=True)
    return split_frame


def load_or_create_splits(
    frame: pd.DataFrame,
    *,
    label_column: str,
    seed: int,
    split_path: Path,
) -> pd.DataFrame:
    split_path.parent.mkdir(parents=True, exist_ok=True)
    if split_path.exists():
        splits = pd.read_csv(split_path)
        required = {"source_index", "split"}
        if not required.issubset(set(splits.columns)):
            raise ValueError(f"Le fichier de split existe mais n'a pas le bon format: {split_path}")
        merged = frame.merge(splits[["source_index", "split"]], on="source_index", how="left", validate="one_to_one")
        if merged["split"].isna().any():
            raise ValueError("Le fichier de split ne couvre pas toutes les lignes du dataset.")
        return merged

    split_frame = create_stratified_splits(frame, label_column=label_column, seed=seed)
    split_frame[["source_index", "split"]].to_csv(split_path, index=False, encoding="utf-8")
    return split_frame


def split_frames(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {name: frame.loc[frame["split"] == name].copy() for name in ["train", "validation", "test"]}


def summarize_split_sizes(frame: pd.DataFrame, label_column: str) -> dict[str, Any]:
    sizes = frame["split"].value_counts().to_dict()
    labels = {
        split: frame.loc[frame["split"] == split, label_column].value_counts().sort_index().to_dict()
        for split in ["train", "validation", "test"]
    }
    return {"sizes": sizes, "by_label": labels}


def threshold_grid(start: float = 0.05, end: float = 0.95, step: float = 0.01) -> np.ndarray:
    values = np.round(np.arange(start, end + step / 2.0, step), 2)
    return values[(values >= start) & (values <= end)]


ERROR_ANALYSIS_COLUMNS = [
    'source_index',
    'record_id',
    'split',
    'text',
    'true_label',
    'predicted_label',
    'probability_ia',
    'threshold',
    'error_type',
]


def error_analysis(frame: pd.DataFrame, scores: np.ndarray, threshold: float) -> pd.DataFrame:
    frame = frame.copy()
    if 'is_ai' not in frame.columns:
        raise KeyError("La colonne 'is_ai' est obligatoire pour l'analyse des erreurs.")

    scores_array = np.asarray(scores, dtype=float).reshape(-1)
    if len(frame) != len(scores_array):
        raise ValueError(
            f"Nombre de lignes et de scores incompatibles : {len(frame)} contre {len(scores_array)}"
        )

    if 'source_index' not in frame.columns:
        frame['source_index'] = frame.index
    for column in ('record_id', 'split', 'text'):
        if column not in frame.columns:
            frame[column] = pd.NA

    true_labels = pd.to_numeric(frame['is_ai'], errors='raise').astype(int)
    score_series = pd.Series(scores_array, index=frame.index, name='probability_ia')
    prediction_series = pd.Series((scores_array >= threshold).astype(int), index=frame.index, name='predicted_label')

    error_mask = prediction_series.ne(true_labels)
    errors = frame.loc[error_mask].copy()
    if errors.empty:
        return pd.DataFrame(columns=ERROR_ANALYSIS_COLUMNS)

    errors['true_label'] = errors['is_ai'].map({0: 'non-IA', 1: 'IA'})
    errors['predicted_label'] = prediction_series.loc[errors.index].map({0: 'non-IA', 1: 'IA'})
    errors['probability_ia'] = score_series.loc[errors.index].to_numpy()
    errors['threshold'] = float(threshold)
    errors['error_type'] = np.where(pd.to_numeric(errors['is_ai'], errors='raise').astype(int) == 1, 'faux négatif', 'faux positif')
    return (
        errors[ERROR_ANALYSIS_COLUMNS]
        .sort_values('probability_ia', ascending=False)
        .reset_index(drop=True)
    )


build_error_analysis = error_analysis


def _score_row(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, float | None]:
    report = evaluate_binary_classification(y_true, y_score, threshold=threshold, model_name="threshold_scan")
    metrics = report.metrics
    return {
        "threshold": float(threshold),
        "accuracy": metrics.get("accuracy"),
        "balanced_accuracy": metrics.get("balanced_accuracy"),
        "precision_ia": metrics.get("precision_ia"),
        "recall_ia": metrics.get("recall_ia"),
        "f1_ia": metrics.get("f1_ia"),
        "precision_non_ia": metrics.get("precision_non_ia"),
        "recall_non_ia": metrics.get("recall_non_ia"),
        "f1_non_ia": metrics.get("f1_non_ia"),
        "precision_macro": float(report.classification_report["macro avg"]["precision"]),
        "recall_macro": float(report.classification_report["macro avg"]["recall"]),
        "f1_macro": metrics.get("f1_macro"),
        "f1_weighted": metrics.get("f1_weighted"),
        "roc_auc": metrics.get("roc_auc"),
        "pr_auc": metrics.get("pr_auc"),
        "mcc": metrics.get("mcc"),
        "cohen_kappa": metrics.get("cohen_kappa"),
        "log_loss": metrics.get("log_loss"),
        "brier_score": metrics.get("brier_score"),
        "specificity": metrics.get("recall_non_ia"),
        "negative_predictive_value": _negative_predictive_value(report.confusion_counts),
        "tp": report.confusion_counts["tp"],
        "fp": report.confusion_counts["fp"],
        "tn": report.confusion_counts["tn"],
        "fn": report.confusion_counts["fn"],
    }


def _negative_predictive_value(confusion_counts: dict[str, int]) -> float | None:
    tn = confusion_counts["tn"]
    fn = confusion_counts["fn"]
    denom = tn + fn
    if denom == 0:
        return None
    return float(tn / denom)


def build_threshold_table(
    y_true: Iterable[int],
    y_score: Iterable[float],
    *,
    thresholds: Iterable[float] | None = None,
) -> pd.DataFrame:
    y_true_arr = np.asarray(list(y_true), dtype=int)
    y_score_arr = np.asarray(list(y_score), dtype=float)
    grid = np.asarray(list(thresholds if thresholds is not None else threshold_grid()), dtype=float)
    rows = [_score_row(y_true_arr, y_score_arr, float(threshold)) for threshold in grid]
    return pd.DataFrame(rows)


def choose_threshold_from_validation(validation_metrics: pd.DataFrame) -> tuple[float, pd.Series]:
    required = {"threshold", "f1_macro", "f1_ia", "recall_ia"}
    missing = required.difference(validation_metrics.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes pour le choix du seuil: {sorted(missing)}")
    table = validation_metrics.copy().reset_index(drop=True)
    table["threshold_distance"] = (table["threshold"] - 0.5).abs()
    table = table.sort_values(
        by=["f1_macro", "f1_ia", "recall_ia", "threshold_distance", "threshold"],
        ascending=[False, False, False, True, True],
    ).reset_index(drop=True)
    best = table.iloc[0]
    return float(best["threshold"]), best


def summarize_metric_table(threshold_table: pd.DataFrame, *, metric_columns: list[str]) -> pd.DataFrame:
    summary_rows = []
    for metric in metric_columns:
        series = threshold_table[metric].astype(float)
        summary_rows.append(
            {
                "metric": metric,
                "mean": float(series.mean()),
                "std": float(series.std(ddof=0)),
                "min": float(series.min()),
                "max": float(series.max()),
            }
        )
    return pd.DataFrame(summary_rows)


def evaluate_with_threshold(
    y_true: Iterable[int],
    y_score: Iterable[float],
    *,
    threshold: float,
    model_name: str,
    inference_time_seconds: float | None = None,
    model_size_bytes: int | None = None,
) -> tuple[dict[str, Any], Any]:
    y_true_arr = np.asarray(list(y_true), dtype=int)
    y_score_arr = np.asarray(list(y_score), dtype=float)
    report = evaluate_binary_classification(
        y_true_arr,
        y_score_arr,
        threshold=float(threshold),
        model_name=model_name,
        inference_time_ms=None if inference_time_seconds is None else float(inference_time_seconds * 1000.0),
        model_size_bytes=model_size_bytes,
    )
    metrics = dict(report.metrics)
    metrics["precision_macro"] = float(report.classification_report["macro avg"]["precision"])
    metrics["recall_macro"] = float(report.classification_report["macro avg"]["recall"])
    counts = report.confusion_counts
    metrics.update(
        {
            "specificity": float(counts["tn"] / max(counts["tn"] + counts["fp"], 1)),
            "negative_predictive_value": float(counts["tn"] / max(counts["tn"] + counts["fn"], 1)),
            "tp": int(counts["tp"]),
            "fp": int(counts["fp"]),
            "tn": int(counts["tn"]),
            "fn": int(counts["fn"]),
            "threshold": float(threshold),
            "inference_time_seconds": None if inference_time_seconds is None else float(inference_time_seconds),
            "inference_time_ms": None if inference_time_seconds is None else float(inference_time_seconds * 1000.0),
            "latency_ms_per_sample": None
            if inference_time_seconds is None or len(y_true_arr) == 0
            else float(inference_time_seconds * 1000.0 / len(y_true_arr)),
            "model_size_bytes": model_size_bytes,
            "model_size_mb": None if model_size_bytes is None else float(model_size_bytes / (1024 * 1024)),
        }
    )
    metrics["classification_report"] = report.classification_report
    metrics["confusion_counts"] = counts
    metrics["per_class"] = report.per_class
    metrics["probability_stats"] = report.probability_stats
    metrics["probability_distribution"] = report.probability_distribution
    metrics["roc_curve"] = report.roc_curve
    metrics["precision_recall_curve"] = report.precision_recall_curve
    metrics["threshold_curve"] = report.threshold_curve
    metrics["calibration_curve"] = report.calibration_curve
    metrics["alerts"] = report.alerts
    return metrics, report


def report_to_flat_row(model_name: str, metrics: dict[str, Any]) -> dict[str, Any]:
    row = {
        "model_name": model_name,
        "accuracy": metrics.get("accuracy"),
        "balanced_accuracy": metrics.get("balanced_accuracy"),
        "precision_IA": metrics.get("precision_ia"),
        "recall_IA": metrics.get("recall_ia"),
        "f1_IA": metrics.get("f1_ia"),
        "precision_non_IA": metrics.get("precision_non_ia"),
        "recall_non_IA": metrics.get("recall_non_ia"),
        "f1_non_IA": metrics.get("f1_non_ia"),
        "precision_macro": metrics.get("precision_macro"),
        "recall_macro": metrics.get("recall_macro"),
        "f1_macro": metrics.get("f1_macro"),
        "f1_weighted": metrics.get("f1_weighted"),
        "roc_auc": metrics.get("roc_auc"),
        "pr_auc": metrics.get("pr_auc"),
        "mcc": metrics.get("mcc"),
        "cohen_kappa": metrics.get("cohen_kappa"),
        "log_loss": metrics.get("log_loss"),
        "brier_score": metrics.get("brier_score"),
        "specificity": metrics.get("specificity"),
        "negative_predictive_value": metrics.get("negative_predictive_value"),
        "tp": metrics.get("tp"),
        "fp": metrics.get("fp"),
        "tn": metrics.get("tn"),
        "fn": metrics.get("fn"),
        "training_time_seconds": metrics.get("training_time_seconds"),
        "inference_time_seconds": metrics.get("inference_time_seconds"),
        "latency_ms_per_sample": metrics.get("latency_ms_per_sample"),
        "model_size_mb": metrics.get("model_size_mb"),
        "threshold": metrics.get("threshold"),
    }
    return row


def model_size_bytes(path: str | Path) -> int:
    candidate = Path(path)
    if candidate.is_file():
        return int(candidate.stat().st_size)
    total = 0
    for subpath in candidate.rglob("*"):
        if subpath.is_file():
            total += subpath.stat().st_size
    return int(total)


def save_json(path: str | Path, payload: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return output


def save_dataframe(path: str | Path, frame: pd.DataFrame) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False, encoding="utf-8")
    return output


def extract_logistic_top_features(pipeline: Any, *, top_n: int = 50) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_union = pipeline.named_steps["features"]
    classifier = pipeline.named_steps["classifier"]
    estimator = classifier
    if hasattr(classifier, "coef_"):
        estimator = classifier
    elif hasattr(classifier, "base_estimator") and hasattr(classifier.base_estimator, "coef_"):
        estimator = classifier.base_estimator
    if not hasattr(estimator, "coef_"):
        empty = pd.DataFrame(columns=["feature", "weight"])
        return empty, empty

    feature_names = feature_union.get_feature_names_out()
    coefficients = np.asarray(estimator.coef_).ravel()
    order = np.argsort(coefficients)
    negative = pd.DataFrame({"feature": feature_names[order[:top_n]], "weight": coefficients[order[:top_n]]})
    positive = pd.DataFrame({"feature": feature_names[order[::-1][:top_n]], "weight": coefficients[order[::-1][:top_n]]})
    return positive, negative


def plot_class_distribution(frame: pd.DataFrame, *, label_column: str, title: str, output_path: str | Path | None = None) -> Path | None:
    counts = frame[label_column].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 4))
    counts.plot(kind="bar", ax=ax, color=["#4c78a8", "#f58518"])
    ax.set_xticklabels(["non-IA", "IA"], rotation=0)
    ax.set_ylabel("Nombre d'exemples")
    ax.set_title(title)
    fig.tight_layout()
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return path
    plt.show()
    return None


def plot_length_distribution(lengths: pd.Series, *, title: str, output_path: str | Path | None = None) -> Path | None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(lengths, bins=30, color="#4c78a8", alpha=0.85)
    ax.set_xlabel("Longueur")
    ax.set_ylabel("Fréquence")
    ax.set_title(title)
    fig.tight_layout()
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return path
    plt.show()
    return None


def plot_confusion_matrix(report: Any, *, title: str, output_path: str | Path | None = None) -> Path | None:
    matrix = np.array(
        [
            [report.confusion_counts["tn"], report.confusion_counts["fp"]],
            [report.confusion_counts["fn"], report.confusion_counts["tp"]],
        ]
    )
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], labels=["non-IA", "IA"])
    ax.set_yticks([0, 1], labels=["non-IA", "IA"])
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Réel")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return path
    plt.show()
    return None


def plot_roc_pr_calibration(report: Any, *, prefix: str, output_dir: str | Path | None = None) -> dict[str, Path | None]:
    out: dict[str, Path | None] = {}
    if output_dir is None:
        output = None
    else:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

    def _save(fig: plt.Figure, name: str) -> Path | None:
        if output is None:
            plt.show()
            return None
        path = output / name
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return path

    fig, ax = plt.subplots(figsize=(6, 5))
    roc = report.roc_curve
    if roc["fpr"] and roc["tpr"]:
        ax.plot(roc["fpr"], roc["tpr"], label=f"AUC={report.metrics.get('roc_auc'):.4f}" if report.metrics.get("roc_auc") is not None else "ROC")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"Courbe ROC - {prefix}")
    ax.legend()
    fig.tight_layout()
    out["roc"] = _save(fig, f"{prefix}_roc.png")

    fig, ax = plt.subplots(figsize=(6, 5))
    pr = report.precision_recall_curve
    if pr["precision"] and pr["recall"]:
        ax.plot(pr["recall"], pr["precision"], label=f"AP={report.metrics.get('pr_auc'):.4f}" if report.metrics.get("pr_auc") is not None else "PR")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Courbe précision-rappel - {prefix}")
    ax.legend()
    fig.tight_layout()
    out["pr"] = _save(fig, f"{prefix}_precision_recall.png")

    fig, ax = plt.subplots(figsize=(6, 5))
    cal = report.calibration_curve
    if cal["predicted"] and cal["observed"]:
        ax.plot(cal["predicted"], cal["observed"], marker="o", label="Calibration")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    ax.set_xlabel("Probabilité moyenne")
    ax.set_ylabel("Taux observé")
    ax.set_title(f"Calibration - {prefix}")
    ax.legend()
    fig.tight_layout()
    out["calibration"] = _save(fig, f"{prefix}_calibration.png")
    return out


def plot_probability_distribution(y_score: Iterable[float], *, title: str, output_path: str | Path | None = None) -> Path | None:
    scores = np.asarray(list(y_score), dtype=float)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(scores, bins=30, color="#72b7b2", alpha=0.9)
    ax.set_xlabel("Probabilité IA")
    ax.set_ylabel("Fréquence")
    ax.set_title(title)
    fig.tight_layout()
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return path
    plt.show()
    return None




def build_tfidf_logistic_regression_pipeline(*, C: float = 1.0, seed: int = SEED) -> Pipeline:
    features = FeatureUnion(
        [
            (
                "word_tfidf",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.98,
                    max_features=50_000,
                    sublinear_tf=True,
                    strip_accents="unicode",
                    lowercase=True,
                ),
            ),
            (
                "char_tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=50_000,
                    sublinear_tf=True,
                    strip_accents="unicode",
                    lowercase=True,
                ),
            ),
        ]
    )
    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=3000,
        random_state=seed,
        solver="liblinear",
        C=float(C),
    )
    return Pipeline([("features", features), ("classifier", classifier)])

def plot_threshold_metrics(threshold_table: pd.DataFrame, *, title: str, output_path: str | Path | None = None) -> Path | None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for metric, color in [("f1_macro", "#4c78a8"), ("f1_ia", "#f58518"), ("recall_ia", "#54a24b"), ("precision_ia", "#e45756")]:
        if metric in threshold_table.columns:
            ax.plot(threshold_table["threshold"], threshold_table[metric], label=metric, color=color)
    ax.set_xlabel("Seuil")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return path
    plt.show()
    return None


def write_metadata(path: str | Path, payload: dict[str, Any]) -> Path:
    payload = dict(payload)
    payload.setdefault("generated_at", utc_now_iso())
    return save_json(path, payload)

