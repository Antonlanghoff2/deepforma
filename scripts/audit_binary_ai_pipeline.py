#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from common.text import clean_text, normalize_for_match
from deepforma.evaluation.binary_classification_metrics import evaluate_binary_classification, optimize_binary_threshold
from deepforma.training.binary_ai_ml import load_binary_ai_ml
from deepforma.training.binary_ai_textcnn import encode_text, load_binary_ai_textcnn


LOGGER = logging.getLogger("audit_binary_ai_pipeline")

SOURCE_DATASET = Path("data/raw/dataset_competences_IA_annotees.xlsx")
SPLITS_DIR = Path("data/training/binary_ai")
OUTPUT_DIR = Path("artifacts/ml")
DOMAIN_SOURCE = Path("data/processed/dataset_a_verifier.csv")
MODEL_DIRS = {
    "binary_ai_ml": Path("models/binary_ai_ml"),
    "binary_ai_textcnn": Path("models/binary_ai_textcnn"),
}
AMBIGUOUS_KEYWORDS = [
    "python",
    "pandas",
    "numpy",
    "statistique",
    "statistiques",
    "analyse de données",
    "analyse de donnees",
    "visualisation",
    "automatisation",
    "gestion de projet",
    "traitement de données",
    "traitement de donnees",
    "sql",
    "power bi",
    "tableau de bord",
    "machine learning",
    "deep learning",
]


@dataclass(frozen=True, slots=True)
class ModelVariant:
    name: str
    feature_set: str
    classifier: str
    class_weight: str | None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit du pipeline binaire IA/non-IA")
    parser.add_argument("--source-dataset", type=Path, default=SOURCE_DATASET)
    parser.add_argument("--splits-dir", type=Path, default=SPLITS_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--skip-training-comparison", action="store_true")
    return parser


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_source_dataset(path: Path) -> pd.DataFrame:
    frame = pd.read_excel(path).copy()
    frame["IA"] = frame["IA"].astype(int)
    frame["compétence"] = frame["compétence"].astype(str)
    frame["normalized_competence"] = frame["compétence"].map(lambda value: normalize_for_match(clean_text(value)))
    frame["text_length"] = frame["compétence"].str.len().astype(int)
    return frame


def _top_terms(frame: pd.DataFrame, label: int, *, top_n: int = 20) -> list[dict[str, Any]]:
    texts = frame.loc[frame["IA"] == label, "compétence"].astype(str).tolist()
    if not texts:
        return []
    vectorizer = CountVectorizer(lowercase=True, ngram_range=(1, 2), min_df=5)
    matrix = vectorizer.fit_transform(texts)
    counts = np.asarray(matrix.sum(axis=0)).ravel()
    features = vectorizer.get_feature_names_out()
    order = np.argsort(counts)[::-1][:top_n]
    return [{"term": str(features[idx]), "count": int(counts[idx])} for idx in order]


def _shared_terms(frame: pd.DataFrame, *, top_n: int = 20) -> list[dict[str, Any]]:
    non_ia = frame.loc[frame["IA"] == 0, "compétence"].astype(str).tolist()
    ia = frame.loc[frame["IA"] == 1, "compétence"].astype(str).tolist()
    if not non_ia or not ia:
        return []
    vectorizer = CountVectorizer(lowercase=True, ngram_range=(1, 2), min_df=5)
    non_matrix = vectorizer.fit_transform(non_ia)
    features = vectorizer.get_feature_names_out()
    non_counts = np.asarray(non_matrix.sum(axis=0)).ravel()
    ia_matrix = CountVectorizer(lowercase=True, ngram_range=(1, 2), min_df=5, vocabulary=vectorizer.vocabulary_).fit_transform(ia)
    ia_counts = np.asarray(ia_matrix.sum(axis=0)).ravel()
    shared = []
    for term, non_count, ia_count in zip(features, non_counts, ia_counts):
        if non_count > 0 and ia_count > 0:
            shared.append(
                {
                    "term": str(term),
                    "count_non_ia": int(non_count),
                    "count_ia": int(ia_count),
                    "shared_score": int(min(non_count, ia_count)),
                }
            )
    shared.sort(key=lambda row: (-row["shared_score"], -(row["count_non_ia"] + row["count_ia"]), row["term"]))
    return shared[:top_n]


def _pick_ambiguous_rows(frame: pd.DataFrame, limit: int = 50) -> pd.DataFrame:
    lower = frame["compétence"].str.lower()
    mask = lower.apply(lambda text: any(keyword in text for keyword in AMBIGUOUS_KEYWORDS))
    ambiguous = frame.loc[mask].copy()
    if len(ambiguous) < limit:
        fallback = frame.sort_values(["text_length", "compétence"]).head(limit)
        ambiguous = pd.concat([ambiguous, fallback], ignore_index=False)
    ambiguous = ambiguous.drop_duplicates(subset=["compétence", "IA"]).head(limit).copy()
    ambiguous["review_status"] = "manual_review"
    ambiguous["reviewer_comment"] = "Compétence potentiellement ambigue; label conserve."
    return ambiguous


def audit_source_dataset(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    counts = frame.groupby("normalized_competence", dropna=False).size().rename("count")
    duplicates = frame.merge(counts.reset_index(), on="normalized_competence", how="left")
    duplicates = duplicates.loc[duplicates["count"] > 1].copy()

    conflicts = (
        frame.groupby("normalized_competence")
        .agg(count=("compétence", "size"), labels=("IA", lambda values: sorted(set(int(value) for value in values))), sample=("compétence", "first"))
        .reset_index()
    )
    conflicts = conflicts.loc[conflicts["labels"].map(len) > 1].copy()
    conflicts = conflicts.sort_values(["count", "normalized_competence"], ascending=[False, True])

    ambiguous = _pick_ambiguous_rows(frame, 50)
    return duplicates, conflicts, ambiguous


def build_dataset_audit_row(frame: pd.DataFrame, conflicts: pd.DataFrame) -> dict[str, Any]:
    lengths = frame["text_length"]
    missing = frame.isna().sum().to_dict()
    return {
        "rows_total": int(len(frame)),
        "class_distribution_0": int((frame["IA"] == 0).sum()),
        "class_distribution_1": int((frame["IA"] == 1).sum()),
        "missing_compétence": int(missing.get("compétence", 0)),
        "missing_IA": int(missing.get("IA", 0)),
        "missing_catégorie_IA": int(missing.get("catégorie IA", 0)),
        "missing_compétence_IA_associée": int(missing.get("compétence IA associée", 0)),
        "exact_duplicates": int(frame.duplicated().sum()),
        "duplicate_compétence_rows": int(frame["compétence"].duplicated().sum()),
        "conflicting_duplicate_groups": int(len(conflicts)),
        "text_length_mean": float(lengths.mean()),
        "text_length_median": float(lengths.median()),
        "text_length_min": int(lengths.min()),
        "text_length_max": int(lengths.max()),
        "shortest_examples": json.dumps(frame.nsmallest(5, "text_length")["compétence"].tolist(), ensure_ascii=False),
        "longest_examples": json.dumps(frame.nlargest(5, "text_length")["compétence"].tolist(), ensure_ascii=False),
        "top_terms_non_ia": json.dumps(_top_terms(frame, 0), ensure_ascii=False),
        "top_terms_ia": json.dumps(_top_terms(frame, 1), ensure_ascii=False),
        "common_terms": json.dumps(_shared_terms(frame), ensure_ascii=False),
        "warnings": json.dumps(["Aucun doublon contradictoire détecté." if conflicts.empty else "Doublons contradictoires détectés."], ensure_ascii=False),
    }


def _load_split_frames(splits_dir: Path) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for name in ["train", "validation", "test"]:
        frame = pd.read_parquet(splits_dir / f"{name}.parquet").copy()
        frame["text_normalized_audit"] = frame["text"].fillna("").astype(str).map(lambda value: normalize_for_match(clean_text(value)))
        frames[name] = frame
    return frames


def _split_intersections(split_frames: dict[str, pd.DataFrame]) -> dict[str, int]:
    sets = {name: set(frame["text_normalized_audit"].tolist()) for name, frame in split_frames.items()}
    return {
        "train_validation": int(len(sets["train"].intersection(sets["validation"]))),
        "train_test": int(len(sets["train"].intersection(sets["test"]))),
        "validation_test": int(len(sets["validation"].intersection(sets["test"]))),
    }


def _majority_baseline(frame: pd.DataFrame) -> dict[str, Any]:
    majority_label = int(frame["is_ai"].value_counts().idxmax())
    prediction = np.full(len(frame), majority_label, dtype=int)
    matrix = confusion_matrix(frame["is_ai"], prediction, labels=[0, 1])
    return {
        "majority_label": majority_label,
        "accuracy": float(accuracy_score(frame["is_ai"], prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(frame["is_ai"], prediction)),
        "precision_ia": float(precision_score(frame["is_ai"], prediction, zero_division=0)),
        "recall_ia": float(recall_score(frame["is_ai"], prediction, zero_division=0)),
        "f1_ia": float(f1_score(frame["is_ai"], prediction, zero_division=0)),
        "tn": int(matrix[0, 0]),
        "fp": int(matrix[0, 1]),
        "fn": int(matrix[1, 0]),
        "tp": int(matrix[1, 1]),
    }


def _feature_pipeline(feature_set: str, classifier: str, class_weight: str | None) -> Pipeline:
    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        max_features=20_000,
        sublinear_tf=True,
        strip_accents="unicode",
        lowercase=True,
    )
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        max_features=20_000,
        sublinear_tf=True,
        strip_accents="unicode",
        lowercase=True,
    )
    if feature_set == "word":
        features: Any = word_vectorizer
    elif feature_set == "char":
        features = char_vectorizer
    elif feature_set == "union":
        features = FeatureUnion([("word", word_vectorizer), ("char", char_vectorizer)])
    else:
        raise ValueError(feature_set)

    if classifier == "logistic":
        estimator: Any = LogisticRegression(C=1.0, class_weight=class_weight, max_iter=2000, random_state=42, solver="liblinear")
    elif classifier == "linearsvc":
        base = LinearSVC(class_weight=class_weight, random_state=42)
        estimator = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3)
    else:
        raise ValueError(classifier)

    return Pipeline([("features", features), ("classifier", estimator)])


def _positive_class_index(estimator: Any) -> int:
    classes = list(getattr(estimator, "classes_", []))
    if 1 not in classes:
        raise ValueError(f"Classe positive absente: {classes}")
    return classes.index(1)


def _predict_scores(pipeline: Pipeline, texts: Iterable[str]) -> np.ndarray:
    probabilities = pipeline.predict_proba(list(texts))
    classifier = pipeline.named_steps["classifier"]
    positive_index = _positive_class_index(classifier)
    return np.asarray(probabilities[:, positive_index], dtype=float)


def _fit_and_evaluate_variants(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    variants = [
        ModelVariant("A_word_logreg_none", "word", "logistic", None),
        ModelVariant("A_word_logreg_balanced", "word", "logistic", "balanced"),
        ModelVariant("B_char_logreg_none", "char", "logistic", None),
        ModelVariant("B_char_logreg_balanced", "char", "logistic", "balanced"),
        ModelVariant("C_union_logreg_none", "union", "logistic", None),
        ModelVariant("C_union_logreg_balanced", "union", "logistic", "balanced"),
        ModelVariant("D_union_linearsvc_none", "union", "linearsvc", None),
        ModelVariant("D_union_linearsvc_balanced", "union", "linearsvc", "balanced"),
    ]
    rows: list[dict[str, Any]] = []
    train_texts = train["text"].fillna("").astype(str).tolist()
    validation_texts = validation["text"].fillna("").astype(str).tolist()
    test_texts = test["text"].fillna("").astype(str).tolist()
    winner: dict[str, Any] | None = None

    for variant in variants:
        pipeline = _feature_pipeline(variant.feature_set, variant.classifier, variant.class_weight)
        pipeline.fit(train_texts, train["is_ai"].astype(int).tolist())
        validation_scores = _predict_scores(pipeline, validation_texts)
        threshold_result = optimize_binary_threshold(validation["is_ai"], validation_scores, mode="maximize_f1")
        test_scores = _predict_scores(pipeline, test_texts)
        test_report = evaluate_binary_classification(test["is_ai"].astype(int), test_scores, threshold=threshold_result.threshold, model_name=variant.name)
        row = {
            "model": variant.name,
            "feature_set": variant.feature_set,
            "classifier": variant.classifier,
            "class_weight": variant.class_weight or "none",
            "validation_threshold": float(threshold_result.threshold),
            "validation_f1_ia": float(threshold_result.optimized_metrics.get("f1_ia") or 0.0),
            "test_accuracy": test_report.metrics.get("accuracy"),
            "test_balanced_accuracy": test_report.metrics.get("balanced_accuracy"),
            "test_precision_ia": test_report.metrics.get("precision_ia"),
            "test_recall_ia": test_report.metrics.get("recall_ia"),
            "test_f1_ia": test_report.metrics.get("f1_ia"),
            "test_f1_macro": test_report.metrics.get("f1_macro"),
            "test_mcc": test_report.metrics.get("mcc"),
            "test_roc_auc": test_report.metrics.get("roc_auc"),
            "test_pr_auc": test_report.metrics.get("pr_auc"),
            "test_fp": test_report.confusion_counts["fp"],
            "test_fn": test_report.confusion_counts["fn"],
            "test_tp": test_report.confusion_counts["tp"],
            "test_tn": test_report.confusion_counts["tn"],
            "test_positive_rate": test_report.predicted_positive_rate,
            "test_real_positive_rate": test_report.positive_rate,
        }
        rows.append(row)
        if winner is None or row["test_f1_ia"] > winner["row"]["test_f1_ia"] or (
            row["test_f1_ia"] == winner["row"]["test_f1_ia"] and row["test_balanced_accuracy"] > winner["row"]["test_balanced_accuracy"]
        ):
            winner = {"row": row, "pipeline": pipeline, "test_scores": test_scores, "validation_threshold": float(threshold_result.threshold)}

    comparison = pd.DataFrame(rows).sort_values(["test_f1_ia", "test_balanced_accuracy"], ascending=False).reset_index(drop=True)
    assert winner is not None
    return comparison, winner


def _plot_confusion_matrix(matrix: np.ndarray, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(4.4, 4))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], labels=["non-IA", "IA"])
    ax.set_yticks([0, 1], labels=["non-IA", "IA"])
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Réel")
    ax.set_title(title)
    for row in range(2):
        for col in range(2):
            ax.text(col, row, str(int(matrix[row, col])), ha="center", va="center", color="black")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _sanity_checks(pipeline: Pipeline, threshold: float) -> pd.DataFrame:
    examples = [
        ("IA évidente", "entrainement d'un réseau de neurones avec PyTorch"),
        ("IA évidente", "construction d'un modèle de machine learning"),
        ("IA évidente", "deploiement d'un modele de deep learning"),
        ("IA évidente", "traitement automatique du langage naturel"),
        ("IA évidente", "creation d'un systeme RAG avec un modele de langage"),
        ("Non-IA évidente", "installation electrique d'un bâtiment"),
        ("Non-IA évidente", "gestion administrative et comptable"),
        ("Non-IA évidente", "maintenance d'un equipement industriel"),
        ("Non-IA évidente", "accueil des clients et gestion des reservations"),
        ("Non-IA évidente", "preparation et cuisson des aliments"),
    ]
    rows = []
    for expected_group, text in examples:
        score = float(_predict_scores(pipeline, [text])[0])
        rows.append(
            {
                "groupe": expected_group,
                "text": text,
                "prediction": "IA" if score >= threshold else "non-IA",
                "probability_ia": score,
                "probability_non_ia": float(1.0 - score),
                "classe_retendue": "IA" if score >= threshold else "non-IA",
                "seuil": float(threshold),
            }
        )
    return pd.DataFrame(rows)


def _domain_validation_sample(pipeline: Pipeline, source_frame: pd.DataFrame, threshold: float, limit: int = 50) -> pd.DataFrame:
    sample = source_frame.copy()
    if "texte_modele" in sample.columns:
        sample["text"] = sample["texte_modele"].fillna("").astype(str)
    else:
        sample["text"] = sample["compétence"].fillna("").astype(str)
    sample["source"] = sample.get("source_file", pd.Series(["source"] * len(sample))).astype(str)
    sample = sample.loc[sample["text"].str.strip().ne("")].copy()
    sample = sample.sort_values("text", key=lambda series: series.str.len(), ascending=False).head(limit).copy()
    scores = _predict_scores(pipeline, sample["text"].tolist())
    sample["true_label"] = ""
    sample["predicted_label"] = np.where(scores >= threshold, "IA", "non-IA")
    sample["probability_ia"] = scores
    sample["review_status"] = "unlabeled"
    sample["reviewer_comment"] = ""
    return sample[["text", "source", "true_label", "predicted_label", "probability_ia", "review_status", "reviewer_comment"]]


def _source_domain_evaluation(pipeline: Pipeline, source_frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    scores = _predict_scores(pipeline, source_frame["compétence"].fillna("").astype(str).tolist())
    predictions = (scores >= threshold).astype(int)
    frame = source_frame.copy()
    frame["probability_ia"] = scores
    frame["predicted_label"] = predictions
    frame["predicted_label_name"] = np.where(predictions == 1, "IA", "non-IA")
    frame["threshold"] = float(threshold)
    frame["prediction_correct"] = frame["predicted_label"] == frame["IA"]
    frame["text_length"] = frame["compétence"].astype(str).str.len().astype(int)
    return frame[["compétence", "IA", "catégorie IA", "compétence IA associée", "predicted_label_name", "probability_ia", "threshold", "prediction_correct", "text_length"]]


def _error_analysis_frame(pipeline: Pipeline, test_frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    scores = _predict_scores(pipeline, test_frame["text"].fillna("").astype(str).tolist())
    classifier = pipeline.named_steps["classifier"]
    top_positive: list[str] = []
    top_negative: list[str] = []
    if hasattr(classifier, "coef_"):
        coefficients = np.asarray(classifier.coef_).ravel()
        feature_names = pipeline.named_steps["features"].get_feature_names_out()
        top_positive = list(feature_names[np.argsort(coefficients)[::-1][:25]])
        top_negative = list(feature_names[np.argsort(coefficients)[:25]])

    frame = test_frame.copy()
    frame["probability_ia"] = scores
    frame["predicted_label"] = np.where(scores >= threshold, 1, 0)
    errors = frame.loc[frame["predicted_label"] != frame["is_ai"]].copy()
    if errors.empty:
        return pd.DataFrame(
            columns=[
                "source_index",
                "text",
                "true_label",
                "predicted_label",
                "probability_ia",
                "threshold",
                "error_type",
                "text_length",
                "influential_terms",
            ]
        )
    errors["source_index"] = errors.index
    errors["true_label"] = errors["is_ai"].map({0: "non-IA", 1: "IA"})
    errors["predicted_label"] = errors["predicted_label"].map({0: "non-IA", 1: "IA"})
    errors["threshold"] = float(threshold)
    errors["error_type"] = np.where(errors["is_ai"] == 1, "faux négatif", "faux positif")
    errors["text_length"] = errors["text"].astype(str).str.len().astype(int)
    if top_positive or top_negative:
        payload = json.dumps({"top_positive": top_positive[:10], "top_negative": top_negative[:10]}, ensure_ascii=False)
        errors["influential_terms"] = [payload for _ in range(len(errors))]
    else:
        errors["influential_terms"] = ""
    return errors[["source_index", "text", "true_label", "predicted_label", "probability_ia", "threshold", "error_type", "text_length", "influential_terms"]].sort_values("probability_ia", ascending=False).reset_index(drop=True)


def _save_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8")


def _current_model_class_logging() -> None:
    if MODEL_DIRS["binary_ai_ml"].exists():
        pipeline, _ = load_binary_ai_ml(MODEL_DIRS["binary_ai_ml"])
        classifier = pipeline.named_steps["classifier"]
        LOGGER.info("binary_ai_ml classes_=%s positive_index=%s", list(classifier.classes_), _positive_class_index(classifier))
    if MODEL_DIRS["binary_ai_textcnn"].exists():
        model, payload, vocab, _ = load_binary_ai_textcnn(MODEL_DIRS["binary_ai_textcnn"])
        LOGGER.info("binary_ai_textcnn threshold=%.4f vocab_size=%d", float(payload.get("threshold", 0.5)), len(vocab))


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    _ensure_dir(args.output_dir)

    source = _load_source_dataset(args.source_dataset)
    duplicates, conflicts, ambiguous = audit_source_dataset(source)
    dataset_audit_row = build_dataset_audit_row(source, conflicts)
    pd.DataFrame([dataset_audit_row]).to_csv(args.output_dir / "dataset_audit.csv", index=False, encoding="utf-8")
    duplicates.to_csv(args.output_dir / "dataset_duplicates.csv", index=False, encoding="utf-8")
    conflicts.to_csv(args.output_dir / "conflicting_duplicates.csv", index=False, encoding="utf-8")
    ambiguous.to_csv(args.output_dir / "ambiguous_annotations.csv", index=False, encoding="utf-8")

    split_frames = _load_split_frames(args.splits_dir)
    split_intersections = _split_intersections(split_frames)
    baseline = _majority_baseline(split_frames["test"])
    pd.DataFrame([baseline | {"split": "test"}]).to_csv(args.output_dir / "baseline_test.csv", index=False, encoding="utf-8")

    summary = {
        "source_dataset": str(args.source_dataset),
        "rows_total": int(len(source)),
        "class_distribution": {"non_ia": int((source["IA"] == 0).sum()), "ia": int((source["IA"] == 1).sum())},
        "split_sizes": {name: int(len(frame)) for name, frame in split_frames.items()},
        "split_intersections": split_intersections,
        "baseline_test": baseline,
        "labels_convention": "0 = non-IA, 1 = IA",
        "ambiguous_sample_size": int(len(ambiguous)),
        "conflicting_duplicates": int(len(conflicts)),
    }

    if not args.skip_training_comparison:
        comparison, winner = _fit_and_evaluate_variants(split_frames["train"], split_frames["validation"], split_frames["test"])
        comparison.to_csv(args.output_dir / "model_comparison.csv", index=False, encoding="utf-8")
        winner_row = winner["row"]
        winner_pipeline: Pipeline = winner["pipeline"]
        winner_threshold = float(winner["validation_threshold"])
        winner_scores = np.asarray(winner["test_scores"], dtype=float)
        winner_frame = split_frames["test"]

        threshold_rows = []
        for threshold in [0.30, 0.40, 0.50, 0.60, 0.70]:
            report = evaluate_binary_classification(winner_frame["is_ai"].astype(int), winner_scores, threshold=threshold, model_name=winner_row["model"])
            threshold_rows.append(
                {
                    "threshold": threshold,
                    "accuracy": report.metrics.get("accuracy"),
                    "balanced_accuracy": report.metrics.get("balanced_accuracy"),
                    "precision_ia": report.metrics.get("precision_ia"),
                    "recall_ia": report.metrics.get("recall_ia"),
                    "f1_ia": report.metrics.get("f1_ia"),
                    "fp": report.confusion_counts["fp"],
                    "fn": report.confusion_counts["fn"],
                }
            )
        pd.DataFrame(threshold_rows).to_csv(args.output_dir / "threshold_comparison.csv", index=False, encoding="utf-8")

        report_05 = evaluate_binary_classification(winner_frame["is_ai"].astype(int), winner_scores, threshold=0.5, model_name=winner_row["model"])
        report_best = evaluate_binary_classification(winner_frame["is_ai"].astype(int), winner_scores, threshold=winner_threshold, model_name=winner_row["model"])
        _plot_confusion_matrix(np.array([[report_05.confusion_counts["tn"], report_05.confusion_counts["fp"]], [report_05.confusion_counts["fn"], report_05.confusion_counts["tp"]]]), args.output_dir / "confusion_matrix_threshold_0_5.png", f"{winner_row['model']} @ 0.5")
        _plot_confusion_matrix(np.array([[report_best.confusion_counts["tn"], report_best.confusion_counts["fp"]], [report_best.confusion_counts["fn"], report_best.confusion_counts["tp"]]]), args.output_dir / "confusion_matrix_validation_threshold.png", f"{winner_row['model']} @ validation threshold")
        _plot_confusion_matrix(np.array([[baseline["tn"], baseline["fp"]], [baseline["fn"], baseline["tp"]]]), args.output_dir / "confusion_matrix_majority_baseline.png", "Majority baseline")

        sanity = _sanity_checks(winner_pipeline, winner_threshold)
        sanity.to_csv(args.output_dir / "sanity_checks.csv", index=False, encoding="utf-8")

        if DOMAIN_SOURCE.exists():
            domain_frame = pd.read_csv(DOMAIN_SOURCE)
            domain_sample = _domain_validation_sample(winner_pipeline, domain_frame, winner_threshold, limit=50)
            domain_sample.to_csv(args.output_dir / "domain_validation_sample.csv", index=False, encoding="utf-8")
            source_domain_eval = _source_domain_evaluation(winner_pipeline, source, threshold=winner_threshold)
            source_domain_eval.to_csv(args.output_dir / "source_domain_evaluation.csv", index=False, encoding="utf-8")
            source_errors = source_domain_eval.loc[~source_domain_eval["prediction_correct"]].copy()
            if not source_errors.empty:
                source_errors["source_index"] = source_errors.index
                source_errors["error_type"] = np.where(source_errors["IA"] == 1, "faux négatif", "faux positif")
                source_errors.to_csv(args.output_dir / "source_domain_error_analysis.csv", index=False, encoding="utf-8")
            else:
                pd.DataFrame(columns=["source_index", "compétence", "IA", "catégorie IA", "compétence IA associée", "predicted_label_name", "probability_ia", "threshold", "prediction_correct", "text_length", "error_type"]).to_csv(args.output_dir / "source_domain_error_analysis.csv", index=False, encoding="utf-8")
            source_scores = _predict_scores(winner_pipeline, source["compétence"].astype(str).tolist())
            source_threshold_rows = []
            for threshold in [0.30, 0.40, 0.50, 0.60, 0.70]:
                report = evaluate_binary_classification(source["IA"].astype(int), source_scores, threshold=threshold, model_name=winner_row["model"])
                source_threshold_rows.append({
                    "threshold": threshold,
                    "accuracy": report.metrics.get("accuracy"),
                    "balanced_accuracy": report.metrics.get("balanced_accuracy"),
                    "precision_ia": report.metrics.get("precision_ia"),
                    "recall_ia": report.metrics.get("recall_ia"),
                    "f1_ia": report.metrics.get("f1_ia"),
                    "fp": report.confusion_counts["fp"],
                    "fn": report.confusion_counts["fn"],
                })
            pd.DataFrame(source_threshold_rows).to_csv(args.output_dir / "source_domain_threshold_comparison.csv", index=False, encoding="utf-8")
            source_report = evaluate_binary_classification(source["IA"].astype(int), source_scores, threshold=winner_threshold, model_name=winner_row["model"])
        else:
            source_domain_eval = None
            source_report = None

        errors = _error_analysis_frame(winner_pipeline, winner_frame, winner_threshold)
        errors.to_csv(args.output_dir / "error_analysis.csv", index=False, encoding="utf-8")

        summary["winner_model"] = winner_row["model"]
        summary["winner_validation_threshold"] = winner_threshold
        summary["winner_test_f1_ia"] = winner_row["test_f1_ia"]
        summary["winner_test_balanced_accuracy"] = winner_row["test_balanced_accuracy"]
        summary["winner_model_comparison"] = winner_row
        summary["sanity_checks_file"] = str(args.output_dir / "sanity_checks.csv")
        summary["domain_validation_sample_file"] = str(args.output_dir / "domain_validation_sample.csv") if DOMAIN_SOURCE.exists() else None
        summary["source_domain_file"] = str(args.output_dir / "source_domain_evaluation.csv") if source_domain_eval is not None else None
        summary["source_domain_threshold_file"] = str(args.output_dir / "source_domain_threshold_comparison.csv") if source_report is not None else None
        summary["source_domain_metrics"] = source_report.metrics if source_report is not None else None
        summary["error_analysis_file"] = str(args.output_dir / "error_analysis.csv")
    else:
        summary["winner_model"] = None

    _current_model_class_logging()
    (args.output_dir / "audit_summary.json").write_text(json.dumps(summary | {"generated_at": pd.Timestamp.now(tz="UTC").isoformat()}, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
