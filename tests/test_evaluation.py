from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np

from src.evaluation import (
    RecommendationCase,
    evaluate_binary_classification,
    evaluate_multilabel_classification,
    evaluate_recommendation,
    evaluate_skill_extraction,
    optimize_binary_threshold,
    optimize_thresholds,
    save_thresholds_json,
    write_evaluation_artifacts,
)
from src.evaluation.skill_extraction import normalize_skill_text


def _auth_headers(username: str = "anton", password: str = "deepforma") -> dict[str, str]:
    token = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
    return {"Authorization": f"Basic {token}"}


def test_binary_metrics_and_serialization(tmp_path):
    y_true = np.array([0, 1, 1, 0])
    y_score = np.array([0.1, 0.9, 0.8, 0.2])
    report = evaluate_binary_classification(y_true, y_score)
    assert report.metrics.accuracy == 1.0
    assert report.metrics.macro_f1 == 1.0
    assert report.per_class["ia"]["f1"] == 1.0
    assert report.total_examples == 4
    assert report.real_positive_count == 2
    assert report.predicted_positive_count == 2
    assert report.confusion_counts == {"tn": 2, "fp": 0, "fn": 0, "tp": 2}
    assert report.probability_stats["max"] == 0.9

    out = write_evaluation_artifacts(report, tmp_path, model_name="binary_ia", task="binary")
    assert (out / "report.json").exists()
    assert (out / "summary.csv").exists()
    assert (out / "report.html").exists()


def test_binary_only_zero_predictions_reports_alerts():
    report = evaluate_binary_classification([0, 1, 1, 0], [0.1, 0.2, 0.3, 0.4])
    assert report.predicted_positive_count == 0
    assert report.predicted_negative_count == 4
    assert any("classe non-IA" in alert for alert in report.alerts)
    assert any("rappel positif est nul" in alert for alert in report.alerts)


def test_binary_only_one_predictions_reports_alerts():
    report = evaluate_binary_classification([0, 1, 1, 0], [0.6, 0.7, 0.8, 0.9])
    assert report.predicted_positive_count == 4
    assert report.predicted_negative_count == 0
    assert any("classe IA" in alert for alert in report.alerts)


def test_binary_threshold_optimization_uses_validation_only():
    validation_true = np.array([0, 1, 1, 0])
    validation_score = np.array([0.10, 0.62, 0.58, 0.20])
    optimized = optimize_binary_threshold(validation_true, validation_score, mode="maximize_f1")
    assert 0.0 <= optimized.threshold <= 1.0
    assert optimized.baseline_metrics["f1"] <= optimized.optimized_metrics["f1"]
    assert optimized.optimized_metrics["f1"] >= optimized.baseline_metrics["f1"]


def test_multilabel_threshold_optimization_and_topk(tmp_path):
    labels = ["skill_a", "skill_b", "skill_c", "skill_d", "skill_e"]
    y_true = np.array([
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
    ])
    y_score = np.array([
        [0.30, 0.15, 0.10, 0.05, 0.02],
        [0.20, 0.60, 0.25, 0.10, 0.05],
        [0.18, 0.57, 0.68, 0.12, 0.08],
        [0.07, 0.10, 0.12, 0.81, 0.09],
    ])
    thresholds = optimize_thresholds(y_true, y_score, labels, mode="maximize_f1")
    assert set(thresholds.thresholds) == set(labels)
    assert thresholds.thresholds["skill_b"] > 0.5

    report = evaluate_multilabel_classification(y_true, y_score, labels, thresholds=thresholds.thresholds)
    assert report.metrics["exact_match_ratio"] == 1.0
    assert report.metrics["micro_f1"] == 1.0
    assert report.precision_at_k["1"] == 1.0
    assert report.recall_at_k["1"] == 1.0
    assert set(report.precision_at_k) == {"1", "3", "5"}
    assert 0.0 <= report.metrics["average_precision_micro"] <= 1.0
    assert report.non_discriminant is False

    thresholds_path = save_thresholds_json(tmp_path / "thresholds.json", thresholds, model_name="demo", version="v1", metric="maximize_f1")
    data = json.loads(thresholds_path.read_text(encoding="utf-8"))
    assert data["model_name"] == "demo"
    assert data["thresholds"]["skill_a"]


def test_multilabel_non_discriminant_and_missing_positive_label():
    labels = ["skill_a", "skill_b"]
    y_true = np.array([
        [1, 0],
        [0, 0],
        [1, 0],
        [0, 0],
    ])
    y_score = np.full((4, 2), 0.5)
    report = evaluate_multilabel_classification(y_true, y_score, labels)
    assert report.non_discriminant is True
    assert report.warnings
    assert report.labels_without_positive_examples == ["skill_b"]
    assert report.labels_never_predicted == []
    assert set(report.labels_always_predicted) == {"skill_a", "skill_b"}
    assert report.per_label[1].average_precision is None
    assert 0.0 <= report.metrics["micro_f1"] <= 0.5


def test_skill_extraction_exact_normalized_and_semantic(monkeypatch, tmp_path):
    model_dir = tmp_path / "embedding-model"
    model_dir.mkdir()

    class FakeEncoder:
        mapping = {
            "python": np.array([1.0, 0.0], dtype=float),
            "machine learning": np.array([0.0, 1.0], dtype=float),
            "apprentissage automatique": np.array([0.0, 1.0], dtype=float),
        }

        def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True, **kwargs):
            vectors = []
            for text in texts:
                key = normalize_skill_text(text)
                vectors.append(self.mapping.get(key, np.array([0.0, 0.0], dtype=float)))
            return np.vstack(vectors)

    import deepforma.cpf.embeddings as embeddings

    monkeypatch.setattr(embeddings, "build_encoder", lambda *_args, **_kwargs: FakeEncoder())
    docs = [
        {
            "document_id": "doc-1",
            "gold_skills": ["Python", "Apprentissage automatique"],
            "predicted_skills": ["Python", "Machine Learning"],
        }
    ]
    report = evaluate_skill_extraction(docs, embedding_model=model_dir, semantic_threshold=0.75)
    assert report.status == "evaluated"
    assert report.exact_match.f1 == 0.5
    assert report.normalized_match.f1 == 0.5
    assert report.semantic_match.f1 == 1.0
    assert report.mean_false_positives_per_document == 0.0
    assert report.mean_missing_skills_per_document == 0.0


def test_skill_extraction_without_gold_dataset(tmp_path):
    docs = [{"document_id": "doc-1", "text": "Sans gold", "predicted_skills": ["Python"]}]
    report = evaluate_skill_extraction(docs, gold_dataset_path=tmp_path / "gold.jsonl")
    assert report.status == "not_evaluated"
    assert report.exact_match is None
    assert report.normalized_match is None
    assert report.semantic_match is None
    assert report.mean_false_positives_per_document is None
    assert report.warnings
    assert "Chemin attendu" in report.warnings[0]


def test_admin_value_formatting_shows_zero_and_none():
    from web_app import _format_eval_value

    assert "—" in str(_format_eval_value(None))
    assert "0.0000" in str(_format_eval_value(0.0))


def test_recommendation_ranking_metrics():
    cases = [
        RecommendationCase(
            case_id="case-1",
            profile="Profil 1",
            owned_skills=["Python"],
            missing_skills=["Machine Learning"],
            target_job="Data Scientist",
            rome_codes=["M1403"],
            territory="93",
            relevant_formations={"form-1": 3},
        ),
        RecommendationCase(
            case_id="case-2",
            profile="Profil 2",
            owned_skills=["SQL"],
            missing_skills=["BI"],
            target_job="Analyste",
            rome_codes=["M1805"],
            territory="75",
            relevant_formations={"form-2": 2},
        ),
    ]
    rankings = {
        "case-1": [{"formation_id": "form-1", "score": 0.91}, {"formation_id": "form-x", "score": 0.10}],
        "case-2": [{"formation_id": "form-2", "score": 0.85}, {"formation_id": "form-y", "score": 0.12}],
    }
    report = evaluate_recommendation(cases, rankings)
    assert report.metrics["precision_at_1"] == 1.0
    assert report.metrics["recall_at_1"] == 1.0
    assert report.metrics["mrr"] == 1.0
    assert report.metrics["map"] == 1.0
    assert report.metrics["ndcg_at_5"] == 1.0
    assert report.catalog_coverage == 1.0


def test_report_serialization(tmp_path):
    labels = ["skill_a"]
    y_true = np.array([[1], [0]])
    y_score = np.array([[0.9], [0.1]])
    report = evaluate_multilabel_classification(y_true, y_score, labels)
    out = write_evaluation_artifacts(report, tmp_path, model_name="ml", task="multilabel")
    payload = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert payload["task"] == "multilabel"
    assert (out / "summary.csv").exists()
    assert (out / "report.html").exists()
