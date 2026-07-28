from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

import scripts.audit_binary_ai_pipeline as audit


class FakeClassifier:
    def __init__(self, classes: list[int]) -> None:
        self.classes_ = np.asarray(classes)
        self.coef_ = np.asarray([[2.0, -1.0, -0.5]])


class FakeFeatures:
    def get_feature_names_out(self):
        return np.asarray(["ia", "machine learning", "gestion"])


class FakePipeline:
    def __init__(self, classes: list[int] | None = None) -> None:
        self.named_steps = {
            "classifier": FakeClassifier(classes or [0, 1]),
            "features": FakeFeatures(),
        }
        self.fit_calls: list[tuple[list[str], list[int]]] = []

    def fit(self, texts, labels):
        self.fit_calls.append((list(texts), list(labels)))
        return self

    def predict_proba(self, texts):
        rows = []
        for text in texts:
            lowered = str(text).lower()
            score_ia = 0.95 if any(keyword in lowered for keyword in ["pytorch", "machine learning", "deep learning", "langage naturel", "rag"]) else 0.05
            if list(self.named_steps["classifier"].classes_) == [1, 0]:
                rows.append([score_ia, 1.0 - score_ia])
            else:
                rows.append([1.0 - score_ia, score_ia])
        return np.asarray(rows, dtype=float)


def test_positive_class_index_and_predict_scores_use_label_one():
    pipeline = FakePipeline(classes=[1, 0])
    scores = audit._predict_scores(pipeline, ["machine learning", "gestion administrative"])

    assert audit._positive_class_index(pipeline.named_steps["classifier"]) == 0
    assert scores.tolist() == [0.95, 0.05]


def test_sanity_checks_marks_ia_examples_high_when_pipeline_is_discriminant():
    pipeline = FakePipeline()
    result = audit._sanity_checks(pipeline, threshold=0.5)

    assert len(result) == 10
    assert result.iloc[:5]["prediction"].tolist() == ["IA"] * 5
    assert result.iloc[5:]["prediction"].tolist() == ["non-IA"] * 5
    assert set(result.columns) == {"groupe", "text", "prediction", "probability_ia", "probability_non_ia", "classe_retendue", "seuil"}


def test_fit_and_evaluate_variants_uses_validation_threshold_only(monkeypatch):
    train = pd.DataFrame({"text": ["a", "b", "c"], "is_ai": [0, 1, 0]})
    validation = pd.DataFrame({"text": ["d", "e"], "is_ai": [1, 0]})
    test = pd.DataFrame({"text": ["f", "g", "h", "i"], "is_ai": [0, 1, 0, 1]})
    threshold_calls: list[int] = []

    def fake_feature_pipeline(*args, **kwargs):
        return FakePipeline()

    def fake_optimize_binary_threshold(y_true, y_score, **kwargs):
        threshold_calls.append(len(list(y_true)))
        return SimpleNamespace(threshold=0.5, optimized_metrics={"f1_ia": 1.0})

    monkeypatch.setattr(audit, "_feature_pipeline", fake_feature_pipeline)
    monkeypatch.setattr(audit, "optimize_binary_threshold", fake_optimize_binary_threshold)

    comparison, winner = audit._fit_and_evaluate_variants(train, validation, test)

    assert len(comparison) == 8
    assert winner["validation_threshold"] == 0.5
    assert threshold_calls == [len(validation)] * 8


def test_split_intersections_helper_detects_overlap():
    train = pd.DataFrame({"text_normalized_audit": ["python", "sql"]})
    validation = pd.DataFrame({"text_normalized_audit": ["sql", "power bi"]})
    test = pd.DataFrame({"text_normalized_audit": ["java"]})

    intersections = audit._split_intersections({"train": train, "validation": validation, "test": test})

    assert intersections == {"train_validation": 1, "train_test": 0, "validation_test": 0}
