from __future__ import annotations

import numpy as np
import pytest

from deepforma.training.binary_ai_common import (
    assess_text_sufficiency,
    classify_binary_probability,
    positive_class_probability,
)


class _FakeEstimator:
    def __init__(self, classes, probabilities):
        self.classes_ = np.asarray(classes)
        self._probabilities = np.asarray(probabilities, dtype=float)

    def predict_proba(self, texts):
        return self._probabilities[: len(list(texts))]


def test_positive_class_probability_uses_label_one_column_even_when_classes_are_inverted():
    estimator = _FakeEstimator([1, 0], [[0.80, 0.20], [0.10, 0.90]])
    scores = positive_class_probability(estimator.predict_proba(["a", "b"]), estimator.classes_)
    assert scores.tolist() == [0.80, 0.10]


def test_positive_class_probability_raises_when_positive_class_is_missing():
    with pytest.raises(ValueError, match="Classe positive absente"):
        positive_class_probability(np.array([[0.2, 0.8]]), [0, 2])


@pytest.mark.parametrize(
    "text,expected_sufficient",
    [
        ("", False),
        ("x", False),
        ("IA", False),
        ("Python", True),
        ("#@!%", False),
        ("qz9x2k7m", False),
        ("analyse de données", True),
    ],
)
def test_text_sufficiency_policy(text, expected_sufficient):
    result = assess_text_sufficiency(text)
    assert result.sufficient is expected_sufficient
    if not expected_sufficient:
        assert result.reason == "texte_insuffisant"


def test_binary_probability_policy_marks_zone_incerte():
    result = classify_binary_probability(0.4935)
    assert result["label"] == "indetermine"
    assert result["requires_review"] is True
    assert result["status"] == "low_confidence"
    assert result["warning"] == "Scores proches de 0,5 : classification peu fiable."


def test_predict_binary_ai_ml_uses_positive_class_index(monkeypatch):
    import deepforma.training.binary_ai_ml as binary_ai_ml

    class _FakeClassifier:
        def __init__(self):
            self.classes_ = np.asarray([1, 0])

    class _FakePipeline:
        def __init__(self):
            self.named_steps = {"classifier": _FakeClassifier()}

        def predict_proba(self, texts):
            rows = [[0.83, 0.17]]
            return np.asarray(rows * len(list(texts)), dtype=float)

    fake_pipeline = _FakePipeline()
    monkeypatch.setattr(binary_ai_ml, "load_binary_ai_ml", lambda model_dir: (fake_pipeline, {"threshold": 0.5, "model_name": "binary_ai_ml", "model_version": "v1"}))

    result = binary_ai_ml.predict_binary_ai_ml("Formation IA", model_dir="/tmp/unused")

    assert result["probability_ai"] == pytest.approx(0.83)
    assert result["label"] == "IA"
