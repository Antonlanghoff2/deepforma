"""Tests pour le classifieur multilabel IA."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.ia_classifier import IAClassifier


TAXONOMY_PATH = Path("config/ia_taxonomy_v2.json")


def test_import():
    assert IAClassifier is not None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

LABEL_NAMES_20 = [
    "Automatisation", "Big Data", "Computer Vision",
    "Data Engineering", "Data Science", "Deep Learning",
    "Ethique IA & RGPD", "Gestion de projet IA",
    "IA Generative", "LangChain / Agents RAG",
    "Machine Learning", "MLOps / Deploiement",
    "NLP / Traitement du langage", "No-code / Low-code",
    "Prompt Engineering", "Python", "Reinforcement Learning",
    "Series temporelles", "SQL / Data Engineering", "Visualisation",
]


def _make_config(num_labels=20, label_names=None):
    if label_names is None:
        label_names = LABEL_NAMES_20[:num_labels]
    return {
        "architectures": ["CamembertForSequenceClassification"],
        "num_labels": num_labels,
        "problem_type": "multi_label_classification",
        "id2label": {str(i): lbl for i, lbl in enumerate(label_names)},
        "label2id": {lbl: i for i, lbl in enumerate(label_names)},
        "transformers_version": "5.12.1",
    }


@pytest.fixture
def mock_model_dir(tmp_path):
    """Cree un dossier modele factice (sans tokenizer/weights)."""
    cfg = _make_config()
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps(cfg))
    return model_dir


@pytest.fixture
def mock_model_dir_with_prediction(tmp_path):
    """Cree un dossier modele factice + mock du tokenizer et du modele."""
    cfg = _make_config()
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps(cfg))

    # --- Mock encode ---
    class MockEncoded(dict):
        def __init__(self, batch_size: int, seq_len: int = 8):
            super().__init__(
                input_ids=torch.randint(0, 1000, (batch_size, seq_len)),
                attention_mask=torch.ones(
                    batch_size, seq_len, dtype=torch.long
                ),
            )

        def to(self, _device):
            return self

    # --- Mock tokenizer ---
    mock_tokenizer = MagicMock()
    mock_tokenizer.side_effect = lambda texts, **kw: MockEncoded(
        len(texts) if isinstance(texts, list) else 1
    )

    # --- Mock model output ---
    class MockLogits:
        def __init__(self, batch_size: int, n_labels: int):
            self.logits = torch.randn(batch_size, n_labels)

    # --- Mock model ---
    mock_model = MagicMock()
    mock_model.side_effect = lambda **kw: MockLogits(
        kw.get("input_ids", torch.zeros(1, 8)).shape[0], 20
    )
    mock_model.to.return_value = mock_model

    # Patch the class-level names so _load_model picks them up
    import src.models.ia_classifier as ic

    orig_tokenizer = ic.AutoTokenizer
    orig_model_cls = ic.AutoModelForSequenceClassification
    ic.AutoTokenizer = MagicMock()
    ic.AutoTokenizer.from_pretrained.return_value = mock_tokenizer
    ic.AutoModelForSequenceClassification = MagicMock()
    ic.AutoModelForSequenceClassification.from_pretrained.return_value = mock_model

    yield model_dir

    ic.AutoTokenizer = orig_tokenizer
    ic.AutoModelForSequenceClassification = orig_model_cls


# ---------------------------------------------------------------------------
# Config / load tests  (no model weights needed)
# ---------------------------------------------------------------------------

def test_config_load(mock_model_dir):
    classifier = IAClassifier(
        mock_model_dir, taxonomy_path=TAXONOMY_PATH,
        _load_model_weights=False,
    )
    assert classifier.num_labels == 20
    assert classifier.problem_type == "multi_label_classification"
    assert len(classifier.labels) == 20


def test_labels_match_taxonomy(mock_model_dir):
    classifier = IAClassifier(
        mock_model_dir, taxonomy_path=TAXONOMY_PATH,
        _load_model_weights=False,
    )
    with open(TAXONOMY_PATH) as f:
        taxonomy = json.load(f)
    assert set(classifier.labels) == set(taxonomy["labels"])


def test_thresholds_defaults_to_fallback(mock_model_dir):
    classifier = IAClassifier(
        mock_model_dir, fallback_threshold=0.42,
        _load_model_weights=False,
    )
    assert classifier.fallback_threshold == 0.42
    for label in classifier.labels[:3]:
        assert classifier._get_threshold(label) == 0.42


def test_thresholds_from_file(mock_model_dir):
    thr = {"Machine Learning": 0.3, "Python": 0.25}
    (mock_model_dir / "thresholds.json").write_text(
        json.dumps({"thresholds": thr})
    )
    classifier = IAClassifier(
        mock_model_dir, fallback_threshold=0.50,
        _load_model_weights=False,
    )
    assert classifier._get_threshold("Machine Learning") == 0.3
    assert classifier._get_threshold("Python") == 0.25
    assert classifier._get_threshold("Big Data") == 0.50


def test_missing_config_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="config.json"):
        IAClassifier(tmp_path / "nonexistent", _load_model_weights=False)


def test_incompatible_taxonomy_raises(tmp_path):
    cfg = {
        "num_labels": 3,
        "problem_type": "multi_label_classification",
        "id2label": {"0": "A", "1": "B", "2": "C"},
        "label2id": {"A": 0, "B": 1, "C": 2},
    }
    model_dir = tmp_path / "bad_model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="INCOMPATIBILITE TAXONOMIE"):
        IAClassifier(
            model_dir, taxonomy_path=TAXONOMY_PATH,
            _load_model_weights=False,
        )


def test_device_setting(mock_model_dir):
    classifier = IAClassifier(
        mock_model_dir, device="cpu",
        _load_model_weights=False,
    )
    assert classifier.device == "cpu"


# ---------------------------------------------------------------------------
# Prediction tests (need mocked tokenizer / model)
# ---------------------------------------------------------------------------

def test_predict_returns_correct_structure(mock_model_dir_with_prediction):
    classifier = IAClassifier(
        mock_model_dir_with_prediction, fallback_threshold=0.50,
    )
    result = classifier.predict("Formation en Python et Machine Learning")
    assert isinstance(result, list)
    assert len(result) == 1
    preds = result[0]
    assert len(preds) == 20
    for p in preds:
        assert "label" in p
        assert "probability" in p
        assert "threshold" in p
        assert "selected" in p
        assert isinstance(p["probability"], float)
        assert 0.0 <= p["probability"] <= 1.0
        assert isinstance(p["selected"], bool)


def test_predict_sorted_by_probability(mock_model_dir_with_prediction):
    classifier = IAClassifier(mock_model_dir_with_prediction)
    result = classifier.predict("Test")[0]
    probs = [p["probability"] for p in result]
    assert probs == sorted(probs, reverse=True)


def test_predict_top_k(mock_model_dir_with_prediction):
    classifier = IAClassifier(
        mock_model_dir_with_prediction, top_k=5,
    )
    result = classifier.predict("Test")[0]
    assert len(result) == 5


def test_predict_labels(mock_model_dir_with_prediction):
    classifier = IAClassifier(
        mock_model_dir_with_prediction, fallback_threshold=0.0,
    )
    result = classifier.predict_labels("Test")
    assert isinstance(result, list)
    assert isinstance(result[0], list)
    assert len(result[0]) == 20


def test_predict_probas_shape(mock_model_dir_with_prediction):
    classifier = IAClassifier(mock_model_dir_with_prediction)
    probas = classifier.predict_probas(["Text A", "Text B"])
    assert probas.shape == (2, 20)
    assert probas.dtype == np.float64 or probas.dtype == np.float32


def test_predict_batch(mock_model_dir_with_prediction):
    classifier = IAClassifier(mock_model_dir_with_prediction)
    texts = ["Formation Python", "Deep Learning avance", "Data Science"]
    result = classifier.predict(texts)
    assert len(result) == 3
    for preds in result:
        assert len(preds) == 20


def test_top_k_none_returns_all(mock_model_dir_with_prediction):
    classifier = IAClassifier(
        mock_model_dir_with_prediction, top_k=None,
    )
    result = classifier.predict("Test")[0]
    assert len(result) == 20
