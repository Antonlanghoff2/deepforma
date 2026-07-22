from __future__ import annotations

from dataclasses import dataclass

import pytest
from flask import render_template

from models.analysis_result import (
    AnalysisResult,
    CheckpointAuditInfo,
    ClassificationInfo,
    IAClassificationInfo,
    ModelMetadata,
    SkillExtractionInfo,
)
from web_app import create_app
import web_app as web_app_module


@dataclass
class DummyPredictor:
    calls: int = 0

    def analyze(self, text, threshold=None):
        self.calls += 1
        return {
            "binary": {
                "is_ia": True,
                "predicted_class": 1,
                "probability_non_ia": 0.25,
                "probability_ia": 0.75,
            },
            "skills": {
                "predictions": [],
                "all_scores": [],
                "score_min": 0.0,
                "score_max": 0.0,
                "score_mean": 0.0,
                "score_std": 0.0,
                "inference_time_ms": 1.0,
                "num_labels": 0,
                "threshold_applied": threshold or 0.35,
            },
            "device": "cpu",
            "inference_time_ms": 1.0,
            "checkpoint_audit": {},
        }


def _build_result() -> AnalysisResult:
    result = AnalysisResult()
    result.summary = {
        "total_skills_extracted": 1,
        "total_tools_detected": 0,
        "total_offers_analyzed": 0,
        "inference_time_ms": 12.0,
    }
    result.classification = ClassificationInfo(
        is_ia=True,
        predicted_class=1,
        probability_ia=0.75,
        probability_non_ia=0.25,
        state="reliable",
        state_description="",
        gap=0.5,
    )
    result.skill_extraction = SkillExtractionInfo(status="success")
    result.ia_classification = IAClassificationInfo(
        status="success",
        categories=[{"label": "IA", "probability": 0.75}],
        families=[],
        scores=[0.75],
        score_min=0.75,
        score_max=0.75,
        score_mean=0.75,
        score_std=0.0,
        discriminating=True,
        warnings=[],
        threshold_applied=0.35,
    )
    result.model_metadata = ModelMetadata(
        binary_model="binary_model",
        multilabel_model="multilabel_model",
        model_name="Classifieur IA",
        taxonomy_version="1.0",
        validation_status="validé",
        binary_checkpoint="binary.ckpt",
        multilabel_checkpoint="multilabel.ckpt",
        device="cpu",
        max_length=512,
        num_labels=1,
        labels=["IA"],
        thresholds={"multilabel": 0.35, "binary": 0.5},
        inference_time_ms=12.0,
    )
    result.checkpoint_audit = CheckpointAuditInfo(
        config_present=True,
        weights_present=True,
        weights_size_bytes=1024,
        architecture_declared="Dummy",
        num_labels_declared=1,
        num_labels_effective=1,
        problem_type="single_label_classification",
        id2label_count=1,
        label2id_count=1,
        strict_load_success=True,
        missing_keys=[],
        unexpected_keys=[],
        ignored_keys=[],
        appears_random_init=False,
        body_params_match_base=False,
        parameter_errors=[],
        classifier_params={},
    )
    return result


@pytest.fixture()
def app():
    return create_app(predictor=DummyPredictor())


@pytest.fixture()
def rendered_result(app):
    def _render(**kwargs):
        with app.test_request_context():
            return render_template(
                "result.html",
                result_dict=_build_result().to_dict(),
                context={"market_status": "unavailable"},
                model_only=True,
                **kwargs,
            )
    return _render


@pytest.fixture()
def client(app):
    return app.test_client()


def _post_analyze(client, monkeypatch, binary_payload):
    monkeypatch.setattr(web_app_module, "build_analysis_result", lambda *args, **kwargs: _build_result())
    monkeypatch.setattr(web_app_module, "predict_binary_ai", binary_payload)
    return client.post(
        "/analyze",
        data={
            "programme": "Formation Python et IA",
            "departement": "75",
            "model_only": "1",
        },
    )


def test_analyze_binary_ai_predictions_normal(client, monkeypatch):
    response = _post_analyze(
        client,
        monkeypatch,
        lambda text, model_name="ml": {
            "label": "IA",
            "probability_ai": 0.82,
            "threshold": 0.5,
            "model_name": model_name,
            "model_version": "v1",
            "pretrained": False,
            "latency_ms": 4.2,
        },
    )
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "Machine learning" in html
    assert "82.0 %" in html
    assert "UndefinedError" not in html


def test_analyze_binary_ai_predictions_low_confidence(client, monkeypatch):
    response = _post_analyze(
        client,
        monkeypatch,
        lambda text, model_name="ml": {
            "label": "IA",
            "probability_ai": 0.4935,
            "probability_non_ia": 0.5065,
            "threshold": 0.5,
            "model_name": model_name,
            "model_version": "v1",
            "pretrained": False,
            "latency_ms": 4.2,
        },
    )
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "Scores proches de 0,5 : classification peu fiable." in html
    assert "49.4 %" in html
    assert "UndefinedError" not in html


def test_analyze_binary_ai_predictions_model_unavailable(client, monkeypatch):
    def _raise(*args, **kwargs):
        raise FileNotFoundError("Modèle introuvable")

    response = _post_analyze(client, monkeypatch, _raise)
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "Modèles indisponibles" in html
    assert "Indisponible" in html
    assert "UndefinedError" not in html


def test_analyze_binary_ai_predictions_inference_exception(client, monkeypatch):
    def _raise(*args, **kwargs):
        raise RuntimeError("Inference error")

    response = _post_analyze(client, monkeypatch, _raise)
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "Inference error" in html
    assert "UndefinedError" not in html


def test_analyze_binary_ai_predictions_empty_dict(client, monkeypatch):
    monkeypatch.setattr(web_app_module, "build_analysis_result", lambda *args, **kwargs: _build_result())
    monkeypatch.setattr(web_app_module, "build_binary_ai_predictions", lambda text, model_names=("ml", "textcnn"): ({}, {}))
    response = client.post(
        "/analyze",
        data={
            "programme": "Formation Python et IA",
            "departement": "75",
            "model_only": "1",
        },
    )
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "Aucune prédiction binaire disponible." in html
    assert "UndefinedError" not in html


def test_result_template_without_binary_predictions(rendered_result):
    html = rendered_result()
    assert "Aucune prédiction binaire disponible." in html


def test_result_template_with_empty_binary_predictions(rendered_result):
    html = rendered_result(binary_ai_predictions={})
    assert "Aucune prédiction binaire disponible." in html


def test_result_template_with_valid_binary_predictions(rendered_result):
    html = rendered_result(
        binary_ai_predictions={
            "camembert_binary": {
                "display_name": "CamemBERT binaire",
                "prediction": "IA",
                "is_ai": True,
                "probability_ia": 0.82,
                "probability_non_ia": 0.18,
                "confidence": 0.82,
                "threshold": 0.5,
                "available": True,
                "status": "ok",
                "warning": None,
                "latency_ms": 4.2,
            }
        }
    )
    assert "CamemBERT binaire" in html
    assert "82.0 %" in html


def test_result_template_with_optional_none_fields(rendered_result):
    html = rendered_result(
        binary_ai_predictions={
            "camembert_binary": {
                "display_name": "CamemBERT binaire",
                "prediction": None,
                "is_ai": None,
                "probability_ia": None,
                "probability_non_ia": None,
                "confidence": None,
                "threshold": 0.5,
                "available": False,
                "status": "error",
                "warning": "Modèle indisponible.",
                "latency_ms": None,
            }
        }
    )
    assert "Indisponible" in html
    assert "Modèle indisponible." in html


def test_post_binary_ai_route_returns_200(client, monkeypatch):
    monkeypatch.setattr(
        web_app_module,
        "predict_binary_ai",
        lambda text, model_name="ml": {
            "label": "IA",
            "probability_ai": 0.82,
            "threshold": 0.5,
            "model_name": model_name,
            "model_version": "v1",
            "pretrained": False,
            "latency_ms": 4.2,
        },
    )
    response = client.post(
        "/binary-ai",
        data={
            "text": "Formation Python et IA",
            "model_name": "ml",
        },
    )
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "Résultat" in html
    assert "82.0000" in html or "0.8200" in html
