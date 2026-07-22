from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from common.text import clean_text
from config.binary_ai import BINARY_AI_SETTINGS
from inference.deepforma_predictor import DeepformaPredictor
from deepforma.training.binary_ai_ml import predict_binary_ai_ml
from deepforma.training.binary_ai_textcnn import predict_binary_ai_textcnn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ML_MODEL_DIR = PROJECT_ROOT / "models" / "binary_ai_ml"
DEFAULT_TEXTCNN_MODEL_DIR = PROJECT_ROOT / "models" / "binary_ai_textcnn"


def _existing_predictor() -> DeepformaPredictor:
    return DeepformaPredictor()


def predict_binary_ai(text: str, model_name: str = "ml") -> dict[str, Any]:
    cleaned = clean_text(text)

    selected_model = (model_name or BINARY_AI_SETTINGS.backend).strip().lower()
    if selected_model in {"ml_from_scratch", "ml"}:
        if not DEFAULT_ML_MODEL_DIR.exists():
            raise FileNotFoundError(f"Modèle ML introuvable: {DEFAULT_ML_MODEL_DIR}")
        return predict_binary_ai_ml(cleaned, model_dir=DEFAULT_ML_MODEL_DIR)
    if selected_model in {"textcnn_from_scratch", "textcnn"}:
        if not DEFAULT_TEXTCNN_MODEL_DIR.exists():
            raise FileNotFoundError(f"Modèle TextCNN introuvable: {DEFAULT_TEXTCNN_MODEL_DIR}")
        return predict_binary_ai_textcnn(cleaned, model_dir=DEFAULT_TEXTCNN_MODEL_DIR)
    if selected_model == "existing":
        if not cleaned:
            raise ValueError("Le texte a analyser est vide.")
        start = time.perf_counter()
        predictor = _existing_predictor()
        result = predictor.predict_binary(cleaned)
        latency_ms = (time.perf_counter() - start) * 1000.0
        probability_ai = float(result["probability_ia"])
        threshold = float(getattr(predictor, "binary_threshold", 0.5) or 0.5)
        return {
            "label": "IA" if result.get("is_ia") else "non-IA",
            "probability_ai": probability_ai,
            "threshold": threshold,
            "model_name": "existing",
            "model_version": getattr(predictor, "binary_model_dir", ""),
            "pretrained": True,
            "latency_ms": latency_ms,
        }
    raise ValueError("model_name doit valoir 'existing', 'ml', 'ml_from_scratch', 'textcnn' ou 'textcnn_from_scratch'")

