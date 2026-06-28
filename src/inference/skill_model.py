from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - compatibilité environnementale
    class _CudaStub:
        @staticmethod
        def is_available() -> bool:
            return False

    class _NoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    class _TorchStub:
        cuda = _CudaStub()

        @staticmethod
        def device(name: str):
            return name

        @staticmethod
        def no_grad():
            return _NoGrad()

    torch = _TorchStub()  # type: ignore[assignment]

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except Exception:  # pragma: no cover - compatibilité environnementale
    class _AutoModelLoaderStub:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            raise ImportError("transformers n'est pas installé.")

    class _AutoTokenizerLoaderStub:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            raise ImportError("transformers n'est pas installé.")

    AutoModelForSequenceClassification = _AutoModelLoaderStub
    AutoTokenizer = _AutoTokenizerLoaderStub

from common.text import clean_text


DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "multilabel_competences_v2" / "final"


@dataclass(frozen=True)
class SkillPrediction:
    label: str
    probability: float
    threshold: float
    source: str = "camembert_multilabel"


def load_label_classes(model_dir: str | Path = DEFAULT_MODEL_DIR) -> list[str]:
    path = Path(model_dir) / "label_classes.json"
    if not path.exists():
        raise FileNotFoundError(f"label_classes.json introuvable: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_thresholds(model_dir: str | Path = DEFAULT_MODEL_DIR) -> dict[str, float | None]:
    path = Path(model_dir) / "thresholds.json"
    if not path.exists():
        raise FileNotFoundError(f"thresholds.json introuvable: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


class SkillModel:
    def __init__(self, model_dir: str | Path = DEFAULT_MODEL_DIR) -> None:
        self.model_dir = Path(model_dir)
        self.labels = load_label_classes(self.model_dir)
        self.thresholds = load_thresholds(self.model_dir)
        self.binary_threshold = self.thresholds.get("binary_threshold")
        self.multilabel_threshold = float(self.thresholds.get("multilabel_threshold") or 0.35)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir).to(self.device)
        self.model.eval()

    @staticmethod
    def build_text(title: str | None, description: str | None, structured_skills: Iterable[dict[str, Any]] | None = None) -> str:
        parts = []
        title = clean_text(title)
        description = clean_text(description)
        if title:
            parts.append(f"Titre : {title}")
        if description:
            parts.append(f"Description : {description}")
        structured_labels = [clean_text(item.get("label", "")) for item in (structured_skills or []) if clean_text(item.get("label", ""))]
        if structured_labels:
            parts.append("Compétences structurées : " + " | ".join(dict.fromkeys(structured_labels)))
        return "\n".join(parts)

    def predict_texts(self, texts: list[str], batch_size: int = 8, threshold: float | None = None) -> list[list[SkillPrediction]]:
        results: list[list[SkillPrediction]] = []
        threshold = float(self.multilabel_threshold if threshold is None else threshold)
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
            ).to(self.device)
            with torch.no_grad():
                logits = self.model(**encoded).logits
            probabilities = torch.sigmoid(logits).cpu().numpy()
            for probs in probabilities:
                batch_predictions = [
                    SkillPrediction(label=label, probability=float(prob), threshold=threshold)
                    for label, prob in zip(self.labels, probs)
                    if float(prob) >= threshold
                ]
                batch_predictions.sort(key=lambda item: item.probability, reverse=True)
                results.append(batch_predictions)
        return results

    def predict_offer(self, title: str | None, description: str | None, structured_skills: Iterable[dict[str, Any]] | None = None, threshold: float | None = None) -> list[dict[str, Any]]:
        text = self.build_text(title, description, structured_skills)
        predictions = self.predict_texts([text], threshold=threshold)[0]
        return [
            {
                "label": item.label,
                "probability": round(item.probability, 4),
                "threshold": item.threshold,
                "source": item.source,
            }
            for item in predictions
        ]

