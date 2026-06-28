from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    import torch
except Exception:  # pragma: no cover - compatibilité environnementale
    import numpy as _np

    class _CudaStub:
        @staticmethod
        def is_available() -> bool:
            return False

    class _NoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    class _InferenceMode(_NoGrad):
        pass

    class _TorchStub:
        cuda = _CudaStub()

        @staticmethod
        def device(name: str):
            return name

        @staticmethod
        def no_grad():
            return _NoGrad()

        @staticmethod
        def inference_mode():
            return _InferenceMode()

        @staticmethod
        def softmax(logits, dim=-1):
            arr = _np.asarray(logits, dtype=float)
            arr = arr - arr.max(axis=dim, keepdims=True)
            exp = _np.exp(arr)
            return exp / exp.sum(axis=dim, keepdims=True)

        @staticmethod
        def sigmoid(logits):
            arr = _np.asarray(logits, dtype=float)
            return 1.0 / (1.0 + _np.exp(-arr))

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
from inference.skill_model import load_label_classes, load_thresholds


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BINARY_MODEL_DIR = PROJECT_ROOT / 'models' / 'binary_ia_v2' / 'final'
DEFAULT_MULTILABEL_MODEL_DIR = PROJECT_ROOT / 'models' / 'multilabel_competences_v2' / 'final'
DEFAULT_MAX_LENGTH = 512


@dataclass(frozen=True)
class ModelBundle:
    model_dir: Path
    tokenizer: Any
    model: Any
    labels: list[str]
    threshold: float | None = None


def _infer_num_labels(model: Any) -> int | None:
    config_num_labels = getattr(getattr(model, 'config', None), 'num_labels', None)
    if isinstance(config_num_labels, int) and config_num_labels > 0:
        return config_num_labels

    classifier = getattr(model, 'classifier', None)
    candidates = []
    if classifier is not None:
        candidates.extend(
            [
                getattr(classifier, 'out_proj', None),
                getattr(classifier, 'score', None),
                classifier,
            ]
        )
    for candidate in candidates:
        if candidate is None:
            continue
        out_features = getattr(candidate, 'out_features', None)
        if isinstance(out_features, int) and out_features > 0:
            return out_features
        weight = getattr(candidate, 'weight', None)
        if weight is not None and hasattr(weight, 'shape') and len(weight.shape) >= 1:
            return int(weight.shape[0])
    return None


def _load_json_list(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f'Fichier de labels introuvable: {path}')
    payload = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(payload, list):
        raise ValueError(f'Format de labels invalide dans {path}: liste attendue.')
    labels = [clean_text(item) for item in payload if clean_text(item)]
    if not labels:
        raise ValueError(f'Aucun label valide trouvé dans {path}.')
    return labels


def _load_json_dict(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f'Fichier de configuration introuvable: {path}')
    payload = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise ValueError(f'Format de configuration invalide dans {path}: objet JSON attendu.')
    return payload


def _prepare_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _to_device(batch: Any, device: torch.device) -> Any:
    if hasattr(batch, 'to'):
        return batch.to(device)
    return {key: value.to(device) for key, value in batch.items()}


def _load_bundle(model_dir: Path, device: torch.device, labels_path: Path | None = None) -> ModelBundle:
    if not model_dir.exists():
        raise FileNotFoundError(f'Modèle introuvable: {model_dir}')
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()
    labels = _load_json_list(labels_path or (model_dir / 'label_classes.json')) if (labels_path or (model_dir / 'label_classes.json')).exists() else []
    return ModelBundle(model_dir=model_dir, tokenizer=tokenizer, model=model, labels=labels)


class DeepformaPredictor:
    def __init__(
        self,
        binary_model_dir: str | Path = DEFAULT_BINARY_MODEL_DIR,
        multilabel_model_dir: str | Path = DEFAULT_MULTILABEL_MODEL_DIR,
        device: torch.device | None = None,
    ) -> None:
        self.binary_model_dir = Path(binary_model_dir)
        self.multilabel_model_dir = Path(multilabel_model_dir)
        self.device = device or _prepare_device()

        self.binary_tokenizer = AutoTokenizer.from_pretrained(self.binary_model_dir)
        self.binary_model = AutoModelForSequenceClassification.from_pretrained(self.binary_model_dir).to(self.device)
        self.binary_model.eval()

        self.multilabel_tokenizer = AutoTokenizer.from_pretrained(self.multilabel_model_dir)
        self.multilabel_model = AutoModelForSequenceClassification.from_pretrained(self.multilabel_model_dir).to(self.device)
        self.multilabel_model.eval()

        self.labels = load_label_classes(self.multilabel_model_dir)
        self.thresholds = load_thresholds(self.multilabel_model_dir)
        self.binary_threshold = self.thresholds.get('binary_threshold')
        self.multilabel_threshold = float(self.thresholds.get('multilabel_threshold') or 0.35)
        self.max_length = int(os.getenv('DEEPFORMA_MAX_LENGTH', str(DEFAULT_MAX_LENGTH)))

        self._validate_model_shapes()

    def _validate_model_shapes(self) -> None:
        binary_num_labels = _infer_num_labels(self.binary_model)
        if binary_num_labels is not None and binary_num_labels != 2:
            raise ValueError(
                f'Le modèle binaire doit exposer 2 labels, obtenu: {binary_num_labels}.'
            )

        multilabel_num_labels = _infer_num_labels(self.multilabel_model)
        if multilabel_num_labels is not None and multilabel_num_labels != len(self.labels):
            raise ValueError(
                'Incompatibilité entre le nombre de labels multi-étiquette '
                f'({len(self.labels)}) et la sortie du modèle ({multilabel_num_labels}).'
            )

    def _encode(self, tokenizer: Any, text: str) -> Any:
        encoded = tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        return _to_device(encoded, self.device)

    def predict_binary(self, text: str) -> dict[str, Any]:
        cleaned = clean_text(text)
        if not cleaned:
            raise ValueError('Le texte à analyser est vide.')

        encoded = self._encode(self.binary_tokenizer, cleaned)
        with torch.inference_mode():
            logits = self.binary_model(**encoded).logits
            probabilities = torch.softmax(logits, dim=-1)[0].detach().cpu().tolist()

        if len(probabilities) != 2:
            raise ValueError(
                f'Sortie binaire invalide: 2 probabilités attendues, obtenu {len(probabilities)}.'
            )

        predicted_class = int(max(range(len(probabilities)), key=probabilities.__getitem__))
        return {
            'is_ia': bool(predicted_class == 1),
            'predicted_class': predicted_class,
            'probability_non_ia': float(probabilities[0]),
            'probability_ia': float(probabilities[1]),
        }

    def predict_skills(self, text: str, threshold: float | None = None) -> list[dict[str, Any]]:
        cleaned = clean_text(text)
        if not cleaned:
            raise ValueError('Le texte à analyser est vide.')

        current_threshold = float(self.multilabel_threshold if threshold is None else threshold)
        encoded = self._encode(self.multilabel_tokenizer, cleaned)
        with torch.inference_mode():
            logits = self.multilabel_model(**encoded).logits
            probabilities = torch.sigmoid(logits)[0].detach().cpu().tolist()

        if len(probabilities) != len(self.labels):
            raise ValueError(
                'Incompatibilité entre les labels chargés et la sortie du modèle '
                f'({len(self.labels)} labels, {len(probabilities)} sorties).'
            )

        predictions = [
            {
                'label': label,
                'probability': float(prob),
                'threshold': current_threshold,
            }
            for label, prob in zip(self.labels, probabilities)
            if float(prob) >= current_threshold
        ]
        predictions.sort(key=lambda item: item['probability'], reverse=True)
        return predictions

    def analyze(self, text: str, threshold: float | None = None) -> dict[str, Any]:
        return {
            'binary': self.predict_binary(text),
            'skills': self.predict_skills(text, threshold=threshold),
            'device': str(self.device),
        }


@lru_cache(maxsize=1)
def get_predictor() -> DeepformaPredictor:
    return DeepformaPredictor()
