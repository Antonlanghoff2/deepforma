from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
except Exception:
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

    torch = _TorchStub()

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from transformers import logging as hf_logging
    hf_logging.set_verbosity_error()
except Exception:
    class _AutoModelLoaderStub:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            raise ImportError("transformers n'est pas installe.")

    class _AutoTokenizerLoaderStub:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            raise ImportError("transformers n'est pas installe.")

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
        candidates.extend([
            getattr(classifier, 'out_proj', None),
            getattr(classifier, 'score', None),
            classifier,
        ])
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
        raise ValueError(f'Aucun label valide trouve dans {path}.')
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


def _fresh_init_classifier_stats(num_labels: int, hidden_size: int = 768) -> dict[str, float]:
    rng = np.random.RandomState(42)
    fresh_weight = rng.randn(num_labels, hidden_size) * 0.02
    return {
        'mean': float(fresh_weight.mean()),
        'std': float(fresh_weight.std()),
        'min': float(fresh_weight.min()),
        'max': float(fresh_weight.max()),
    }


def _audit_checkpoint(model_dir: Path) -> dict[str, Any]:
    audit: dict[str, Any] = {
        'config_present': False,
        'weights_present': False,
        'weights_size_bytes': 0,
        'architecture_declared': '',
        'num_labels_declared': 0,
        'problem_type': '',
        'id2label_count': 0,
        'label2id_count': 0,
        'strict_load_success': False,
        'missing_keys': [],
        'unexpected_keys': [],
        'ignored_keys': [],
        'classifier_weight_shape': '',
        'classifier_weight_mean': 0.0,
        'classifier_weight_std': 0.0,
        'classifier_weight_min': 0.0,
        'classifier_weight_max': 0.0,
        'classifier_bias_mean': None,
        'appears_random_init': True,
    }

    config_path = model_dir / 'config.json'
    if config_path.exists():
        cfg = json.loads(config_path.read_text(encoding='utf-8'))
        audit['config_present'] = True
        audit['architecture_declared'] = str(cfg.get('architectures', [''])[0])
        audit['num_labels_declared'] = int(cfg.get('num_labels', 0))
        audit['problem_type'] = str(cfg.get('problem_type', ''))
        audit['id2label_count'] = len(cfg.get('id2label', {}))
        audit['label2id_count'] = len(cfg.get('label2id', {}))

    weights_path = model_dir / 'model.safetensors'
    if weights_path.exists():
        audit['weights_present'] = True
        audit['weights_size_bytes'] = weights_path.stat().st_size

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_dir, return_dict=True
        )
        strict_model = AutoModelForSequenceClassification.from_pretrained(
            model_dir, return_dict=True,
        )
        _ = strict_model  # loaded without strict=False
        audit['strict_load_success'] = True
    except Exception as exc:
        msg = str(exc)
        if 'is not in the model' in msg or 'are missing' in msg:
            audit['missing_keys'] = _extract_keys_from_error(msg)

        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_dir, return_dict=True, ignore_mismatched_sizes=True
            )
        except Exception as fallback_exc:
            logger.error('Echec chargement checkpoint %s: %s', model_dir, fallback_exc)
            return audit

    classifier = getattr(model, 'classifier', None)
    if classifier is not None:
        weight = getattr(classifier, 'weight', None)
        if weight is not None:
            w = weight.data.detach().cpu().numpy()
            audit['classifier_weight_shape'] = str(list(w.shape))
            audit['classifier_weight_mean'] = float(w.mean())
            audit['classifier_weight_std'] = float(w.std())
            audit['classifier_weight_min'] = float(w.min())
            audit['classifier_weight_max'] = float(w.max())

            hidden_size = w.shape[1] if len(w.shape) > 1 else 768
            fresh_stats = _fresh_init_classifier_stats(w.shape[0], hidden_size)
            weight_mean_close_to_zero = abs(audit['classifier_weight_mean']) < 0.01
            weight_std_close_to_fresh = abs(
                audit['classifier_weight_std'] - fresh_stats['std']
            ) < 0.005

            if weight_mean_close_to_zero and weight_std_close_to_fresh:
                audit['appears_random_init'] = True
                logger.warning(
                    'Poids du classifier (mean=%.4f, std=%.4f) sont coherents avec '
                    'une initialisation aleatoire (fresh std=%.4f). '
                    'Le checkpoint semble ne pas avoir ete entraine.',
                    audit['classifier_weight_mean'], audit['classifier_weight_std'],
                    fresh_stats['std']
                )
            else:
                audit['appears_random_init'] = False
                logger.info(
                    'Poids du classifier coherents avec un entrainement: '
                    'mean=%.4f, std=%.4f (fresh init std=%.4f)',
                    audit['classifier_weight_mean'], audit['classifier_weight_std'],
                    fresh_stats['std']
                )

        bias = getattr(classifier, 'bias', None)
        if bias is not None:
            b = bias.data.detach().cpu().numpy()
            audit['classifier_bias_mean'] = float(b.mean())

    model.to('cpu')
    return audit


def _extract_keys_from_error(msg: str) -> list[str]:
    keys = []
    for line in msg.split('\n'):
        line = line.strip()
        if line.startswith('weight') or line.startswith('bias') or line.startswith('classifier'):
            keys.append(line.split()[0] if ' ' in line else line)
    return keys


def _check_model_weights_loaded(model: Any, model_dir: Path) -> dict[str, Any]:
    safetensors_path = model_dir / 'model.safetensors'
    if safetensors_path.exists():
        expected_size = safetensors_path.stat().st_size
        logger.info('Checkpoint present: %s (%d octets)', safetensors_path, expected_size)
    else:
        logger.warning('Aucun fichier model.safetensors trouve dans %s', model_dir)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('Parametres entrainables: %d', param_count)

    stats = {}
    classifier = getattr(model, 'classifier', None)
    if classifier is not None:
        weight = getattr(classifier, 'weight', None)
        if weight is not None:
            stats = {
                'mean': float(weight.data.mean()),
                'std': float(weight.data.std()),
                'min': float(weight.data.min()),
                'max': float(weight.data.max()),
            }
            logger.info('Poids du classifier: %s', stats)
            if abs(stats['mean']) < 0.01 and stats['std'] < 0.03:
                logger.warning(
                    'Les poids du classifier semblent proches de zero '
                    '(mean=%.4f, std=%.4f) — possible initialisation aleatoire.',
                    stats['mean'], stats['std']
                )
    return stats


def _filter_out_random_params(state_dict: dict[str, Any]) -> dict[str, Any]:
    return state_dict


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

        logger.info('Device utilise: %s', self.device)

        logger.info('Chargement du modele binaire depuis: %s', self.binary_model_dir)
        self.binary_tokenizer = AutoTokenizer.from_pretrained(self.binary_model_dir)
        self.binary_model = AutoModelForSequenceClassification.from_pretrained(
            self.binary_model_dir
        ).to(self.device)
        self.binary_model.eval()
        _check_model_weights_loaded(self.binary_model, self.binary_model_dir)

        logger.info('Chargement du modele multilabel depuis: %s', self.multilabel_model_dir)
        self.multilabel_tokenizer = AutoTokenizer.from_pretrained(self.multilabel_model_dir)
        self.multilabel_model = AutoModelForSequenceClassification.from_pretrained(
            self.multilabel_model_dir
        ).to(self.device)
        self.multilabel_model.eval()
        self.checkpoint_audit = _audit_checkpoint(self.multilabel_model_dir)
        logger.info('Audit checkpoint: appears_random_init=%s, shape=%s, mean=%.4f, std=%.4f',
                     self.checkpoint_audit['appears_random_init'],
                     self.checkpoint_audit['classifier_weight_shape'],
                     self.checkpoint_audit['classifier_weight_mean'],
                     self.checkpoint_audit['classifier_weight_std'])

        self.labels = load_label_classes(self.multilabel_model_dir)
        self.thresholds = load_thresholds(self.multilabel_model_dir)
        self.binary_threshold = self.thresholds.get('binary_threshold')
        self.multilabel_threshold = float(self.thresholds.get('multilabel_threshold') or 0.35)
        self.max_length = int(os.getenv('DEEPFORMA_MAX_LENGTH', str(DEFAULT_MAX_LENGTH)))

        logger.info('Labels charges (%d): %s', len(self.labels), self.labels)
        logger.info('Seuils: multilabel=%s, binaire=%s', self.multilabel_threshold, self.binary_threshold)
        logger.info('Max length: %d', self.max_length)

        self._validate_model_shapes()

    def _validate_model_shapes(self) -> None:
        binary_num_labels = _infer_num_labels(self.binary_model)
        if binary_num_labels is not None and binary_num_labels != 2:
            raise ValueError(
                f'Le modele binaire doit exposer 2 labels, obtenu: {binary_num_labels}.'
            )
        multilabel_num_labels = _infer_num_labels(self.multilabel_model)
        if multilabel_num_labels is not None and multilabel_num_labels != len(self.labels):
            raise ValueError(
                'Incompatibilite entre le nombre de labels multi-etiquette '
                f'({len(self.labels)}) et la sortie du modele ({multilabel_num_labels}).'
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
            raise ValueError('Le texte a analyser est vide.')
        logger.debug('Entree binaire (tronquee, %d car.): %s ...', len(cleaned), cleaned[:200])
        encoded = self._encode(self.binary_tokenizer, cleaned)
        with torch.inference_mode():
            logits = self.binary_model(**encoded).logits
            logger.debug('Logits bruts (binaire): %s', logits.detach().cpu().tolist())
            probabilities = torch.softmax(logits, dim=-1)[0].detach().cpu().tolist()
        logger.debug('Probabilites (softmax): non-IA=%.4f, IA=%.4f', probabilities[0], probabilities[1])
        if len(probabilities) != 2:
            raise ValueError(
                f'Sortie binaire invalide: 2 probabilites attendues, obtenu {len(probabilities)}.'
            )
        predicted_class = int(max(range(len(probabilities)), key=probabilities.__getitem__))
        return {
            'is_ia': bool(predicted_class == 1),
            'predicted_class': predicted_class,
            'probability_non_ia': float(probabilities[0]),
            'probability_ia': float(probabilities[1]),
        }

    def predict_skills(self, text: str, threshold: float | None = None) -> dict[str, Any]:
        cleaned = clean_text(text)
        if not cleaned:
            raise ValueError('Le texte a analyser est vide.')
        current_threshold = float(self.multilabel_threshold if threshold is None else threshold)
        encoded = self._encode(self.multilabel_tokenizer, cleaned)
        t0 = time.time()
        with torch.inference_mode():
            logits = self.multilabel_model(**encoded).logits
            probabilities = torch.sigmoid(logits)[0].detach().cpu().tolist()
        inference_time = (time.time() - t0) * 1000
        raw_logits = logits[0].detach().cpu().tolist() if hasattr(logits, 'detach') else logits
        if len(probabilities) != len(self.labels):
            raise ValueError(
                'Incompatibilite entre les labels charges et la sortie du modele '
                f'({len(self.labels)} labels, {len(probabilities)} sorties).'
            )
        scores_array = probabilities
        score_min = float(min(scores_array))
        score_max = float(max(scores_array))
        score_mean = float(sum(scores_array) / len(scores_array))
        if len(scores_array) > 1:
            variance = sum((x - score_mean) ** 2 for x in scores_array) / len(scores_array)
            score_std = float(variance ** 0.5)
        else:
            score_std = 0.0
        above_high = sum(1 for s in scores_array if s >= 0.80)
        above_medium = sum(1 for s in scores_array if s >= 0.60)
        above_low = sum(1 for s in scores_array if s >= 0.40)
        above_threshold = sum(1 for s in scores_array if s >= current_threshold)
        logger.info('=== Diagnostic inference multilabel ===')
        logger.info('Taille entree: %d caracteres', len(cleaned))
        logger.info('Entree (debut): %s ...', cleaned[:200])
        logger.info('Checkpoint: %s', self.multilabel_model_dir)
        logger.info('Device: %s', self.device)
        logger.info('Logits bruts: %s', raw_logits)
        logger.info('Probabilites (sigmoid): %s', scores_array)
        logger.info('Min=%.4f, Max=%.4f, Moy=%.4f, Ecart-type=%.4f', score_min, score_max, score_mean, score_std)
        logger.info('Nb scores >= 0.80: %d', above_high)
        logger.info('Nb scores >= 0.60: %d', above_medium)
        logger.info('Nb scores >= 0.40: %d', above_low)
        logger.info('Nb scores >= seuil (%.2f): %d', current_threshold, above_threshold)
        logger.info('Temps inference: %.2f ms', inference_time)
        detecting = score_std > 0.05 or score_max > 0.70
        if not detecting:
            logger.warning(
                'Les scores sont tous regroupes autour de %.3f (std=%.4f, max=%.4f). '
                'Le modele ne discrimine pas.', score_mean, score_std, score_max
            )
        predictions = [
            {'label': label, 'probability': float(prob), 'threshold': current_threshold}
            for label, prob in zip(self.labels, probabilities)
        ]
        predictions.sort(key=lambda item: item['probability'], reverse=True)
        all_scores = [p['probability'] for p in predictions]
        return {
            'predictions': predictions,
            'all_scores': all_scores,
            'score_min': score_min,
            'score_max': score_max,
            'score_mean': score_mean,
            'score_std': score_std,
            'inference_time_ms': round(inference_time, 2),
            'num_labels': len(self.labels),
            'threshold_applied': current_threshold,
            'raw_logits': raw_logits,
        }

    def analyze(self, text: str, threshold: float | None = None) -> dict[str, Any]:
        t0 = time.time()
        binary_result = self.predict_binary(text)
        skills_result = self.predict_skills(text, threshold=threshold)
        total_time = (time.time() - t0) * 1000
        logger.info('Analyse completee en %.2f ms', total_time)
        logger.info('Classification binaire: IA=%s (p_IA=%.4f, p_nonIA=%.4f)',
                     binary_result['is_ia'], binary_result['probability_ia'],
                     binary_result['probability_non_ia'])
        return {
            'binary': binary_result,
            'skills': skills_result,
            'device': str(self.device),
            'inference_time_ms': round(total_time, 2),
            'checkpoint_audit': self.checkpoint_audit,
        }


@lru_cache(maxsize=1)
def get_predictor() -> DeepformaPredictor:
    return DeepformaPredictor()
