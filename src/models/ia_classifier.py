"""Classifieur multilabel de competences IA.

Enrobe un modele HuggingFace pour la classification multilabel avec :
    - chargement avec verification de coherence
    - seuils par label (optimises sur validation)
    - support top_k
    - seuil global de secours
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("ia_classifier")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except ImportError:
    torch = None
    AutoModelForSequenceClassification = None
    AutoTokenizer = None


class IAClassifier:
    """Classifieur multilabel de competences IA.

    Args:
        model_dir: Chemin vers le dossier du checkpoint.
        taxonomy_path: Chemin vers la taxonomie JSON (optionnel, pour verification).
        device: Device torch ('cpu', 'cuda', etc.). Auto-detecte si None.
        fallback_threshold: Seuil global utilise si thresholds.json absent.
        top_k: Nombre max de labels a retourner (None = tous).
    """

    def __init__(
        self,
        model_dir: str | Path,
        taxonomy_path: str | Path | None = None,
        device: str | None = None,
        fallback_threshold: float = 0.50,
        top_k: int | None = None,
        _load_model_weights: bool = True,
    ):
        if AutoModelForSequenceClassification is None:
            raise ImportError("transformers n'est pas disponible")

        self.model_dir = Path(model_dir)
        self.fallback_threshold = fallback_threshold
        self.top_k = top_k
        self._model_loaded = False

        self.device = (
            device
            or ("cuda" if torch and torch.cuda.is_available() else "cpu")
        )

        self._load_config()
        if taxonomy_path:
            self._verify_taxonomy(taxonomy_path)
        self._check_consistency()
        self._load_thresholds()
        if _load_model_weights:
            self._load_model()

    def _load_config(self) -> None:
        config_path = self.model_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"config.json introuvable: {config_path}")
        with open(config_path) as f:
            self.config = json.load(f)

        self.num_labels = self.config.get("num_labels") or len(
            self.config.get("id2label", {})
        )
        raw_id2label = self.config.get("id2label", {})
        self.id2label = {
            int(k): v for k, v in raw_id2label.items()
        }
        # Build label2id from id2label to ensure consistency
        self.label2id = {
            v: int(k) for k, v in raw_id2label.items()
        }
        self.problem_type = self.config.get("problem_type", "")
        self.labels = [
            self.id2label[i] for i in range(self.num_labels)
        ]

        logger.info(
            "Configuration: %d labels, problem_type=%s",
            self.num_labels, self.problem_type,
        )

    def _verify_taxonomy(self, taxonomy_path: str | Path) -> None:
        with open(taxonomy_path) as f:
            taxonomy = json.load(f)
        taxonomy_labels = set(taxonomy.get("labels", []))
        model_labels = set(self.labels)

        missing_in_model = taxonomy_labels - model_labels
        extra_in_model = model_labels - taxonomy_labels

        if missing_in_model:
            logger.warning(
                "Labels de la taxonomie absents du modele (%d): %s",
                len(missing_in_model), sorted(missing_in_model),
            )
        if extra_in_model:
            logger.warning(
                "Labels du modele absents de la taxonomie (%d): %s",
                len(extra_in_model), sorted(extra_in_model),
            )

        if len(missing_in_model) == len(taxonomy_labels):
            raise ValueError(
                "INCOMPATIBILITE TAXONOMIE: Le modele ne contient aucun label "
                "de la taxonomie. Un nouvel entrainement est necessaire."
            )

    def _check_consistency(self) -> None:
        if self.problem_type != "multi_label_classification":
            logger.warning(
                "problem_type attendu 'multi_label_classification', "
                "obtenu '%s'", self.problem_type
            )

        if self.num_labels < 2:
            raise ValueError(
                f"num_labels invalide: {self.num_labels}. "
                "Au moins 2 labels attendus."
            )

        if len(self.id2label) != self.num_labels:
            logger.warning(
                "id2label contient %d entrees pour %d labels",
                len(self.id2label), self.num_labels,
            )

    def _load_model(self) -> None:
        logger.info("Chargement du modele depuis %s...", self.model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)  # type: ignore
        self.model = AutoModelForSequenceClassification.from_pretrained(  # type: ignore
            self.model_dir
        ).to(self.device)
        self.model.eval()
        logger.info("Modele charge sur %s.", self.device)

    def _load_thresholds(self) -> None:
        thr_path = self.model_dir / "thresholds.json"
        if thr_path.exists():
            with open(thr_path) as f:
                data = json.load(f)
            raw = data.get("thresholds", data)
            self.thresholds: dict[str, float] = {}
            for k, v in raw.items():
                if isinstance(v, (int, float)):
                    if k.isdigit():
                        label = self.id2label.get(int(k), k)
                    else:
                        label = k
                    self.thresholds[label] = float(v)
            logger.info(
                "Seuils charges: %d labels", len(self.thresholds)
            )
        else:
            self.thresholds = {}
            logger.info(
                "thresholds.json absent, seuil global par defaut: %.2f",
                self.fallback_threshold,
            )

    def _get_threshold(self, label: str) -> float:
        return self.thresholds.get(label, self.fallback_threshold)

    @torch.no_grad()
    def predict(
        self,
        texts: str | list[str],
        return_probas: bool = True,
    ) -> list[list[dict[str, Any]]]:
        if isinstance(texts, str):
            texts = [texts]

        encoded = self.tokenizer(  # type: ignore
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        ).to(self.device)

        logits = self.model(**encoded).logits  # type: ignore
        probas = torch.sigmoid(logits).cpu().numpy()

        results: list[list[dict[str, Any]]] = []
        for row_probas in probas:
            preds = []
            for i in range(len(self.labels)):
                label = self.labels[i]
                probability = float(row_probas[i])
                threshold = self._get_threshold(label)
                selected = probability >= threshold
                preds.append({
                    "label": label,
                    "probability": round(probability, 6),
                    "threshold": threshold,
                    "selected": bool(selected),
                })
            preds.sort(key=lambda x: -x["probability"])
            if self.top_k is not None and self.top_k > 0:
                preds = preds[: self.top_k]
            results.append(preds)

        return results

    def predict_labels(
        self,
        texts: str | list[str],
    ) -> list[list[str]]:
        preds = self.predict(texts, return_probas=True)
        return [
            [p["label"] for p in row if p["selected"]] for row in preds
        ]

    @torch.no_grad()
    def predict_probas(
        self,
        texts: str | list[str],
    ) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        encoded = self.tokenizer(  # type: ignore
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        ).to(self.device)

        logits = self.model(**encoded).logits  # type: ignore
        return torch.sigmoid(logits).cpu().numpy()
