"""Tests pour le pipeline d'entraînement du modèle multilabel.

Vérifie que :
  - Les poids du classifieur changent après entraînement (non identiques à l'init)
  - La loss diminue
  - Les scores d'inférence sont discriminants
  - Le config.json contient id2label avec des vrais IDs de taxonomie (pas LABEL_N)
  - Le modèle chargé correspond au nombre de labels du dataset
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from ast import literal_eval
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except ImportError:
    torch = None
    AutoModelForSequenceClassification = None
    AutoTokenizer = None


MODEL_DIR = Path("models/multilabel_competences_v2/retrained/final")
DATASET_INFO = Path("data/multilabel/multilabel_dataset_info.json")
OLD_MODEL_DIR = Path("models/multilabel_competences_v2/final.untrained")


# ======================================================================
#  Helpers
# ======================================================================

def _param_stats(tensor: torch.Tensor) -> dict:
    t = tensor.float().detach().cpu().numpy()
    return {
        "mean": float(t.mean()),
        "std": float(t.std()),
        "min": float(t.min()),
        "max": float(t.max()),
        "l2_norm": float(np.sqrt((t ** 2).sum())),
        "nonzero_frac": float((np.abs(t) > 1e-8).mean()),
    }


def _classifier_weights(model_dir: Path) -> dict:
    weights = load_file(str(model_dir / "model.safetensors"))
    cls_params = {
        k.replace("classifier.", ""): v
        for k, v in weights.items() if k.startswith("classifier.")
    }
    assert len(cls_params) == 4, (
        f"Attendu 4 paramètres classifieur, trouvé {len(cls_params)}"
    )
    return cls_params


def _has_taxonomy_ids(config_path: Path) -> bool:
    cfg = json.loads(config_path.read_text())
    id2label = cfg.get("id2label", {})
    # Real taxonomy IDs look like "ml.intro", "dl.intro", etc.
    sample = list(id2label.values())[0] if id2label else ""
    return "." in sample and not sample.startswith("LABEL_")


# ======================================================================
#  Tests
# ======================================================================

class TestMultilabelTraining:

    def test_config_has_taxonomy_ids(self):
        """Le config.json doit contenir des IDs de taxonomie (pas LABEL_N)."""
        assert MODEL_DIR.exists(), f"Modèle introuvable: {MODEL_DIR}"
        config_path = MODEL_DIR / "config.json"
        assert config_path.exists()
        cfg = json.loads(config_path.read_text())
        id2label = cfg.get("id2label", {})
        assert len(id2label) == 18, f"id2label: 18 attendus, {len(id2label)} trouvés"
        assert _has_taxonomy_ids(config_path), (
            "id2label contient LABEL_N au lieu des IDs de taxonomie"
        )

    def test_model_num_labels_matches_dataset(self):
        """Le modèle doit avoir le même nombre de labels que le dataset."""
        dataset_info = json.loads(DATASET_INFO.read_text())
        expected = dataset_info["num_labels"]
        config_path = MODEL_DIR / "config.json"
        cfg = json.loads(config_path.read_text())
        # Some HF versions don't save num_labels explicitly, infer from id2label
        num_labels = cfg.get("num_labels") or len(cfg.get("id2label", {}))
        assert num_labels == expected, (
            f"num_labels: {expected} attendus, {num_labels} dans config"
        )

    def test_classifier_weights_not_identical_to_base_init(self):
        """Les poids du classifieur doivent différer de l'initialisation aléatoire.

        Vérifie que les biais ne sont pas tous zéro (signe d'entraînement reçu).
        """
        cls_params = _classifier_weights(MODEL_DIR)
        for name in ("dense.bias", "out_proj.bias"):
            stats = _param_stats(cls_params[name])
            assert stats["std"] > 1e-6, (
                f"{name}: std={stats['std']:.8f} (biais tous à zéro = pas d'entraînement)"
            )
            assert stats["nonzero_frac"] > 0.5, (
                f"{name}: {stats['nonzero_frac']:.1%} des valeurs non nulles"
            )

    def test_classifier_weights_changed_from_previous_checkpoint(self):
        """Les poids d'un modèle entraîné doivent différer de l'ancien checkpoint.

        Compare les poids du classifieur entre l'ancien checkpoint (non entraîné)
        et le nouveau (entraîné). La différence moyenne doit être significative.
        """
        if not OLD_MODEL_DIR.exists():
            pytest.skip("Ancien checkpoint non présent pour comparaison")
        old_cls = _classifier_weights(OLD_MODEL_DIR)
        new_cls = _classifier_weights(MODEL_DIR)
        for name in ("dense.weight", "dense.bias", "out_proj.weight", "out_proj.bias"):
            old_t = old_cls[name].float().numpy()
            new_t = new_cls[name].float().numpy()
            mean_diff = float(np.abs(old_t - new_t).mean())
            assert mean_diff > 1e-6, (
                f"{name}: différence moyenne {mean_diff:.8f} (poids identiques)"
            )

    def test_loss_decreased_during_training(self):
        """La loss doit avoir diminué entre la première et la dernière époque."""
        report_path = MODEL_DIR.parent / "training_report.json"
        assert report_path.exists(), "training_report.json absent"
        report = json.loads(report_path.read_text())
        metrics = report.get("epoch_metrics", [])
        assert len(metrics) >= 2, f"Besoin d'au moins 2 époques, {len(metrics)} trouvées"
        first_loss = metrics[0]["loss"]
        last_loss = metrics[-1]["loss"]
        assert last_loss < first_loss - 0.01, (
            f"Loss a augmenté: {first_loss:.4f} -> {last_loss:.4f}"
        )

    def test_f1_improved_during_training(self):
        """Le F1 micro doit s'être amélioré entre la première et la dernière époque."""
        report_path = MODEL_DIR.parent / "training_report.json"
        assert report_path.exists()
        report = json.loads(report_path.read_text())
        metrics = report.get("epoch_metrics", [])
        assert len(metrics) >= 2
        first_f1 = metrics[0]["f1_micro"]
        last_f1 = metrics[-1]["f1_micro"]
        assert last_f1 > first_f1, (
            f"F1 micro a diminué: {first_f1:.4f} -> {last_f1:.4f}"
        )

    def test_inference_scores_are_discriminant(self):
        """Les scores d'inférence doivent montrer une discrimination significative.

        Pour un texte IA (Python pour ML/DL), au moins un label doit avoir
        un score > 0.60 et l'écart-type des scores > 0.05.
        """
        if AutoModelForSequenceClassification is None:
            pytest.skip("transformers non disponible")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
        model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
        model.to(device)
        model.eval()

        text = "Formation avancee en Python pour le Machine Learning et le Deep Learning"
        encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)
        with torch.no_grad():
            logits = model(**encoded).logits
            probs = torch.sigmoid(logits)[0].cpu().numpy()

        score_std = float(probs.std())
        score_max = float(probs.max())

        model.to("cpu")
        assert score_std > 0.05, (
            f"Écart-type des scores trop faible: {score_std:.4f} (modèle non discriminant)"
        )
        assert score_max > 0.60, (
            f"Score max trop faible: {score_max:.4f} (modèle non discriminant)"
        )

    def test_validation_report_exists_and_has_f1(self):
        """Le rapport de validation doit exister et contenir des métriques F1."""
        report_path = Path("reports/multilabel_validation_report.json")
        assert report_path.exists(), "multilabel_validation_report.json absent"
        report = json.loads(report_path.read_text())
        assert "threshold_results" in report
        thr05 = [r for r in report["threshold_results"] if r["threshold"] == 0.5]
        assert len(thr05) == 1
        assert thr05[0]["f1_micro"] > 0.5, (
            f'F1 micro < 0.5 au seuil 0.5: {thr05[0]["f1_micro"]}'
        )

    def test_old_checkpoint_labeled_untrained(self):
        """L'ancien checkpoint doit être clairement identifié comme non entraîné."""
        if not OLD_MODEL_DIR.exists():
            pytest.skip("Ancien checkpoint non présent")
        old_cls = _classifier_weights(OLD_MODEL_DIR)
        for name in ("dense.bias", "out_proj.bias"):
            stats = _param_stats(old_cls[name])
            assert stats["std"] < 1e-8, (
                f"{name} dans l'ancien checkpoint: std={stats['std']:.8f} "
                "(devrait être ~0 pour une init aléatoire)"
            )
            assert stats["nonzero_frac"] < 1e-8, (
                f"{name}: {stats['nonzero_frac']:.1%} des valeurs non nulles"
                " (devrait être 0 pour init aléatoire)"
            )
        # Vérifier que l'ancien config a LABEL_N
        old_config = OLD_MODEL_DIR / "config.json"
        assert not _has_taxonomy_ids(old_config), (
            "L'ancien checkpoint ne devrait pas avoir d'IDs de taxonomie"
        )
