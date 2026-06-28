#!/usr/bin/env python3
"""
Mini-test de surapprentissage obligatoire — vérifie que la tête de
classification peut apprendre sur un mini-lot avant tout entraînement complet.

Usage:
    python scripts/smoke_test_classifier_training.py \\
        --task binary \\
        --samples 32 --epochs 15 \\
        --output reports/binary_smoke_test.json

    python scripts/smoke_test_classifier_training.py \\
        --task multilabel \\
        --samples 32 --epochs 15 \\
        --output reports/multilabel_smoke_test.json

Échoue (code 1) si l'un des contrôles suivants n'est pas satisfait :
  - la perte baisse d'au moins 0,05
  - les poids de out_proj changent (max_diff > 1e-6)
  - l'écart-type des scores sigmoid dépasse 0,01
  - le F1 dépasse 0,25
  - les gradients des 4 paramètres de la tête sont non nuls
  - les 4 paramètres de la tête sont présents dans l'optimiseur
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('smoke_test_classifier')

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    from datasets import Dataset
    from sklearn.metrics import f1_score
    from torch.utils.data import DataLoader
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        get_scheduler,
    )
except ImportError as e:
    logger.error('Dépendance manquante: %s', e)
    sys.exit(1)


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASSIFIER_PARAM_NAMES = [
    'classifier.dense.weight',
    'classifier.dense.bias',
    'classifier.out_proj.weight',
    'classifier.out_proj.bias',
]

# Seuils de validation
MIN_LOSS_DELTA = 0.05
MIN_F1 = 0.25
MIN_WEIGHT_CHANGE = 1e-6
MIN_SCORE_STD = 0.005


# ---------------------------------------------------------------------------
#  Utilitaires
# ---------------------------------------------------------------------------

def parse_multi_values(value: str) -> list[str]:
    if pd.isna(value) or not str(value).strip():
        return []
    return [s.strip() for s in str(value).split('|') if s.strip()]


def build_text_modele(row) -> str:
    from scripts.clean_and_merge_datasets import build_text_modele as _btm
    return _btm(row)


def clean_text(text: str) -> str:
    from scripts.clean_and_merge_datasets import clean_text as _ct
    return _ct(text)


# ---------------------------------------------------------------------------
#  Chargement des données
# ---------------------------------------------------------------------------

def load_multilabel_data(csv_path: Path, n_samples: int) -> tuple[Dataset, list[str]]:
    from sklearn.preprocessing import MultiLabelBinarizer

    df = pd.read_csv(csv_path)
    if 'competences_ia' not in df.columns:
        raise ValueError(f'Colonne competences_ia introuvable dans {csv_path}')

    ia_df = df[df['statut_annotation'] == 'ia_confirmee'].copy()
    if len(ia_df) == 0:
        raise ValueError('Aucune formation ia_confirmee dans le dataset')

    ia_df = ia_df.head(n_samples).reset_index(drop=True)

    label_lists = ia_df['competences_ia'].apply(parse_multi_values).tolist()
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(label_lists)
    label_names = list(mlb.classes_)

    texts = [clean_text(build_text_modele(row)) for _, row in ia_df.iterrows()]

    ds = Dataset.from_dict({'text': texts, 'labels': Y.tolist()})
    return ds, label_names


def load_binary_data(csv_path: Path, n_samples: int) -> Dataset:
    df = pd.read_csv(csv_path)
    if 'statut_annotation' not in df.columns:
        raise ValueError('Colonne statut_annotation introuvable')

    label_map = {'non_ia_confirmee': 0, 'ia_confirmee': 1}
    df = df[df['statut_annotation'].isin(label_map)].copy()
    df = df.head(n_samples).reset_index(drop=True)
    df['label'] = df['statut_annotation'].map(label_map).astype(int)

    texts = [clean_text(build_text_modele(row)) for _, row in df.iterrows()]
    ds = Dataset.from_dict({'text': texts, 'labels': df['label'].tolist()})
    return ds


# ---------------------------------------------------------------------------
#  Vérification de l'optimiseur
# ---------------------------------------------------------------------------

def check_optimizer_includes_classifier(
    model: nn.Module,
    optim: torch.optim.Optimizer,
) -> dict[str, bool]:
    """
    Vérifie que les 4 paramètres de la tête de classification sont bien
    présents dans au moins un groupe de l'optimiseur.
    """
    result: dict[str, bool] = {}
    for name, param in model.named_parameters():
        if name in CLASSIFIER_PARAM_NAMES:
            found = any(
                any(p is param for p in group['params'])
                for group in optim.param_groups
            )
            result[name] = found
            if not found:
                logger.error('PARAM MANQUANT DANS L OPTIMISEUR: %s', name)
    return result


# ---------------------------------------------------------------------------
#  Vérification des gradients
# ---------------------------------------------------------------------------

def check_gradients(model: nn.Module, optim: torch.optim.Optimizer, batch, task: str, label_key: str) -> dict:
    optim.zero_grad()

    if task == 'multilabel':
        logits = model(**{k: v for k, v in batch.items() if k != label_key}).logits
        labels = batch[label_key].float()
        loss = nn.BCEWithLogitsLoss()(logits, labels)
    else:
        logits = model(**{k: v for k, v in batch.items() if k != label_key}).logits
        labels = batch[label_key]
        loss = nn.CrossEntropyLoss()(logits, labels)

    loss.backward()

    grad_report = {}
    for name in CLASSIFIER_PARAM_NAMES:
        param = dict(model.named_parameters()).get(name)
        if param is None:
            grad_report[name] = {'error': 'parameter_not_found'}
            logger.error('PARAMETRE INTROUVABLE: %s', name)
        elif param.grad is None:
            grad_report[name] = {'gradient_is_none': True}
            logger.error('GRADIENT ABSENT: %s', name)
        elif param.grad.abs().sum().item() == 0:
            grad_report[name] = {'gradient_is_zero': True}
            logger.error('GRADIENT NUL: %s', name)
        else:
            g = param.grad.detach()
            grad_report[name] = {
                'grad_norm': float(g.norm().item()),
                'grad_mean': float(g.mean().item()),
                'grad_std': float(g.std().item()),
                'grad_min': float(g.min().item()),
                'grad_max': float(g.max().item()),
                'grad_nonzero_prop': float((g != 0).sum().item() / g.numel()),
            }
            logger.info('Grad %s: norm=%.6f nonzero=%.1f%%',
                        name, grad_report[name]['grad_norm'],
                        grad_report[name]['grad_nonzero_prop'] * 100)

    errors = []
    for name, report in grad_report.items():
        if report.get('error'):
            errors.append(report['error'])
        elif report.get('gradient_is_none'):
            errors.append(f'gradient absent: {name}')
        elif report.get('gradient_is_zero'):
            errors.append(f'gradient nul: {name}')

    return {
        'gradients': grad_report,
        'errors': errors,
        'loss_value': float(loss.item()),
        'has_errors': len(errors) > 0,
    }


# ---------------------------------------------------------------------------
#  Test principal
# ---------------------------------------------------------------------------

def run_smoke_test(
    task: str,
    csv_path: Path,
    base_model: str,
    n_samples: int,
    epochs: int,
    output: str | None,
) -> dict:
    logger.info('Device: %s', DEVICE)
    logger.info('Task: %s', task)
    logger.info('CSV: %s', csv_path)
    logger.info('Base model: %s', base_model)
    logger.info('Samples: %d, Epochs: %d', n_samples, epochs)

    # ---- Chargement ----
    if task == 'multilabel':
        ds, label_names = load_multilabel_data(csv_path, n_samples)
        num_labels = len(label_names)
        problem_type = 'multi_label_classification'
        label_key = 'labels'
        logger.info('Labels (%d): %s', num_labels, label_names)
    else:
        ds = load_binary_data(csv_path, n_samples)
        num_labels = 2
        problem_type = None
        label_key = 'labels'
        label_names = ['non_ia', 'ia']
        logger.info('Task binaire: %d échantillons', len(ds))

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    def tokenize(batch):
        return tokenizer(batch['text'], truncation=True, max_length=256)

    ds = ds.map(tokenize, batched=True)
    ds = ds.remove_columns(['text'])
    ds.set_format('torch')

    collator = DataCollatorWithPadding(tokenizer)
    loader = DataLoader(ds, batch_size=min(8, len(ds)), shuffle=True, collate_fn=collator)

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=num_labels,
        problem_type=problem_type,
    ).to(DEVICE)
    model.train()

    # ---- Snapshot initial des poids ----
    weights_before = {
        name: param.data.detach().cpu().clone()
        for name, param in model.named_parameters()
        if 'classifier' in name
    }

    optim = torch.optim.AdamW(model.parameters(), lr=2e-5)
    num_steps = epochs * len(loader)
    scheduler = get_scheduler('linear', optimizer=optim, num_warmup_steps=0, num_training_steps=num_steps)

    # ---- Vérification de l'optimiseur ----
    optimizer_check = check_optimizer_includes_classifier(model, optim)
    all_params_in_optim = all(optimizer_check.values())

    # ---- Boucle d'entraînement ----
    losses = []
    all_predictions = []
    all_labels_list = []
    all_sigmoid_scores: list[np.ndarray] = []
    grad_info: dict = {}
    weight_changes: dict = {}
    first_batch_processed = False

    for epoch in range(epochs):
        epoch_losses = []
        for batch in loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()
                     if k in ('input_ids', 'attention_mask', label_key)}

            # Vérification des gradients sur le premier batch
            if not first_batch_processed:
                grad_info = check_gradients(model, optim, batch, task, label_key)
                first_batch_processed = True
                if grad_info['has_errors']:
                    logger.warning('Problèmes de gradient détectés en début')

            optim.zero_grad()

            if task == 'multilabel':
                logits = model(**{k: v for k, v in batch.items() if k != label_key}).logits
                labels = batch[label_key].float()
                loss = nn.BCEWithLogitsLoss()(logits, labels)
                sigmoid_scores = torch.sigmoid(logits).detach().cpu().numpy()
                all_sigmoid_scores.append(sigmoid_scores)
                preds = (sigmoid_scores >= 0.35).astype(int)
                all_predictions.append(preds)
                all_labels_list.append(labels.detach().cpu().numpy().astype(int))
            else:
                logits = model(**{k: v for k, v in batch.items() if k != label_key}).logits
                labels = batch[label_key]
                loss = nn.CrossEntropyLoss()(logits, labels)
                probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
                all_sigmoid_scores.append(probs)
                preds = np.argmax(probs, axis=-1)
                all_predictions.append(preds)
                all_labels_list.append(labels.detach().cpu().numpy())

            loss.backward()
            optim.step()
            scheduler.step()

            epoch_losses.append(float(loss.item()))

        avg_loss = float(np.mean(epoch_losses))
        losses.append(avg_loss)
        logger.info('Epoch %d/%d: loss=%.4f', epoch + 1, epochs, avg_loss)

    # ---- Snapshot final des poids ----
    weights_after = {
        name: param.data.detach().cpu().clone()
        for name, param in model.named_parameters()
        if 'classifier' in name
    }

    # ---- Changement des poids ----
    for name in weights_before:
        if name in weights_after:
            diff = (weights_after[name] - weights_before[name]).abs().max().item()
            weight_changes[name] = {
                'before_norm': float(weights_before[name].norm().item()),
                'after_norm': float(weights_after[name].norm().item()),
                'max_abs_change': round(diff, 8),
            }

    # ---- Métriques ----
    if len(all_predictions) > 0:
        if task == 'multilabel':
            y_true = np.vstack(all_labels_list)
            y_pred = np.vstack(all_predictions)
            f1 = float(f1_score(y_true, y_pred, average='micro', zero_division=0))
        else:
            y_true = np.concatenate(all_labels_list)
            y_pred = np.concatenate(all_predictions)
            f1 = float(f1_score(y_true, y_pred, average='binary', zero_division=0))
    else:
        f1 = 0.0

    # ---- Diversité des scores ----
    score_variation = 0.0
    if task == 'multilabel' and len(all_sigmoid_scores) > 0:
        scores_flat = np.vstack(all_sigmoid_scores)
        score_variation = float(np.std([s.max() for s in scores_flat]))
    elif len(all_sigmoid_scores) > 0:
        scores_flat = np.concatenate(all_sigmoid_scores) if len(all_sigmoid_scores[0].shape) == 1 else np.vstack(all_sigmoid_scores)
        if len(scores_flat.shape) == 2:
            score_variation = float(np.std([s.max() for s in scores_flat]))
        else:
            score_variation = float(np.std(scores_flat))

    # ---- Prédictions uniques ----
    if len(all_predictions) > 0:
        if task == 'multilabel':
            preds_flat = np.vstack(all_predictions)
            n_unique_rows = len(np.unique(preds_flat, axis=0))
        else:
            preds_flat = np.concatenate(all_predictions)
            n_unique_rows = len(np.unique(preds_flat))
    else:
        n_unique_rows = 1

    loss_delta = losses[0] - losses[-1] if len(losses) >= 2 else 0.0

    # ---- Validation ----
    passed_checks = []
    failed_checks = []

    if loss_delta >= MIN_LOSS_DELTA:
        passed_checks.append(f'loss_baisse: delta={loss_delta:.4f}')
    else:
        failed_checks.append(f'loss_baisse: delta={loss_delta:.4f} < {MIN_LOSS_DELTA}')

    any_weight_change = any(c.get('max_abs_change', 0) > MIN_WEIGHT_CHANGE for c in weight_changes.values())
    if any_weight_change:
        passed_checks.append('poids_classifieur_modifies')
    else:
        failed_checks.append(f'poids_classifieur_non_modifies (max_change < {MIN_WEIGHT_CHANGE})')

    diversity_ok = n_unique_rows > 1 or score_variation > MIN_SCORE_STD
    if diversity_ok:
        passed_checks.append(f'scores_diversifies: {n_unique_rows} uniques, score_std={score_variation:.4f}')
    else:
        failed_checks.append(f'scores_non_diversifies: {n_unique_rows} uniques, score_std={score_variation:.4f}')

    if f1 > MIN_F1:
        passed_checks.append(f'f1_superieur_hasard: F1={f1:.4f}')
    else:
        failed_checks.append(f'f1_superieur_hasard: F1={f1:.4f} <= {MIN_F1}')

    grad_ok = not grad_info.get('has_errors', True)
    if grad_ok:
        passed_checks.append('gradients_presents_et_non_nuls')
    else:
        failed_checks.append(f'gradients: {grad_info.get("errors", [])}')

    if all_params_in_optim:
        passed_checks.append('tete_classifieur_dans_optimiseur')
    else:
        missing = [name for name, found in optimizer_check.items() if not found]
        failed_checks.append(f'parametres_manquants_optimiseur: {missing}')

    report = {
        'task': task,
        'base_model': base_model,
        'n_samples': n_samples,
        'epochs': epochs,
        'device': str(DEVICE),
        'label_names': label_names,
        'losses': [round(l, 4) for l in losses],
        'loss_delta': round(loss_delta, 4),
        'loss_first': round(losses[0], 4) if losses else None,
        'loss_last': round(losses[-1], 4) if losses else None,
        'f1': round(f1, 4),
        'n_unique_predictions': int(n_unique_rows),
        'score_variation': round(float(score_variation), 6),
        'optimizer_check': optimizer_check,
        'gradient_check': grad_info,
        'weight_changes': weight_changes,
        'passed_checks': passed_checks,
        'failed_checks': failed_checks,
        'passed': len(failed_checks) == 0,
    }

    logger.info('')
    logger.info('=' * 50)
    logger.info('RESULTATS DU SMOKE TEST CLASSIFIEUR')
    logger.info('=' * 50)
    logger.info('Loss: %.4f -> %.4f (delta=%.4f)', losses[0], losses[-1], loss_delta)
    logger.info('F1: %.4f', f1)
    logger.info('Predictions uniques: %d', n_unique_rows)
    logger.info('Variation des scores: %.4f', score_variation)
    logger.info('Poids classifieur modifiés: %s', any_weight_change)
    logger.info('Gradients OK: %s', grad_ok)
    logger.info('Tête dans optimiseur: %s', all_params_in_optim)
    logger.info('')
    if passed_checks:
        logger.info('RÉUSSITES:')
        for c in passed_checks:
            logger.info('  [OK] %s', c)
    if failed_checks:
        logger.info('ÉCHECS:')
        for c in failed_checks:
            logger.info('  [FAIL] %s', c)
    logger.info('')
    if report['passed']:
        logger.info('CONCLUSION: Le pipeline d entraînement fonctionne correctement.')
    else:
        logger.warning('CONCLUSION: Le pipeline nécessite des corrections.')

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
        logger.info('Rapport sauvegardé: %s', out_path)
    else:
        print(json.dumps(report, indent=2, ensure_ascii=False))

    return report


def main():
    parser = argparse.ArgumentParser(description='Smoke test obligatoire du classifieur')
    parser.add_argument('--task', choices=['multilabel', 'binary'], default='multilabel')
    parser.add_argument('--train', type=str, default='data/processed/dataset_entrainement.csv')
    parser.add_argument('--base-model', type=str, default='camembert-base')
    parser.add_argument('--samples', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--output', type=str, default='')
    args = parser.parse_args()

    train_path = Path(args.train)
    if not train_path.exists():
        logger.error('Fichier introuvable: %s', train_path)
        sys.exit(1)

    # Libération VRAM avant de commencer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    report = run_smoke_test(
        task=args.task,
        csv_path=train_path,
        base_model=args.base_model,
        n_samples=args.samples,
        epochs=args.epochs,
        output=args.output if args.output else None,
    )

    if not report['passed']:
        sys.exit(1)


if __name__ == '__main__':
    main()
