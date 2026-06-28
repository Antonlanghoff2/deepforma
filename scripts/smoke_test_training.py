#!/usr/bin/env python3
"""
Mini-test de surapprentissage pour valider le pipeline d'entraînement.

Usage:
    python scripts/smoke_test_training.py \\
        --task multilabel \\
        --train data/processed/dataset_entrainement.csv \\
        --base-model camembert-base \\
        --samples 32 \\
        --epochs 10 \\
        --output reports/multilabel_smoke_test.json

Verifie :
  - baisse nette de la loss
  - changement mesurable des poids de sortie
  - predictions differentes selon les textes
  - F1 nettement superieure au hasard

Conditions d'echec :
  - loss ne baisse pas (delta < 0.05)
  - predictions toutes identiques
  - F1 <= 0.25 (hasard)
  - norme du gradient nulle
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('smoke_test')

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
    logger.error('Dependance manquante: %s', e)
    sys.exit(1)


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info('Device: %s', DEVICE)


def load_multilabel_data(csv_path: Path, n_samples: int) -> tuple[Dataset, list[str]]:
    from sklearn.preprocessing import MultiLabelBinarizer

    df = pd.read_csv(csv_path)
    if 'competences_ia' not in df.columns:
        raise ValueError('Colonne competences_ia introuvable dans %s' % csv_path)

    ia_df = df[df['statut_annotation'] == 'ia_confirmee'].copy()
    if len(ia_df) == 0:
        raise ValueError('Aucune formation ia_confirmee dans le dataset')

    ia_df = ia_df.head(n_samples).reset_index(drop=True)

    def parse_labels(value):
        if pd.isna(value) or not str(value).strip():
            return []
        return [s.strip() for s in str(value).split('|') if s.strip()]

    label_lists = ia_df['competences_ia'].apply(parse_labels).tolist()
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(label_lists)
    label_names = list(mlb.classes_)

    from scripts.clean_and_merge_datasets import clean_text, build_text_modele

    texts = []
    for _, row in ia_df.iterrows():
        row_series = build_text_modele(row)
        texts.append(clean_text(row_series))

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

    from scripts.clean_and_merge_datasets import clean_text, build_text_modele

    texts = []
    for _, row in df.iterrows():
        row_series = build_text_modele(row)
        texts.append(clean_text(row_series))

    ds = Dataset.from_dict({'text': texts, 'label': df['label'].tolist()})
    return ds


def check_gradients(model: nn.Module, optim: torch.optim.Optimizer, batch, task: str) -> dict:
    optim.zero_grad()

    if task == 'multilabel':
        logits = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']).logits
        labels = batch['labels'].float()
        loss = nn.BCEWithLogitsLoss()(logits, labels)
    else:
        logits = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']).logits
        labels = batch['labels']
        loss = nn.CrossEntropyLoss()(logits, labels)

    loss.backward()

    grad_report = {}
    classifier_params = {
        'classifier.dense.weight': None,
        'classifier.dense.bias': None,
        'classifier.out_proj.weight': None,
        'classifier.out_proj.bias': None,
    }

    for name, param in model.named_parameters():
        if name in classifier_params:
            if param.grad is None:
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
                }
                logger.info('Gradient %s: norm=%.6f', name, grad_report[name]['grad_norm'])

    errors = []
    for name, report in grad_report.items():
        if report.get('gradient_is_none'):
            errors.append('Gradient absent pour %s' % name)
        elif report.get('gradient_is_zero'):
            errors.append('Gradient nul pour %s' % name)

    return {
        'gradients': grad_report,
        'errors': errors,
        'loss_before_step': float(loss.item()),
        'has_errors': len(errors) > 0,
    }


def weight_change(model: nn.Module, before: dict, after: dict) -> dict:
    changes = {}
    for name in before:
        if name in after:
            diff = (after[name] - before[name]).abs().max().item()
            changes[name] = {
                'before_norm': float(before[name].norm().item()),
                'after_norm': float(after[name].norm().item()),
                'max_abs_change': round(diff, 8),
            }
    return changes


def run_smoke_test(
    task: str,
    csv_path: Path,
    base_model: str,
    n_samples: int,
    epochs: int,
    output: str | None,
) -> dict:
    logger.info('Task: %s', task)
    logger.info('CSV: %s', csv_path)
    logger.info('Base model: %s', base_model)
    logger.info('Samples: %d, Epochs: %d', n_samples, epochs)

    if task == 'multilabel':
        ds, label_names = load_multilabel_data(csv_path, n_samples)
        num_labels = len(label_names)
        problem_type = 'multi_label_classification'
        logger.info('Labels (%d): %s', num_labels, label_names)
    else:
        ds = load_binary_data(csv_path, n_samples)
        num_labels = 2
        problem_type = None
        label_names = ['non_ia', 'ia']
        logger.info('Task binaire: %d echantillons', len(ds))

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

    # Snapshot weights before training
    weights_before = {
        name: param.data.detach().cpu().clone()
        for name, param in model.named_parameters()
        if 'classifier' in name
    }

    optim = torch.optim.AdamW(model.parameters(), lr=2e-5)
    num_steps = epochs * len(loader)
    scheduler = get_scheduler('linear', optimizer=optim, num_warmup_steps=0, num_training_steps=num_steps)

    losses = []
    all_predictions = []
    all_labels = []
    all_sigmoid_scores: list[np.ndarray] = []
    grad_info = {}
    weight_changes = {}

    first_batch = None

    for epoch in range(epochs):
        epoch_losses = []
        for batch in loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items() if k in ('input_ids', 'attention_mask', 'labels', 'label')}

            # Check gradients on first batch
            if first_batch is None:
                grad_info = check_gradients(model, optim, batch, task)
                first_batch = True
                if grad_info['has_errors']:
                    logger.warning('Problemes de gradient detectes en debut')

            optim.zero_grad()

            if task == 'multilabel':
                logits = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']).logits
                labels = batch['labels'].float()
                loss = nn.BCEWithLogitsLoss()(logits, labels)
                sigmoid_scores = torch.sigmoid(logits).detach().cpu().numpy()
                all_sigmoid_scores.append(sigmoid_scores)
                preds = (sigmoid_scores >= 0.35).astype(int)
                all_predictions.append(preds)
                all_labels.append(labels.detach().cpu().numpy().astype(int))
            else:
                logits = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']).logits
                labels = batch['labels']
                loss = nn.CrossEntropyLoss()(logits, labels)
                probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
                all_sigmoid_scores.append(probs)
                preds = np.argmax(probs, axis=-1)
                all_predictions.append(preds)
                all_labels.append(labels.detach().cpu().numpy())

            loss.backward()
            optim.step()
            scheduler.step()

            epoch_losses.append(float(loss.item()))

        avg_loss = float(np.mean(epoch_losses))
        losses.append(avg_loss)
        logger.info('Epoch %d/%d: loss=%.4f', epoch + 1, epochs, avg_loss)

    # Snapshot weights after training
    weights_after = {
        name: param.data.detach().cpu().clone()
        for name, param in model.named_parameters()
        if 'classifier' in name
    }

    weight_changes = weight_change(model, weights_before, weights_after)

    # Compute F1
    if len(all_predictions) > 0:
        if task == 'multilabel':
            y_true = np.vstack(all_labels)
            y_pred = np.vstack(all_predictions)
            f1 = float(f1_score(y_true, y_pred, average='micro', zero_division=0))
        else:
            y_true = np.concatenate(all_labels)
            y_pred = np.concatenate(all_predictions)
            f1 = float(f1_score(y_true, y_pred, average='binary', zero_division=0))
    else:
        f1 = 0.0

    # Determine diversity
    score_variation = 0.0
    if task == 'multilabel':
        if len(all_predictions) > 0:
            preds_flat = np.vstack(all_predictions)
            n_unique_rows = len(np.unique(preds_flat, axis=0))
        else:
            n_unique_rows = 1
        if len(all_sigmoid_scores) > 0:
            scores_flat = np.vstack(all_sigmoid_scores)
            score_variation = float(np.std([s.max() for s in scores_flat]))
    else:
        if len(all_predictions) > 0:
            preds_flat = np.concatenate(all_predictions)
            n_unique_rows = len(np.unique(preds_flat))
        else:
            n_unique_rows = 1

    loss_delta = losses[0] - losses[-1] if len(losses) >= 2 else 0.0

    passed_checks = []
    failed_checks = []

    if loss_delta >= 0.05:
        passed_checks.append('loss_baisse: delta=%.4f' % loss_delta)
    else:
        failed_checks.append('loss_baisse: delta=%.4f < 0.05' % loss_delta)

    if task == 'multilabel':
        diversity_ok = n_unique_rows > 1 or score_variation > 0.01
    else:
        diversity_ok = n_unique_rows > 1
    if diversity_ok:
        passed_checks.append('predictions_diversifiees: %d uniques, score_variation=%.4f' % (n_unique_rows, score_variation if task == 'multilabel' else 0))
    else:
        failed_checks.append('predictions_diversifiees: %d uniques, score_variation=%.4f' % (n_unique_rows, score_variation if task == 'multilabel' else 0))

    if f1 > 0.25:
        passed_checks.append('f1_superieur_hasard: F1=%.4f' % f1)
    else:
        failed_checks.append('f1_superieur_hasard: F1=%.4f <= 0.25' % f1)

    any_weight_change = any(c.get('max_abs_change', 0) > 1e-6 for c in weight_changes.values())
    if any_weight_change:
        passed_checks.append('poids_classifieur_modifies')
    else:
        failed_checks.append('poids_classifieur_non_modifies')

    grad_ok = not grad_info.get('has_errors', True)
    if grad_ok:
        passed_checks.append('gradients_presents_et_non_nuls')
    else:
        failed_checks.append('gradients: %s' % grad_info.get('errors', []))

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
        'gradient_check': grad_info,
        'weight_changes': weight_changes,
        'passed_checks': passed_checks,
        'failed_checks': failed_checks,
        'passed': len(failed_checks) == 0,
    }

    logger.info('')
    logger.info('=' * 50)
    logger.info('RESULTATS DU SMOKE TEST')
    logger.info('=' * 50)
    logger.info('Loss: %.4f -> %.4f (delta=%.4f)', losses[0], losses[-1], loss_delta)
    logger.info('F1: %.4f', f1)
    logger.info('Predictions uniques: %d', n_unique_rows)
    logger.info('Poids classifieur modifies: %s', any_weight_change)
    logger.info('Gradients OK: %s', grad_ok)
    logger.info('')
    if passed_checks:
        logger.info('REUSSITES:')
        for c in passed_checks:
            logger.info('  [OK] %s', c)
    if failed_checks:
        logger.info('ECHECS:')
        for c in failed_checks:
            logger.info('  [FAIL] %s', c)
    logger.info('')
    if report['passed']:
        logger.info('CONCLUSION: Le pipeline d entrainement fonctionne correctement.')
    else:
        logger.warning('CONCLUSION: Le pipeline d entrainement necessite des corrections.')

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
        logger.info('Rapport sauvegarde: %s', out_path)
    else:
        print(json.dumps(report, indent=2, ensure_ascii=False))

    return report


def main():
    parser = argparse.ArgumentParser(description='Smoke test d entrainement')
    parser.add_argument('--task', choices=['multilabel', 'binary'], default='multilabel')
    parser.add_argument('--train', type=str, default='data/processed/dataset_entrainement.csv')
    parser.add_argument('--base-model', type=str, default='camembert-base')
    parser.add_argument('--samples', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--output', type=str, default='')
    args = parser.parse_args()

    train_path = Path(args.train)
    if not train_path.exists():
        logger.error('Fichier introuvable: %s', train_path)
        sys.exit(1)

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
