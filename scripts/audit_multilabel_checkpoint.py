#!/usr/bin/env python3
"""
Audit CLI du checkpoint multilabel.

Usage:
    python scripts/audit_multilabel_checkpoint.py \
        --model models/multilabel_competences_v2/final \
        --output reports/multilabel_checkpoint_audit.json

Verifie :
  - presence et validite du config.json
  - presence des poids
  - architecture declaree
  - num_labels
  - id2label / label2id
  - problem_type
  - chargement strict
  - statistiques detailees de la tete de classification
  - coherence avec une initialisation aleatoire
  - comportement sur entrees de test controlees
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger('audit')

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import torch
except ImportError:
    torch = None

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except ImportError:
    AutoModelForSequenceClassification = None
    AutoTokenizer = None

from common.text import clean_text


AUDIT_REPORT: dict = {}


def section(title: str) -> None:
    logger.info('')
    logger.info('=' * 60)
    logger.info('  %s', title)
    logger.info('=' * 60)


def check_config(model_dir: Path) -> dict:
    section('CONFIGURATION')
    report: dict = {
        'config_present': False,
        'architecture': '',
        'num_labels': 0,
        'problem_type': '',
        'id2label_count': 0,
        'label2id_count': 0,
        'id2label': {},
        'label2id': {},
        'issues': [],
    }

    config_path = model_dir / 'config.json'
    if not config_path.exists():
        report['issues'].append('config.json absent')
        logger.error('config.json absent dans %s', model_dir)
        return report

    cfg = json.loads(config_path.read_text(encoding='utf-8'))
    report['config_present'] = True
    report['architecture'] = str(cfg.get('architectures', [''])[0])
    report['num_labels'] = int(cfg.get('num_labels', 0))
    report['problem_type'] = str(cfg.get('problem_type', ''))
    report['id2label'] = cfg.get('id2label', {})
    report['label2id'] = cfg.get('label2id', {})
    report['id2label_count'] = len(report['id2label'])
    report['label2id_count'] = len(report['label2id'])

    logger.info('Architecture : %s', report['architecture'])
    logger.info('num_labels  : %d', report['num_labels'])
    logger.info('problem_type: %s', report['problem_type'])
    logger.info('id2label    : %d entrees', report['id2label_count'])
    logger.info('label2id    : %d entrees', report['label2id_count'])

    if report['num_labels'] != 18:
        report['issues'].append(f'num_labels={report["num_labels"]}, attendu 18')
    if report['problem_type'] != 'multi_label_classification':
        report['issues'].append(f'problem_type={report["problem_type"]}, attendu multi_label_classification')
    if report['id2label_count'] != report['num_labels']:
        report['issues'].append(f'id2label contient {report["id2label_count"]} entrees, num_labels={report["num_labels"]}')
    if report['label2id_count'] != report['num_labels']:
        report['issues'].append(f'label2id contient {report["label2id_count"]} entrees, num_labels={report["num_labels"]}')

    if report['issues']:
        for issue in report['issues']:
            logger.warning('ISSUE: %s', issue)
    else:
        logger.info('Configuration OK')

    return report


def check_weights(model_dir: Path) -> dict:
    section('POIDS')
    report: dict = {
        'weights_present': False,
        'weights_size_bytes': 0,
        'files': [],
    }

    for ext in ('.safetensors', '.bin', '.pt', '.pth'):
        for f in model_dir.glob(f'*{ext}'):
            sz = f.stat().st_size
            report['files'].append({'name': f.name, 'size_bytes': sz})
            logger.info('Fichier: %s (%d octets)', f.name, sz)

    safetensors = list(model_dir.glob('*.safetensors'))
    bin_files = list(model_dir.glob('*.bin'))

    if safetensors:
        report['weights_present'] = True
        report['weights_size_bytes'] = sum(f.stat().st_size for f in safetensors)
        logger.info('Poids presents au format safetensors: %d octets', report['weights_size_bytes'])
    elif bin_files:
        report['weights_present'] = True
        report['weights_size_bytes'] = sum(f.stat().st_size for f in bin_files)
        logger.info('Poids presents au format bin: %d octets', report['weights_size_bytes'])
    else:
        logger.error('AUCUN fichier de poids trouve dans %s', model_dir)
        report['issues'] = ['Aucun fichier de poids trouve']

    return report


def check_strict_loading(model_dir: Path) -> dict:
    section('CHARGEMENT STRICT')
    report: dict = {
        'strict_load_success': False,
        'missing_keys': [],
        'unexpected_keys': [],
        'classifier_weight_shape': '',
        'classifier_weight_mean': 0.0,
        'classifier_weight_std': 0.0,
        'classifier_weight_min': 0.0,
        'classifier_weight_max': 0.0,
        'classifier_bias_mean': None,
        'appears_random_init': True,
        'issues': [],
    }

    if AutoModelForSequenceClassification is None:
        report['issues'].append('transformers non disponible')
        return report

    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_dir, return_dict=True
        )
        report['strict_load_success'] = True
        logger.info('Chargement strict: OK')
    except Exception as exc:
        msg = str(exc)
        logger.warning('Chargement strict echoue: %s', msg)
        if 'is not in the model' in msg:
            report['unexpected_keys'] = [msg]
        if 'are missing' in msg:
            report['missing_keys'] = [msg]

        logger.warning('Missing keys: %s', report['missing_keys'])
        logger.warning('Unexpected keys: %s', report['unexpected_keys'])

        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_dir, return_dict=True, ignore_mismatched_sizes=True
            )
            logger.info('Chargement avec ignore_mismatched_sizes: OK')
        except Exception as exc2:
            report['issues'].append(f'Echec chargement: {exc2}')
            return report

    classifier = getattr(model, 'classifier', None)
    if classifier is None:
        report['issues'].append('Pas de tete de classification (classifier)')
        logger.error('Pas de tete de classification')
        return report

    weight = getattr(classifier, 'weight', None)
    if weight is not None:
        w = weight.data.detach().cpu().numpy()
        report['classifier_weight_shape'] = str(list(w.shape))
        report['classifier_weight_mean'] = float(w.mean())
        report['classifier_weight_std'] = float(w.std())
        report['classifier_weight_min'] = float(w.min())
        report['classifier_weight_max'] = float(w.max())
        logger.info('Poids du classifier: shape=%s', report['classifier_weight_shape'])
        logger.info('  mean=%.6f, std=%.6f', report['classifier_weight_mean'],
                     report['classifier_weight_std'])
        logger.info('  min=%.6f, max=%.6f', report['classifier_weight_min'],
                     report['classifier_weight_max'])

        hidden_size = w.shape[1] if len(w.shape) > 1 else 768
        rng = np.random.RandomState(42)
        fresh_weight = rng.randn(w.shape[0], hidden_size) * 0.02
        fresh_std = float(fresh_weight.std())

        logger.info('Comparaison initialisation aleatoire: std checkpoint=%.6f, std fresh=%.6f',
                     report['classifier_weight_std'], fresh_std)

        if abs(report['classifier_weight_mean']) < 0.01 and abs(report['classifier_weight_std'] - fresh_std) < 0.005:
            report['appears_random_init'] = True
            logger.warning('ATTENTION: les poids du classifier sont coherents avec une initialisation aleatoire')
        else:
            report['appears_random_init'] = False
            logger.info('Les poids du classifier semblent avoir ete entraines')

    bias = getattr(classifier, 'bias', None)
    if bias is not None:
        b = bias.data.detach().cpu().numpy()
        report['classifier_bias_mean'] = float(b.mean())
        logger.info('Biais du classifier: mean=%.6f', report['classifier_bias_mean'])

    model.to('cpu')
    return report


def check_label_classes(model_dir: Path) -> dict:
    section('LABELS')
    report: dict = {
        'labels_file_present': False,
        'num_labels': 0,
        'labels': [],
        'issues': [],
    }

    labels_path = model_dir / 'label_classes.json'
    if not labels_path.exists():
        report['issues'].append('label_classes.json absent')
        logger.error('label_classes.json absent')
        return report

    report['labels_file_present'] = True
    labels = json.loads(labels_path.read_text(encoding='utf-8'))
    report['labels'] = [clean_text(l) for l in labels if clean_text(l)]
    report['num_labels'] = len(report['labels'])

    logger.info('Labels (%d):', report['num_labels'])
    for i, lbl in enumerate(report['labels']):
        logger.info('  [%d] %s', i, lbl)

    config_path = model_dir / 'config.json'
    if config_path.exists():
        cfg = json.loads(config_path.read_text(encoding='utf-8'))
        expected_labels = cfg.get('num_labels', 0)
        if report['num_labels'] != expected_labels:
            report['issues'].append(
                f'label_classes.json contient {report["num_labels"]} labels, '
                f'mais config.json declare num_labels={expected_labels}'
            )

    if report['num_labels'] != 18:
        report['issues'].append(f'{report["num_labels"]} labels trouves, 18 attendus')

    return report


def check_thresholds(model_dir: Path) -> dict:
    section('SEUILS')
    report: dict = {
        'thresholds_file_present': False,
        'thresholds': {},
        'issues': [],
    }

    thr_path = model_dir / 'thresholds.json'
    if not thr_path.exists():
        report['issues'].append('thresholds.json absent')
        logger.error('thresholds.json absent')
        return report

    report['thresholds_file_present'] = True
    report['thresholds'] = json.loads(thr_path.read_text(encoding='utf-8'))
    logger.info('Seuils: %s', report['thresholds'])
    return report


def check_inference(model_dir: Path, device_str: str = 'cpu') -> dict:
    section('INFERENCE SUR TEXTES DE TEST')

    if AutoTokenizer is None:
        return {'issues': ['transformers non disponible']}

    test_texts = [
        ('Python pour IA', 'Formation avancee en Python pour le Machine Learning et le Deep Learning'),
        ('Non IA', 'Formation en comptabilite et gestion d entreprise'),
        ('NLP', 'Traitement automatique du langage naturel avec BERT et GPT'),
        ('Computer Vision', 'Vision par ordinateur avec CNN, YOLO et detection d objets'),
        ('Machine Learning', 'Algorithmes de regression, classification et clustering'),
        ('Deep Learning', 'Reseaux de neurones, backpropagation, architectures transformer'),
        ('BTP', 'Formation aux metiers du batiment et travaux publics'),
        ('Coiffure', 'Formation en coiffure et soins capillaires'),
        ('Vide', ''),
        ('Ambigu', 'Techniques modernes appliquees aux processus'),
    ]

    report: dict = {
        'test_examples': [],
        'score_stats': {},
    }

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    all_score_stds: list[float] = []
    all_score_maxs: list[float] = []
    all_score_means: list[float] = []

    device = torch.device(device_str)
    model.to(device)

    for name, text in test_texts:
        cleaned = clean_text(text)
        if not cleaned:
            example = {
                'name': name,
                'text': '(vide)',
                'error': 'texte vide',
            }
            report['test_examples'].append(example)
            logger.info('[%s] TEXTE VIDE', name)
            continue

        encoded = tokenizer(
            cleaned, return_tensors='pt', padding=True,
            truncation=True, max_length=512,
        ).to(device)

        with torch.no_grad() if torch else _noop():
            logits = model(**encoded).logits
            probs = torch.sigmoid(logits) if torch else _sigmoid(logits)
            scores = probs[0].detach().cpu().tolist() if torch else probs[0].tolist()

        score_min = float(min(scores))
        score_max = float(max(scores))
        score_mean = float(sum(scores) / len(scores))
        score_std = float((sum((s - score_mean) ** 2 for s in scores) / len(scores)) ** 0.5)

        all_score_stds.append(score_std)
        all_score_maxs.append(score_max)
        all_score_means.append(score_mean)

        label_scores = [
            {'label': lbl, 'score': round(s, 4)}
            for lbl, s in zip([f'LABEL_{i}' for i in range(len(scores))], scores)
        ]
        label_scores.sort(key=lambda x: -x['score'])

        example = {
            'name': name,
            'text': cleaned[:200],
            'score_min': round(score_min, 4),
            'score_max': round(score_max, 4),
            'score_mean': round(score_mean, 4),
            'score_std': round(score_std, 4),
            'top3': label_scores[:3],
        }
        report['test_examples'].append(example)
        logger.info('[%s] min=%.4f max=%.4f mean=%.4f std=%.4f',
                     name, score_min, score_max, score_mean, score_std)

    if all_score_stds:
        report['score_stats'] = {
            'global_score_std_mean': round(float(np.mean(all_score_stds)), 4),
            'global_score_std_min': round(float(np.min(all_score_stds)), 4),
            'global_score_std_max': round(float(np.max(all_score_stds)), 4),
            'global_score_max_mean': round(float(np.mean(all_score_maxs)), 4),
            'global_score_max_min': round(float(np.min(all_score_maxs)), 4),
            'global_score_max_max': round(float(np.max(all_score_maxs)), 4),
            'any_discriminating': any(s > 0.05 or m > 0.70
                                       for s, m in zip(all_score_stds, all_score_maxs)),
        }
        if report['score_stats']['any_discriminating']:
            logger.info('Au moins un texte de test produit des scores discriminants')
        else:
            logger.warning('AUCUN texte de test ne produit de scores discriminants')

    model.to('cpu')
    return report


def _noop():
    class _Noop:
        def __enter__(self):
            return None
        def __exit__(self, *args):
            return False
    return _Noop()


def _sigmoid(logits):
    import numpy as np
    arr = logits.detach().cpu().numpy() if hasattr(logits, 'detach') else np.asarray(logits)
    return 1.0 / (1.0 + np.exp(-arr))


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Audit du checkpoint multilabel competences'
    )
    parser.add_argument(
        '--model', type=str,
        default='models/multilabel_competences_v2/final',
        help='Chemin vers le dossier du checkpoint'
    )
    parser.add_argument(
        '--output', type=str, default='',
        help='Chemin du fichier JSON de sortie'
    )
    parser.add_argument(
        '--device', type=str, default='cpu',
        help='Device pour inference (cpu ou cuda, defaut: cpu)'
    )
    args = parser.parse_args()

    model_dir = Path(args.model)
    if not model_dir.exists():
        logger.error('Dossier modele introuvable: %s', model_dir)
        sys.exit(1)

    logger.info('Audit du checkpoint: %s', model_dir.resolve())

    report = {
        'model_path': str(model_dir.resolve()),
        'audit_timestamp': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'config': check_config(model_dir),
        'weights': check_weights(model_dir),
        'strict_loading': check_strict_loading(model_dir),
        'label_classes': check_label_classes(model_dir),
        'thresholds': check_thresholds(model_dir),
        'inference': check_inference(model_dir, device_str=args.device),
    }

    section('RESUME')
    issues = []
    for section_name, section_data in report.items():
        if isinstance(section_data, dict):
            section_issues = section_data.get('issues', [])
            issues.extend([f'[{section_name}] {i}' for i in section_issues])

    if issues:
        logger.warning('ISSUES DETECTEES (%d):', len(issues))
        for issue in issues:
            logger.warning('  - %s', issue)
    else:
        logger.info('Aucune issue detectee.')

    if report['strict_loading']['appears_random_init']:
        logger.warning('CONCLUSION: Le checkpoint semble etre une initialisation aleatoire.')
        logger.warning('  Les poids du classifier sont coherents avec une init fraiche.')
        logger.warning('  Les scores d inference sont probablement tous autour de 0.5.')
    else:
        logger.info('CONCLUSION: Le checkpoint semble avoir ete entraine.')

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding='utf-8'
        )
        logger.info('Rapport sauvegarde: %s', output_path)
    else:
        print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
