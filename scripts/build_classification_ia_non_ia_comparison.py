
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def row_from_artifacts(model_name: str, model_dir: Path) -> dict[str, Any]:
    metrics = load_json(model_dir / 'metrics_test.json')
    threshold_payload = load_json(model_dir / 'threshold.json')
    metadata = load_json(model_dir / 'metadata.json')
    return {
        'model_name': model_name,
        'accuracy': metrics.get('accuracy'),
        'balanced_accuracy': metrics.get('balanced_accuracy'),
        'precision_IA': metrics.get('precision_ia'),
        'recall_IA': metrics.get('recall_ia'),
        'f1_IA': metrics.get('f1_ia'),
        'f1_macro': metrics.get('f1_macro'),
        'f1_weighted': metrics.get('f1_weighted'),
        'roc_auc': metrics.get('roc_auc'),
        'pr_auc': metrics.get('pr_auc'),
        'mcc': metrics.get('mcc'),
        'cohen_kappa': metrics.get('cohen_kappa'),
        'log_loss': metrics.get('log_loss'),
        'brier_score': metrics.get('brier_score'),
        'specificity': metrics.get('specificity'),
        'training_time_seconds': metrics.get('training_time_seconds'),
        'inference_time_seconds': metrics.get('inference_time_seconds'),
        'latency_ms_per_sample': metrics.get('latency_ms_per_sample'),
        'model_size_mb': metrics.get('model_size_mb'),
        'threshold': threshold_payload.get('threshold'),
        'model_type': metadata.get('model_type'),
    }


def render_report(df: pd.DataFrame) -> str:
    ranked = df.set_index('model_name')
    best_accuracy = ranked['accuracy'].astype(float).idxmax()
    best_f1_macro = ranked['f1_macro'].astype(float).idxmax()
    best_generalization = ranked['balanced_accuracy'].astype(float).idxmax()
    fastest = ranked['inference_time_seconds'].astype(float).idxmin()
    lightest = ranked['model_size_mb'].astype(float).idxmin()

    lines = [
        '# Comparaison des modèles IA / non-IA',
        '',
        '## Résumé',
        '',
        f'- Meilleure accuracy : `{best_accuracy}`',
        f'- Meilleure F1 macro : `{best_f1_macro}`',
        f'- Meilleure généralisation observée via balanced accuracy : `{best_generalization}`',
        f"- Le plus rapide à l'inférence : `{fastest}`",
        f'- Le plus léger sur disque : `{lightest}`',
        '',
        '## Lecture méthodologique',
        '',
        'Un modèle ML peut battre un modèle DL sur un petit jeu de données comme celui-ci pour trois raisons principales :',
        '',
        '- les TF-IDF capturent directement des indices lexicaux discriminants ;',
        '- la régression logistique apprend vite et généralise bien sur peu de données ;',
        "- un TextCNN from scratch a plus de capacité mais moins de signal pour l'exploiter avec seulement 1 800 lignes.",
        '',
        '## Tableau des résultats',
        '',
        markdown_table(df),
        '',
        '## Recommandation',
        '',
        f"La recommandation dépend des métriques finales. Ici, le modèle à privilégier est `{best_f1_macro}` si l'objectif principal est la robustesse globale, ou `{best_accuracy}` si l'objectif est la précision brute. En pratique, il faut aussi arbitrer avec la latence et la taille.",
        '',
        "## Types d'erreurs",
        '',
        'Comparer les faux positifs et faux négatifs via les fichiers `error_analysis.csv` des deux notebooks. Le modèle qui sur-prédit la classe IA aura plus de faux positifs ; celui qui reste trop conservateur aura plus de faux négatifs.',
    ]
    return '\n'.join(lines)




def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        '| ' + ' | '.join(columns) + ' |',
        '| ' + ' | '.join(['---'] * len(columns)) + ' |',
    ]
    for _, row in frame.iterrows():
        values = ["" if pd.isna(row[col]) else str(row[col]) for col in columns]
        lines.append('| ' + ' | '.join(values) + ' |')
    return '\n'.join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Génère la comparaison IA/non-IA')
    parser.add_argument('--ml-dir', type=Path, default=Path('artifacts/classification_ia_non_ia_ml'))
    parser.add_argument('--textcnn-dir', type=Path, default=Path('artifacts/classification_ia_non_ia_textcnn'))
    parser.add_argument('--output-dir', type=Path, default=Path('artifacts/classification_ia_non_ia_comparison'))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        row_from_artifacts('machine_learning', args.ml_dir),
        row_from_artifacts('textcnn', args.textcnn_dir),
    ]
    df = pd.DataFrame(rows)
    df.to_csv(args.output_dir / 'model_comparison.csv', index=False, encoding='utf-8')
    (args.output_dir / 'model_comparison.md').write_text(render_report(df), encoding='utf-8')
    print(df.to_string(index=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
