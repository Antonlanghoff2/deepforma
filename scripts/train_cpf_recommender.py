from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from deepforma.training.cpf_dataset import load_jsonl, validate_rows
from deepforma.training.cpf_trainer import CPFRecommenderTrainer, TrainingConfig


LOGGER = logging.getLogger(__name__)
DEFAULT_OUTPUT_DIR = Path('models/cpf-recommender')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Entraîne le recommender CPF Sentence-Transformer')
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--validation', type=Path, required=True)
    parser.add_argument('--base-model', type=str, default='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--learning-rate', type=float, default=2e-5)
    parser.add_argument('--warmup-ratio', type=float, default=0.1)
    parser.add_argument('--max-seq-length', type=int, default=256)
    parser.add_argument('--loss', type=str, default='MultipleNegativesRankingLoss', choices=['MultipleNegativesRankingLoss', 'TripletLoss'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--gradient-accumulation', type=int, default=2)
    parser.add_argument('--mixed-precision', action='store_true', default=True)
    parser.add_argument('--no-mixed-precision', dest='mixed_precision', action='store_false')
    parser.add_argument('--resume-from-checkpoint', type=Path, default=None)
    return parser


def _print_preflight(train_rows: list[dict], validation_rows: list[dict]) -> None:
    combined = train_rows + validation_rows
    validation = validate_rows(combined, min_positives=1)
    summary = validation.summary
    print(f'Formations: {summary.formation_count}')
    print(f'Textes exploitables: {summary.text_count}')
    print(f'Formations avec compétences: {summary.skill_count}')
    print(f'Requêtes: {summary.query_count}')
    print(f'Positifs: {summary.positive_count}')
    print(f'Négatifs: {summary.negative_count}')
    print(f'Répartition négatifs: {json.dumps(summary.negative_type_counts, ensure_ascii=False)}')
    print(f'Splits: {json.dumps(summary.split_counts, ensure_ascii=False)}')
    print(f'Taux de labels heuristiques: {summary.heuristic_label_rate:.4f}')
    print(f'Recouvrement certifications train/test: {summary.certification_overlap_rate:.4f}')
    if train_rows[:3]:
        print('Exemples:')
        for row in train_rows[:3]:
            print(json.dumps({'query': row['query'], 'positive_uid': row['positive_uid'], 'negative_type': row['negative_type']}, ensure_ascii=False))
    if not validation.ok:
        raise ValueError(' ; '.join(validation.errors))
    if not validation_rows:
        raise ValueError("Le split de validation est vide.")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s %(message)s')
    args = build_parser().parse_args()
    train_rows = load_jsonl(args.train)
    validation_rows = load_jsonl(args.validation)
    if not train_rows:
        raise ValueError("Le dataset d'entraînement est vide.")
    if not validation_rows:
        raise ValueError("Le split de validation est vide.")
    _print_preflight(train_rows, validation_rows)
    config = TrainingConfig(
        base_model=args.base_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        max_seq_length=args.max_seq_length,
        loss=args.loss,
        seed=args.seed,
        device=args.device,
        gradient_accumulation=args.gradient_accumulation,
        mixed_precision=args.mixed_precision,
        output_dir=str(args.output_dir),
    )
    trainer = CPFRecommenderTrainer(config)
    result = trainer.train(args.train, args.validation, resume_from_checkpoint=args.resume_from_checkpoint)
    LOGGER.info('Modèle final écrit dans %s', result['model_path'])
    LOGGER.info('Métriques validation: %s', json.dumps(result['metrics'], ensure_ascii=False))


if __name__ == '__main__':
    main()
