#!/usr/bin/env python3
"""Entraine le modele de recommandation CPF avec Sentence-Transformers.

Le chemin d'entraînement reste compatible avec `make cpf-train`, mais
les paires JSONL sont converties en ``InputExample`` avant l'appel à
``SentenceTransformer.fit()``.
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import random
import sys
from collections import defaultdict
from contextlib import nullcontext
from unittest.mock import patch
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_cpf_recommender")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import numpy as np
    import torch
    from sentence_transformers import InputExample, SentenceTransformer, losses
    try:
        from sentence_transformers.datasets import NoDuplicatesDataLoader
    except Exception:  # pragma: no cover - compatibilite version
        NoDuplicatesDataLoader = None
    from torch.utils.data import DataLoader
except ImportError as exc:
    logger.error("Dependance manquante: %s", exc)
    sys.exit(1)

from common.text import clean_text, normalize_for_match
from deepforma.training.cpf_dataset import build_group_id, load_jsonl, normalize_training_row, split_by_group, timestamp_iso


DEFAULT_BASE_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
DEFAULT_BATCH_SIZE = 16
DEFAULT_GRADIENT_ACCUMULATION = 2
DEFAULT_EPOCHS = 3
DEFAULT_MAX_SEQ_LENGTH = 256
DEFAULT_LR = 2e-5
DEFAULT_WARMUP_RATIO = 0.1
DEFAULT_MAX_TRAIN_SAMPLES = None
DEFAULT_MAX_PAIRS_PER_FORMATION = 10

PAIR_TEXT_KEYS: tuple[tuple[str, str], ...] = (
    ('anchor_text', 'positive_text'),
    ('query', 'positive_text'),
    ('text_a', 'text_b'),
    ('source_text', 'target_text'),
    ('anchor', 'positive'),
)
KNOWN_PAIR_KEYS = {key for pair in PAIR_TEXT_KEYS for key in pair} | {
    'negative_text', 'negative', 'candidate_text', 'pair_text', 'text',
}

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingConfig:
    base_model: str = DEFAULT_BASE_MODEL
    epochs: int = DEFAULT_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    learning_rate: float = DEFAULT_LR
    warmup_ratio: float = DEFAULT_WARMUP_RATIO
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH
    loss: str = 'MultipleNegativesRankingLoss'
    seed: int = 42
    device: str | None = None
    gradient_accumulation: int = DEFAULT_GRADIENT_ACCUMULATION
    mixed_precision: bool = True
    output_dir: str = 'models/cpf-recommender'
    max_train_samples: int | None = DEFAULT_MAX_TRAIN_SAMPLES
    max_pairs_per_formation: int = DEFAULT_MAX_PAIRS_PER_FORMATION
    trainer_api: str = 'fit'


@dataclass(frozen=True)
class TrainingMetrics:
    validation_examples: int
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    ndcg_at_10: float
    mean_positive_similarity: float
    mean_negative_similarity: float


def resolve_device(requested: str | None = None) -> str:
    if requested:
        return requested
    if not torch.cuda.is_available():
        return 'cpu'
    try:
        free_bytes, _total_bytes = torch.cuda.mem_get_info()
        if free_bytes < 1_000_000_000:
            return 'cpu'
    except Exception:
        pass
    return 'cuda'


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _infer_loss(name: str):
    normalized = clean_text(name).lower()
    if 'triplet' in normalized:
        return losses.TripletLoss
    return losses.MultipleNegativesRankingLoss


def _normalize_pair_text(value: Any) -> str:
    return clean_text(value)


def _available_keys(row: dict[str, Any]) -> list[str]:
    return sorted(key for key, value in row.items() if clean_text(value))


def _pair_texts(row: dict[str, Any]) -> tuple[str, str]:
    if not isinstance(row, dict):
        raise TypeError(f'Chaque ligne doit etre un dictionnaire, recu: {type(row)!r}')
    if not any(key in row for key in KNOWN_PAIR_KEYS):
        available = _available_keys(row)
        expected = ', '.join(f'{a}/{b}' for a, b in PAIR_TEXT_KEYS)
        raise ValueError(
            "Structure de paire inconnue. "
            f"Cles disponibles: {available}. Cles attendues: {expected}."
        )
    for left_key, right_key in PAIR_TEXT_KEYS:
        left = clean_text(row.get(left_key))
        right = clean_text(row.get(right_key))
        if not left or not right:
            continue
        return left, right
    return '', ''


def row_to_input_example(row: dict[str, Any]) -> InputExample | None:
    left, right = _pair_texts(row)
    if not left or not right or normalize_for_match(left) == normalize_for_match(right):
        return None
    return InputExample(texts=[left, right])


def _pair_text_key(row: dict[str, Any]) -> tuple[str, str] | None:
    left, right = _pair_texts(row)
    if not left or not right:
        return None
    return normalize_for_match(left), normalize_for_match(right)


def _filter_positive_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int, int]:
    valid_rows: list[dict[str, Any]] = []
    invalid_rows = 0
    identical_rows = 0
    for row in rows:
        try:
            left, right = _pair_texts(row)
        except ValueError:
            raise
        if not left or not right:
            invalid_rows += 1
            continue
        if normalize_for_match(left) == normalize_for_match(right):
            identical_rows += 1
            invalid_rows += 1
            continue
        valid_rows.append(row)
    return valid_rows, invalid_rows, identical_rows


def _dedupe_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    seen: set[tuple[str, str]] = set()
    deduped: list[dict[str, Any]] = []
    removed = 0
    for row in rows:
        key = _pair_text_key(row)
        if key is None:
            continue
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        deduped.append(row)
    return deduped, removed


def _round_robin_limit(rows: list[dict[str, Any]], max_samples: int | None, seed: int) -> list[dict[str, Any]]:
    if max_samples is None or max_samples <= 0 or len(rows) <= max_samples:
        return rows
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        group_id = str(row.get('group_id') or build_group_id(row))
        grouped[group_id].append(row)
    for group_rows in grouped.values():
        group_rows.sort(key=lambda item: (
            normalize_for_match(item.get('query') or item.get('anchor_text') or item.get('positive_text') or ''),
            normalize_for_match(item.get('positive_text') or item.get('candidate') or item.get('text_b') or ''),
            clean_text(item.get('query_id') or item.get('positive_uid') or ''),
        ))
    group_ids = list(grouped)
    rng = random.Random(seed)
    rng.shuffle(group_ids)
    selected: list[dict[str, Any]] = []
    cursor = 0
    while len(selected) < max_samples and group_ids:
        group_id = group_ids[cursor % len(group_ids)]
        bucket = grouped[group_id]
        if bucket:
            selected.append(bucket.pop(0))
        else:
            group_ids.remove(group_id)
            if not group_ids:
                break
            cursor -= 1
        cursor += 1
    return selected


def load_pairs(path: str | Path) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    with Path(path).open(encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


def _build_loader(examples: list[InputExample], batch_size: int):
    if NoDuplicatesDataLoader is not None:
        return NoDuplicatesDataLoader(examples, batch_size=batch_size)
    return DataLoader(examples, shuffle=True, batch_size=batch_size, collate_fn=lambda batch: batch)


def _candidate_text(row: dict[str, Any]) -> str:
    return clean_text(row.get('positive') or row.get('positive_text') or row.get('target_text') or row.get('candidate_text'))


def _build_candidate_corpus(rows: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], int]:
    corpus: dict[str, dict[str, Any]] = {}
    conflicts = 0
    for row in rows:
        normalized = normalize_training_row(row)
        uid = normalized['positive_uid']
        text = _candidate_text(normalized)
        if not text:
            continue
        existing = corpus.get(uid)
        if existing is None:
            corpus[uid] = normalized
            continue
        existing_text = _candidate_text(existing)
        if normalize_for_match(existing_text) == normalize_for_match(text):
            continue
        conflicts += 1
        if len(text) > len(existing_text):
            LOGGER.warning(
                'Conflit positive_uid=%s, remplacement par texte plus complet: %s -> %s',
                uid,
                existing_text[:80],
                text[:80],
            )
            corpus[uid] = normalized
        else:
            LOGGER.warning(
                'Conflit positive_uid=%s, texte conserve: %s vs %s',
                uid,
                existing_text[:80],
                text[:80],
            )
    return corpus, conflicts


def _validation_metrics(model: SentenceTransformer, rows: list[dict[str, Any]], *, candidate_rows: list[dict[str, Any]] | None = None) -> TrainingMetrics:
    if not rows:
        raise ValueError('Le split de validation est vide.')
    candidates_raw = candidate_rows or rows
    candidate_corpus, conflicts = _build_candidate_corpus(candidates_raw)
    validation_rows = [normalize_training_row(row) for row in rows]
    missing_positive_uids = [row['positive_uid'] for row in validation_rows if row['positive_uid'] not in candidate_corpus]
    LOGGER.info('Candidats bruts: %d', len(candidates_raw))
    LOGGER.info('Candidats uniques: %d', len(candidate_corpus))
    LOGGER.info('Conflits uid/texte: %d', conflicts)
    LOGGER.info('Positive_uid absents: %d', len(missing_positive_uids))
    if candidate_corpus:
        sample_uid, sample_row = next(iter(candidate_corpus.items()))
        LOGGER.info('Exemple candidat: %s -> %s', sample_uid, _candidate_text(sample_row)[:120])
    if missing_positive_uids:
        missing_preview = ', '.join(missing_positive_uids[:10])
        raise ValueError(
            'Des positive_uid de validation sont absents du corpus candidat: ' + missing_preview
        )
    candidate_ids = list(candidate_corpus)
    candidate_texts = [_candidate_text(candidate_corpus[uid]) for uid in candidate_ids]
    candidate_embeddings = torch.as_tensor(model.encode(candidate_texts, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False))
    query_embeddings = torch.as_tensor(model.encode([row['anchor'] for row in validation_rows], convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False))
    positive_scores: list[float] = []
    negative_scores: list[float] = []
    ranks: list[int] = []
    candidate_index = {uid: idx for idx, uid in enumerate(candidate_ids)}
    for idx, row in enumerate(validation_rows):
        sims = torch.matmul(candidate_embeddings, query_embeddings[idx]).detach().cpu().numpy().tolist()
        ranking = sorted(zip(candidate_ids, sims), key=lambda item: item[1], reverse=True)
        positive_uid = row['positive_uid']
        positive_rank = next((rank for rank, (uid, _) in enumerate(ranking, start=1) if uid == positive_uid), len(ranking) + 1)
        ranks.append(positive_rank)
        positive_scores.append(float(sims[candidate_index[positive_uid]]))
        negative_uid = clean_text(row.get('negative_uid'))
        if negative_uid and negative_uid in candidate_index:
            negative_scores.append(float(sims[candidate_index[negative_uid]]))
    total = len(ranks)
    recall_at_1 = sum(1 for rank in ranks if rank <= 1) / total
    recall_at_5 = sum(1 for rank in ranks if rank <= 5) / total
    recall_at_10 = sum(1 for rank in ranks if rank <= 10) / total
    precision_at_5 = sum(1 for rank in ranks if rank <= 5) / (total * 5)
    mrr = sum(1.0 / rank for rank in ranks) / total
    ndcg_at_10 = sum(1.0 / np.log2(rank + 1) if rank <= 10 else 0.0 for rank in ranks) / total
    return TrainingMetrics(
        validation_examples=len(validation_rows),
        recall_at_1=round(recall_at_1, 4),
        recall_at_5=round(recall_at_5, 4),
        recall_at_10=round(recall_at_10, 4),
        mrr=round(mrr, 4),
        ndcg_at_10=round(ndcg_at_10, 4),
        mean_positive_similarity=round(float(np.mean(positive_scores)) if positive_scores else 0.0, 4),
        mean_negative_similarity=round(float(np.mean(negative_scores)) if negative_scores else 0.0, 4),
    )


class CPFRecommenderTrainer:
    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.device = resolve_device(config.device)
        self.output_dir = Path(config.output_dir)
        self.checkpoints_dir = self.output_dir / 'checkpoints'
        self.final_dir = self.output_dir / 'final'

    def load_model(self) -> SentenceTransformer:
        try:
            model = SentenceTransformer(self.config.base_model, device=self.device)
        except torch.OutOfMemoryError:
            if self.device != 'cuda':
                raise
            torch.cuda.empty_cache()
            self.device = 'cpu'
            model = SentenceTransformer(self.config.base_model, device=self.device)
        except RuntimeError as exc:
            if self.device == 'cuda' and 'out of memory' in str(exc).lower():
                torch.cuda.empty_cache()
                self.device = 'cpu'
                model = SentenceTransformer(self.config.base_model, device=self.device)
            else:
                raise
        model.max_seq_length = self.config.max_seq_length
        return model

    def _load_dataset(self, path: str | Path) -> list[dict[str, Any]]:
        rows = load_jsonl(path)
        if not rows:
            raise ValueError("Le dataset d'entraînement est vide.")
        return rows

    def _prepare_training_rows(self, rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
        valid_rows, invalid_rows, identical_rows = _filter_positive_rows(rows)
        deduped_rows, duplicate_rows = _dedupe_rows(valid_rows)
        normalized_rows = [normalize_training_row(row) for row in deduped_rows]
        if self.config.max_train_samples:
            normalized_rows = _round_robin_limit(normalized_rows, self.config.max_train_samples, self.config.seed)
        counts = {
            'loaded': len(rows),
            'valid': len(valid_rows),
            'invalid': invalid_rows,
            'identical_removed': identical_rows,
            'duplicate_rows': duplicate_rows,
            'train_rows': len(normalized_rows),
        }
        return normalized_rows, counts

    def _build_input_examples(self, rows: list[dict[str, Any]]) -> list[InputExample]:
        examples: list[InputExample] = []
        for row in rows:
            example = row_to_input_example(row)
            if example is not None:
                if len(example.texts) != 2:
                    raise ValueError('Chaque InputExample doit contenir exactement deux textes.')
                examples.append(example)
        return examples

    def train(
        self,
        input_pairs_path: str | Path,
        *,
        resume_from_checkpoint: str | Path | None = None,
    ) -> dict[str, Any]:
        all_rows = self._load_dataset(input_pairs_path)
        training_rows, counts = self._prepare_training_rows(all_rows)
        if counts['train_rows'] < 2:
            raise ValueError('Pas assez de paires valides apres nettoyage pour entrainer le modele.')
        splits = split_by_group(training_rows, seed=self.config.seed)
        train_rows = splits['train']
        validation_rows = splits['validation']
        if not validation_rows:
            raise ValueError('Le split de validation est vide apres separation par groupe.')

        _seed_everything(self.config.seed)
        model = self.load_model()
        train_examples = self._build_input_examples(train_rows)
        validation_examples = self._build_input_examples(validation_rows)
        if not train_examples:
            raise ValueError("Aucun InputExample valide n'a pu etre construit.")
        if any(len(example.texts) != 2 for example in train_examples):
            raise ValueError('Tous les InputExample doivent contenir exactement deux textes.')

        train_loader = _build_loader(train_examples, self.config.batch_size)
        batches_per_epoch = max(1, len(train_loader))
        total_steps = batches_per_epoch * self.config.epochs
        loss_cls = _infer_loss(self.config.loss)
        warmup_steps = max(1, int(batches_per_epoch * self.config.epochs * self.config.warmup_ratio))

        LOGGER.info('Lignes chargees: %d', counts['loaded'])
        LOGGER.info('Paires valides: %d', counts['valid'])
        LOGGER.info('Lignes invalides: %d', counts['invalid'])
        LOGGER.info('Paires identiques supprimees: %d', counts['identical_removed'])
        LOGGER.info('Doublons exacts supprimes: %d', counts['duplicate_rows'])
        LOGGER.info('Paires gardees pour entrainement: %d', len(train_examples))
        LOGGER.info('Batches par epoque: %d', batches_per_epoch)
        LOGGER.info('Etapes totales: %d', total_steps)
        if train_examples:
            preview = ' | '.join(text[:120] for text in train_examples[0].texts)
            LOGGER.info('Premiere paire (anonymisee): %s', preview)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        train_loss = loss_cls(model)
        fit_kwargs = {
            'train_objectives': [(train_loader, train_loss)],
            'epochs': self.config.epochs,
            'warmup_steps': warmup_steps,
            'optimizer_params': {'lr': self.config.learning_rate},
            'output_path': str(self.final_dir),
            'show_progress_bar': False,
            'use_amp': bool(self.config.mixed_precision and self.device == 'cuda'),
            'checkpoint_path': str(self.checkpoints_dir),
            'checkpoint_save_steps': max(1, batches_per_epoch),
            'checkpoint_save_total_limit': 3,
            'save_best_model': False,
        }
        if resume_from_checkpoint:
            fit_kwargs['resume_from_checkpoint'] = str(resume_from_checkpoint)
        fit_signature = inspect.signature(model.fit)
        if 'gradient_accumulation_steps' in fit_signature.parameters:
            fit_kwargs['gradient_accumulation_steps'] = self.config.gradient_accumulation
        device_context = patch.object(torch.cuda, 'is_available', lambda: False) if self.device == 'cpu' else nullcontext()
        try:
            with device_context:
                model.fit(**fit_kwargs)
        except Exception as exc:
            message = f'{type(exc).__name__}: {exc}'.lower()
            if self.device != 'cuda' or 'out of memory' not in message:
                raise
            LOGGER.warning("Bascule vers CPU apres OOM CUDA pendant l'entrainement.")
            torch.cuda.empty_cache()
            self.device = 'cpu'
            model = self.load_model()
            train_loss = loss_cls(model)
            fit_kwargs['use_amp'] = False
            fit_kwargs['train_objectives'] = [(train_loader, train_loss)]
            device_context = patch.object(torch.cuda, 'is_available', lambda: False)
            with device_context:
                model.fit(**fit_kwargs)

        model.save(str(self.final_dir))
        metrics = _validation_metrics(model, validation_rows, candidate_rows=train_rows + validation_rows)
        manifest = {
            'base_model': self.config.base_model,
            'device': self.device,
            'seed': self.config.seed,
            'loss': self.config.loss,
            'epochs': self.config.epochs,
            'batch_size': self.config.batch_size,
            'learning_rate': self.config.learning_rate,
            'warmup_ratio': self.config.warmup_ratio,
            'max_seq_length': self.config.max_seq_length,
            'gradient_accumulation': self.config.gradient_accumulation,
            'mixed_precision': self.config.mixed_precision,
            'dataset_hash': self._dataset_hash(train_rows + validation_rows),
            'training_at': timestamp_iso(),
            'validation_metrics': asdict(metrics),
            'train_examples': len(train_examples),
            'validation_examples': len(validation_examples),
            'batches_per_epoch': batches_per_epoch,
            'total_steps': total_steps,
        }
        (self.output_dir / 'training_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        model_card = {
            'base_model': self.config.base_model,
            'device': self.device,
            'loss': self.config.loss,
            'seed': self.config.seed,
            'max_train_samples': self.config.max_train_samples,
            'max_pairs_per_formation': self.config.max_pairs_per_formation,
        }
        (self.output_dir / 'config.json').write_text(json.dumps(model_card, ensure_ascii=False, indent=2), encoding='utf-8')
        LOGGER.info('Modele final ecrit dans %s', self.final_dir)
        LOGGER.info('MetrIques validation: %s', json.dumps(asdict(metrics), ensure_ascii=False))
        return {'model_path': str(self.final_dir), 'manifest': manifest, 'metrics': asdict(metrics)}

    @staticmethod
    def _dataset_hash(rows: list[dict[str, Any]]) -> str:
        payload = json.dumps(rows, ensure_ascii=False, sort_keys=True)
        return __import__('hashlib').sha256(payload.encode('utf-8')).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Entrainement du recommender CPF')
    parser.add_argument('--input-pairs', type=str, default='data/processed/cpf/pairs_generalistes.jsonl')
    parser.add_argument('--output-dir', type=str, default='models/cpf-recommender')
    parser.add_argument('--base-model', type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--learning-rate', '--lr', dest='learning_rate', type=float, default=DEFAULT_LR)
    parser.add_argument('--warmup-ratio', type=float, default=DEFAULT_WARMUP_RATIO)
    parser.add_argument('--max-seq-length', type=int, default=DEFAULT_MAX_SEQ_LENGTH)
    parser.add_argument('--loss', type=str, default='MultipleNegativesRankingLoss')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--gradient-accumulation', type=int, default=DEFAULT_GRADIENT_ACCUMULATION)
    parser.add_argument('--mixed-precision', dest='mixed_precision', action='store_true', default=True)
    parser.add_argument('--no-mixed-precision', dest='mixed_precision', action='store_false')
    parser.add_argument('--resume-from-checkpoint', type=str, default=None)
    parser.add_argument('--max-train-samples', type=int, default=None)
    parser.add_argument('--max-pairs-per-formation', type=int, default=DEFAULT_MAX_PAIRS_PER_FORMATION)
    parser.add_argument('--trainer-api', type=str, choices=['fit', 'trainer'], default='fit')
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.trainer_api != 'fit':
        raise NotImplementedError("L option --trainer-api=trainer n'est pas encore implementee.")
    trainer = CPFRecommenderTrainer(
        TrainingConfig(
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
            output_dir=args.output_dir,
            max_train_samples=args.max_train_samples,
            max_pairs_per_formation=args.max_pairs_per_formation,
            trainer_api=args.trainer_api,
        )
    )
    trainer.train(args.input_pairs, resume_from_checkpoint=args.resume_from_checkpoint)


if __name__ == '__main__':
    main()
