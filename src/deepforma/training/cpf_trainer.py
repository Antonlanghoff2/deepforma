from __future__ import annotations

import inspect
import json
import logging
import random
from contextlib import nullcontext
from unittest.mock import patch
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader
from sentence_transformers import InputExample, SentenceTransformer, losses

from common.text import clean_text
from deepforma.training.cpf_dataset import CPFTrainingExample, load_jsonl, normalize_training_row, save_jsonl, timestamp_iso, validate_rows


DEFAULT_BASE_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
DEFAULT_BATCH_SIZE = 16
DEFAULT_GRADIENT_ACCUMULATION = 2
DEFAULT_EPOCHS = 3
DEFAULT_MAX_SEQ_LENGTH = 256
DEFAULT_LR = 2e-5
DEFAULT_WARMUP_RATIO = 0.1


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


def _example_to_input_example(example: dict[str, Any], loss_name: str) -> InputExample:
    anchor = clean_text(example.get('anchor') or example.get('query') or example.get('anchor_text'))
    positive = clean_text(example.get('positive') or example.get('positive_text') or example.get('candidate_text'))
    negative = clean_text(example.get('negative') or example.get('negative_text'))
    if 'triplet' in clean_text(loss_name).lower():
        return InputExample(texts=[anchor, positive, negative])
    return InputExample(texts=[anchor, positive])


def _is_positive_pair_row(row: dict[str, Any]) -> bool:
    anchor = clean_text(row.get('anchor_text') or row.get('query') or row.get('text_a') or row.get('source_text') or row.get('anchor'))
    positive = clean_text(row.get('positive_text') or row.get('candidate_text') or row.get('text_b') or row.get('target_text') or row.get('positive'))
    return bool(anchor and positive)


def _positive_pair_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if _is_positive_pair_row(row)]


def _read_examples(path: str | Path) -> list[dict[str, Any]]:
    return load_jsonl(path)


def _ranking_metrics(ranks: list[int]) -> dict[str, float]:
    if not ranks:
        return {'recall@1': 0.0, 'recall@5': 0.0, 'recall@10': 0.0, 'precision@5': 0.0, 'mrr': 0.0, 'ndcg@10': 0.0}
    total = len(ranks)
    recall_at_1 = sum(1 for rank in ranks if rank <= 1) / total
    recall_at_5 = sum(1 for rank in ranks if rank <= 5) / total
    recall_at_10 = sum(1 for rank in ranks if rank <= 10) / total
    precision_at_5 = sum(1 for rank in ranks if rank <= 5) / (total * 5)
    mrr = sum(1.0 / rank for rank in ranks) / total
    ndcg_at_10 = sum(1.0 / np.log2(rank + 1) if rank <= 10 else 0.0 for rank in ranks) / total
    return {
        'recall@1': round(recall_at_1, 4),
        'recall@5': round(recall_at_5, 4),
        'recall@10': round(recall_at_10, 4),
        'precision@5': round(precision_at_5, 4),
        'mrr': round(mrr, 4),
        'ndcg@10': round(ndcg_at_10, 4),
    }


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
        rows = _read_examples(path)
        if not rows:
            raise ValueError("Le dataset d'entraînement est vide.")
        return rows

    def _build_dataloader(self, rows: list[dict[str, Any]]) -> DataLoader:
        examples = [_example_to_input_example(row, self.config.loss) for row in rows]
        return DataLoader(examples, shuffle=True, batch_size=self.config.batch_size)

    def _validation_metrics(self, model: SentenceTransformer, rows: list[dict[str, Any]], *, candidate_rows: list[dict[str, Any]] | None = None) -> TrainingMetrics:
        if not rows:
            raise ValueError("Le split de validation est vide.")
        candidates = candidate_rows or rows
        candidate_corpus: dict[str, dict[str, Any]] = {}
        conflicts = 0
        for row in candidates:
            if not _is_positive_pair_row(row):
                continue
            normalized = normalize_training_row(row)
            uid = normalized['positive_uid']
            text = clean_text(normalized.get('positive'))
            if not text:
                continue
            existing = candidate_corpus.get(uid)
            if existing is None:
                candidate_corpus[uid] = normalized
                continue
            existing_text = clean_text(existing.get('positive'))
            if existing_text == text:
                continue
            conflicts += 1
            if len(text) > len(existing_text):
                candidate_corpus[uid] = normalized
        normalized_rows = [normalize_training_row(row) for row in rows]
        missing_positive_uids = [row['positive_uid'] for row in normalized_rows if row['positive_uid'] not in candidate_corpus]
        if missing_positive_uids:
            raise ValueError('Des positive_uid de validation sont absents du corpus candidat: ' + ', '.join(missing_positive_uids[:10]))
        candidate_ids = list(candidate_corpus)
        candidate_texts = [clean_text(candidate_corpus[uid].get('positive')) for uid in candidate_ids]
        candidate_embeddings = torch.as_tensor(model.encode(candidate_texts, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False))
        query_embeddings = torch.as_tensor(model.encode([row['anchor'] for row in normalized_rows], convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False))
        positive_scores: list[float] = []
        negative_scores: list[float] = []
        ranks: list[int] = []
        candidate_index = {uid: idx for idx, uid in enumerate(candidate_ids)}
        for idx, row in enumerate(normalized_rows):
            sims = torch.matmul(candidate_embeddings, query_embeddings[idx]).detach().cpu().numpy().tolist()
            ranking = sorted(zip(candidate_ids, sims), key=lambda item: item[1], reverse=True)
            positive_uid = row['positive_uid']
            positive_rank = next((rank for rank, (uid, _) in enumerate(ranking, start=1) if uid == positive_uid), len(ranking) + 1)
            ranks.append(positive_rank)
            positive_scores.append(float(sims[candidate_index[positive_uid]]))
            negative_uid = clean_text(row.get('negative_uid'))
            if negative_uid and negative_uid in candidate_index:
                negative_scores.append(float(sims[candidate_index[negative_uid]]))
        ranking_metrics = _ranking_metrics(ranks)
        LOGGER.info('Candidats bruts: %d', len(candidates))
        LOGGER.info('Candidats uniques: %d', len(candidate_corpus))
        LOGGER.info('Conflits uid/texte: %d', conflicts)
        LOGGER.info('Positive_uid absents: %d', len(missing_positive_uids))
        if candidate_ids:
            LOGGER.info('Exemple candidat: %s -> %s', candidate_ids[0], clean_text(candidate_corpus[candidate_ids[0]].get('positive'))[:120])
        return TrainingMetrics(
            validation_examples=len(rows),
            recall_at_1=ranking_metrics['recall@1'],
            recall_at_5=ranking_metrics['recall@5'],
            recall_at_10=ranking_metrics['recall@10'],
            mrr=ranking_metrics['mrr'],
            ndcg_at_10=ranking_metrics['ndcg@10'],
            mean_positive_similarity=round(float(np.mean(positive_scores)) if positive_scores else 0.0, 4),
            mean_negative_similarity=round(float(np.mean(negative_scores)) if negative_scores else 0.0, 4),
        )

    def train(
        self,
        train_path: str | Path,
        validation_path: str | Path,
        *,
        resume_from_checkpoint: str | Path | None = None,
    ) -> dict[str, Any]:
        train_rows = _positive_pair_rows(self._load_dataset(train_path))
        validation_rows = _positive_pair_rows(self._load_dataset(validation_path))
        validation_result = validate_rows(train_rows + validation_rows)
        if not validation_result.ok:
            raise ValueError(' ; '.join(validation_result.errors))
        _seed_everything(self.config.seed)
        model = self.load_model()
        train_loader = self._build_dataloader(train_rows)
        loss_cls = _infer_loss(self.config.loss)
        warmup_steps = max(1, int(len(train_loader) * self.config.epochs * self.config.warmup_ratio))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        def _fit(active_model: SentenceTransformer, *, active_device: str) -> None:
            train_loss = loss_cls(active_model)
            fit_kwargs = {
                'train_objectives': [(train_loader, train_loss)],
                'epochs': self.config.epochs,
                'warmup_steps': warmup_steps,
                'optimizer_params': {'lr': self.config.learning_rate},
                'output_path': str(self.final_dir),
                'show_progress_bar': False,
                'use_amp': bool(self.config.mixed_precision and active_device == 'cuda'),
                'checkpoint_path': str(self.checkpoints_dir),
                'checkpoint_save_steps': max(1, len(train_loader)),
                'checkpoint_save_total_limit': 3,
                'resume_from_checkpoint': str(resume_from_checkpoint) if resume_from_checkpoint else None,
            }
            fit_signature = inspect.signature(active_model.fit)
            if 'gradient_accumulation_steps' in fit_signature.parameters:
                fit_kwargs['gradient_accumulation_steps'] = self.config.gradient_accumulation
            device_context = patch.object(torch.cuda, 'is_available', lambda: False) if active_device == 'cpu' else nullcontext()
            with device_context:
                active_model.fit(**fit_kwargs)

        try:
            _fit(model, active_device=self.device)
        except Exception as exc:
            message = f'{type(exc).__name__}: {exc}'.lower()
            if self.device != 'cuda' or 'out of memory' not in message:
                raise
            LOGGER.warning("Bascule vers CPU après OOM CUDA pendant l'entraînement.")
            torch.cuda.empty_cache()
            self.device = 'cpu'
            model = self.load_model()
            _fit(model, active_device='cpu')
        model.save(str(self.final_dir))
        metrics = self._validation_metrics(model, validation_rows, candidate_rows=train_rows + validation_rows)
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
        }
        (self.output_dir / 'config.json').write_text(json.dumps(asdict(self.config), ensure_ascii=False, indent=2), encoding='utf-8')
        (self.output_dir / 'training_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        return {'manifest': manifest, 'metrics': asdict(metrics), 'model_path': str(self.final_dir)}

    @staticmethod
    def _dataset_hash(rows: list[dict[str, Any]]) -> str:
        payload = json.dumps(rows, ensure_ascii=False, sort_keys=True)
        return __import__('hashlib').sha256(payload.encode('utf-8')).hexdigest()


__all__ = ['CPFRecommenderTrainer', 'TrainingConfig', 'TrainingMetrics', 'resolve_device']
