#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import shutil
import subprocess
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from common.text import normalize_for_match, stable_hash
from continual_learning.dataset_export import example_id

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('train_continual_skill_extractor')

LABELS = ['O', 'B-SKILL', 'I-SKILL']
LABEL2ID = {label: idx for idx, label in enumerate(LABELS)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}
PROVENANCE_WEIGHTS = {
    'human_review': 1.0,
    'imported_gold_dataset': 1.0,
    'france_travail_api': 0.9,
    'exact_reference_match': 0.7,
    'semantic_match': 0.4,
    'model_prediction': 0.0,
}


@dataclass(frozen=True)
class Example:
    id: str
    text: str
    entities: list[dict[str, Any]]
    document_skills: list[dict[str, Any]]
    metadata: dict[str, Any]
    source_path: str

    @property
    def text_hash(self) -> str:
        return stable_hash(normalize_for_match(self.text), length=24)

    @property
    def sample_weight(self) -> float:
        weights = []
        for entity in self.entities:
            weights.append(PROVENANCE_WEIGHTS.get(entity.get('provenance'), 0.0))
        for skill in self.document_skills:
            weights.append(PROVENANCE_WEIGHTS.get(skill.get('provenance'), 0.0))
        if not weights:
            return 0.0
        return float(max(weights))


class NERDataset(Dataset):
    def __init__(self, examples: list[Example], tokenizer: Any, max_length: int = 256) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def _encode(self, example: Example) -> dict[str, Any]:
        encoding = self.tokenizer(
            example.text,
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
        )
        offsets = encoding.pop('offset_mapping')
        labels = [-100] * len(encoding['input_ids'])
        for entity in example.entities:
            start = entity.get('start')
            end = entity.get('end')
            if start is None or end is None:
                continue
            try:
                start = int(start)
                end = int(end)
            except Exception:
                continue
            if end <= start:
                continue
            for idx, (tok_start, tok_end) in enumerate(offsets):
                if tok_end <= tok_start:
                    continue
                if tok_end <= start or tok_start >= end:
                    continue
                labels[idx] = LABEL2ID['B-SKILL'] if tok_start == start else LABEL2ID['I-SKILL']
        if labels:
            labels[0] = -100
        if labels and labels[-1] == LABEL2ID['O']:
            labels[-1] = -100
        encoding['labels'] = labels
        encoding['sample_weight'] = float(example.sample_weight)
        return encoding

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self._encode(self.examples[idx])


class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        sample_weight = inputs.pop('sample_weight', None)
        labels = inputs.get('labels')
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        if sample_weight is not None:
            weight = sample_weight.to(loss.device).repeat_interleave(labels.shape[1])
            valid = labels.view(-1) != -100
            if valid.any():
                loss = loss[valid] * weight[valid]
            else:
                loss = loss[valid]
        else:
            loss = loss[labels.view(-1) != -100]
        final_loss = loss.mean() if loss.numel() else torch.tensor(0.0, device=logits.device)
        return (final_loss, outputs) if return_outputs else final_loss


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path or not path.exists():
        return []
    records = []
    with path.open(encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open('rb') as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def build_examples(records: list[dict[str, Any]], source_path: str) -> list[Example]:
    examples: list[Example] = []
    for record in records:
        text = record.get('text', '')
        entities = record.get('entities', []) or []
        document_skills = record.get('document_skills', []) or []
        metadata = record.get('metadata', {}) or {}
        record_id = record.get('id') or example_id(source_path, text)
        examples.append(
            Example(
                id=str(record_id),
                text=str(text),
                entities=[e for e in entities if isinstance(e, dict)],
                document_skills=[d for d in document_skills if isinstance(d, dict)],
                metadata=metadata,
                source_path=source_path,
            )
        )
    return examples


def deduplicate_examples(examples: list[Example]) -> list[Example]:
    seen: set[str] = set()
    deduped: list[Example] = []
    for example in examples:
        key = example.text_hash
        if key in seen:
            continue
        seen.add(key)
        deduped.append(example)
    return deduped


def split_rehearsal(base_examples: list[Example], incremental_examples: list[Example], max_samples: int | None, seed: int) -> list[Example]:
    rng = random.Random(seed)
    combined = base_examples + incremental_examples
    if max_samples is None or len(combined) <= max_samples:
        return combined
    base_target = min(len(base_examples), max(1, int(max_samples * 0.6)))
    inc_target = max_samples - base_target
    base_sample = rng.sample(base_examples, base_target) if len(base_examples) > base_target else list(base_examples)
    inc_sample = rng.sample(incremental_examples, min(len(incremental_examples), inc_target)) if incremental_examples else []
    remainder = [ex for ex in combined if ex not in base_sample and ex not in inc_sample]
    rng.shuffle(remainder)
    selected = base_sample + inc_sample + remainder
    return selected[:max_samples]


def build_manifest(args: argparse.Namespace, train_examples: list[Example], val_examples: list[Example], test_examples: list[Example]) -> dict[str, Any]:
    return {
        'seed': args.seed,
        'base_dataset': str(args.base_dataset),
        'incremental_dataset': str(args.incremental_dataset),
        'validation_dataset': str(args.validation_dataset),
        'test_dataset': str(args.test_dataset),
        'base_model': args.base_model,
        'resume_from_model': args.resume_from_model,
        'example_ids': {
            'train': [ex.id for ex in train_examples],
            'validation': [ex.id for ex in val_examples],
            'test': [ex.id for ex in test_examples],
        },
        'text_hashes': {
            'train': [ex.text_hash for ex in train_examples],
            'validation': [ex.text_hash for ex in val_examples],
            'test': [ex.text_hash for ex in test_examples],
        },
        'dataset_hashes': {
            'base': sha256_file(args.base_dataset),
            'incremental': sha256_file(args.incremental_dataset),
            'validation': sha256_file(args.validation_dataset),
            'test': sha256_file(args.test_dataset),
        },
        'provenance_weights': PROVENANCE_WEIGHTS,
        'label_space': LABELS,
    }


def compute_token_metrics(eval_pred) -> dict[str, float]:
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=-1)
    mask = labels != -100
    true = labels[mask]
    pred = preds[mask]
    if true.size == 0:
        return {'token_accuracy': 0.0, 'token_f1': 0.0}
    accuracy = float((true == pred).mean())
    tp = float(((pred != LABEL2ID['O']) & (true == pred)).sum())
    fp = float(((pred != LABEL2ID['O']) & (true == LABEL2ID['O'])).sum())
    fn = float(((pred == LABEL2ID['O']) & (true != LABEL2ID['O'])).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {'token_accuracy': accuracy, 'token_f1': f1}


def train(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    base_records = load_jsonl(args.base_dataset)
    incremental_records = load_jsonl(args.incremental_dataset)
    validation_records = load_jsonl(args.validation_dataset)
    test_records = load_jsonl(args.test_dataset)

    base_examples = build_examples(base_records, str(args.base_dataset))
    incremental_examples = build_examples(incremental_records, str(args.incremental_dataset))
    val_examples = build_examples(validation_records, str(args.validation_dataset))
    test_examples = build_examples(test_records, str(args.test_dataset))

    val_hashes = {ex.text_hash for ex in val_examples}
    test_hashes = {ex.text_hash for ex in test_examples}
    base_examples = [ex for ex in base_examples if ex.text_hash not in val_hashes and ex.text_hash not in test_hashes]
    incremental_examples = [ex for ex in incremental_examples if ex.text_hash not in val_hashes and ex.text_hash not in test_hashes]

    base_examples = deduplicate_examples(base_examples)
    incremental_examples = deduplicate_examples(incremental_examples)
    val_examples = deduplicate_examples(val_examples)
    test_examples = deduplicate_examples(test_examples)

    train_examples = split_rehearsal(base_examples, incremental_examples, args.max_samples, args.seed)
    if not train_examples:
        raise SystemExit('Aucun exemple d entraînement exploitable après déduplication.')

    tokenizer_source = args.resume_from_model or args.base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    model_source = args.resume_from_model or args.base_model
    model = AutoModelForTokenClassification.from_pretrained(
        model_source,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    train_ds = NERDataset(train_examples, tokenizer)
    val_ds = NERDataset(val_examples, tokenizer)
    test_ds = NERDataset(test_examples, tokenizer)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        logging_strategy='steps',
        logging_steps=25,
        save_total_limit=2,
        learning_rate=5e-5,
        weight_decay=0.01,
        fp16=bool(args.fp16 and torch.cuda.is_available()),
        seed=args.seed,
        report_to=[],
        load_best_model_at_end=True,
        metric_for_best_model='token_f1',
        greater_is_better=True,
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_token_metrics,
    )

    logger.info('Training on %d examples (%d base, %d incremental)', len(train_examples), len(base_examples), len(incremental_examples))
    trainer.train(resume_from_checkpoint=args.resume_from_model)
    val_metrics = trainer.evaluate(val_ds)
    test_metrics = trainer.evaluate(test_ds)

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    manifest = build_manifest(args, train_examples, val_examples, test_examples)
    manifest['validation_metrics'] = {k: float(v) for k, v in val_metrics.items() if isinstance(v, (int, float))}
    manifest['test_metrics'] = {k: float(v) for k, v in test_metrics.items() if isinstance(v, (int, float))}
    manifest['sample_counts'] = {
        'base': len(base_examples),
        'incremental': len(incremental_examples),
        'train': len(train_examples),
        'validation': len(val_examples),
        'test': len(test_examples),
    }
    manifest_path = output_dir / 'training_manifest.json'
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding='utf-8')

    report = {
        'output_dir': str(output_dir),
        'train_examples': len(train_examples),
        'validation_examples': len(val_examples),
        'test_examples': len(test_examples),
        'validation_metrics': manifest['validation_metrics'],
        'test_metrics': manifest['test_metrics'],
    }
    (output_dir / 'training_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding='utf-8')
    logger.info('Training complete: %s', report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Train the continual skill extractor')
    parser.add_argument('--base-dataset', type=Path, required=True)
    parser.add_argument('--incremental-dataset', type=Path, required=True)
    parser.add_argument('--validation-dataset', type=Path, required=True)
    parser.add_argument('--test-dataset', type=Path, required=True)
    parser.add_argument('--base-model', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--resume-from-model', default=None)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default=None)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=3)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train(args)


if __name__ == '__main__':
    main()
