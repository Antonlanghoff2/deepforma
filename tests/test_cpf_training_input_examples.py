from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from sentence_transformers import InputExample, losses

from scripts.build_cpf_training_pairs import build_pairs
from scripts.train_cpf_recommender import _build_candidate_corpus, _build_loader, _dedupe_rows, _validation_metrics, row_to_input_example, normalize_training_row


def test_row_to_input_example_valid():
    row = {'anchor_text': 'Programme Python', 'positive_text': 'Formation Python'}
    example = row_to_input_example(row)
    assert isinstance(example, InputExample)
    assert example.texts == ['Programme Python', 'Formation Python']


def test_row_to_input_example_empty_text_returns_none():
    row = {'anchor_text': '   ', 'positive_text': 'Formation Python'}
    assert row_to_input_example(row) is None


def test_row_to_input_example_identical_texts_returns_none():
    row = {'anchor_text': 'Python', 'positive_text': 'Python'}
    assert row_to_input_example(row) is None


def test_row_to_input_example_unknown_structure_raises():
    with pytest.raises(ValueError) as exc:
        row_to_input_example({'foo': 'bar'})
    message = str(exc.value)
    assert 'Cles disponibles' in message
    assert 'Cles attendues' in message


def test_normalize_training_row_with_positive_uid_alias():
    row = {'anchor_text': 'Programme Python', 'positive_text': 'Formation Python', 'formation_id': 'F-123'}
    normalized = normalize_training_row(row)
    assert normalized['anchor'] == 'Programme Python'
    assert normalized['positive'] == 'Formation Python'
    assert normalized['positive_uid'] == 'F-123'


def test_normalize_training_row_with_candidate_id_alias():
    row = {'query': 'Programme Python', 'positive_text': 'Formation Python', 'candidate_id': 'C-456'}
    normalized = normalize_training_row(row)
    assert normalized['positive_uid'] == 'C-456'


def test_normalize_training_row_missing_positive_uid_raises():
    with pytest.raises(ValueError) as exc:
        normalize_training_row({'anchor_text': 'Programme Python', 'positive_text': 'Formation Python'})
    assert 'positive_uid' in str(exc.value)


def test_dedup_rows_removes_normalized_duplicates():
    rows = [
        {'anchor_text': 'Python', 'positive_text': 'Analyse de données'},
        {'anchor_text': ' Python ', 'positive_text': 'Analyse de données '},
        {'anchor_text': 'Java', 'positive_text': 'API REST'},
    ]
    deduped, removed = _dedupe_rows(rows)
    assert removed == 1
    assert len(deduped) == 2


class _FakeModel:
    def tokenize(self, texts):
        batch_size = len(texts)
        max_len = 4
        tensor = torch.zeros((batch_size, max_len), dtype=torch.long)
        for i, text in enumerate(texts):
            values = [ord(char) % 13 + 1 for char in text[:max_len]]
            tensor[i, : len(values)] = torch.tensor(values, dtype=torch.long)
        return {'input_ids': tensor, 'attention_mask': (tensor > 0).long()}

    def __call__(self, sentence_feature):
        input_ids = sentence_feature['input_ids'].float()
        embeddings = torch.stack(
            [input_ids.sum(dim=1), input_ids[:, 0], input_ids[:, -1]],
            dim=1,
        )
        return {'sentence_embedding': embeddings}


def test_loader_and_mnrl_smoke():
    rows = [
        {'anchor_text': f'Programme {i}', 'positive_text': f'Competence {i}', 'formation_id': f'F-{i}'}
        for i in range(128)
    ]
    examples = [row_to_input_example(row) for row in rows]
    examples = [example for example in examples if example is not None]
    loader = _build_loader(examples, batch_size=16)
    batch = next(iter(loader))
    assert len(batch) == 16
    assert all(isinstance(example, InputExample) for example in batch)

    model = _FakeModel()
    loss_fn = losses.MultipleNegativesRankingLoss(model)
    sentence_features = [
        model.tokenize([example.texts[0] for example in batch]),
        model.tokenize([example.texts[1] for example in batch]),
    ]
    loss = loss_fn(sentence_features, torch.tensor([0]))
    assert torch.isfinite(loss).item() is True


def test_candidate_corpus_deduplicates_uid_and_prefers_longer_text():
    rows = [
        {'anchor_text': 'Python', 'positive_text': 'Formation courte', 'formation_id': 'F-1'},
        {'anchor_text': 'Python', 'positive_text': 'Formation plus longue et plus detaillee', 'formation_id': 'F-1'},
    ]
    corpus, conflicts = _build_candidate_corpus(rows)
    assert conflicts == 1
    assert list(corpus) == ['F-1']
    assert 'plus detaillee' in corpus['F-1']['positive']


def test_validation_metrics_unique_corpus_and_missing_uid_raises():
    class _EvalModel(_FakeModel):
        def encode(self, texts, **kwargs):
            import numpy as np
            return np.array([[float(len(str(text))), 0.0] for text in texts], dtype=float)

    model = _EvalModel()
    rows = [
        {'anchor_text': 'Programme Python', 'positive_text': 'Formation Python', 'formation_id': 'F-1', 'query': 'Programme Python'},
        {'anchor_text': 'Programme Python', 'positive_text': 'Formation Python detaillee', 'formation_id': 'F-1', 'query': 'Programme Python'},
    ]
    metrics = _validation_metrics(model, rows, candidate_rows=rows)
    assert metrics.validation_examples == 2
    assert metrics.mean_positive_similarity >= 0.0

    with pytest.raises(ValueError) as exc:
        _validation_metrics(model, [{'anchor_text': 'Programme Python', 'positive_text': 'Formation Python', 'formation_id': 'F-999', 'query': 'Programme Python'}], candidate_rows=rows)
    assert 'absents du corpus candidat' in str(exc.value)


def test_build_pairs_deduplicates_positive_pairs(tmp_path):
    records = [
        {
            'formation_id': 'f-1',
            'group_id': 'cert:1',
            'title': 'Formation Python',
            'sector': 'Tech',
            'source_text': 'Python analyse de donnees',
            'skills': ['Python', 'Data'],
            'tags': ['python'],
            'rome_codes': ['M1805'],
            'modality': 'Distance',
        },
        {
            'formation_id': 'f-2',
            'group_id': 'cert:1',
            'title': 'Formation Python',
            'sector': 'Tech',
            'source_text': 'Python analyse de donnees',
            'skills': ['Python', 'Data'],
            'tags': ['python'],
            'rome_codes': ['M1805'],
            'modality': 'Distance',
        },
        {
            'formation_id': 'f-3',
            'group_id': 'cert:1',
            'title': 'Formation Python',
            'sector': 'Tech',
            'source_text': 'Python analyse de donnees',
            'skills': ['Python', 'Data'],
            'tags': ['python'],
            'rome_codes': ['M1805'],
            'modality': 'Distance',
        },
    ]
    input_path = tmp_path / 'formations.jsonl'
    with input_path.open('w', encoding='utf-8') as fh:
        for row in records:
            fh.write(json.dumps(row, ensure_ascii=False) + '\n')

    class Args:
        input = str(input_path)
        output_dir = str(tmp_path / 'output')
        output_pairs = 'pairs.jsonl'
        seed = 42
        max_pairs_per_formation = 1
        max_train_samples = None

    summary = build_pairs(Args())
    assert summary['positive_pairs_removed'] > 0
    assert summary['total_pairs'] >= 1
    assert Path(Args.output_dir, Args.output_pairs).exists()
    with Path(Args.output_dir, Args.output_pairs).open(encoding='utf-8') as fh:
        first_row = json.loads(fh.readline())
    assert 'positive_uid' in first_row
