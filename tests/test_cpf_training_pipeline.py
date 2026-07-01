from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from deepforma.training.cpf_dataset import build_group_id, load_jsonl, save_jsonl, split_by_group, validate_rows
from deepforma.training.cpf_trainer import CPFRecommenderTrainer, TrainingConfig
from scripts.evaluate_cpf_recommender import evaluate_model, _metrics_from_ranks
from scripts.extract_cpf_skills import extract_cpf_skills


def _write_parquet(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame.from_records(rows).to_parquet(path, index=False)


def _sample_formations() -> list[dict[str, object]]:
    return [
        {
            'formation_uid': 'f-python-1',
            'title': 'Développeur Python Data',
            'description': 'Python, analyse de données et reporting.',
            'objectives': 'Acquérir Python et analyse de données.',
            'certification': 'RNCP Python Data',
            'exit_level': '5',
            'nsf': '326',
            'organization': 'Org A',
            'organization_siret': '12345678901234',
            'region': 'Île-de-France',
            'region_code': '11',
            'department': 'Paris',
            'department_code': '75',
            'referential_type': 'RNCP',
            'search_text': 'Développeur Python Data | Python analyse de données',
            'skills_normalized': ['Python', 'Analyse de données'],
            'skills_explicit': [],
            'skills_inferred': [],
            'skills_confidence': {'Python': 0.9},
            'skills_evidence': {'Python': 'title'},
        },
        {
            'formation_uid': 'f-python-2',
            'title': 'Python en ligne',
            'description': 'Formation Python accessible à distance.',
            'objectives': 'Python pour la data.',
            'certification': 'RS Python',
            'exit_level': '4',
            'nsf': '326',
            'organization': 'Org B',
            'organization_siret': '12345678901235',
            'region': 'Auvergne-Rhône-Alpes',
            'region_code': '84',
            'department': 'Rhône',
            'department_code': '69',
            'referential_type': 'RS',
            'search_text': 'Python en ligne | Python pour la data',
            'skills_normalized': ['Python', 'Analyse de données'],
            'skills_explicit': [],
            'skills_inferred': [],
            'skills_confidence': {'Python': 0.9},
            'skills_evidence': {'Python': 'title'},
            'distance_compatible': True,
        },
        {
            'formation_uid': 'f-java-1',
            'title': 'Développeur Java',
            'description': 'Java et API REST.',
            'objectives': 'Construire des APIs.',
            'certification': 'RNCP Java',
            'exit_level': '5',
            'nsf': '326',
            'organization': 'Org C',
            'organization_siret': '12345678901236',
            'region': 'Auvergne-Rhône-Alpes',
            'region_code': '84',
            'department': 'Rhône',
            'department_code': '69',
            'referential_type': 'RNCP',
            'search_text': 'Développeur Java | Java API REST',
            'skills_normalized': ['Java', 'API REST'],
            'skills_explicit': [],
            'skills_inferred': [],
            'skills_confidence': {'Python': 0.9},
            'skills_evidence': {'Python': 'title'},
        },
        {
            'formation_uid': 'f-agri-1',
            'title': 'Agriculture biologique',
            "description": "Conduite d'une exploitation agricole.",
            'objectives': 'Installer une exploitation.',
            'certification': 'RS Agriculture',
            'exit_level': '4',
            'nsf': '210',
            'organization': 'Org D',
            'organization_siret': '12345678901237',
            'region': 'Normandie',
            'region_code': '28',
            'department': 'Eure',
            'department_code': '27',
            'referential_type': 'RS',
            'search_text': 'Agriculture biologique | exploitation',
            'skills_normalized': ['Communication'],
            'skills_explicit': [],
            'skills_inferred': [],
            'skills_confidence': {'Python': 0.9},
            'skills_evidence': {'Python': 'title'},
        },
    ]


def test_extract_cpf_skills_creates_expected_columns(tmp_path):
    input_path = tmp_path / 'formations.parquet'
    output_path = tmp_path / 'formations_with_skills.parquet'
    _write_parquet(input_path, _sample_formations())

    enriched = extract_cpf_skills(input_path, output_path)
    assert output_path.exists()
    assert not enriched.empty
    expected = {
        'formation_uid', 'title', 'description', 'objectives', 'certification_code', 'certification_label',
        'referential_type', 'level', 'nsf', 'organization', 'siret', 'region', 'region_code', 'department',
        'department_code', 'search_text', 'skills_explicit', 'skills_inferred', 'skills_normalized',
        'skills_confidence', 'skills_evidence',
    }
    assert expected.issubset(set(enriched.columns))





def test_split_by_group_is_reproducible_and_leak_free(tmp_path):
    rows = [
        {
            'query_id': '1',
            'query': 'Python',
            'target_job': 'Développeur Python',
            'required_skills': ['Python'],
            'missing_skills': ['Python'],
            'region_code': '11',
            'department_code': '75',
            'positive_uid': 'f-python-1',
            'positive_text': 'Python',
            'negative_uid': 'f-java-1',
            'negative_text': 'Java',
            'negative_type': 'easy',
            'label_source': 'heuristic',
            'label_confidence': 0.8,
            'group_id': build_group_id({'certification_code': 'RNCP Python Data', 'title': 'Développeur Python Data', 'search_text': 'Python'}),
        },
        {
            'query_id': '2',
            'query': 'Java',
            'target_job': 'Développeur Java',
            'required_skills': ['Java'],
            'missing_skills': ['Java'],
            'region_code': '84',
            'department_code': '69',
            'positive_uid': 'f-java-1',
            'positive_text': 'Java',
            'negative_uid': 'f-agri-1',
            'negative_text': 'Agriculture',
            'negative_type': 'easy',
            'label_source': 'heuristic',
            'label_confidence': 0.8,
            'group_id': build_group_id({'certification_code': 'RNCP Java', 'title': 'Développeur Java', 'search_text': 'Java'}),
        },
        {
            'query_id': '3',
            'query': 'Agriculture',
            'target_job': 'Agriculteur',
            'required_skills': ['Communication'],
            'missing_skills': ['Communication'],
            'region_code': '28',
            'department_code': '27',
            'positive_uid': 'f-agri-1',
            'positive_text': 'Agriculture',
            'negative_uid': 'f-python-1',
            'negative_text': 'Python',
            'negative_type': 'territorial',
            'label_source': 'heuristic',
            'label_confidence': 0.7,
            'group_id': build_group_id({'certification_code': 'RS Agriculture', 'title': 'Agriculture biologique', 'search_text': 'Agriculture'}),
        },
    ]
    first = split_by_group(rows, seed=42)
    second = split_by_group(rows, seed=42)
    assert first == second
    split_groups = {name: {row['group_id'] for row in split_rows} for name, split_rows in first.items()}
    assert split_groups['train'].isdisjoint(split_groups['validation'])
    assert split_groups['train'].isdisjoint(split_groups['test'])
    assert split_groups['validation'].isdisjoint(split_groups['test'])


def test_load_dataset_and_validate(tmp_path):
    rows = [
        {
            'query_id': f'1-{i}',
            'query': 'Python',
            'target_job': 'Développeur Python',
            'required_skills': ['Python'],
            'missing_skills': ['Python'],
            'region_code': '11',
            'department_code': '75',
            'positive_uid': f'f-python-{i}',
            'positive_text': 'Python',
            'negative_uid': f'f-java-{i}',
            'negative_text': 'Java',
            'negative_type': 'easy',
            'label_source': 'heuristic',
            'label_confidence': 0.8,
            'group_id': f'g{i}',
            'certification_code': f'RNCP Python Data {i}',
        }
        for i in range(10)
    ]
    path = tmp_path / 'pairs.jsonl'
    save_jsonl(path, rows)
    loaded = load_jsonl(path)
    assert len(loaded) == 10
    validation = validate_rows(loaded, min_positives=1)
    assert validation.ok





def test_resolve_device_prefers_cpu_when_cuda_memory_low(monkeypatch):
    from deepforma.training import cpf_trainer as trainer_module

    monkeypatch.setattr(trainer_module.torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(trainer_module.torch.cuda, 'mem_get_info', lambda: (500_000_000, 8_000_000_000))
    assert trainer_module.resolve_device() == 'cpu'


def test_trainer_falls_back_to_cpu_on_cuda_oom(monkeypatch, tmp_path):
    from deepforma.training import cpf_trainer as trainer_module

    class FakeModel:
        def __init__(self):
            self.max_seq_length = None

        def fit(self, *args, **kwargs):
            pass

        def save(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)

    load_calls = []

    def fake_sentence_transformer(*args, **kwargs):
        load_calls.append(kwargs.get('device'))
        if kwargs.get('device') == 'cuda':
            raise trainer_module.torch.OutOfMemoryError('CUDA out of memory')
        return FakeModel()

    monkeypatch.setattr(trainer_module.torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(trainer_module.torch.cuda, 'mem_get_info', lambda: (8_000_000_000, 8_000_000_000))
    monkeypatch.setattr(trainer_module, 'SentenceTransformer', fake_sentence_transformer)
    trainer = CPFRecommenderTrainer(TrainingConfig(base_model='dummy', output_dir=str(tmp_path / 'model'), epochs=1, batch_size=2, mixed_precision=False))
    model = trainer.load_model()
    assert trainer.device == 'cpu'
    assert load_calls == ['cuda', 'cpu']
    assert model.max_seq_length == 256



def test_trainer_retries_training_on_cpu_after_cuda_oom(monkeypatch, tmp_path):
    from deepforma.training import cpf_trainer as trainer_module

    class FakeModel:
        def __init__(self, device):
            self.device = device
            self.max_seq_length = None
            self.fit_calls = 0

        def fit(self, *args, **kwargs):
            self.fit_calls += 1
            if self.device == 'cuda' and self.fit_calls == 1:
                raise trainer_module.torch.OutOfMemoryError('CUDA out of memory')

        def save(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)

        def encode(self, texts, **kwargs):
            import numpy as np
            return np.array([[1.0, 0.0] for _ in texts], dtype=float)

    def fake_sentence_transformer(*args, **kwargs):
        return FakeModel(kwargs.get('device'))

    monkeypatch.setattr(trainer_module.torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(trainer_module.torch.cuda, 'mem_get_info', lambda: (8_000_000_000, 8_000_000_000))
    monkeypatch.setattr(trainer_module, 'SentenceTransformer', fake_sentence_transformer)
    trainer = CPFRecommenderTrainer(TrainingConfig(base_model='dummy', output_dir=str(tmp_path / 'model'), epochs=1, batch_size=2, mixed_precision=False))

    train_rows = [
        {
            'query_id': f'q{i}',
            'query': 'Python',
            'target_job': 'Développeur Python',
            'required_skills': ['Python'],
            'missing_skills': ['Python'],
            'region_code': '11',
            'department_code': '75',
            'positive_uid': f'p{i}',
            'positive_text': 'Python',
            'negative_uid': f'n{i}',
            'negative_text': 'Java',
            'negative_type': 'easy',
            'label_source': 'heuristic',
            'label_confidence': 0.8,
            'group_id': f'g{i}',
            'certification_code': f'CERT{i}',
        }
        for i in range(10)
    ]
    validation_rows = [
        {
            'query_id': f'vq{i}',
            'query': 'Python',
            'target_job': 'Développeur Python',
            'required_skills': ['Python'],
            'missing_skills': ['Python'],
            'region_code': '11',
            'department_code': '75',
            'positive_uid': f'vp{i}',
            'positive_text': 'Python',
            'negative_uid': f'vn{i}',
            'negative_text': 'Java',
            'negative_type': 'easy',
            'label_source': 'heuristic',
            'label_confidence': 0.8,
            'group_id': f'vg{i}',
            'certification_code': f'VCERT{i}',
        }
        for i in range(10)
    ]
    train_path = tmp_path / 'train.jsonl'
    val_path = tmp_path / 'validation.jsonl'
    save_jsonl(train_path, train_rows)
    save_jsonl(val_path, validation_rows)

    result = trainer.train(train_path, val_path)
    assert trainer.device == 'cpu'
    assert Path(result['model_path']).exists()
    assert result['manifest']['device'] == 'cpu'


def test_trainer_load_model_and_train_mock(monkeypatch, tmp_path):
    from deepforma.training import cpf_trainer as trainer_module

    class FakeModel:
        def __init__(self, *args, **kwargs):
            self.max_seq_length = None
            self.saved = None

        def fit(self, *args, **kwargs):
            self.fit_args = (args, kwargs)

        def save(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)
            self.saved = path

        def encode(self, texts, **kwargs):
            import numpy as np
            return np.array([[1.0, 0.0] for _ in texts], dtype=float)

    monkeypatch.setattr(trainer_module, 'SentenceTransformer', lambda *args, **kwargs: FakeModel())
    trainer = CPFRecommenderTrainer(TrainingConfig(base_model='dummy', output_dir=str(tmp_path / 'model'), epochs=1, batch_size=2, mixed_precision=False))
    model = trainer.load_model()
    assert model.max_seq_length == 256

    train_rows = [
        {
            'query_id': f'q{i}',
            'query': 'Python',
            'target_job': 'Développeur Python',
            'required_skills': ['Python'],
            'missing_skills': ['Python'],
            'region_code': '11',
            'department_code': '75',
            'positive_uid': f'p{i}',
            'positive_text': 'Python',
            'negative_uid': f'n{i}',
            'negative_text': 'Java',
            'negative_type': 'easy',
            'label_source': 'heuristic',
            'label_confidence': 0.8,
            'group_id': f'g{i}',
            'certification_code': f'CERT{i}',
        }
        for i in range(10)
    ]
    validation_rows = [
        {
            'query_id': f'vq{i}',
            'query': 'Python',
            'target_job': 'Développeur Python',
            'required_skills': ['Python'],
            'missing_skills': ['Python'],
            'region_code': '11',
            'department_code': '75',
            'positive_uid': f'vp{i}',
            'positive_text': 'Python',
            'negative_uid': f'vn{i}',
            'negative_text': 'Java',
            'negative_type': 'easy',
            'label_source': 'heuristic',
            'label_confidence': 0.8,
            'group_id': f'vg{i}',
            'certification_code': f'VCERT{i}',
        }
        for i in range(10)
    ]
    train_path = tmp_path / 'train.jsonl'
    val_path = tmp_path / 'validation.jsonl'
    save_jsonl(train_path, train_rows)
    save_jsonl(val_path, validation_rows)
    result = trainer.train(train_path, val_path)
    assert Path(result['model_path']).exists()
    assert 'validation_metrics' in result['manifest']


def test_ranking_metrics_and_evaluation_mock():
    metrics = _metrics_from_ranks([1, 3, 5])
    assert metrics['recall_at_1'] == pytest.approx(0.3333, rel=1e-3)
    assert metrics['recall_at_5'] == 1.0

    queries = [
        {
            'query': 'Python',
            'positive_uid': 'f-python-1',
            'negative_uid': 'f-java-1',
            'required_skills': ['Python'],
            'missing_skills': ['Python'],
            'region_code': '11',
            'department_code': '75',
        }
    ]
    candidates = [
        {'formation_uid': 'f-python-1', 'title': 'Python', 'search_text': 'Python', 'skills_normalized': ['Python'], 'region_code': '11', 'department_code': '75'},
        {'formation_uid': 'f-java-1', 'title': 'Java', 'search_text': 'Java', 'skills_normalized': ['Java'], 'region_code': '84', 'department_code': '69'},
    ]
    similarities = np.array([[0.9, 0.1]])
    result, errors = evaluate_model('demo', similarities, queries, candidates)
    assert result['metrics']['recall_at_1'] == 1.0
    assert errors == []


def test_cli_help_commands(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    scripts = [
        'extract_cpf_skills.py',
        'build_cpf_training_pairs.py',
        'train_cpf_recommender.py',
        'evaluate_cpf_recommender.py',
    ]
    for script_name in scripts:
        result = subprocess.run(
            [sys.executable, str(repo_root / 'scripts' / script_name), '--help'],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0
        assert 'usage' in result.stdout.lower()
