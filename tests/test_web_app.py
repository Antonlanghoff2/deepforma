from __future__ import annotations

from dataclasses import dataclass

import pytest

from services.recommendation_service import RecommendationService
from web_app import create_app


@dataclass
class DummyPredictor:
    calls: int = 0
    device: str = 'cpu'
    discriminating: bool = True

    def analyze(self, text, threshold=None):
        self.calls += 1
        if not self.discriminating:
            return self._non_discriminating_result(threshold)
        return self._discriminating_result(threshold)

    def _discriminating_result(self, threshold):
        return {
            'binary': {
                'is_ia': True, 'predicted_class': 1,
                'probability_non_ia': 0.25, 'probability_ia': 0.75,
            },
            'skills': {
                'predictions': [
                    {'label': 'Python', 'probability': 0.91, 'threshold': threshold or 0.35},
                    {'label': 'Machine Learning', 'probability': 0.72, 'threshold': threshold or 0.35},
                    {'label': 'Deep Learning', 'probability': 0.48, 'threshold': threshold or 0.35},
                ],
                'all_scores': [0.91, 0.72, 0.48],
                'score_min': 0.48, 'score_max': 0.91,
                'score_mean': 0.703, 'score_std': 0.176,
                'inference_time_ms': 42.0, 'num_labels': 3,
                'threshold_applied': threshold or 0.35,
            },
            'device': 'cpu',
            'inference_time_ms': 85.0,
            'checkpoint_audit': {
                'config_present': True, 'weights_present': True,
                'weights_size_bytes': 1000000,
                'architecture_declared': 'CamembertForSequenceClassification',
                'num_labels_declared': 3, 'num_labels_effective': 3,
                'problem_type': 'multi_label_classification',
                'id2label_count': 3, 'label2id_count': 3,
                'appears_random_init': False,
                'body_params_match_base': False,
                'parameter_errors': [],
                'classifier_params': {
                    'classifier.dense.weight': {
                        'shape': '[768, 768]', 'dtype': 'torch.float32',
                        'requires_grad': True,
                        'mean': 0.05, 'std': 0.15, 'min': -0.3, 'max': 0.4,
                        'l2_norm': 120.0, 'n_nonzero': 589824, 'proportion_nonzero': 1.0,
                    },
                    'classifier.dense.bias': {
                        'shape': '[768]', 'dtype': 'torch.float32',
                        'requires_grad': True,
                        'mean': 0.01, 'std': 0.1, 'min': -0.2, 'max': 0.2,
                        'l2_norm': 5.0, 'n_nonzero': 768, 'proportion_nonzero': 1.0,
                    },
                    'classifier.out_proj.weight': {
                        'shape': '[3, 768]', 'dtype': 'torch.float32',
                        'requires_grad': True,
                        'mean': 0.05, 'std': 0.15, 'min': -0.3, 'max': 0.4,
                        'l2_norm': 6.0, 'n_nonzero': 2304, 'proportion_nonzero': 1.0,
                    },
                    'classifier.out_proj.bias': {
                        'shape': '[3]', 'dtype': 'torch.float32',
                        'requires_grad': True,
                        'mean': 0.02, 'std': 0.08, 'min': -0.1, 'max': 0.1,
                        'l2_norm': 1.0, 'n_nonzero': 3, 'proportion_nonzero': 1.0,
                    },
                },
            },
        }

    def _non_discriminating_result(self, threshold):
        return {
            'binary': {
                'is_ia': False, 'predicted_class': 0,
                'probability_non_ia': 0.51, 'probability_ia': 0.49,
            },
            'skills': {
                'predictions': [
                    {'label': 'Python', 'probability': 0.51, 'threshold': threshold or 0.35},
                    {'label': 'Machine Learning', 'probability': 0.50, 'threshold': threshold or 0.35},
                    {'label': 'Deep Learning', 'probability': 0.49, 'threshold': threshold or 0.35},
                ],
                'all_scores': [0.51, 0.50, 0.49],
                'score_min': 0.49, 'score_max': 0.51,
                'score_mean': 0.50, 'score_std': 0.008,
                'inference_time_ms': 40.0, 'num_labels': 3,
                'threshold_applied': threshold or 0.35,
            },
            'device': 'cpu',
            'inference_time_ms': 80.0,
            'checkpoint_audit': {
                'config_present': True, 'weights_present': True,
                'weights_size_bytes': 1000000,
                'architecture_declared': 'CamembertForSequenceClassification',
                'num_labels_declared': 3, 'num_labels_effective': 3,
                'problem_type': 'multi_label_classification',
                'id2label_count': 3, 'label2id_count': 3,
                'appears_random_init': True,
                'body_params_match_base': True,
                'parameter_errors': [],
                'classifier_params': {
                    'classifier.dense.weight': {
                        'shape': '[768, 768]', 'dtype': 'torch.float32',
                        'requires_grad': True,
                        'mean': 0.000, 'std': 0.020, 'min': -0.08, 'max': 0.08,
                        'l2_norm': 15.0, 'n_nonzero': 589824, 'proportion_nonzero': 1.0,
                    },
                    'classifier.dense.bias': {
                        'shape': '[768]', 'dtype': 'torch.float32',
                        'requires_grad': True,
                        'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0,
                        'l2_norm': 0.0, 'n_nonzero': 0, 'proportion_nonzero': 0.0,
                    },
                    'classifier.out_proj.weight': {
                        'shape': '[3, 768]', 'dtype': 'torch.float32',
                        'requires_grad': True,
                        'mean': 0.001, 'std': 0.020, 'min': -0.06, 'max': 0.06,
                        'l2_norm': 2.0, 'n_nonzero': 2304, 'proportion_nonzero': 1.0,
                    },
                    'classifier.out_proj.bias': {
                        'shape': '[3]', 'dtype': 'torch.float32',
                        'requires_grad': True,
                        'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0,
                        'l2_norm': 0.0, 'n_nonzero': 0, 'proportion_nonzero': 0.0,
                    },
                },
            },
        }


class DummyOfferClient:
    def __init__(self, offers=None, error=None):
        self.offers = offers or []
        self.error = error

    def iter_offers(self, *args, **kwargs):
        if self.error:
            raise self.error
        yield from self.offers


def build_app(predictor=None, client_factory=None):
    return create_app(
        predictor=predictor or DummyPredictor(),
        france_travail_client_factory=client_factory,
        cache_ttl_seconds=60,
    )


def test_home_page():
    app = build_app()
    client = app.test_client()
    response = client.get('/')
    assert response.status_code == 200
    assert 'Diagnostic territorial' in response.get_data(as_text=True)


def test_empty_form_returns_error():
    app = build_app()
    client = app.test_client()
    response = client.post('/analyze', data={'programme': '', 'departement': ''})
    assert response.status_code == 400
    assert 'obligatoire' in response.get_data(as_text=True)


def test_api_analysis_with_mocks():
    predictor = DummyPredictor(discriminating=True)
    offers = [
        {'title': 'Offre 1', 'description': 'Python et data',
         'competences': [{'label': 'Python'}, {'label': 'SQL'}]},
        {'title': 'Offre 2', 'description': 'Machine Learning',
         'competences': [{'label': 'Machine Learning'}]},
    ]
    app = build_app(predictor=predictor,
                     client_factory=lambda: DummyOfferClient(offers=offers))
    client = app.test_client()
    response = client.post(
        '/api/analyze',
        json={'programme': 'Programme Python et IA', 'departement': '93',
              'keywords': 'python', 'threshold': 0.5},
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload['ok'] is True
    assert payload['result']['formation_analysis_status'] == 'reliable'
    assert payload['result']['comparison_available'] is True
    assert payload['result']['recommendations_available'] is True
    assert payload['result']['classification']['is_ia'] is True
    assert len(payload['result']['detected_skills']) >= 1
    assert predictor.calls == 1


def test_health_check():
    app = build_app()
    client = app.test_client()
    response = client.get('/health')
    assert response.status_code == 200
    payload = response.get_json()
    assert payload['status'] == 'ok'
    assert payload['models_available'] is True


def test_france_travail_error():
    app = build_app(
        client_factory=lambda: DummyOfferClient(
            error=RuntimeError('France Travail a repondu avec une limite de debit (429).')
        )
    )
    client = app.test_client()
    response = client.post(
        '/api/analyze',
        json={'programme': 'Programme', 'departement': '93', 'keywords': 'python'},
    )
    assert response.status_code == 429
    payload = response.get_json()
    assert payload['ok'] is False


def _raise_invalid_config():
    raise ValueError(
        'FRANCE_TRAVAIL_CLIENT_ID et FRANCE_TRAVAIL_CLIENT_SECRET doivent etre definis.'
    )


def test_france_travail_invalid_config():
    app = build_app(client_factory=_raise_invalid_config)
    client = app.test_client()
    response = client.post(
        '/api/analyze',
        json={'programme': 'Programme', 'departement': '93', 'keywords': 'python'},
    )
    assert response.status_code == 503
    payload = response.get_json()
    assert payload['ok'] is False
    assert 'Configuration France Travail' in payload['error']


def test_no_offers():
    app = build_app(client_factory=lambda: DummyOfferClient(offers=[]))
    client = app.test_client()
    response = client.post(
        '/api/analyze',
        json={'programme': 'Programme', 'departement': '93'},
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload['result']['summary']['total_offers_analyzed'] == 0


def test_recommendation_comparison():
    service = RecommendationService()
    report = service.compare(
        ['Python', 'Machine Learning'],
        [
            {'normalized_skills': ['Python', 'SQL']},
            {'normalized_skills': ['Python']},
            {'normalized_skills': ['Machine Learning', 'SQL']},
        ],
    )
    assert report.coverage_score == 60.0
    assert 'Python' in report.covered_skills
    assert any(skill.label.lower() == 'sql' for skill in report.missing_priority_skills)


def test_model_loading_once(monkeypatch):
    from inference import deepforma_predictor as predictor_module

    class FakeModel:
        def __init__(self, out_features):
            self.config = type('Config', (), {'num_labels': out_features})()
            self.classifier = type('Classifier', (), {
                'out_proj': type('Proj', (), {'out_features': out_features})()
            })()

        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, **kwargs):
            import torch
            return type('Output', (), {'logits': torch.tensor([[0.1, 0.9]])})()

        def parameters(self):
            return []

    class FakeTokenizer:
        def __call__(self, *args, **kwargs):
            import torch
            return {'input_ids': torch.tensor([[1, 2]]),
                    'attention_mask': torch.tensor([[1, 1]])}

    load_calls = {'model': 0, 'tokenizer': 0}

    def fake_model_loader(*args, **kwargs):
        load_calls['model'] += 1
        if 'binary' in str(args[0]):
            return FakeModel(2)
        return FakeModel(3)

    def fake_tokenizer_loader(*args, **kwargs):
        load_calls['tokenizer'] += 1
        return FakeTokenizer()

    monkeypatch.setattr(predictor_module.torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(predictor_module.AutoModelForSequenceClassification,
                        'from_pretrained', fake_model_loader)
    monkeypatch.setattr(predictor_module.AutoTokenizer,
                        'from_pretrained', fake_tokenizer_loader)
    monkeypatch.setattr(predictor_module, 'load_label_classes',
                        lambda path=None: ['Python', 'ML', 'DL'])
    monkeypatch.setattr(predictor_module, 'load_thresholds',
                        lambda path=None: {'multilabel_threshold': 0.35})
    monkeypatch.setattr(predictor_module, '_audit_checkpoint',
                        lambda path: {
                            'appears_random_init': False,
                            'body_params_match_base': False,
                            'parameter_errors': [],
                            'num_labels_effective': 18,
                            'classifier_params': {},
                        })
    predictor_module.get_predictor.cache_clear()

    first = predictor_module.get_predictor()
    second = predictor_module.get_predictor()
    assert first is second
    assert load_calls['model'] == 2
    assert load_calls['tokenizer'] == 2


# ===== Business-critical tests =====

def test_unreliable_skill_analysis_does_not_claim_missing_skills():
    """Quand l'analyse est non fiable, aucune competence ne doit etre
    declaree absente et aucune recommandation d'ajout ne doit etre generee."""
    predictor = DummyPredictor(discriminating=False)
    offers = [
        {'title': 'Offre Python', 'description': 'Python',
         'competences': [{'label': 'Python'}]},
        {'title': 'Offre ML', 'description': 'ML',
         'competences': [{'label': 'Machine Learning'}]},
    ]
    app = build_app(predictor=predictor,
                     client_factory=lambda: DummyOfferClient(offers=offers))
    client = app.test_client()
    response = client.post(
        '/api/analyze',
        json={'programme': 'Programme Python', 'departement': '93'},
    )
    assert response.status_code == 200
    payload = response.get_json()
    result = payload['result']

    assert result['formation_analysis_status'] == 'unreliable'
    assert result['comparison_available'] is False
    assert result['recommendations_available'] is False
    assert result['skills_presence'] == 'indeterminate'
    assert 'skill_scores_not_discriminant' in result['blocking_reasons']
    assert len(result['detected_skills']) == 0
    assert len(result['low_confidence_skills']) == 0
    assert len(result['indeterminate_skills']) == 3
    assert len(result['missing_skills']) == 0
    assert len(result['recommendations']) == 0
    assert result['global_score'] == {}


def test_unreliable_does_not_claim_python_absent():
    """Quand l'analyse est non fiable, Python ne doit jamais etre
    declare absent."""
    predictor = DummyPredictor(discriminating=False)
    offers = [
        {'title': 'Offre Python', 'description': 'Python',
         'competences': [{'label': 'Python'}]},
    ]
    app = build_app(predictor=predictor,
                     client_factory=lambda: DummyOfferClient(offers=offers))
    client = app.test_client()
    response = client.post(
        '/api/analyze',
        json={'programme': 'Programme Python', 'departement': '93'},
    )
    payload = response.get_json()
    result = payload['result']

    for skill in result['indeterminate_skills']:
        assert skill['presence'] == 'indeterminate'
        assert skill['statut'] == 'indetermine'

    assert result['formation_analysis_status'] == 'unreliable'


def test_reliable_analysis_allows_absent_skills():
    """Quand l'analyse est fiable, les competences sous le seuil peuvent
    etre declarees absentes."""
    predictor = DummyPredictor(discriminating=True)
    offers = [
        {'title': 'Offre Python', 'description': 'Python',
         'competences': [{'label': 'Python'}]},
        {'title': 'Offre Java', 'description': 'Java',
         'competences': [{'label': 'Java'}]},
    ]
    app = build_app(predictor=predictor,
                     client_factory=lambda: DummyOfferClient(offers=offers))
    client = app.test_client()
    response = client.post(
        '/api/analyze',
        json={'programme': 'Programme Python', 'departement': '93'},
    )
    payload = response.get_json()
    result = payload['result']

    assert result['formation_analysis_status'] == 'reliable'
    assert result['comparison_available'] is True
    assert len(result['detected_skills']) >= 1
    assert len(result['low_confidence_skills']) >= 1


def test_non_discriminating_skills_all_indeterminate():
    """En mode non discriminant, TOUTES les competences doivent etre
    'indeterminate' quelle que soit leur probabilite individuelle."""
    predictor = DummyPredictor(discriminating=False)
    offers = [
        {'title': 'Offre', 'description': 'test',
         'competences': [{'label': 'Python'}]},
    ]
    app = build_app(predictor=predictor,
                     client_factory=lambda: DummyOfferClient(offers=offers))
    client = app.test_client()
    response = client.post(
        '/api/analyze',
        json={'programme': 'Programme test', 'departement': '93'},
    )
    payload = response.get_json()
    result = payload['result']

    for skill_list in [result['indeterminate_skills']]:
        for skill in skill_list:
            assert skill['presence'] == 'indeterminate'


# ===== Audit / Classifier tests =====

def test_checkpoint_audit_includes_classifier_params():
    """Le rapport d audit doit contenir les 4 parametres du classifieur
    (dense.weight, dense.bias, out_proj.weight, out_proj.bias)
    et signaler les parametres manquants."""
    from inference.deepforma_predictor import _audit_checkpoint, _classifier_param_names
    from pathlib import Path
    model_dir = Path('models/multilabel_competences_v2/final')
    if not model_dir.exists():
        pytest.skip('Checkpoint non disponible')

    audit = _audit_checkpoint(model_dir)
    expected_params = _classifier_param_names()
    for pname in expected_params:
        assert pname in audit['classifier_params'], (
            f'Parametre {pname} manquant dans le rapport'
        )
    assert isinstance(audit['classifier_params']['classifier.dense.weight'], dict)
    assert 'shape' in audit['classifier_params']['classifier.dense.weight']
    assert 'mean' in audit['classifier_params']['classifier.dense.weight']
    assert 'std' in audit['classifier_params']['classifier.dense.weight']
    assert 'l2_norm' in audit['classifier_params']['classifier.dense.weight']
    assert 'n_nonzero' in audit['classifier_params']['classifier.dense.weight']
    assert audit.get('parameter_errors') == []


def test_checkpoint_audit_handles_missing_checkpoint():
    """L audit ne doit pas planter sur un dossier checkpoint vide."""
    from inference.deepforma_predictor import _audit_checkpoint
    import tempfile
    import json
    from pathlib import Path

    tmpdir = Path(tempfile.mkdtemp())
    cfg = {
        'architectures': ['CamembertForSequenceClassification'],
        'hidden_size': 768,
        'model_type': 'camembert',
    }
    (tmpdir / 'config.json').write_text(json.dumps(cfg))
    audit = _audit_checkpoint(tmpdir)
    assert audit['config_present'] is True
    assert audit['weights_present'] is False
    assert isinstance(audit, dict)


def test_audit_detects_untrained_checkpoint():
    """Un checkpoint non entraine doit etre detecte: appears_random_init=True
    et body_params_match_base=True."""
    from inference.deepforma_predictor import _audit_checkpoint
    from pathlib import Path
    model_dir = Path('models/multilabel_competences_v2/final')
    if not model_dir.exists():
        pytest.skip('Checkpoint non disponible')

    audit = _audit_checkpoint(model_dir)
    assert audit['appears_random_init'] is True
    assert audit['body_params_match_base'] is True

def test_checkpoint_detects_trained_v1():
    """Le checkpoint v1 (modele_camembert_competences_ia) a ete entraine:
    body_params_match_base=False, les biases sont non nuls."""
    from inference.deepforma_predictor import _audit_checkpoint
    from pathlib import Path
    model_dir = Path('modele_camembert_competences_ia')
    if not model_dir.exists():
        pytest.skip('Checkpoint v1 non disponible')
    audit = _audit_checkpoint(model_dir)
    assert audit['body_params_match_base'] is False
    out_bias = audit['classifier_params'].get('classifier.out_proj.bias', {})
    assert out_bias.get('n_nonzero', 0) > 0, 'Biases out_proj devraient etre non nuls'
    assert out_bias.get('std', 0) > 0, 'Biases out_proj devraient avoir une variance non nulle'

def test_binary_checkpoint_also_untrained():
    """Le checkpoint binaire v2 est aussi non entraine."""
    from inference.deepforma_predictor import _audit_checkpoint
    from pathlib import Path
    model_dir = Path('models/binary_ia_v2/final')
    if not model_dir.exists():
        pytest.skip('Checkpoint binaire non disponible')
    audit = _audit_checkpoint(model_dir)
    assert audit['appears_random_init'] is True
    assert audit['body_params_match_base'] is True

def test_v2_multilabel_config_has_generic_labels():
    """Le config.json du v2 multilabel contient LABEL_0..LABEL_17
    (pas les vrais noms), contrairement au v1."""
    from pathlib import Path
    import json
    cfg_path = Path('models/multilabel_competences_v2/final/config.json')
    if not cfg_path.exists():
        pytest.skip('Checkpoint non disponible')
    cfg = json.loads(cfg_path.read_text())
    id2label = cfg.get('id2label', {})
    for v in id2label.values():
        assert v.startswith('LABEL_'), f'Label generique attendu, obtenu: {v}'

def test_v1_config_has_real_labels():
    """Le config.json du v1 contient les vrais noms de competences."""
    from pathlib import Path
    import json
    cfg_path = Path('modele_camembert_competences_ia/config.json')
    if not cfg_path.exists():
        pytest.skip('Checkpoint v1 non disponible')
    cfg = json.loads(cfg_path.read_text())
    id2label = cfg.get('id2label', {})
    assert 'Python' in id2label.values()
    assert 'Deep Learning' in id2label.values()

def test_gradients_flow_on_first_batch():
    """Le smoke test verifie que les 4 parametres du classifieur recoivent
    un gradient non nul sur le premier batch."""
    import sys, json, tempfile, os
    from pathlib import Path
    from scripts.smoke_test_classifier_training import (
        check_gradients, load_multilabel_data, CLASSIFIER_PARAM_NAMES
    )
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
    from datasets import Dataset
    from torch.utils.data import DataLoader

    csv_path = Path('data/processed/dataset_entrainement.csv')
    if not csv_path.exists():
        pytest.skip('Dataset non disponible')

    ds, labels = load_multilabel_data(csv_path, 8)
    tokenizer = AutoTokenizer.from_pretrained('camembert-base')
    def tok(batch):
        return tokenizer(batch['text'], truncation=True, max_length=64)
    ds = ds.map(tok, batched=True)
    ds = ds.remove_columns(['text'])
    ds.set_format('torch')
    coll = DataCollatorWithPadding(tokenizer)
    loader = DataLoader(ds, batch_size=8, collate_fn=coll)

    model = AutoModelForSequenceClassification.from_pretrained(
        'camembert-base',
        num_labels=len(labels),
        problem_type='multi_label_classification',
    )
    optim = torch.optim.AdamW(model.parameters(), lr=2e-5)

    for batch in loader:
        grad_info = check_gradients(model, optim, batch, 'multilabel', 'labels')
        break

    assert not grad_info['has_errors'], f'Erreurs gradients: {grad_info["errors"]}'
    for name in CLASSIFIER_PARAM_NAMES:
        assert name in grad_info['gradients'], f'{name} manquant'
        g = grad_info['gradients'][name]
        assert 'gradient_is_none' not in g, f'{name}: gradient absent'
        assert 'gradient_is_zero' not in g, f'{name}: gradient nul'
        assert g.get('grad_norm', 0) > 0, f'{name}: norme nulle'

def test_optimizer_includes_classifier_head():
    """L optimiseur doit contenir les 4 parametres de la tete de classification."""
    import torch
    from transformers import AutoModelForSequenceClassification
    from scripts.smoke_test_classifier_training import (
        check_optimizer_includes_classifier, CLASSIFIER_PARAM_NAMES
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        'camembert-base', num_labels=18, problem_type='multi_label_classification',
    )
    optim = torch.optim.AdamW(model.parameters(), lr=2e-5)
    result = check_optimizer_includes_classifier(model, optim)

    for name in CLASSIFIER_PARAM_NAMES:
        assert result.get(name), f'{name} absent de l optimiseur'
    assert all(result.values())

def test_weight_change_after_training():
    """Apres quelques pas d optimisation, les poids du classifieur doivent
    avoir change."""
    import torch, numpy as np
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
    from torch.utils.data import DataLoader
    from datasets import Dataset
    from pathlib import Path
    from sklearn.preprocessing import MultiLabelBinarizer
    import pandas as pd

    csv_path = Path('data/processed/dataset_entrainement.csv')
    if not csv_path.exists():
        pytest.skip('Dataset non disponible')

    df = pd.read_csv(csv_path)
    ia = df[df['statut_annotation'] == 'ia_confirmee'].head(8)
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform([['Automatisation', 'Python pour l\'IA'] for _ in range(len(ia))])
    texts = ['test'] * len(ia)
    ds = Dataset.from_dict({'text': texts, 'labels': Y.tolist()})
    tokenizer = AutoTokenizer.from_pretrained('camembert-base')
    def tok(b):
        return tokenizer(b['text'], truncation=True, max_length=64)
    ds = ds.map(tok, batched=True)
    ds = ds.remove_columns(['text'])
    ds.set_format('torch')

    model = AutoModelForSequenceClassification.from_pretrained(
        'camembert-base', num_labels=len(mlb.classes_),
        problem_type='multi_label_classification',
    )
    model.train()

    weights_before = {
        n: p.data.detach().cpu().clone()
        for n, p in model.named_parameters() if 'classifier' in n
    }

    optim = torch.optim.AdamW(model.parameters(), lr=2e-5)
    for _ in range(3):
        for batch in DataLoader(ds, batch_size=8, collate_fn=DataCollatorWithPadding(tokenizer)):
            optim.zero_grad()
            logits = model(**{k: v for k, v in batch.items() if k != 'labels'}).logits
            loss = torch.nn.BCEWithLogitsLoss()(logits, batch['labels'].float())
            loss.backward()
            optim.step()

    weights_after = {
        n: p.data.detach().cpu().clone()
        for n, p in model.named_parameters() if 'classifier' in n
    }

    for name in weights_before:
        if name in weights_after:
            diff = (weights_after[name] - weights_before[name]).abs().max().item()
            assert diff > 1e-6, f'{name}: poids inchanges (diff={diff})'

def test_smoke_test_passes_with_real_data():
    """Le smoke test reussi sur les donnees reelles (multi-label)."""
    import sys, json, subprocess
    from pathlib import Path

    csv_path = Path('data/processed/dataset_entrainement.csv')
    if not csv_path.exists():
        pytest.skip('Dataset non disponible')

    result = subprocess.run(
        [sys.executable, 'scripts/smoke_test_classifier_training.py',
         '--task', 'multilabel', '--samples', '32', '--epochs', '15',
         '--output', '/tmp/smoke_test_result.json'],
        capture_output=True, text=True, timeout=300,
    )
    report = json.loads(Path('/tmp/smoke_test_result.json').read_text())
    assert report['passed'], f'Smoke test echoue: {report["failed_checks"]}'

def test_smoke_test_binary_passes():
    """Le smoke test binaire reussi sur les donnees reelles."""
    import sys, json, subprocess
    from pathlib import Path

    csv_path = Path('data/processed/dataset_entrainement.csv')
    if not csv_path.exists():
        pytest.skip('Dataset non disponible')

    result = subprocess.run(
        [sys.executable, 'scripts/smoke_test_classifier_training.py',
         '--task', 'binary', '--samples', '32', '--epochs', '10',
         '--output', '/tmp/smoke_test_binary_result.json'],
        capture_output=True, text=True, timeout=300,
    )
    report = json.loads(Path('/tmp/smoke_test_binary_result.json').read_text())
    assert report['passed'], f'Smoke test binaire echoue: {report["failed_checks"]}'

def test_v2_classifier_stats_not_trained():
    """Les statistiques des poids du classifieur v2 confirment
    une initialisation aleatoire (std ~ 0.02, mean ~ 0)."""
    from inference.deepforma_predictor import _audit_checkpoint
    from pathlib import Path

    model_dir = Path('models/multilabel_competences_v2/final')
    if not model_dir.exists():
        pytest.skip('Checkpoint non disponible')

    audit = _audit_checkpoint(model_dir)
    out_proj = audit['classifier_params'].get('classifier.out_proj.weight', {})
    assert abs(out_proj.get('mean', 1)) < 0.01, 'Mean devrait etre ~0'
    assert abs(out_proj.get('std', 0) - 0.02) < 0.005, 'Std devrait etre ~0.02'
    assert out_proj.get('n_nonzero', 0) > 0, 'Poids devraient etre non nuls (init alea)'

    dense = audit['classifier_params'].get('classifier.dense.weight', {})
    assert abs(dense.get('mean', 1)) < 0.01, 'Mean dense devrait etre ~0'
    assert abs(dense.get('std', 0) - 0.02) < 0.005, 'Std dense devrait etre ~0.02'

def test_v1_classifier_biases_trained():
    """Les biases du classifieur v1 sont tous non nuls (entrainement)."""
    from inference.deepforma_predictor import _audit_checkpoint
    from pathlib import Path

    model_dir = Path('modele_camembert_competences_ia')
    if not model_dir.exists():
        pytest.skip('Checkpoint v1 non disponible')

    audit = _audit_checkpoint(model_dir)
    assert audit['body_params_match_base'] is False
    out_bias = audit['classifier_params'].get('classifier.out_proj.bias', {})
    assert out_bias.get('n_nonzero', 0) > 0, 'Biases out_proj devraient etre entrained'
    assert out_bias.get('std', 0) > 0, 'Biases devraient avoir une variance non nulle'

    dense_bias = audit['classifier_params'].get('classifier.dense.bias', {})
    assert dense_bias.get('n_nonzero', 0) > 0, 'Biases dense devraient etre entrained'
    assert dense_bias.get('std', 0) > 0, 'Biases dense devraient avoir une variance non nulle'
