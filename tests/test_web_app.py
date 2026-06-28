from __future__ import annotations

from dataclasses import dataclass

import pytest

from services.recommendation_service import RecommendationService
from web_app import create_app


@dataclass
class DummyPredictor:
    calls: int = 0
    device: str = 'cpu'

    def analyze(self, text, threshold=None):
        self.calls += 1
        return {
            'binary': {
                'is_ia': True,
                'predicted_class': 1,
                'probability_non_ia': 0.25,
                'probability_ia': 0.75,
            },
            'skills': [
                {'label': 'Python', 'probability': 0.91, 'threshold': threshold or 0.35},
                {'label': 'Machine Learning', 'probability': 0.72, 'threshold': threshold or 0.35},
            ],
            'device': 'cpu',
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
    assert 'Diagnostic territorial des compétences de formation' in response.get_data(as_text=True)


def test_empty_form_returns_error():
    app = build_app()
    client = app.test_client()
    response = client.post('/analyze', data={'programme': '', 'departement': ''})
    assert response.status_code == 400
    assert 'obligatoire' in response.get_data(as_text=True)


def test_api_analysis_with_mocks():
    predictor = DummyPredictor()
    offers = [
        {
            'title': 'Offre 1',
            'description': 'Python et data',
            'competences': [{'label': 'Python'}, {'label': 'SQL'}],
        },
        {
            'title': 'Offre 2',
            'description': 'Machine Learning',
            'competences': [{'label': 'Machine Learning'}],
        },
    ]
    app = build_app(predictor=predictor, client_factory=lambda: DummyOfferClient(offers=offers))
    client = app.test_client()
    response = client.post(
        '/api/analyze',
        json={
            'programme': 'Programme Python et IA',
            'departement': '93',
            'keywords': 'python',
            'threshold': 0.5,
        },
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload['ok'] is True
    assert payload['analysis']['binary']['is_ia'] is True
    assert payload['market']['offer_count'] == 2
    assert payload['market']['recommendation']['coverage_score'] >= 0
    assert predictor.calls == 1


def test_health_check():
    app = build_app()
    client = app.test_client()
    response = client.get('/health')
    assert response.status_code == 200
    payload = response.get_json()
    assert payload['status'] == 'ok'
    assert payload['models_available'] is True
    assert payload['device'] == 'cpu'


def test_france_travail_error():
    app = build_app(client_factory=lambda: DummyOfferClient(error=RuntimeError('France Travail a répondu avec une limite de débit (429).')))
    client = app.test_client()
    response = client.post(
        '/api/analyze',
        json={'programme': 'Programme', 'departement': '93', 'keywords': 'python'},
    )
    assert response.status_code == 429
    payload = response.get_json()
    assert payload['ok'] is False



def _raise_invalid_config():
    raise ValueError('FRANCE_TRAVAIL_CLIENT_ID et FRANCE_TRAVAIL_CLIENT_SECRET doivent être définis.')


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
    assert payload['market']['offer_count'] == 0
    assert payload['market']['recommendation']['offer_count'] == 0


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
            self.classifier = type('Classifier', (), {'out_proj': type('Proj', (), {'out_features': out_features})()})()

        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, **kwargs):
            import torch
            return type('Output', (), {'logits': torch.tensor([[0.1, 0.9]])})()

    class FakeTokenizer:
        def __call__(self, *args, **kwargs):
            import torch
            return {'input_ids': torch.tensor([[1, 2]]), 'attention_mask': torch.tensor([[1, 1]])}

    load_calls = {'model': 0, 'tokenizer': 0}

    def fake_model_loader(*args, **kwargs):
        load_calls['model'] += 1
        if 'binary' in str(args[0]):
            return FakeModel(2)
        return FakeModel(18)

    def fake_tokenizer_loader(*args, **kwargs):
        load_calls['tokenizer'] += 1
        return FakeTokenizer()

    monkeypatch.setattr(predictor_module.torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(predictor_module.AutoModelForSequenceClassification, 'from_pretrained', fake_model_loader)
    monkeypatch.setattr(predictor_module.AutoTokenizer, 'from_pretrained', fake_tokenizer_loader)
    monkeypatch.setattr(predictor_module, 'load_label_classes', lambda path=None: [f'LABEL_{i}' for i in range(18)])
    monkeypatch.setattr(predictor_module, 'load_thresholds', lambda path=None: {'multilabel_threshold': 0.35})
    predictor_module.get_predictor.cache_clear()

    first = predictor_module.get_predictor()
    second = predictor_module.get_predictor()
    assert first is second
    assert load_calls['model'] == 2
    assert load_calls['tokenizer'] == 2
