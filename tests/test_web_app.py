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
                'num_labels_declared': 3, 'problem_type': 'multi_label_classification',
                'id2label_count': 3, 'label2id_count': 3,
                'classifier_weight_shape': '[3, 768]',
                'classifier_weight_mean': 0.05, 'classifier_weight_std': 0.15,
                'classifier_weight_min': -0.3, 'classifier_weight_max': 0.4,
                'appears_random_init': False,
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
                'num_labels_declared': 3, 'problem_type': 'multi_label_classification',
                'id2label_count': 3, 'label2id_count': 3,
                'classifier_weight_shape': '[3, 768]',
                'classifier_weight_mean': 0.001, 'classifier_weight_std': 0.02,
                'appears_random_init': True,
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
                            'classifier_weight_shape': '[18, 768]',
                            'classifier_weight_mean': 0.05,
                            'classifier_weight_std': 0.15,
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
