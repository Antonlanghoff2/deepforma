from __future__ import annotations

import base64
from dataclasses import dataclass
from io import BytesIO
from urllib.parse import urlencode

import pytest

from referential_import.models import DerivedSkill, EvaluationCriterion, ImportReport, OfficialCompetency, ReferentialActivity, ReferentialBlock, ReferentialDocument
from continual_learning.store import ContinualLearningStore
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
    html = response.get_data(as_text=True)
    assert "Analyse d'un référentiel RNCP" in html
    assert 'PDF du référentiel' in html
    assert 'Département' in html


def test_empty_form_returns_error():
    app = build_app()
    client = app.test_client()
    response = client.post('/analyze', data={'programme': '', 'departement': ''})
    assert response.status_code == 400
    assert 'obligatoire' in response.get_data(as_text=True)


def test_referential_analysis_route_uses_pdf_upload(monkeypatch):
    from referential_import.import_service import ReferentialImportService
    import web_app as web_app_module

    document = ReferentialDocument(
        id='doc-2',
        source_path='/tmp/referentiel.pdf',
        file_name='referentiel.pdf',
        sha256='def456',
        page_count=1,
        collected_at='2026-07-03T00:00:00+00:00',
        text_extraction_method='pdftotext-layout',
        title='',
    )
    competency = OfficialCompetency(
        code='C1.1',
        official_label='Organiser la coordination',
        normalized_label='organiser la coordination',
        block_code='BLOC_1',
        activity_code='A1.1',
        page_start=1,
        page_end=1,
        confidence=0.95,
        source_pages=[1],
    )
    derived = DerivedSkill(
        label='Coordination',
        canonical_label='Coordination',
        category='action',
        source_code='C1.1',
        source_type='competency',
        surface_form='coordination',
        normalized_surface='coordination',
        confidence=0.9,
        explicit=True,
        page_start=1,
        page_end=1,
        context='Organiser la coordination',
    )
    competency.derived_skills = [derived]
    analysis = {
        'document': document,
        'report': ImportReport(
            schema_version='1.0',
            importer_version='0.1.0',
            document_id='doc-2',
            source_hash='def456',
            pages=1,
            blocks=1,
            activities=1,
            competencies=1,
            criteria=0,
            derived_skills=1,
            tools_methods=0,
            errors=[],
            warnings=[],
            review_items=[],
            score_global=0.8,
            coverage_score=1.0,
            duplicate_document=False,
            extraction_mode='layout',
        ),
        'blocks': [ReferentialBlock(code='BLOC_1', label='Bloc 1', page_start=1, page_end=1, confidence=0.9, source_pages=[1])],
        'activities': [ReferentialActivity(code='A1.1', block_code='BLOC_1', label='Activité 1', page_start=1, page_end=1, confidence=0.9, source_pages=[1])],
        'competencies': [competency],
        'criteria': [],
        'derived_skills': [derived],
        'tools_methods': [derived],
    }

    class DummyPage:
        def __init__(self, text):
            self.text = text

    class DummyPdf:
        def __init__(self, pages):
            self.pages = pages

    monkeypatch.setattr(ReferentialImportService, 'analyze', lambda self, input_path: analysis)
    monkeypatch.setattr(web_app_module, 'load_pdf_document', lambda path: DummyPdf([DummyPage("Responsable commercial\nRéférentiel d'activités\nC1.1 Organiser la coordination")]))

    app = build_app(client_factory=lambda: DummyOfferClient(offers=[]))
    client = app.test_client()
    response = client.post(
        '/analyze',
        data={
            'pdf': (BytesIO(b'%PDF-1.4 fake'), 'referentiel.pdf'),
            'departement': '93',
        },
        content_type='multipart/form-data',
    )
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert 'Tableau de bord' in html
    assert 'Contenu extrait du PDF' in html
    assert 'Intitulé détecté' in html
    assert 'Intitulé utilisé pour France Travail' in html
    assert 'Compétences officielles du référentiel' in html
    assert 'Organiser la coordination' in html


def test_market_offers_are_listed_with_preview_and_details(monkeypatch):
    from referential_import.import_service import ReferentialImportService
    import web_app as web_app_module

    document = ReferentialDocument(
        id='doc-5',
        source_path='/tmp/referentiel.pdf',
        file_name='referentiel.pdf',
        sha256='mno345',
        page_count=1,
        collected_at='2026-07-03T00:00:00+00:00',
        text_extraction_method='pdftotext-layout',
        title='',
    )
    competency = OfficialCompetency(
        code='C1.1',
        official_label='Organiser la coordination',
        normalized_label='organiser la coordination',
        block_code='BLOC_1',
        activity_code='A1.1',
        page_start=1,
        page_end=1,
        confidence=0.95,
        source_pages=[1],
    )
    analysis = {
        'document': document,
        'report': ImportReport(
            schema_version='1.0',
            importer_version='0.1.0',
            document_id='doc-5',
            source_hash='mno345',
            pages=1,
            blocks=1,
            activities=1,
            competencies=1,
            criteria=0,
            derived_skills=0,
            tools_methods=0,
            errors=[],
            warnings=[],
            review_items=[],
            score_global=0.0,
            coverage_score=0.0,
            duplicate_document=False,
            extraction_mode='layout',
        ),
        'blocks': [],
        'activities': [],
        'competencies': [competency],
        'criteria': [],
        'derived_skills': [],
        'tools_methods': [],
    }

    class DummyPage:
        def __init__(self, text):
            self.text = text

    class DummyPdf:
        def __init__(self, pages):
            self.pages = pages

    class QueryAwareClient:
        def __init__(self):
            self.seen_keywords = []

        def iter_offers(self, criteria, **kwargs):
            self.seen_keywords.append(criteria.keywords)
            if criteria.keywords and 'Responsable commercial' in criteria.keywords:
                for i in range(11):
                    yield {
                        'id': f'offer-{i + 1}',
                        'title': f'Responsable commercial {i + 1}',
                        'description': 'Pilotage commercial et coordination des equipes',
                        'competences': [
                            {'label': 'Coordination'} if i == 0 else {'label': 'Négociation'},
                            {'label': 'Commerce'} if i == 0 else {'label': 'Relation client'},
                        ],
                        'location': {'label': 'Paris'},
                        'contract': {'label': 'CDI'},
                        'url': f'https://example.com/offer-{i + 1}',
                    }

    tracking_client = QueryAwareClient()
    monkeypatch.setattr(ReferentialImportService, 'analyze', lambda self, input_path: analysis)
    monkeypatch.setattr(web_app_module, 'load_pdf_document', lambda path: DummyPdf([DummyPage("Responsable commercial\nRéférentiel d'activités\nC1.1 Organiser la coordination")]))

    app = build_app(client_factory=lambda: tracking_client)
    client = app.test_client()
    response = client.post(
        '/analyze',
        data={
            'pdf': (BytesIO(b'%PDF-1.4 fake'), 'referentiel.pdf'),
            'departement': '93',
        },
        content_type='multipart/form-data',
    )
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "Offres utilisées pour l'étude de marché (11)" in html
    assert 'Voir plus (1 offres supplémentaires)' in html
    assert 'Compétences exactes utilisées' in html
    assert 'data-market-filter="exact"' in html
    assert 'data-market-filter="structured"' in html
    assert 'https://example.com/offer-1' in html
    assert html.index('Responsable commercial 1') < html.index('Responsable commercial 2')
    assert tracking_client.seen_keywords[0] == 'Responsable commercial'


def test_referential_analysis_falls_back_to_broader_market_queries(monkeypatch):
    from referential_import.import_service import ReferentialImportService
    import web_app as web_app_module

    document = ReferentialDocument(
        id='doc-4',
        source_path='/tmp/referentiel.pdf',
        file_name='referentiel.pdf',
        sha256='jkl012',
        page_count=1,
        collected_at='2026-07-03T00:00:00+00:00',
        text_extraction_method='pdftotext-layout',
        title='',
    )
    competency = OfficialCompetency(
        code='C1.1',
        official_label='Organiser la coordination',
        normalized_label='organiser la coordination',
        block_code='BLOC_1',
        activity_code='A1.1',
        page_start=1,
        page_end=1,
        confidence=0.95,
        source_pages=[1],
    )
    analysis = {
        'document': document,
        'report': ImportReport(
            schema_version='1.0',
            importer_version='0.1.0',
            document_id='doc-4',
            source_hash='jkl012',
            pages=1,
            blocks=1,
            activities=1,
            competencies=1,
            criteria=0,
            derived_skills=0,
            tools_methods=0,
            errors=[],
            warnings=[],
            review_items=[],
            score_global=0.0,
            coverage_score=0.0,
            duplicate_document=False,
            extraction_mode='layout',
        ),
        'blocks': [],
        'activities': [],
        'competencies': [competency],
        'criteria': [],
        'derived_skills': [],
        'tools_methods': [],
    }

    class DummyPage:
        def __init__(self, text):
            self.text = text

    class DummyPdf:
        def __init__(self, pages):
            self.pages = pages

    class QueryAwareClient:
        def __init__(self):
            self.seen_keywords = []

        def iter_offers(self, criteria, **kwargs):
            self.seen_keywords.append(criteria.keywords)
            if criteria.keywords and 'Responsable commercial' in criteria.keywords:
                yield {
                    'id': 'offer-1',
                    'title': 'Responsable commercial',
                    'description': 'Pilotage commercial',
                    'competences': [{'label': 'Négociation'}],
                }

    tracking_client = QueryAwareClient()
    monkeypatch.setattr(ReferentialImportService, 'analyze', lambda self, input_path: analysis)
    monkeypatch.setattr(web_app_module, 'load_pdf_document', lambda path: DummyPdf([DummyPage("Responsable commercial\nRéférentiel d'activités\nC1.1 Organiser la coordination")]))

    app = build_app(client_factory=lambda: tracking_client)
    client = app.test_client()
    response = client.post(
        '/analyze',
        data={
            'pdf': (BytesIO(b'%PDF-1.4 fake'), 'referentiel.pdf'),
            'departement': '93',
        },
        content_type='multipart/form-data',
    )
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert 'Intitulé utilisé pour France Travail' in html
    assert "Offres utilisées pour l'étude de marché" in html
    assert 'Responsable commercial' in html
    assert tracking_client.seen_keywords[0] == 'Responsable commercial'

def test_referential_analysis_continues_if_france_travail_fails(monkeypatch):
    from france_travail.client import FranceTravailRateLimitError
    from referential_import.import_service import ReferentialImportService
    import web_app as web_app_module

    document = ReferentialDocument(
        id='doc-3',
        source_path='/tmp/referentiel.pdf',
        file_name='referentiel.pdf',
        sha256='ghi789',
        page_count=1,
        collected_at='2026-07-03T00:00:00+00:00',
        text_extraction_method='pdftotext-layout',
    )
    analysis = {
        'document': document,
        'report': ImportReport(
            schema_version='1.0',
            importer_version='0.1.0',
            document_id='doc-3',
            source_hash='ghi789',
            pages=1,
            blocks=1,
            activities=1,
            competencies=1,
            criteria=0,
            derived_skills=0,
            tools_methods=0,
            errors=[],
            warnings=[],
            review_items=[],
            score_global=0.0,
            coverage_score=0.0,
            duplicate_document=False,
            extraction_mode='layout',
        ),
        'blocks': [],
        'activities': [],
        'competencies': [],
        'criteria': [],
        'derived_skills': [],
        'tools_methods': [],
    }

    class DummyPage:
        def __init__(self, text):
            self.text = text

    class DummyPdf:
        def __init__(self, pages):
            self.pages = pages

    monkeypatch.setattr(ReferentialImportService, 'analyze', lambda self, input_path: analysis)
    monkeypatch.setattr(web_app_module, 'load_pdf_document', lambda path: DummyPdf([DummyPage('C1.1 Organiser la coordination')]))

    app = build_app(client_factory=lambda: DummyOfferClient(error=FranceTravailRateLimitError('429')))
    client = app.test_client()
    response = client.post(
        '/analyze',
        data={
            'pdf': (BytesIO(b'%PDF-1.4 fake'), 'referentiel.pdf'),
            'departement': '93',
        },
        content_type='multipart/form-data',
    )
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert 'France Travail a repondu avec une limite de debit (429).' in html
    assert 'Analyse du referentiel reste disponible' in html or 'Analyse territoriale partielle' in html


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


def test_referential_import_preview_on_home_page(monkeypatch):
    from referential_import.import_service import ReferentialImportService

    document = ReferentialDocument(
        id='doc-1',
        source_path='/tmp/referentiel.pdf',
        file_name='referentiel.pdf',
        sha256='abc123',
        page_count=2,
        collected_at='2026-07-03T00:00:00+00:00',
        text_extraction_method='pdftotext-layout',
    )
    report = ImportReport(
        schema_version='1.0',
        importer_version='0.1.0',
        document_id='doc-1',
        source_hash='abc123',
        pages=2,
        blocks=1,
        activities=1,
        competencies=1,
        criteria=1,
        derived_skills=1,
        tools_methods=1,
        errors=[],
        warnings=[],
        review_items=[],
        score_global=0.9,
        coverage_score=1.0,
        duplicate_document=False,
        extraction_mode='pdftotext-layout',
    )
    competency = OfficialCompetency(
        code='C1.1',
        official_label='Déployer Excel',
        normalized_label='deployer excel',
        block_code='BLOC_1',
        activity_code='A1.1',
        page_start=1,
        page_end=1,
        confidence=0.9,
        source_pages=[1],
    )
    criterion = EvaluationCriterion(
        code='CE1.1.1',
        competency_code='C1.1',
        criterion_label='Mobiliser Excel',
        normalized_label='mobiliser excel',
        page_start=1,
        page_end=1,
        confidence=0.9,
        source_pages=[1],
    )
    competency.evaluation_criteria = [criterion]
    derived = DerivedSkill(
        label='Excel',
        canonical_label='Excel',
        category='tool',
        source_code='C1.1',
        source_type='competency',
        surface_form='Excel',
        normalized_surface='excel',
        confidence=0.95,
        explicit=True,
        page_start=1,
        page_end=1,
        context='Déployer Excel',
    )
    competency.derived_skills = [derived]
    analysis = {
        'document': document,
        'report': report,
        'blocks': [ReferentialBlock(code='BLOC_1', label='Bloc 1', page_start=1, page_end=1, confidence=0.9, source_pages=[1])],
        'activities': [ReferentialActivity(code='A1.1', block_code='BLOC_1', label='Activité 1', page_start=1, page_end=1, confidence=0.9, source_pages=[1])],
        'competencies': [competency],
        'criteria': [criterion],
        'derived_skills': [derived],
        'tools_methods': [derived],
    }

    monkeypatch.setattr(ReferentialImportService, 'analyze', lambda self, input_path: analysis)
    app = build_app()
    client = app.test_client()
    response = client.post(
        '/referential/import',
        data={'pdf': (BytesIO(b'%PDF-1.4 fake'), 'referentiel.pdf')},
        content_type='multipart/form-data',
    )
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "Rapport d'extraction référentiel" in html
    assert 'Déployer Excel' in html
    assert "Rapport d'extraction référentiel" in html




def test_referential_import_validation_step_then_analysis_uses_corrected_title(monkeypatch, tmp_path):
    import json
    from referential_import.import_service import ReferentialImportService, build_export_payload
    from referential_import.store import ReferentialImportStore
    import web_app as web_app_module

    store = ReferentialImportStore(tmp_path / 'imports.sqlite3')
    service = ReferentialImportService(store=store, output_dir=tmp_path / 'out')
    monkeypatch.setattr(web_app_module, 'ReferentialImportService', lambda *args, **kwargs: service)

    document = ReferentialDocument(
        id='doc-validation',
        source_path='/tmp/referentiel.pdf',
        file_name='referentiel.pdf',
        sha256='val123',
        page_count=1,
        collected_at='2026-07-03T00:00:00+00:00',
        text_extraction_method='pdftotext-layout',
        title='Manager d’affaires REFERENTIEL D’ACTIVITES décrit les situations de travail et les activités exercées, les métiers ou emplois visés',
    )
    report = ImportReport(
        schema_version='1.0',
        importer_version='0.1.0',
        document_id='doc-validation',
        source_hash='val123',
        pages=1,
        blocks=1,
        activities=1,
        competencies=1,
        criteria=0,
        derived_skills=1,
        tools_methods=1,
        errors=[],
        warnings=[],
        review_items=[],
        score_global=0.9,
        coverage_score=1.0,
        duplicate_document=False,
        extraction_mode='pdftotext-layout',
    )
    competency = OfficialCompetency(
        code='C1.1',
        official_label='Organiser la coordination',
        normalized_label='organiser la coordination',
        block_code='BLOC_1',
        activity_code='A1.1',
        page_start=1,
        page_end=1,
        confidence=0.95,
        source_pages=[1],
    )
    analysis = {
        'document': document,
        'report': report,
        'blocks': [ReferentialBlock(code='BLOC_1', label='Bloc 1', page_start=1, page_end=1, confidence=0.9, source_pages=[1])],
        'activities': [ReferentialActivity(code='A1.1', block_code='BLOC_1', label='Activité 1', page_start=1, page_end=1, confidence=0.9, source_pages=[1])],
        'competencies': [competency],
        'criteria': [],
        'derived_skills': [DerivedSkill(label='Coordination', canonical_label='Coordination', category='action', source_code='C1.1', source_type='competency', surface_form='coordination', normalized_surface='coordination', confidence=0.9, explicit=True, page_start=1, page_end=1, context='coordination')],
        'tools_methods': [],
    }

    class DummyPage:
        def __init__(self, text):
            self.text = text

    class DummyPdf:
        def __init__(self, pages):
            self.pages = pages

    class QueryAwareClient:
        def __init__(self):
            self.seen_keywords = []

        def iter_offers(self, criteria, **kwargs):
            self.seen_keywords.append(criteria.keywords)
            if criteria.keywords and "Manager d'affaires" in criteria.keywords:
                yield {
                    'id': 'offer-1',
                    'title': 'Manager d’affaires',
                    'description': 'Gestion commerciale',
                    'competences': [{'label': 'Gestion de projet'}],
                }

    tracking_client = QueryAwareClient()
    monkeypatch.setattr(ReferentialImportService, 'analyze', lambda self, input_path: analysis)
    monkeypatch.setattr(web_app_module, 'load_pdf_document', lambda path: DummyPdf([DummyPage("Manager d’affaires REFERENTIEL D’ACTIVITES décrit les situations de travail et les activités exercées, les métiers ou emplois visés\nC1.1 Organiser la coordination")]))

    app = build_app(client_factory=lambda: tracking_client)
    client = app.test_client()
    preview = client.post(
        '/referential/import',
        data={
            'pdf': (BytesIO(b'%PDF-1.4 fake'), 'referentiel.pdf'),
            'departement': '93',
        },
        content_type='multipart/form-data',
    )
    assert preview.status_code == 200
    preview_html = preview.get_data(as_text=True)
    assert 'Validation humaine' in preview_html
    assert 'Compétences à corriger' in preview_html
    assert 'Valider le référentiel et lancer l’analyse' in preview_html

    validation_payload = {
        'action': 'validate_referential',
        'analysis_json': json.dumps(build_export_payload(analysis), ensure_ascii=False),
        'source_path': '/tmp/referentiel.pdf',
        'departement': '93',
        'validated_title': "Manager d'affaires",
        'validation_note': 'Titre confirmé par lecture humaine',
        'competency_label__C1.1': 'Coordonner les parties prenantes',
        'competency_status__C1.1': 'corrected',
    }
    response = client.post(
        '/analyze',
        data=validation_payload,
    )
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert 'Tableau de bord' in html
    assert tracking_client.seen_keywords[0] == "Manager d'affaires"
    validated_competencies = store.list_annotations('code = ?', ('C1.1',))
    assert validated_competencies
    import json as _json
    competency_payload = _json.loads(validated_competencies[0]['payload_json'])
    assert competency_payload['official_label'] == 'Coordonner les parties prenantes'
    assert competency_payload['review_status'] == 'corrected'
    approved_imports = store.list_imports(status='approved')
    assert len(approved_imports) == 1
    assert approved_imports[0]['review_status'] == 'approved'




def test_admin_referential_import_edits_are_persisted(monkeypatch, tmp_path):
    import json
    from referential_import.import_service import ReferentialImportService, build_export_payload
    from referential_import.store import ReferentialImportStore
    import web_app as web_app_module

    import_store = ReferentialImportStore(tmp_path / 'referential_imports.sqlite3')
    service = ReferentialImportService(store=import_store, output_dir=tmp_path / 'imports')
    monkeypatch.setenv('DEEPFORMA_ADMIN_USER', 'anton')
    monkeypatch.setenv('DEEPFORMA_ADMIN_PASSWORD', 'deepforma')
    monkeypatch.setattr(web_app_module, 'ReferentialImportService', lambda *args, **kwargs: service)

    document = ReferentialDocument(
        id='doc-admin-1',
        source_path='/tmp/referentiel.pdf',
        file_name='referentiel.pdf',
        sha256='abc123',
        page_count=1,
        collected_at='2026-07-03T00:00:00+00:00',
        text_extraction_method='pdftotext-layout',
        title='Titre initial',
    )
    competency = OfficialCompetency(
        code='C1.1',
        official_label='Coordonner les parties prenantes',
        normalized_label='coordonner les parties prenantes',
        block_code='BLOC_1',
        activity_code='A1.1',
        page_start=1,
        page_end=1,
        confidence=0.95,
        source_pages=[1],
    )
    derived = DerivedSkill(
        label='Coordination',
        canonical_label='Coordination',
        category='skill',
        source_code='C1.1',
        source_type='competency',
        surface_form='Coordination',
        normalized_surface='coordination',
        confidence=0.9,
        explicit=True,
        page_start=1,
        page_end=1,
        context='Coordonner les parties prenantes',
    )
    competency.derived_skills = [derived]
    analysis = {
        'document': document,
        'report': ImportReport(
            schema_version='1.0',
            importer_version='0.1.0',
            document_id='doc-admin-1',
            source_hash='abc123',
            pages=1,
            blocks=1,
            activities=1,
            competencies=1,
            criteria=0,
            derived_skills=1,
            tools_methods=0,
            errors=[],
            warnings=[],
            review_items=[],
            score_global=0.8,
            coverage_score=1.0,
            duplicate_document=False,
            extraction_mode='layout',
        ),
        'blocks': [ReferentialBlock(code='BLOC_1', label='Bloc 1', page_start=1, page_end=1, confidence=0.9, source_pages=[1])],
        'activities': [ReferentialActivity(code='A1.1', block_code='BLOC_1', label='Activité 1', page_start=1, page_end=1, confidence=0.9, source_pages=[1])],
        'competencies': [competency],
        'criteria': [],
        'derived_skills': [derived],
        'tools_methods': [derived],
    }
    analysis_json = build_export_payload(analysis)

    app = build_app(client_factory=lambda: DummyOfferClient(offers=[]))
    app.testing = True
    client = app.test_client()
    auth = base64.b64encode(b'anton:deepforma').decode('ascii')
    response = client.post(
        '/admin/referential-import',
        data={
            'action': 'approve',
            'analysis_json': json.dumps(analysis_json, ensure_ascii=False),
            'source_path': '/tmp/referentiel.pdf',
            'validated_title': 'Data Scientist',
            'validation_note': 'Titre corrigé manuellement',
            'competency_label__C1.1': '',
            'remove_competency__C1.1': 'on',
            'new_competency_labels': 'Modélisation prédictive',
            'derived_skill_label__0': 'Coordination',
            'derived_skill_canonical__0': 'Coordination',
            'derived_skill_category__0': 'skill',
            'remove_derived_skill__0': 'on',
            'new_derived_skill_labels': 'TensorFlow | tool | TensorFlow',
            'validated_by': 'anton',
        },
        headers={'Authorization': f'Basic {auth}'},
    )
    assert response.status_code == 200
    approved_imports = import_store.list_imports(status='approved')
    assert len(approved_imports) == 1
    document_payload = json.loads(approved_imports[0]['document_json'])
    assert document_payload['title'] == 'Data Scientist'
    assert approved_imports[0]['validated_by'] == 'anton'
    output_path = tmp_path / 'imports' / 'referentiel.pdf.json'
    assert output_path.exists()
    output_payload = json.loads(output_path.read_text(encoding='utf-8'))
    assert output_payload['document']['title'] == 'Data Scientist'
    assert len(output_payload['competencies']) == 1
    assert output_payload['competencies'][0]['official_label'] == 'Modélisation prédictive'
    assert len(output_payload['derived_skills']) == 1
    assert output_payload['derived_skills'][0]['canonical_label'] == 'TensorFlow'
    assert output_payload['derived_skills'][0]['category'] == 'tool'

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
    """Le classifieur IA non fiable ne bloque plus l extraction.
    L extraction ouverte reussit, la comparaison est disponible."""
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

    # Extraction reussie meme si IA classifieur non discriminant
    assert result['formation_analysis_status'] == 'reliable'
    assert result['comparison_available'] is True
    assert result['recommendations_available'] is True
    assert payload['result']['territorial_market']['offer_count'] == 1
    # IA classification est unreliable
    assert result['ia_classification']['status'] == 'unreliable'
    assert result['ia_classification']['discriminating'] is False
    # Extraction a trouve Python
    assert len(result['skill_extraction']['skills']) >= 0
    assert len(result['skill_extraction']['tools']) >= 1
    tool_labels = [t['source_label'] for t in result['skill_extraction']['tools']]
    assert 'Python' in tool_labels


def test_unreliable_does_not_claim_python_absent():
    """Quand le classifieur IA est non discriminant, Python extrait
    par l extracteur ouvert reste present dans skill_extraction."""
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

    # Les 18 labels IA sont tous indetermines (classifieur non discriminant)
    for skill in result['indeterminate_skills']:
        assert skill['presence'] == 'indeterminate'
        assert skill['statut'] == 'indetermine'

    # Mais l extraction ouverte a reussi
    assert result['formation_analysis_status'] == 'reliable'
    assert result['skill_extraction']['status'] in ('success', 'partial')
    tool_labels = [t['source_label'] for t in result['skill_extraction']['tools']]
    assert 'Python' in tool_labels


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


def test_audit_detects_trained_checkpoint():
    """Le checkpoint multilabel v2 a ete entraine: appears_random_init=False,
    biasses non nuls."""
    from inference.deepforma_predictor import _audit_checkpoint
    from pathlib import Path
    model_dir = Path('models/multilabel_competences_v2/final')
    if not model_dir.exists():
        pytest.skip('Checkpoint non disponible')

    audit = _audit_checkpoint(model_dir)
    assert audit['config_present'] is True
    assert audit['weights_present'] is True
    assert audit['appears_random_init'] is False, (
        'Le checkpoint entraine ne devrait pas etre marque comme non entraine. '
        'Verifiez que les biais du classifier sont non nuls.'
    )
    assert audit['biases_all_zero'] is False

def test_audit_detects_untrained_checkpoint():
    """Un checkpoint non entraine doit etre detecte: appears_random_init=True.
    Utilise le backup final.untrained."""
    from inference.deepforma_predictor import _audit_checkpoint
    from pathlib import Path
    model_dir = Path('models/multilabel_competences_v2/final.untrained')
    if not model_dir.exists():
        pytest.skip('Backup untrained non disponible')

    audit = _audit_checkpoint(model_dir)
    assert audit['appears_random_init'] is True
    assert audit['biases_all_zero'] is True

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

def test_v2_multilabel_config_has_taxonomy_ids():
    """Le config.json du v2 multilabel (retrained) contient des IDs de taxonomie
    (ml.intro, dl.intro, etc.), pas LABEL_N."""
    from pathlib import Path
    import json
    cfg_path = Path('models/multilabel_competences_v2/final/config.json')
    if not cfg_path.exists():
        pytest.skip('Checkpoint non disponible')
    cfg = json.loads(cfg_path.read_text())
    id2label = cfg.get('id2label', {})
    assert len(id2label) == 18
    first_val = list(id2label.values())[0]
    assert '.' in first_val, (
        f'ID de taxonomie attendu (ex: ml.intro), obtenu: {first_val}'
    )
    assert not first_val.startswith('LABEL_'), (
        f'Label generique incorrect apres entrainement: {first_val}'
    )

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


# ===== Independence tests: IA classifier vs skill extraction =====

def test_unreliable_ia_classifier_does_not_block_skill_extraction():
    """Le classifieur IA peut etre unreliable alors que l extraction est success."""
    predictor = DummyPredictor(discriminating=False)
    app = build_app(predictor=predictor,
                     client_factory=lambda: DummyOfferClient(offers=[]))
    client = app.test_client()
    response = client.post(
        '/api/analyze',
        json={'programme': 'Maîtrise Python et SQL', 'departement': '93'},
    )
    payload = response.get_json()
    result = payload['result']

    assert result['ia_classification']['status'] == 'unreliable'
    assert result['ia_classification']['discriminating'] is False
    assert result['skill_extraction']['status'] in ('success', 'partial')
    assert result['formation_analysis_status'] == 'reliable'


def test_comparison_available_when_ia_unreliable():
    """La comparaison reste disponible meme si le classifieur IA est unreliable."""
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
        json={'programme': 'Maîtrise Python', 'departement': '93'},
    )
    payload = response.get_json()
    result = payload['result']

    assert result['ia_classification']['status'] == 'unreliable'
    assert result['comparison_available'] is True
    assert result['recommendations_available'] is True


def test_ia_warning_does_not_become_global_warning():
    """L avertissement IA ne devient pas un avertissement global."""
    predictor = DummyPredictor(discriminating=False)
    app = build_app(predictor=predictor,
                     client_factory=lambda: DummyOfferClient(offers=[]))
    client = app.test_client()
    response = client.post(
        '/api/analyze',
        json={'programme': 'Maîtrise Python et Docker',
              'departement': '93'},
    )
    payload = response.get_json()
    result = payload['result']

    # Les warnings IA sont scopes
    assert len(result['ia_classification']['warnings']) > 0
    # Aucun warning global bloquant
    assert result['formation_analysis_status'] == 'reliable'
    assert result['skill_extraction']['status'] in ('success', 'partial')


def test_skill_extracted_without_ia_label_btp():
    """Une competence BTP peut etre extraite sans label IA."""
    from skills.open_extractor import extract_skills
    text = "Maîtrise des techniques de maçonnerie et lecture de plans."
    results = extract_skills(text)
    labels = [r.source_label.lower() for r in results]
    assert any('maçonnerie' in l or 'maconnerie' in l for l in labels)
    assert any('plan' in l for l in labels)


def test_skill_extracted_without_ia_label_comptabilite():
    """Une competence comptable peut etre extraite sans label IA."""
    from skills.open_extractor import extract_skills
    text = "Savoir établir un bilan comptable et maîtriser la comptabilité générale."
    results = extract_skills(text)
    labels = [r.source_label.lower() for r in results]
    assert any('comptabil' in l or 'bilan' in l for l in labels)


def test_skill_extracted_without_ia_label_coiffure():
    """Une competence coiffure peut etre extraite sans label IA."""
    from skills.open_extractor import extract_skills
    text = "Maîtrise des techniques de coupe et savoir réaliser des colorations."
    results = extract_skills(text)
    labels = [r.source_label.lower() for r in results]
    assert any('coupe' in l for l in labels)
    assert any('coloration' in l for l in labels)


def test_unknown_skill_preserved():
    """Une competence inconnue du referentiel est conservee."""
    from skills.open_extractor import extract_skills
    text = "Savoir pratiquer la médecine traditionnelle chinoise."
    results = extract_skills(text)
    labels = [r.source_label.lower() for r in results]
    assert any('médecine' in l or 'medecine' in l for l in labels)
    assert any('chinoise' in l or 'traditionnelle' in l for l in labels)


def test_eighteen_labels_not_used_as_primary_skills():
    """Les 18 scores IA ne sont jamais utilises comme liste principale de competences."""
    predictor = DummyPredictor(discriminating=True)
    app = build_app(predictor=predictor,
                     client_factory=lambda: DummyOfferClient(offers=[]))
    client = app.test_client()
    response = client.post(
        '/api/analyze',
        json={'programme': 'Maîtrise Kubernetes et Docker',
              'departement': '93'},
    )
    payload = response.get_json()
    result = payload['result']

    # Les competences extraites viennent de l extracteur ouvert (tools)
    assert len(result['skill_extraction']['tools']) >= 1
    # detected_skills ne contient que les labels du modele 18-IA
    assert len(result['detected_skills']) >= 1 or len(result['low_confidence_skills']) >= 1
    # comparison_available est pilote par skill_extraction, pas par les 18 labels
    assert result['comparison_available'] is True


def test_market_comparison_uses_open_extracted_skills():
    """La comparaison marche utilise les competences extraites, pas les 18 labels."""
    predictor = DummyPredictor(discriminating=True)
    offers = [
        {'title': 'Offre Kubernetes', 'description': 'Kubernetes',
         'competences': [{'label': 'Kubernetes'}]},
        {'title': 'Offre Docker', 'description': 'Docker',
         'competences': [{'label': 'Docker'}]},
    ]
    app = build_app(predictor=predictor,
                     client_factory=lambda: DummyOfferClient(offers=offers))
    client = app.test_client()
    response = client.post(
        '/api/analyze',
        json={'programme': 'Maîtrise Kubernetes et Docker',
              'departement': '93'},
    )
    payload = response.get_json()
    result = payload['result']

    # Les competences extraites contiennent Kubernetes, Docker
    tool_labels = [t['source_label'].lower() for t in result['skill_extraction']['tools']]
    assert 'kubernetes' in tool_labels or any('kubernetes' in t for t in tool_labels)
    assert 'docker' in tool_labels or any('docker' in t for t in tool_labels)
    # La comparaison est disponible
    assert result['comparison_available'] is True
    # Il y a des offres analysees
    assert result['summary']['total_offers_analyzed'] > 0



def _admin_auth_headers(username: str = 'anton', password: str = 'deepforma') -> dict[str, str]:
    token = base64.b64encode(f'{username}:{password}'.encode('utf-8')).decode('ascii')
    return {'Authorization': f'Basic {token}'}


def _admin_csrf_token(client, offer_row_id: int | None = None) -> str:
    url = '/admin/continual-learning'
    if offer_row_id is not None:
        url = f'{url}?offer_row_id={offer_row_id}'
    response = client.get(url, headers=_admin_auth_headers())
    assert response.status_code == 200
    with client.session_transaction() as session_data:
        return session_data['_csrf_token']


def _seed_continual_learning_offer(
    store: ContinualLearningStore,
    *,
    offer_id: str,
    title: str,
    description: str,
    territory: str = '93',
    job_family: str = 'Commerce',
    model_scores: list[tuple[str, float]] | None = None,
    ft_skills: list[dict[str, object]] | None = None,
    text_skills: list[dict[str, object]] | None = None,
    rejected_skills: list[dict[str, object]] | None = None,
    corrected_skills: list[dict[str, object]] | None = None,
    human_additions: list[dict[str, object]] | None = None,
) -> int:
    structured_skills = [
        {
            'canonical_name': str(item['canonical_name']),
            'label': str(item.get('label', item['canonical_name'])),
            'referential_code': str(item.get('referential_code', f'FT-{index}')),
            'referential_label': str(item.get('referential_label', item['canonical_name'])),
        }
        for index, item in enumerate(ft_skills or [], start=1)
    ]
    predicted_skills = [
        {
            'label': label,
            'canonical_name': label,
            'confidence': confidence,
            'source': 'camembert_multilabel',
            'provenance': 'model_prediction',
        }
        for label, confidence in (model_scores or [])
    ]
    offer = store.upsert_offer(
        offer_id=offer_id,
        title=title,
        description_original=description,
        collected_at='2026-07-03T00:00:00+00:00',
        location_label='Paris',
        territory=territory,
        job_family=job_family,
        structured_skills=structured_skills,
        predicted_skills=predicted_skills,
        detected_forms=[],
        offsets=[],
        confidence={'score': 0.5},
        sources=[{'source': 'unit-test'}],
        model_version='camembert_multilabel:v-test',
        validation_status='pending',
    )
    for item in ft_skills or []:
        store.upsert_annotation(
            offer_row_id=offer.offer_row_id,
            offer_id=offer_id,
            content_version=offer.content_version,
            canonical_name=str(item['canonical_name']),
            surface_form=str(item.get('surface_form', item['canonical_name'])),
            normalized_name=str(item['canonical_name']),
            label='SKILL',
            start=int(item['start']) if item.get('start') is not None else None,
            end=int(item['end']) if item.get('end') is not None else None,
            confidence=float(item.get('confidence', 0.95)),
            source='france_travail_api',
            provenance='france_travail_api',
            is_explicit=False,
            text_sentence=str(item.get('text_sentence', description)),
            referential_code=str(item.get('referential_code', 'FT-1')),
            referential_label=str(item.get('referential_label', item['canonical_name'])),
            validation_status=str(item.get('validation_status', 'approved')),
        )
    for item in text_skills or []:
        store.upsert_annotation(
            offer_row_id=offer.offer_row_id,
            offer_id=offer_id,
            content_version=offer.content_version,
            canonical_name=str(item['canonical_name']),
            surface_form=str(item.get('surface_form', item['canonical_name'])),
            normalized_name=str(item['canonical_name']),
            label='SKILL',
            start=int(item['start']) if item.get('start') is not None else None,
            end=int(item['end']) if item.get('end') is not None else None,
            confidence=float(item.get('confidence', 0.9)),
            source=str(item.get('source', 'text_explicit')),
            provenance=str(item.get('provenance', 'exact_reference_match')),
            is_explicit=bool(item.get('is_explicit', True)),
            text_sentence=str(item.get('text_sentence', description)),
            referential_code=str(item.get('referential_code')) if item.get('referential_code') is not None else None,
            referential_label=str(item.get('referential_label')) if item.get('referential_label') is not None else None,
            validation_status=str(item.get('validation_status', 'pending')),
        )
    for item in rejected_skills or []:
        store.upsert_annotation(
            offer_row_id=offer.offer_row_id,
            offer_id=offer_id,
            content_version=offer.content_version,
            canonical_name=str(item['canonical_name']),
            surface_form=str(item.get('surface_form', item['canonical_name'])),
            normalized_name=str(item['canonical_name']),
            label='SKILL',
            start=int(item['start']) if item.get('start') is not None else None,
            end=int(item['end']) if item.get('end') is not None else None,
            confidence=float(item.get('confidence', 0.1)),
            source=str(item.get('source', 'text_explicit')),
            provenance=str(item.get('provenance', 'semantic_match')),
            is_explicit=bool(item.get('is_explicit', True)),
            text_sentence=str(item.get('text_sentence', description)),
            validation_status='rejected',
            rejected_reason=str(item.get('rejected_reason', 'absent')),
        )
    for item in corrected_skills or []:
        store.upsert_annotation(
            offer_row_id=offer.offer_row_id,
            offer_id=offer_id,
            content_version=offer.content_version,
            canonical_name=str(item['canonical_name']),
            surface_form=str(item.get('surface_form', item['canonical_name'])),
            normalized_name=str(item['canonical_name']),
            label='SKILL',
            start=int(item['start']) if item.get('start') is not None else None,
            end=int(item['end']) if item.get('end') is not None else None,
            confidence=float(item.get('confidence', 0.8)),
            source=str(item.get('source', 'text_explicit')),
            provenance=str(item.get('provenance', 'exact_reference_match')),
            is_explicit=bool(item.get('is_explicit', True)),
            text_sentence=str(item.get('text_sentence', description)),
            validation_status='corrected',
            correction={
                'corrected_name': str(item.get('corrected_name', item['canonical_name'])),
                'corrected_surface': str(item.get('corrected_surface', item.get('surface_form', item['canonical_name']))),
            },
        )
    for item in human_additions or []:
        store.upsert_annotation(
            offer_row_id=offer.offer_row_id,
            offer_id=offer_id,
            content_version=offer.content_version,
            canonical_name=str(item['canonical_name']),
            surface_form=str(item.get('surface_form', item['canonical_name'])),
            normalized_name=str(item['canonical_name']),
            label='SKILL',
            start=int(item['start']) if item.get('start') is not None else None,
            end=int(item['end']) if item.get('end') is not None else None,
            confidence=float(item.get('confidence', 1.0)),
            source='human_review',
            provenance='human_review',
            is_explicit=bool(item.get('is_explicit', True)),
            text_sentence=str(item.get('text_sentence', description)),
            validation_status='approved',
            validated_at='2026-07-03T00:00:00+00:00',
            validated_by='admin',
        )
    for label, confidence in model_scores or []:
        store.upsert_annotation(
            offer_row_id=offer.offer_row_id,
            offer_id=offer_id,
            content_version=offer.content_version,
            canonical_name=label,
            surface_form=label,
            normalized_name=label,
            label='SKILL',
            start=None,
            end=None,
            confidence=confidence,
            source='camembert_multilabel',
            provenance='model_prediction',
            is_explicit=False,
            text_sentence=None,
            validation_status='pending',
        )
    return offer.offer_row_id


def _build_admin_app(monkeypatch, tmp_path, predictor=None):
    import web_app as web_app_module

    store = ContinualLearningStore(tmp_path / 'continual_learning.sqlite3')
    monkeypatch.setenv('DEEPFORMA_ADMIN_USER', 'anton')
    monkeypatch.setenv('DEEPFORMA_ADMIN_PASSWORD', 'deepforma')
    monkeypatch.setattr(web_app_module, 'ContinualLearningStore', lambda *args, **kwargs: store)
    app = build_app(
        predictor=predictor or DummyPredictor(discriminating=True),
        client_factory=lambda: DummyOfferClient(offers=[]),
    )
    app.testing = True
    return app, store


def test_admin_continual_learning_separates_text_ft_and_model_categories(monkeypatch, tmp_path):
    app, store = _build_admin_app(monkeypatch, tmp_path, predictor=DummyPredictor(discriminating=False))
    offer_row_id = _seed_continual_learning_offer(
        store,
        offer_id='offer-bid-1',
        title='Bid Manager (H/F)',
        description="Gestion des appels d'offres, rédaction de propositions commerciales, coordination d'équipes et négociation en anglais professionnel.",
        model_scores=[
            ('Big Data', 0.48),
            ("Python pour l'IA", 0.48),
            ('Data Science', 0.49),
            ('No-code / Low-code', 0.49),
            ('RAG', 0.49),
        ],
        ft_skills=[
            {'canonical_name': 'Gestion de projet', 'surface_form': 'gestion de projet', 'referential_code': 'FT-1', 'referential_label': 'Gestion de projet', 'confidence': 0.94},
        ],
        text_skills=[
            {
                'canonical_name': "Gestion des appels d'offres",
                'surface_form': "Gestion des appels d'offres",
                'start': 0,
                'end': 27,
                'confidence': 0.96,
                'provenance': 'exact_reference_match',
                'source': 'text_explicit',
                'text_sentence': "Gestion des appels d'offres et rédaction de propositions commerciales.",
                'validation_status': 'approved',
            },
            {
                'canonical_name': 'Rédaction de propositions commerciales',
                'surface_form': 'rédaction de propositions commerciales',
                'start': 29,
                'end': 69,
                'confidence': 0.95,
                'provenance': 'semantic_match',
                'source': 'text_explicit',
                'text_sentence': "Gestion des appels d'offres et rédaction de propositions commerciales.",
                'validation_status': 'pending',
            },
        ],
        rejected_skills=[
            {
                'canonical_name': 'Marketing',
                'surface_form': 'marketing',
                'start': 71,
                'end': 80,
                'rejected_reason': 'Absente du texte.',
            },
        ],
        corrected_skills=[
            {
                'canonical_name': 'Communication',
                'surface_form': 'communication client',
                'start': 82,
                'end': 102,
                'corrected_name': 'Communication',
                'corrected_surface': 'communication client',
            },
        ],
        human_additions=[
            {
                'canonical_name': 'Anglais professionnel',
                'surface_form': 'anglais professionnel',
                'start': 104,
                'end': 125,
                'text_sentence': 'Anglais professionnel pour les échanges clients.',
            },
        ],
    )

    client = app.test_client()
    response = client.get(
        f'/admin/continual-learning?offer_row_id={offer_row_id}',
        headers=_admin_auth_headers(),
    )
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert 'A. Offre' in html
    assert 'B. Compétences France Travail' in html
    assert 'C. Compétences trouvées dans le texte' in html
    assert 'D. Catégories générales suggérées par le modèle' in html
    assert 'E. Compétences rejetées' in html
    assert 'F. Compétences ajoutées manuellement' in html
    assert 'G. Décision globale' in html
    assert 'Classifieur non fiable' in html
    assert 'Catégorie IA' in html
    assert "Gestion des appels d&#39;offres" in html
    assert 'Rédaction de propositions commerciales' in html
    assert 'Gestion de projet' in html
    assert 'Aucune compétence France Travail enregistrée' not in html
    assert 'model-category-actions' not in html
    assert 'data-validate-button' in html and 'disabled' in html
    assert 'Prédictions ignorées' in html
    assert 'Prédiction du modèle' not in html


def test_admin_continual_learning_requires_explicit_confirmation(monkeypatch, tmp_path):
    app, store = _build_admin_app(monkeypatch, tmp_path)
    offer_row_id = _seed_continual_learning_offer(
        store,
        offer_id='offer-bid-2',
        title='Bid Manager (H/F)',
        description="Rédaction de propositions commerciales et gestion de projet.",
        model_scores=[('Data Science', 0.49), ('RAG', 0.48)],
        text_skills=[
            {
                'canonical_name': 'Rédaction de propositions commerciales',
                'surface_form': 'rédaction de propositions commerciales',
                'start': 0,
                'end': 38,
                'confidence': 0.95,
                'provenance': 'exact_reference_match',
                'source': 'text_explicit',
                'validation_status': 'pending',
            },
        ],
    )
    client = app.test_client()
    csrf_token = _admin_csrf_token(client, offer_row_id)
    with pytest.raises(ValueError, match='validation explicite'):
        client.post(
            '/admin/continual-learning/action',
            data={
                'offer_row_id': offer_row_id,
                'action': 'mark_offer_approved',
                'csrf_token': csrf_token,
            },
            headers=_admin_auth_headers(),
        )
    response = client.post(
        '/admin/continual-learning/action',
        data={
            'offer_row_id': offer_row_id,
            'action': 'mark_offer_approved',
            'csrf_token': csrf_token,
            'confirm_pending': '1',
        },
        headers=_admin_auth_headers(),
        follow_redirects=False,
    )
    assert response.status_code == 302
    assert store.get_offer(offer_row_id)['validation_status'] == 'approved'


def test_admin_continual_learning_actions_update_annotation_statuses(monkeypatch, tmp_path):
    app, store = _build_admin_app(monkeypatch, tmp_path)
    offer_row_id = _seed_continual_learning_offer(
        store,
        offer_id='offer-bid-3',
        title='Bid Manager (H/F)',
        description="Gestion de projet, communication et négociation.",
        model_scores=[('Big Data', 0.48), ('RAG', 0.49)],
        text_skills=[
            {
                'canonical_name': 'Gestion de projet',
                'surface_form': 'gestion de projet',
                'start': 0,
                'end': 16,
                'confidence': 0.96,
                'provenance': 'exact_reference_match',
                'source': 'text_explicit',
                'validation_status': 'pending',
            },
            {
                'canonical_name': 'Négociation',
                'surface_form': 'négociation',
                'start': 18,
                'end': 29,
                'confidence': 0.91,
                'provenance': 'semantic_match',
                'source': 'text_explicit',
                'validation_status': 'pending',
            },
        ],
    )
    text_annotations = [
        row for row in store.list_annotations('offer_row_id = ?', (offer_row_id,))
        if row['provenance'] != 'model_prediction'
    ]
    text_annotations.sort(key=lambda row: row['id'])
    first_annotation_id = text_annotations[0]['id']
    second_annotation_id = text_annotations[1]['id']

    client = app.test_client()
    csrf_token = _admin_csrf_token(client, offer_row_id)
    response = client.post(
        '/admin/continual-learning/action',
        data={
            'offer_row_id': offer_row_id,
            'annotation_id': first_annotation_id,
            'action': 'approve_annotation',
            'csrf_token': csrf_token,
        },
        headers=_admin_auth_headers(),
        follow_redirects=False,
    )
    assert response.status_code == 302
    assert store.get_annotation(first_annotation_id)['validation_status'] == 'approved'

    response = client.post(
        '/admin/continual-learning/action',
        data={
            'offer_row_id': offer_row_id,
            'annotation_id': second_annotation_id,
            'action': 'reject_annotation',
            'csrf_token': csrf_token,
            'note': 'Absente du texte',
        },
        headers=_admin_auth_headers(),
        follow_redirects=False,
    )
    assert response.status_code == 302
    assert store.get_annotation(second_annotation_id)['validation_status'] == 'rejected'

    corrected_id = _seed_continual_learning_offer(
        store,
        offer_id='offer-bid-4',
        title='Bid Manager (H/F)',
        description='Relation client.',
        model_scores=[('Data Science', 0.48)],
        corrected_skills=[
            {
                'canonical_name': 'Relation client',
                'surface_form': 'relation client',
                'start': 0,
                'end': 15,
                'corrected_name': 'Relation client',
                'corrected_surface': 'relation client',
            },
        ],
    )
    corrected_annotation = [
        row for row in store.list_annotations('offer_row_id = ?', (corrected_id,))
        if row['validation_status'] == 'corrected'
    ][0]
    response = client.post(
        '/admin/continual-learning/action',
        data={
            'offer_row_id': corrected_id,
            'annotation_id': corrected_annotation['id'],
            'action': 'correct_annotation',
            'csrf_token': csrf_token,
            'corrected_name': 'Relation client',
            'corrected_surface': 'relation client',
            'evidence': 'Formulation normalisée',
        },
        headers=_admin_auth_headers(),
        follow_redirects=False,
    )
    assert response.status_code == 302
    updated = store.get_annotation(corrected_annotation['id'])
    assert updated['validation_status'] == 'corrected'
    assert 'Relation client' in updated['canonical_name']


def test_admin_continual_learning_action_payload_remains_small(monkeypatch, tmp_path):
    app, store = _build_admin_app(monkeypatch, tmp_path)
    offer_row_id = _seed_continual_learning_offer(
        store,
        offer_id='offer-bid-6',
        title='Bid Manager (H/F)',
        description='Gestion de projet et communication.',
        model_scores=[('Python', 0.91)],
        text_skills=[
            {
                'canonical_name': 'Gestion de projet',
                'surface_form': 'gestion de projet',
                'start': 0,
                'end': 16,
                'confidence': 0.96,
                'provenance': 'exact_reference_match',
                'source': 'text_explicit',
                'validation_status': 'pending',
            },
        ],
    )
    client = app.test_client()
    csrf_token = _admin_csrf_token(client, offer_row_id)
    text_annotations = [
        row for row in store.list_annotations('offer_row_id = ?', (offer_row_id,))
        if row['provenance'] != 'model_prediction'
    ]
    payload = {
        'offer_row_id': offer_row_id,
        'annotation_id': text_annotations[0]['id'],
        'action': 'approve_annotation',
        'csrf_token': csrf_token,
    }
    assert len(urlencode(payload)) < 512
    response = client.post(
        '/admin/continual-learning/action',
        data=payload,
        headers=_admin_auth_headers(),
        follow_redirects=False,
    )
    assert response.status_code == 302


def test_admin_continual_learning_returns_readable_413(monkeypatch, tmp_path):
    app, store = _build_admin_app(monkeypatch, tmp_path)
    app.config['MAX_FORM_MEMORY_SIZE'] = 64
    app.config['MAX_CONTENT_LENGTH'] = 128
    offer_row_id = _seed_continual_learning_offer(
        store,
        offer_id='offer-bid-7',
        title='Bid Manager (H/F)',
        description='Gestion de projet.',
        model_scores=[('Python', 0.91)],
        text_skills=[
            {
                'canonical_name': 'Gestion de projet',
                'surface_form': 'gestion de projet',
                'start': 0,
                'end': 16,
                'confidence': 0.96,
                'provenance': 'exact_reference_match',
                'source': 'text_explicit',
                'validation_status': 'pending',
            },
        ],
    )
    client = app.test_client()
    csrf_token = _admin_csrf_token(client, offer_row_id)
    annotations = [row for row in store.list_annotations('offer_row_id = ?', (offer_row_id,)) if row['provenance'] != 'model_prediction']
    response = client.post(
        '/admin/continual-learning/action',
        data={
            'offer_row_id': offer_row_id,
            'annotation_id': annotations[0]['id'],
            'action': 'correct_annotation',
            'csrf_token': csrf_token,
            'corrected_name': 'X' * 5000,
            'corrected_surface': 'Y',
            'evidence': 'Z',
        },
        headers=_admin_auth_headers(),
    )
    assert response.status_code == 413
    body = response.get_data(as_text=True)
    assert 'trop volumineuse' in body


def test_admin_continual_learning_reliable_model_categories_and_empty_sections(monkeypatch, tmp_path):
    app, store = _build_admin_app(monkeypatch, tmp_path, predictor=DummyPredictor(discriminating=True))
    offer_row_id = _seed_continual_learning_offer(
        store,
        offer_id='offer-bid-5',
        title='Bid Manager (H/F)',
        description='Gestion de projet et coordination.',
        model_scores=[('Python', 0.91), ('Machine Learning', 0.72), ('Deep Learning', 0.48)],
        ft_skills=[],
        text_skills=[],
    )
    client = app.test_client()
    response = client.get(
        f'/admin/continual-learning?offer_row_id={offer_row_id}',
        headers=_admin_auth_headers(),
    )
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert 'Oui' in html
    assert 'model-category-actions' in html
    assert 'Aucune compétence France Travail enregistrée.' in html
    assert "Aucune compétence n'a été extraite du texte pour cette offre." in html
