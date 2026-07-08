from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any
from io import BytesIO

from referential_import.import_service import build_export_payload
from referential_import.models import DerivedSkill, ImportReport, OfficialCompetency, ReferentialActivity, ReferentialBlock, ReferentialDocument
from referentials.rome_referential import RomeJob
from web_app import create_app


@dataclass
class DummyRomeStats:
    rome_code: str
    rome_label: str
    raw_count: int
    accepted_count: int
    rejected_count: int
    pages_count: int
    error: str | None = None


@dataclass
class DummyMultiRomeResult:
    requested_rome_codes: list[str]
    offers: list[dict[str, Any]]
    stats_by_rome: list[DummyRomeStats]
    rejected_offers: list[Any]
    warnings: list[str]


@dataclass
class DummyPredictor:
    def analyze(self, text, threshold=None):
        return {
            'binary': {
                'is_ia': True,
                'predicted_class': 1,
                'probability_non_ia': 0.25,
                'probability_ia': 0.75,
            },
            'skills': {
                'predictions': [
                    {'label': 'Python', 'probability': 0.91, 'threshold': threshold or 0.35},
                ],
                'all_scores': [0.91],
                'score_min': 0.91,
                'score_max': 0.91,
                'score_mean': 0.91,
                'score_std': 0.0,
                'inference_time_ms': 42.0,
                'num_labels': 1,
                'threshold_applied': threshold or 0.35,
            },
            'device': 'cpu',
            'inference_time_ms': 42.0,
            'checkpoint_audit': {
                'config_present': True,
                'weights_present': True,
                'weights_size_bytes': 1000000,
                'architecture_declared': 'CamembertForSequenceClassification',
                'num_labels_declared': 1,
                'num_labels_effective': 1,
                'problem_type': 'multi_label_classification',
                'id2label_count': 1,
                'label2id_count': 1,
                'appears_random_init': False,
                'body_params_match_base': True,
                'parameter_errors': [],
                'classifier_params': {},
            },
        }


class DummyOfferClient:
    def iter_offers(self, *args, **kwargs):
        return iter(())


def _build_analysis():
    document = ReferentialDocument(
        id='doc-rome',
        source_path='/tmp/referentiel.pdf',
        file_name='referentiel.pdf',
        sha256='abc123',
        page_count=1,
        collected_at='2026-07-03T00:00:00+00:00',
        text_extraction_method='pdftotext-layout',
        title='Ingénieur en intelligence artificielle',
    )
    competency = OfficialCompetency(
        code='A1-C1',
        official_label='Organiser la coordination',
        normalized_label='organiser la coordination',
        block_code='B1',
        activity_code='A1',
        page_start=1,
        page_end=1,
        confidence=0.95,
        source_pages=[1],
    )
    activity = ReferentialActivity(
        code='A1',
        block_code='B1',
        label='Coordonner un projet',
        page_start=1,
        page_end=1,
        confidence=0.95,
        source_pages=[1],
    )
    block = ReferentialBlock(
        code='B1',
        label='Pilotage',
        page_start=1,
        page_end=1,
        confidence=0.95,
        source_pages=[1],
    )
    derived = DerivedSkill(
        label='Coordination',
        canonical_label='Coordination',
        category='action',
        source_code='A1-C1',
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
            document_id='doc-rome',
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
        'blocks': [block],
        'activities': [activity],
        'competencies': [competency],
        'criteria': [],
        'derived_skills': [derived],
        'tools_methods': [derived],
    }
    return analysis


def build_app(monkeypatch):
    app = create_app(
        predictor=DummyPredictor(),
        france_travail_client_factory=lambda: DummyOfferClient(),
        cache_ttl_seconds=60,
    )
    monkeypatch.setattr('web_app.RomeService.search', lambda self, query, limit=10: [RomeJob('M1805', 'Data Scientist', 'Concevoir des solutions data.', ['Machine Learning Engineer'], [], [], 'Data'), RomeJob('M1802', 'Expertise et support en systèmes d’information', 'Support SI.', ['Support informatique'], [], [], 'IT')])
    monkeypatch.setattr('web_app.RomeService.get', lambda self, code: RomeJob(code, 'Data Scientist' if code == 'M1805' else 'Expertise et support en systèmes d’information', 'Concevoir des solutions data.', ['Machine Learning Engineer'], [], [], 'Data'))

    def fake_fetch_offers_by_rome_codes(rome_codes, territory, **kwargs):
        offers: list[dict[str, Any]] = []
        stats: list[DummyRomeStats] = []
        for code in rome_codes:
            if code == 'M1805':
                offers.append({
                    'offer_id': 'offer-1',
                    'title': 'Data Scientist',
                    'description': 'Travail sur la coordination des données',
                    'normalized_skills': ['Organiser la coordination'],
                    'structured_skills': [{'canonical_label': 'Organiser la coordination'}],
                    'contract_type': 'CDI',
                    'location_label': 'Paris',
                    'department_code': '75',
                    'offer_url': 'https://example.com/offer-1',
                    'rome_code': 'M1805',
                    'rome_label': 'Data Scientist',
                    'matched_requested_rome_codes': ['M1805'],
                })
                stats.append(DummyRomeStats('M1805', 'Data Scientist', 1, 1, 0, 1, None))
            elif code == 'M1802':
                offers.append({
                    'offer_id': 'offer-2',
                    'title': 'Support SI',
                    'description': 'Maintenance et support.',
                    'normalized_skills': ['Support informatique'],
                    'structured_skills': [{'canonical_label': 'Support informatique'}],
                    'contract_type': 'CDI',
                    'location_label': 'Paris',
                    'department_code': '75',
                    'offer_url': 'https://example.com/offer-2',
                    'rome_code': 'M1802',
                    'rome_label': 'Expertise et support en systèmes d’information',
                    'matched_requested_rome_codes': ['M1802'],
                })
                stats.append(DummyRomeStats('M1802', 'Expertise et support en systèmes d’information', 1, 1, 0, 1, None))
            else:
                stats.append(DummyRomeStats(code, code, 0, 0, 0, 0, 'Aucune offre'))
        return DummyMultiRomeResult(list(rome_codes), offers, stats, [], [])

    monkeypatch.setattr('web_app.fetch_offers_by_rome_codes', fake_fetch_offers_by_rome_codes)
    return app


def test_pdf_analysis_then_rome_confirmation_and_market_search(monkeypatch):
    import web_app as web_app_module
    from referential_import.import_service import ReferentialImportService

    analysis = _build_analysis()
    export_json = json.dumps(build_export_payload(analysis), ensure_ascii=False, indent=2)

    class DummyPage:
        def __init__(self, text):
            self.text = text

    class DummyPdf:
        def __init__(self, pages):
            self.pages = pages

    monkeypatch.setattr(ReferentialImportService, 'analyze', lambda self, input_path: analysis)
    monkeypatch.setattr(web_app_module, 'load_pdf_document', lambda path: DummyPdf([DummyPage('Ingénieur en intelligence artificielle\nRéférentiel de compétences')]))

    app = build_app(monkeypatch)
    client = app.test_client()

    preview_response = client.post(
        '/referential/import',
        data={
            'pdf': (BytesIO(b'%PDF-1.4 fake'), 'referentiel.pdf'),
            'departement': '75',
        },
        content_type='multipart/form-data',
    )
    assert preview_response.status_code == 302
    assert preview_response.headers.get('Location', '').startswith('/referential/import/')
    preview_follow = client.get(preview_response.headers['Location'])
    assert preview_follow.status_code == 200
    preview_html = preview_follow.get_data(as_text=True)
    assert 'cible' in preview_html
    match = re.search(r'name="analysis_id" value="([^"]+)"', preview_html)
    assert match, preview_html
    analysis_id = match.group(1)

    search_response = client.post(
        '/analyze',
        data={
            'action': 'search_rome_candidates',
            'analysis_id': analysis_id,
            'analysis_json': export_json,
            'source_path': '/tmp/referentiel.pdf',
            'departement': '75',
            'rome_query': 'Data Scientist',
        },
    )
    assert search_response.status_code == 200
    search_html = search_response.get_data(as_text=True)
    assert 'M1805' in search_html
    assert 'Data Scientist' in search_html

    confirm_response = client.post(
        '/analyze',
        data={
            'action': 'add_market_target',
            'analysis_id': analysis_id,
            'analysis_json': export_json,
            'source_path': '/tmp/referentiel.pdf',
            'departement': '75',
            'rome_code': 'M1805',
            'rome_label': 'Data Scientist',
            'territory_code': '75',
            'territory_label': 'Paris',
            'radius_km': '20',
        },
    )
    assert confirm_response.status_code == 200
    confirm_html = confirm_response.get_data(as_text=True)
    assert 'M1805' in confirm_html
    assert 'Paris' in confirm_html

    second_confirm_response = client.post(
        '/analyze',
        data={
            'action': 'add_market_target',
            'analysis_id': analysis_id,
            'analysis_json': export_json,
            'source_path': '/tmp/referentiel.pdf',
            'departement': '75',
            'rome_code': 'M1802',
            'rome_label': 'Expertise et support en systèmes d’information',
            'territory_code': '75',
            'territory_label': 'Paris',
            'radius_km': '20',
        },
    )
    assert second_confirm_response.status_code == 200
    second_confirm_html = second_confirm_response.get_data(as_text=True)
    assert 'M1805' in second_confirm_html and 'M1802' in second_confirm_html

    result_response = client.post(
        '/analyze',
        data={
            'action': 'run_market_search',
            'analysis_id': analysis_id,
            'analysis_json': export_json,
            'source_path': '/tmp/referentiel.pdf',
            'departement': '75',
            'territory_code': '75',
            'territory_label': 'Paris',
            'radius_km': '20',
            'contract_type': 'CDI',
        },
    )
    assert result_response.status_code == 200
    result_html = result_response.get_data(as_text=True)
    assert 'Codes ROME analysés' in result_html
    assert 'M1805' in result_html
    assert 'M1802' in result_html
    assert "Offres utilisées pour l'étude de marché" in result_html
    assert 'Organiser la coordination' in result_html
    assert 'Support informatique' in result_html
    assert 'Offres rejetées' in result_html or 'rejected_count' in result_html
