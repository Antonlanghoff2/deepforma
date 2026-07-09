from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from services.certification_market_comparison import CertificationMarketComparator
from web_app import create_app



class DummyPredictor:
    def analyze(self, text, threshold=None):
        return {
            'binary': {'is_ia': False, 'predicted_class': 0, 'probability_non_ia': 1.0, 'probability_ia': 0.0},
            'skills': {'predictions': [], 'all_scores': [], 'score_min': 0.0, 'score_max': 0.0, 'score_mean': 0.0, 'score_std': 0.0, 'inference_time_ms': 0.0, 'num_labels': 0, 'threshold_applied': threshold or 0.35},
            'device': 'cpu',
            'inference_time_ms': 0.0,
            'checkpoint_audit': {},
        }

class DummyOfferClient:
    def __init__(self, offers=None):
        self.offers = offers or []

    def iter_offers(self, *args, **kwargs):
        yield from self.offers


def _make_referential(path: Path) -> Path:
    payload = {
        'referential_id': 'ingenieur_ia_2025',
        'title': 'Ingénieur en intelligence artificielle',
        'version': '2025-01',
        'skills': [
            {
                'id': 'B1-A1-C1',
                'block': 'B1',
                'block_name': 'Bloc 1',
                'activity': 'A1',
                'code': 'A1-C1',
                'label': 'Machine Learning',
                'official_description': 'Apprentissage automatique.',
                'normalized_label': 'machine learning',
                'category': 'Machine Learning',
                'subcategory': 'Classification',
                'technical_keywords': ['Machine Learning'],
                'origin_document': 'sample.pdf',
                'aliases': ['apprentissage automatique'],
                'source_page': 2,
                'active': True,
            },
        ],
        'metadata': {'source_pdf': 'sample.pdf'},
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return path


def _auth_headers() -> dict[str, str]:
    auth = base64.b64encode(b'anton:deepforma').decode('ascii')
    return {'Authorization': f'Basic {auth}'}


def _mock_options(referential_path: Path) -> list:
    from referentials.referential_registry import ReferentialOption
    return [
        ReferentialOption(
            id='ingenieur_ia_2025',
            label='Ingénieur en intelligence artificielle',
            type='certification',
            path=str(referential_path),
            record_id=None,
            status='active',
            source='json_file',
            skill_count=1,
        )
    ]


def _setup_app(monkeypatch, *, referential_path: Path, offers: list | None = None):
    monkeypatch.setenv('DEEPFORMA_ADMIN_USER', 'anton')
    monkeypatch.setenv('DEEPFORMA_ADMIN_PASSWORD', 'deepforma')
    app = create_app(
        predictor=DummyPredictor(),
        france_travail_client_factory=lambda: DummyOfferClient(offers or []),
        cache_ttl_seconds=60,
    )
    return app


def test_admin_ai_certification_market_comparison_route(tmp_path, monkeypatch):
    referential_path = _make_referential(tmp_path / 'referential.json')
    mock_opts = _mock_options(referential_path)
    monkeypatch.setattr('web_app.list_available_referentials', lambda: mock_opts)
    monkeypatch.setattr('web_app.get_referential_option', lambda rid: next((o for o in mock_opts if o.id == rid), None))
    app = _setup_app(monkeypatch, referential_path=referential_path, offers=[
        {
            'offer_id': 'offer-1',
            'title': 'Assistant administratif',
            'description': 'Bac+5, personne dynamique, télétravail deux jours par semaine.',
            'creation_date': '2026-07-01T00:00:00+00:00',
            'location_label': 'Paris',
            'contract_label': 'CDI',
        }
    ])
    monkeypatch.setattr('web_app.write_comparison_outputs', lambda report, output_dir: {'json': tmp_path / 'report.json', 'validation_csv': tmp_path / 'validation.csv', 'gaps_csv': tmp_path / 'gaps.csv'})
    client = app.test_client()
    response = client.post(
        '/admin/ai-certification-market-comparison',
        data={
            'referential_id': 'ingenieur_ia_2025',
            'territory': '75056',
            'commune': '75056',
            'departement': '75',
            'job_titles': 'Data Scientist',
            'rome_codes': 'M1805',
            'max_pages': '1',
            'max_offers': '10',
            'page_size': '10',
        },
        headers=_auth_headers(),
    )
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert 'Comparaison du référentiel IA avec le marché' in html
    assert 'Score global' in html
    assert 'Machine Learning' in html


def test_get_affiche_liste_referentiels(tmp_path, monkeypatch):
    referential_path = _make_referential(tmp_path / 'referential.json')
    mock_opts = _mock_options(referential_path)
    monkeypatch.setattr('web_app.list_available_referentials', lambda: mock_opts)
    monkeypatch.setattr('web_app.get_referential_option', lambda rid: next((o for o in mock_opts if o.id == rid), None))
    app = _setup_app(monkeypatch, referential_path=referential_path)
    client = app.test_client()
    response = client.get('/admin/ai-certification-market-comparison', headers=_auth_headers())
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert 'Référentiel à comparer' in html
    assert 'Ingénieur en intelligence artificielle' in html
    assert '1 compétences' in html
    assert '<select' in html


def test_get_sans_referentiel_affiche_message(tmp_path, monkeypatch):
    monkeypatch.setattr('web_app.list_available_referentials', lambda: [])
    app = _setup_app(monkeypatch, referential_path=tmp_path / 'dummy.json')
    client = app.test_client()
    response = client.get('/admin/ai-certification-market-comparison', headers=_auth_headers())
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert 'Aucun référentiel disponible' in html
    assert 'disabled' in html


def test_post_sans_referential_id_retourne_erreur(tmp_path, monkeypatch):
    referential_path = _make_referential(tmp_path / 'referential.json')
    mock_opts = _mock_options(referential_path)
    monkeypatch.setattr('web_app.list_available_referentials', lambda: mock_opts)
    app = _setup_app(monkeypatch, referential_path=referential_path)
    client = app.test_client()
    response = client.post(
        '/admin/ai-certification-market-comparison',
        data={
            'referential_id': '',
            'territory': '75056',
        },
        headers=_auth_headers(),
    )
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert 'Veuillez sélectionner un référentiel' in html
    assert 'Résumé' not in html


def test_post_referential_id_inconnu_retourne_erreur(tmp_path, monkeypatch):
    referential_path = _make_referential(tmp_path / 'referential.json')
    mock_opts = _mock_options(referential_path)
    monkeypatch.setattr('web_app.list_available_referentials', lambda: mock_opts)
    monkeypatch.setattr('web_app.get_referential_option', lambda rid: None)
    app = _setup_app(monkeypatch, referential_path=referential_path)
    client = app.test_client()
    response = client.post(
        '/admin/ai-certification-market-comparison',
        data={
            'referential_id': 'unknown_id',
            'territory': '75056',
        },
        headers=_auth_headers(),
    )
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert 'introuvable' in html
    assert 'Résumé' not in html


def test_post_referentiel_fichier_absent_pas_500(tmp_path, monkeypatch):
    from referentials.referential_registry import ReferentialOption
    missing_path = tmp_path / 'nonexistent.json'
    mock_opts = [
        ReferentialOption(
            id='missing_ref',
            label='Référentiel manquant',
            type='certification',
            path=str(missing_path),
            record_id=None,
            status='active',
            source='json_file',
            skill_count=1,
        )
    ]
    monkeypatch.setattr('web_app.list_available_referentials', lambda: mock_opts)
    monkeypatch.setattr('web_app.get_referential_option', lambda rid: next((o for o in mock_opts if o.id == rid), None))
    app = _setup_app(monkeypatch, referential_path=tmp_path / 'dummy.json')
    client = app.test_client()
    response = client.post(
        '/admin/ai-certification-market-comparison',
        data={
            'referential_id': 'missing_ref',
            'territory': '75056',
        },
        headers=_auth_headers(),
    )
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert 'introuvable' in html


def test_selection_conservee_apres_erreur(tmp_path, monkeypatch):
    referential_path = _make_referential(tmp_path / 'referential.json')
    mock_opts = _mock_options(referential_path)
    monkeypatch.setattr('web_app.list_available_referentials', lambda: mock_opts)
    monkeypatch.setattr('web_app.get_referential_option', lambda rid: None)
    app = _setup_app(monkeypatch, referential_path=referential_path)
    client = app.test_client()
    response = client.post(
        '/admin/ai-certification-market-comparison',
        data={
            'referential_id': 'ingenieur_ia_2025',
            'territory': '75056',
        },
        headers=_auth_headers(),
    )
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert 'selected' in html
    referential_path = _make_referential(tmp_path / 'referential.json')
    from referentials.referential_registry import ReferentialOption
    mock_options = [
        ReferentialOption(
            id='ingenieur_ia_2025',
            label='Ingénieur en intelligence artificielle',
            type='certification',
            path=str(referential_path),
            record_id=None,
            status='active',
            source='json_file',
            skill_count=1,
        )
    ]
    monkeypatch.setattr('web_app.list_available_referentials', lambda: mock_options)
    monkeypatch.setattr('web_app.get_referential_option', lambda rid: next((o for o in mock_options if o.id == rid), None))
    monkeypatch.setenv('DEEPFORMA_ADMIN_USER', 'anton')
    monkeypatch.setenv('DEEPFORMA_ADMIN_PASSWORD', 'deepforma')
    app = create_app(
        predictor=DummyPredictor(),
        france_travail_client_factory=lambda: DummyOfferClient([
            {
                'offer_id': 'offer-1',
                'title': 'Assistant administratif',
                'description': 'Bac+5, personne dynamique, télétravail deux jours par semaine.',
                'creation_date': '2026-07-01T00:00:00+00:00',
                'location_label': 'Paris',
                'contract_label': 'CDI',
            }
        ]),
        cache_ttl_seconds=60,
    )
    monkeypatch.setattr('web_app.write_comparison_outputs', lambda report, output_dir: {'json': tmp_path / 'report.json', 'validation_csv': tmp_path / 'validation.csv', 'gaps_csv': tmp_path / 'gaps.csv'})
    client = app.test_client()
    auth = base64.b64encode(b'anton:deepforma').decode('ascii')
    response = client.post(
        '/admin/ai-certification-market-comparison',
        data={
            'referential_id': 'ingenieur_ia_2025',
            'territory': '75056',
            'commune': '75056',
            'departement': '75',
            'job_titles': 'Data Scientist',
            'rome_codes': 'M1805',
            'max_pages': '1',
            'max_offers': '10',
            'page_size': '10',
        },
        headers={'Authorization': f'Basic {auth}'},
    )
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert 'Comparaison du référentiel IA avec le marché' in html
    assert 'Score global' in html
    assert 'Machine Learning' in html
