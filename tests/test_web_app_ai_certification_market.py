from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import pytest

from referentials.referential_registry import (
    ReferentialOption,
    convert_imported_to_skills_format,
    ensure_loadable_path,
)
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
            is_selectable=True,
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
    monkeypatch.setattr('web_app.list_available_referentials', lambda **kw: mock_opts)
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
    monkeypatch.setattr('web_app.list_available_referentials', lambda **kw: mock_opts)
    monkeypatch.setattr('web_app.get_referential_option', lambda rid: next((o for o in mock_opts if o.id == rid), None))
    app = _setup_app(monkeypatch, referential_path=referential_path)
    client = app.test_client()
    response = client.get('/admin/ai-certification-market-comparison', headers=_auth_headers())
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert 'Référentiel à comparer' in html
    assert 'Ingénieur en intelligence artificielle' in html
    assert '1 compétence' in html
    assert '<select' in html


def test_get_sans_referentiel_affiche_message(tmp_path, monkeypatch):
    monkeypatch.setattr('web_app.list_available_referentials', lambda **kw: [])
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
    monkeypatch.setattr('web_app.list_available_referentials', lambda **kw: mock_opts)
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
    monkeypatch.setattr('web_app.list_available_referentials', lambda **kw: mock_opts)
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
            is_selectable=True,
        )
    ]
    monkeypatch.setattr('web_app.list_available_referentials', lambda **kw: mock_opts)
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
    monkeypatch.setattr('web_app.list_available_referentials', lambda **kw: mock_opts)
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
            is_selectable=True,
        )
    ]
    monkeypatch.setattr('web_app.list_available_referentials', lambda **kw: mock_options)
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


def _make_imported_referential(path: Path) -> Path:
    payload = {
        'referential_id': 'manager_affaires',
        'title': 'Manager affaires',
        'schema_version': '1.0',
        'document': {
            'file_name': 'tmpgaqzgv7s.pdf',
            'source_path': '/tmp/sample.pdf',
            'sha256': 'abc123',
            'title': 'Manager affaires',
        },
        'blocks': [
            {'code': 'BLOC_1', 'label': 'Stratégie commerciale'},
            {'code': 'BLOC_2', 'label': 'Pilotage d\'activité'},
        ],
        'competencies': [
            {
                'code': 'C1.1',
                'official_label': 'Réaliser une étude de marché',
                'normalized_label': 'réaliser une étude de marché',
                'block_code': 'BLOC_1',
                'activity_code': 'A1',
                'source_pages': [5],
                'derived_skills': ['étude de marché', 'veille'],
            },
            {
                'code': 'C2.1',
                'official_label': 'Piloter un budget',
                'normalized_label': 'piloter un budget',
                'block_code': 'BLOC_2',
                'activity_code': 'A2',
                'source_pages': [10],
                'derived_skills': ['budget', 'pilotage'],
            },
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return path


def test_convert_imported_to_skills_format(tmp_path):
    imported_path = _make_imported_referential(tmp_path / 'imported.json')
    payload = json.loads(imported_path.read_text(encoding='utf-8'))
    result = convert_imported_to_skills_format(payload)
    assert result['referential_id'] == 'manager_affaires'
    assert result['title'] == 'Manager affaires'
    assert len(result['skills']) == 2
    skill = result['skills'][0]
    assert skill['id'] == 'C1.1'
    assert skill['code'] == 'C1.1'
    assert skill['label'] == 'Réaliser une étude de marché'
    assert skill['block'] == 'BLOC_1'
    assert skill['block_name'] == 'Stratégie commerciale'
    assert skill['technical_keywords'] == ['étude de marché', 'veille']
    assert skill['source_page'] == 5
    assert skill['active'] is True
    assert result['metadata']['source'] == 'imported_pdf'


def test_ensure_loadable_path_standard_format(tmp_path):
    path = _make_referential(tmp_path / 'standard.json')
    option = ReferentialOption(
        id='ingenieur_ia_2025',
        label='Ingénieur IA',
        type='certification',
        path=str(path),
        record_id=None,
        status='active',
        source='json_file',
        skill_count=1,
        is_selectable=True,
    )
    result = ensure_loadable_path(option)
    assert result == str(path)


def test_ensure_loadable_path_imported_format(tmp_path):
    imported_path = _make_imported_referential(tmp_path / 'imported.json')
    option = ReferentialOption(
        id='manager_affaires',
        label='Manager affaires',
        type='certification',
        path=str(imported_path),
        record_id=None,
        status='active',
        source='imported_pdf',
        skill_count=2,
        is_selectable=True,
    )
    result = ensure_loadable_path(option)
    assert result is not None
    converted_path = Path(result)
    assert converted_path.exists()
    assert converted_path.suffix == '.json'
    assert '.converted.' in converted_path.name
    payload = json.loads(converted_path.read_text(encoding='utf-8'))
    assert 'skills' in payload
    assert len(payload['skills']) == 2


def test_ensure_loadable_path_missing_file_returns_none(tmp_path):
    missing = tmp_path / 'nonexistent.json'
    option = ReferentialOption(
        id='missing',
        label='Missing',
        type='certification',
        path=str(missing),
        record_id=None,
        status='active',
        source='json_file',
        skill_count=1,
        is_selectable=True,
    )
    assert ensure_loadable_path(option) is None


def test_ensure_loadable_path_empty_competencies_returns_none(tmp_path):
    path = tmp_path / 'empty.json'
    path.write_text(json.dumps({'competencies': []}), encoding='utf-8')
    option = ReferentialOption(
        id='empty',
        label='Empty',
        type='certification',
        path=str(path),
        record_id=None,
        status='active',
        source='imported_pdf',
        skill_count=0,
        is_selectable=True,
    )
    assert ensure_loadable_path(option) is None


def test_liste_inclut_referentiels_importes(tmp_path, monkeypatch):
    from referentials.referential_registry import list_available_referentials
    _make_referential(tmp_path / 'standard.json')
    imported = _make_imported_referential(tmp_path / 'imported.json')
    imported_dir = tmp_path / 'imported'
    imported_dir.mkdir()
    (imported_dir / 'manager.json').write_text(imported.read_text())
    options = list_available_referentials(referentials_dir=tmp_path)
    ids = [o.id for o in options]
    assert 'ingenieur_ia_2025' in ids
    assert 'manager_affaires' in ids


def test_route_importe_converti_et_comparaison_reussit(tmp_path, monkeypatch):
    imported_path = _make_imported_referential(tmp_path / 'imported.json')
    mock_opts = [
        ReferentialOption(
            id='manager_affaires',
            label='Manager affaires',
            type='certification',
            path=str(imported_path),
            record_id=None,
            status='active',
            source='imported_pdf',
            skill_count=2,
            is_selectable=True,
        )
    ]
    monkeypatch.setattr('web_app.list_available_referentials', lambda **kw: mock_opts)
    monkeypatch.setattr('web_app.get_referential_option', lambda rid: next((o for o in mock_opts if o.id == rid), None))
    monkeypatch.setenv('DEEPFORMA_ADMIN_USER', 'anton')
    monkeypatch.setenv('DEEPFORMA_ADMIN_PASSWORD', 'deepforma')
    app = create_app(
        predictor=DummyPredictor(),
        france_travail_client_factory=lambda: DummyOfferClient([
            {
                'offer_id': 'offer-1',
                'title': 'Chef de projet',
                'description': 'Pilotage de budget et étude de marché.',
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
            'referential_id': 'manager_affaires',
            'territory': '75056',
            'commune': '75056',
            'departement': '75',
            'job_titles': 'Chef de projet',
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
    assert 'Manager affaires' in html


def test_upload_referential_ok(tmp_path, monkeypatch):
    from referentials.referential_registry import DEFAULT_REFERENTIALS_DIR as _REG_DIR
    monkeypatch.setattr('web_app.DEFAULT_REFERENTIALS_DIR', tmp_path)
    monkeypatch.setattr('referentials.referential_registry.DEFAULT_REFERENTIALS_DIR', tmp_path)
    monkeypatch.setenv('DEEPFORMA_ADMIN_USER', 'anton')
    monkeypatch.setenv('DEEPFORMA_ADMIN_PASSWORD', 'deepforma')
    app = create_app(
        predictor=DummyPredictor(),
        france_travail_client_factory=lambda: DummyOfferClient([]),
        cache_ttl_seconds=60,
    )
    client = app.test_client()
    payload = json.dumps({
        'referential_id': 'test_ref',
        'title': 'Test',
        'skills': [{'id': 'S1', 'label': 'Skill 1'}],
    })
    response = client.post(
        '/admin/referential/upload',
        data={'file': (io.BytesIO(payload.encode()), 'test_ref.json')},
        headers=_auth_headers(),
    )
    assert response.status_code == 302
    assert (tmp_path / 'test_ref.json').exists()


def test_upload_referential_no_file_returns_error(tmp_path, monkeypatch):
    monkeypatch.setenv('DEEPFORMA_ADMIN_USER', 'anton')
    monkeypatch.setenv('DEEPFORMA_ADMIN_PASSWORD', 'deepforma')
    app = create_app(
        predictor=DummyPredictor(),
        france_travail_client_factory=lambda: DummyOfferClient([]),
        cache_ttl_seconds=60,
    )
    client = app.test_client()
    response = client.post('/admin/referential/upload', data={}, headers=_auth_headers())
    assert response.status_code == 302
    assert 'error=' in response.location


def test_upload_referential_invalid_json_returns_error(tmp_path, monkeypatch):
    monkeypatch.setenv('DEEPFORMA_ADMIN_USER', 'anton')
    monkeypatch.setenv('DEEPFORMA_ADMIN_PASSWORD', 'deepforma')
    app = create_app(
        predictor=DummyPredictor(),
        france_travail_client_factory=lambda: DummyOfferClient([]),
        cache_ttl_seconds=60,
    )
    client = app.test_client()
    response = client.post(
        '/admin/referential/upload',
        data={'file': (io.BytesIO(b'not json'), 'bad.json')},
        headers=_auth_headers(),
    )
    assert response.status_code == 302
    assert 'error=' in response.location


def test_upload_referential_no_skills_returns_error(tmp_path, monkeypatch):
    monkeypatch.setenv('DEEPFORMA_ADMIN_USER', 'anton')
    monkeypatch.setenv('DEEPFORMA_ADMIN_PASSWORD', 'deepforma')
    app = create_app(
        predictor=DummyPredictor(),
        france_travail_client_factory=lambda: DummyOfferClient([]),
        cache_ttl_seconds=60,
    )
    client = app.test_client()
    payload = json.dumps({'name': 'test'})
    response = client.post(
        '/admin/referential/upload',
        data={'file': (io.BytesIO(payload.encode()), 'no_skills.json')},
        headers=_auth_headers(),
    )
    assert response.status_code == 302
    assert 'error=' in response.location


def test_delete_referential_ok(tmp_path, monkeypatch):
    from referentials.referential_registry import DEFAULT_REFERENTIALS_DIR
    monkeypatch.setattr('referentials.referential_registry.DEFAULT_REFERENTIALS_DIR', tmp_path)
    monkeypatch.setenv('DEEPFORMA_ADMIN_USER', 'anton')
    monkeypatch.setenv('DEEPFORMA_ADMIN_PASSWORD', 'deepforma')
    json_path = _make_referential(tmp_path / 'to_delete.json')
    app = create_app(
        predictor=DummyPredictor(),
        france_travail_client_factory=lambda: DummyOfferClient([]),
        cache_ttl_seconds=60,
    )
    client = app.test_client()
    response = client.post(
        '/admin/referential/ingenieur_ia_2025/delete',
        headers=_auth_headers(),
    )
    assert response.status_code == 302
    assert not json_path.exists()


def test_delete_referential_unknown_returns_redirect(tmp_path, monkeypatch):
    monkeypatch.setenv('DEEPFORMA_ADMIN_USER', 'anton')
    monkeypatch.setenv('DEEPFORMA_ADMIN_PASSWORD', 'deepforma')
    app = create_app(
        predictor=DummyPredictor(),
        france_travail_client_factory=lambda: DummyOfferClient([]),
        cache_ttl_seconds=60,
    )
    client = app.test_client()
    response = client.post(
        '/admin/referential/nonexistent/delete',
        headers=_auth_headers(),
    )
    assert response.status_code == 302


# ── Tests pour la source de vérité unique ──────────────────────────────────


def test_gestion_liste_tous_les_referentiels(tmp_path, monkeypatch):
    from referentials.referential_registry import ReferentialOption
    mock_opts = [
        ReferentialOption(id='r1', label='Actif', type='certification', path=str(tmp_path / 'r1.json'), record_id=None, status='active', source='json_file', skill_count=5, is_selectable=True, reason=None),
        ReferentialOption(id='r2', label='Vide', type='certification', path=str(tmp_path / 'r2.json'), record_id=None, status='empty', source='json_file', skill_count=0, is_selectable=False, reason='Aucune compétence détectée'),
        ReferentialOption(id='r3', label='Invalide', type='certification', path=str(tmp_path / 'r3.json'), record_id=None, status='invalid', source='json_file', skill_count=0, is_selectable=False, reason='Fichier JSON invalide'),
    ]
    monkeypatch.setattr('web_app.list_available_referentials', lambda **kw: mock_opts)
    monkeypatch.setattr('web_app.get_referential_option', lambda rid: next((o for o in mock_opts if o.id == rid), None))
    monkeypatch.setenv('DEEPFORMA_ADMIN_USER', 'anton')
    monkeypatch.setenv('DEEPFORMA_ADMIN_PASSWORD', 'deepforma')
    app = create_app(predictor=DummyPredictor(), france_travail_client_factory=lambda: DummyOfferClient([]), cache_ttl_seconds=60)
    client = app.test_client()
    response = client.get('/admin/ai-certification-market-comparison', headers=_auth_headers())
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert 'Actif' in html
    assert 'Vide' in html
    assert 'Invalide' in html
    assert 'exploitable' in html
    assert 'incomplet' in html
    assert 'invalide' in html


def test_comparaison_dropdown_liste_tous_les_referentiels(tmp_path, monkeypatch):
    from referentials.referential_registry import ReferentialOption
    mock_opts = [
        ReferentialOption(id='r1', label='Actif', type='certification', path=str(tmp_path / 'r1.json'), record_id=None, status='active', source='json_file', skill_count=5, is_selectable=True, reason=None),
        ReferentialOption(id='r2', label='Vide', type='certification', path=str(tmp_path / 'r2.json'), record_id=None, status='empty', source='json_file', skill_count=0, is_selectable=False, reason='Aucune compétence détectée'),
    ]
    monkeypatch.setattr('web_app.list_available_referentials', lambda **kw: mock_opts)
    monkeypatch.setattr('web_app.get_referential_option', lambda rid: next((o for o in mock_opts if o.id == rid), None))
    monkeypatch.setenv('DEEPFORMA_ADMIN_USER', 'anton')
    monkeypatch.setenv('DEEPFORMA_ADMIN_PASSWORD', 'deepforma')
    app = create_app(predictor=DummyPredictor(), france_travail_client_factory=lambda: DummyOfferClient([]), cache_ttl_seconds=60)
    client = app.test_client()
    response = client.get('/admin/ai-certification-market-comparison', headers=_auth_headers())
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert 'Actif' in html
    assert 'Vide' in html
    assert 'disabled' in html
    assert 'incomplet' in html
    assert 'Seuls les référentiels contenant au moins une compétence' in html


def test_referentiel_actif_selectable(tmp_path, monkeypatch):
    path = _make_referential(tmp_path / 'test.json')
    from referentials.referential_registry import ReferentialOption
    mock_opts = [
        ReferentialOption(id='r1', label='Actif', type='certification', path=str(path), record_id=None, status='active', source='json_file', skill_count=5, is_selectable=True, reason=None),
    ]
    monkeypatch.setattr('web_app.list_available_referentials', lambda **kw: mock_opts)
    monkeypatch.setattr('web_app.get_referential_option', lambda rid: next((o for o in mock_opts if o.id == rid), None))
    monkeypatch.setenv('DEEPFORMA_ADMIN_USER', 'anton')
    monkeypatch.setenv('DEEPFORMA_ADMIN_PASSWORD', 'deepforma')
    app = create_app(predictor=DummyPredictor(), france_travail_client_factory=lambda: DummyOfferClient([]), cache_ttl_seconds=60)
    client = app.test_client()
    response = client.get('/admin/ai-certification-market-comparison', headers=_auth_headers())
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert '<option value="r1"' in html
    assert 'disabled' not in html[html.index('<option value="r1"'):html.index('</option>', html.index('<option value="r1"'))] if '<option value="r1"' in html else True


def test_referentiel_vide_desactive_dans_dropdown(tmp_path, monkeypatch):
    from referentials.referential_registry import ReferentialOption
    mock_opts = [
        ReferentialOption(id='r2', label='Vide', type='certification', path=str(tmp_path / 'r2.json'), record_id=None, status='empty', source='json_file', skill_count=0, is_selectable=False, reason='Aucune compétence détectée'),
    ]
    monkeypatch.setattr('web_app.list_available_referentials', lambda **kw: mock_opts)
    monkeypatch.setattr('web_app.get_referential_option', lambda rid: next((o for o in mock_opts if o.id == rid), None))
    monkeypatch.setenv('DEEPFORMA_ADMIN_USER', 'anton')
    monkeypatch.setenv('DEEPFORMA_ADMIN_PASSWORD', 'deepforma')
    app = create_app(predictor=DummyPredictor(), france_travail_client_factory=lambda: DummyOfferClient([]), cache_ttl_seconds=60)
    client = app.test_client()
    response = client.get('/admin/ai-certification-market-comparison', headers=_auth_headers())
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert 'disabled' in html
    assert 'incomplet' in html


def test_referentiel_invalide_desactive_dans_dropdown(tmp_path, monkeypatch):
    from referentials.referential_registry import ReferentialOption
    mock_opts = [
        ReferentialOption(id='r3', label='Invalide', type='certification', path=str(tmp_path / 'r3.json'), record_id=None, status='invalid', source='json_file', skill_count=0, is_selectable=False, reason='Fichier JSON invalide'),
    ]
    monkeypatch.setattr('web_app.list_available_referentials', lambda **kw: mock_opts)
    monkeypatch.setattr('web_app.get_referential_option', lambda rid: next((o for o in mock_opts if o.id == rid), None))
    monkeypatch.setenv('DEEPFORMA_ADMIN_USER', 'anton')
    monkeypatch.setenv('DEEPFORMA_ADMIN_PASSWORD', 'deepforma')
    app = create_app(predictor=DummyPredictor(), france_travail_client_factory=lambda: DummyOfferClient([]), cache_ttl_seconds=60)
    client = app.test_client()
    response = client.get('/admin/ai-certification-market-comparison', headers=_auth_headers())
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert 'disabled' in html
    assert 'invalide' in html


def test_post_referentiel_empty_retourne_erreur_propre(tmp_path, monkeypatch):
    from referentials.referential_registry import ReferentialOption
    mock_opts = [
        ReferentialOption(id='r2', label='Vide', type='certification', path=str(tmp_path / 'r2.json'), record_id=None, status='empty', source='json_file', skill_count=0, is_selectable=False, reason='Aucune compétence détectée'),
    ]
    monkeypatch.setattr('web_app.list_available_referentials', lambda **kw: mock_opts)
    monkeypatch.setattr('web_app.get_referential_option', lambda rid: next((o for o in mock_opts if o.id == rid), None))
    monkeypatch.setenv('DEEPFORMA_ADMIN_USER', 'anton')
    monkeypatch.setenv('DEEPFORMA_ADMIN_PASSWORD', 'deepforma')
    app = create_app(predictor=DummyPredictor(), france_travail_client_factory=lambda: DummyOfferClient([]), cache_ttl_seconds=60)
    client = app.test_client()
    response = client.post(
        '/admin/ai-certification-market-comparison',
        data={'referential_id': 'r2', 'territory': '75056'},
        headers=_auth_headers(),
    )
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert 'aucune compétence exploitable' in html.lower()


def test_post_referentiel_invalide_retourne_erreur_propre(tmp_path, monkeypatch):
    from referentials.referential_registry import ReferentialOption
    mock_opts = [
        ReferentialOption(id='r3', label='Invalide', type='certification', path=str(tmp_path / 'r3.json'), record_id=None, status='invalid', source='json_file', skill_count=0, is_selectable=False, reason='Fichier JSON invalide'),
    ]
    monkeypatch.setattr('web_app.list_available_referentials', lambda **kw: mock_opts)
    monkeypatch.setattr('web_app.get_referential_option', lambda rid: next((o for o in mock_opts if o.id == rid), None))
    monkeypatch.setenv('DEEPFORMA_ADMIN_USER', 'anton')
    monkeypatch.setenv('DEEPFORMA_ADMIN_PASSWORD', 'deepforma')
    app = create_app(predictor=DummyPredictor(), france_travail_client_factory=lambda: DummyOfferClient([]), cache_ttl_seconds=60)
    client = app.test_client()
    response = client.post(
        '/admin/ai-certification-market-comparison',
        data={'referential_id': 'r3', 'territory': '75056'},
        headers=_auth_headers(),
    )
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert 'invalide' in html.lower()
    assert '500' not in response.status


def test_count_referential_skills_structures(tmp_path):
    from referentials.referential_registry import count_referential_skills
    assert count_referential_skills({'skills': [{'id': 'S1'}, {'id': 'S2'}]}) == 2
    assert count_referential_skills({'competencies': [{'code': 'C1'}]}) == 1
    assert count_referential_skills({'competences': [{'code': 'C1'}]}) == 1
    assert count_referential_skills({'criteria': [{'id': 'C1'}]}) == 1
    assert count_referential_skills({'official_skills': [{'id': 'S1'}]}) == 1
    assert count_referential_skills({'detected_skills': [{'id': 'S1'}]}) == 1
    assert count_referential_skills({'subskills': [{'id': 'S1'}]}) == 1
    assert count_referential_skills({'derived_competencies': [{'id': 'S1'}]}) == 1
    assert count_referential_skills({'blocks': [{'code': 'B1'}]}) == 1
    assert count_referential_skills({'blocs': [{'code': 'B1'}]}) == 1
    assert count_referential_skills({'skills': []}) == 0
    assert count_referential_skills({'competencies': []}) == 0
    assert count_referential_skills({}) == 0
    assert count_referential_skills([]) == 0
    assert count_referential_skills('not a dict') == 0


def test_gestion_et_comparaison_meme_source_de_verite(tmp_path, monkeypatch):
    from referentials.referential_registry import ReferentialOption
    mock_opts = [
        ReferentialOption(id='a', label='A', type='certification', path=str(tmp_path / 'a.json'), record_id=None, status='active', source='json_file', skill_count=3, is_selectable=True, reason=None),
        ReferentialOption(id='b', label='B', type='certification', path=str(tmp_path / 'b.json'), record_id=None, status='empty', source='json_file', skill_count=0, is_selectable=False, reason='Aucune compétence détectée'),
        ReferentialOption(id='c', label='C', type='certification', path=str(tmp_path / 'c.json'), record_id=None, status='invalid', source='json_file', skill_count=0, is_selectable=False, reason='Fichier JSON invalide'),
    ]
    monkeypatch.setattr('web_app.list_available_referentials', lambda **kw: mock_opts)
    monkeypatch.setattr('web_app.get_referential_option', lambda rid: next((o for o in mock_opts if o.id == rid), None))
    monkeypatch.setenv('DEEPFORMA_ADMIN_USER', 'anton')
    monkeypatch.setenv('DEEPFORMA_ADMIN_PASSWORD', 'deepforma')
    app = create_app(predictor=DummyPredictor(), france_travail_client_factory=lambda: DummyOfferClient([]), cache_ttl_seconds=60)
    client = app.test_client()
    response = client.get('/admin/ai-certification-market-comparison', headers=_auth_headers())
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    for label in ('A', 'B', 'C'):
        assert label in html
    assert 'disabled' in html
    assert html.count('<option ') == 4  # 3 referentials + 1 empty placeholder


def test_pas_de_fallback_vers_chemin_hardcode(tmp_path, monkeypatch):
    referential_path = _make_referential(tmp_path / 'referential.json')
    from referentials.referential_registry import ReferentialOption
    mock_opts = [
        ReferentialOption(id='custom_ref', label='Custom', type='certification', path=str(referential_path), record_id=None, status='active', source='json_file', skill_count=1, is_selectable=True, reason=None),
    ]
    monkeypatch.setattr('web_app.list_available_referentials', lambda **kw: mock_opts)
    monkeypatch.setattr('web_app.get_referential_option', lambda rid: next((o for o in mock_opts if o.id == rid), None))
    monkeypatch.setenv('DEEPFORMA_ADMIN_USER', 'anton')
    monkeypatch.setenv('DEEPFORMA_ADMIN_PASSWORD', 'deepforma')
    app = create_app(
        predictor=DummyPredictor(),
        france_travail_client_factory=lambda: DummyOfferClient([
            {'offer_id': 'o1', 'title': 'Test', 'description': 'Machine Learning test.', 'creation_date': '2026-07-01T00:00:00+00:00', 'location_label': 'Paris', 'contract_label': 'CDI'},
        ]),
        cache_ttl_seconds=60,
    )
    monkeypatch.setattr('web_app.write_comparison_outputs', lambda report, output_dir: {'json': tmp_path / 'r.json', 'validation_csv': tmp_path / 'v.csv', 'gaps_csv': tmp_path / 'g.csv'})
    client = app.test_client()
    response = client.post(
        '/admin/ai-certification-market-comparison',
        data={'referential_id': 'custom_ref', 'territory': '75056', 'commune': '75056', 'departement': '75', 'job_titles': 'Data Scientist', 'rome_codes': 'M1805', 'max_pages': '1', 'max_offers': '10', 'page_size': '10'},
        headers=_auth_headers(),
    )
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert 'Score global' in html
