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


def test_admin_ai_certification_market_comparison_route(tmp_path, monkeypatch):
    referential_path = _make_referential(tmp_path / 'referential.json')
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
    app.extensions['certification_market_comparator'] = CertificationMarketComparator(referential_path=referential_path, semantic_threshold=0.2)
    monkeypatch.setattr('web_app.write_comparison_outputs', lambda report, output_dir: {'json': tmp_path / 'report.json', 'validation_csv': tmp_path / 'validation.csv', 'gaps_csv': tmp_path / 'gaps.csv'})
    client = app.test_client()
    auth = base64.b64encode(b'anton:deepforma').decode('ascii')
    response = client.post(
        '/admin/ai-certification-market-comparison',
        data={
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
