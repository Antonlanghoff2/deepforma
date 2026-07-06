from __future__ import annotations

import json
from pathlib import Path

from services.certification_market_comparison import CertificationMarketComparator, write_comparison_outputs
from referential_learning.ai_certification_taxonomy import infer_skill_taxonomy, normalize_market_alias


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
                'official_description': 'Apprentissage automatique et modélisation prédictive.',
                'normalized_label': 'machine learning',
                'category': 'Machine Learning',
                'subcategory': 'Classification',
                'technical_keywords': ['Machine Learning', 'Python'],
                'origin_document': 'sample.pdf',
                'aliases': ['ML', 'apprentissage automatique'],
                'source_page': 2,
                'active': True,
            },
            {
                'id': 'B1-A1-C2',
                'block': 'B1',
                'block_name': 'Bloc 1',
                'activity': 'A1',
                'code': 'A1-C2',
                'label': 'TensorFlow',
                'official_description': 'Framework de deep learning.',
                'normalized_label': 'tensorflow',
                'category': 'Deep Learning',
                'subcategory': 'Déploiement',
                'technical_keywords': ['TensorFlow'],
                'origin_document': 'sample.pdf',
                'aliases': ['Tensor Flow'],
                'source_page': 3,
                'active': True,
            },
            {
                'id': 'B2-A1-C1',
                'block': 'B2',
                'block_name': 'Bloc 2',
                'activity': 'A1',
                'code': 'A1-C1',
                'label': 'RGPD',
                'official_description': 'Protection des données personnelles.',
                'normalized_label': 'rgpd',
                'category': 'RGPD et sécurité',
                'subcategory': 'RGPD',
                'technical_keywords': ['RGPD'],
                'origin_document': 'sample.pdf',
                'aliases': ['GDPR'],
                'source_page': 5,
                'active': True,
            },
        ],
        'metadata': {'source_pdf': 'sample.pdf'},
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return path


def test_market_alias_normalization():
    assert normalize_market_alias('Tensor Flow') == 'TensorFlow'
    assert normalize_market_alias('python 3') == 'Python'
    assert infer_skill_taxonomy('apprentissage automatique', None, []).category == 'Machine Learning'


def test_comparator_scores_coverage_and_exports(tmp_path):
    referential_path = _make_referential(tmp_path / 'referential.json')
    comparator = CertificationMarketComparator(referential_path=referential_path, semantic_threshold=0.2)

    offers = [
        {
            'offer_id': 'offer-1',
            'title': 'Data Scientist',
            'description': 'Nous utilisons l’apprentissage automatique et Tensor Flow pour nos modèles.',
            'structured_skills': [{'label': 'Tensor Flow', 'confidence': 1.0}],
            'creation_date': '2026-07-01T00:00:00+00:00',
            'location_label': 'Paris',
            'contract_label': 'CDI',
        },
        {
            'offer_id': 'offer-2',
            'title': 'Assistant administratif',
            'description': 'Bac+5, personne dynamique, télétravail deux jours par semaine.',
            'creation_date': '2026-07-01T00:00:00+00:00',
            'location_label': 'Paris',
            'contract_label': 'CDI',
        },
    ]

    report = comparator.compare(
        offers,
        territory='75056',
        job_titles=['Data Scientist'],
        rome_codes=['M1805'],
        source_queries=['Data Scientist', 'M1805'],
    )

    assert report.offer_count == 2
    assert report.covered_offer_count == 1
    assert report.global_coverage_score > 0
    assert any(row.label == 'Machine Learning' and row.status == 'covered' for row in report.covered_skills)
    assert any(row.label == 'TensorFlow' and row.status == 'covered' for row in report.covered_skills)
    assert any(row.label == 'RGPD' and row.status == 'missing' for row in report.missing_skills)
    assert report.block_summaries

    paths = write_comparison_outputs(report, tmp_path / 'out')
    assert paths['json'].exists()
    assert paths['validation_csv'].exists()
    assert paths['gaps_csv'].exists()


def test_comparator_returns_zero_when_no_common_skill(tmp_path):
    referential_path = _make_referential(tmp_path / 'referential.json')
    comparator = CertificationMarketComparator(referential_path=referential_path, semantic_threshold=0.2)

    report = comparator.compare(
        [
            {
                'offer_id': 'offer-1',
                'title': 'Chef de projet marketing',
                'description': 'Bac+5, personne dynamique, télétravail deux jours par semaine.',
                'creation_date': '2026-07-01T00:00:00+00:00',
            }
        ],
        territory='75056',
        job_titles=['Chef de projet marketing'],
        rome_codes=['M1805'],
    )

    assert report.global_coverage_score == 0.0
    assert report.covered_offer_count == 0
    assert all(row.offer_count == 0 for row in report.covered_skills)
    assert any(row.status == 'missing' for row in report.missing_skills)
