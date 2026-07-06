from __future__ import annotations

import json
from pathlib import Path

import pytest

from continual_learning.store import ContinualLearningStore
from referentials.ai_certification_referential import AICertificationReferential
from skill_extraction.ai_certification_extractor import AICertificationSkillExtractor
from scripts.build_ai_certification_skill_referential import build_referential


pytest.importorskip('fitz')


def _make_referential(path: Path) -> Path:
    payload = {
        'referential_id': 'ingenieur_ia_2025',
        'title': 'Ingénieur en intelligence artificielle',
        'version': '2025-01',
        'skills': [
            {
                'id': 'B1-A1-C1',
                'block': 'B1',
                'activity': 'A1',
                'code': 'A1-C1',
                'label': 'SVM',
                'official_description': 'Utiliser un SVM pour classer les données.',
                'normalized_label': 'svm',
                'aliases': ['support vector machine'],
                'source_page': 2,
                'active': True,
            },
            {
                'id': 'B1-A1-C2',
                'block': 'B1',
                'activity': 'A1',
                'code': 'A1-C2',
                'label': 'Random Forest',
                'official_description': 'Utiliser une forêt aléatoire pour la classification.',
                'normalized_label': 'random forest',
                'aliases': ['RandomForest'],
                'source_page': 2,
                'active': True,
            },
            {
                'id': 'B2-A2-C3',
                'block': 'B2',
                'activity': 'A2',
                'code': 'A2-C3',
                'label': 'Préparer le texte pour l’apprentissage',
                'official_description': 'Préparer et normaliser les corpus textuels avant l’apprentissage.',
                'normalized_label': 'preparer le texte pour l apprentissage',
                'aliases': ['nettoyage et normalisation des corpus textuels'],
                'source_page': 7,
                'active': True,
            },
            {
                'id': 'B2-A5-C1',
                'block': 'B2',
                'activity': 'A5',
                'code': 'A5-C1',
                'label': 'Comparer les modèles',
                'official_description': 'Comparer plusieurs modèles avec des métriques de précision, rappel et F-mesure.',
                'normalized_label': 'comparer les modeles',
                'aliases': [],
                'source_page': 9,
                'active': True,
            },
            {
                'id': 'B3-A1-C1',
                'block': 'B3',
                'activity': 'A1',
                'code': 'A1-C1',
                'label': 'SVM avancé',
                'official_description': 'Utiliser SVM dans un autre bloc.',
                'normalized_label': 'svm avance',
                'aliases': [],
                'source_page': 12,
                'active': True,
            },
        ],
        'metadata': {'source_pdf': 'sample.pdf'},
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return path


def test_referential_loader_indexes_aliases_and_block_collision(tmp_path):
    referential_path = _make_referential(tmp_path / 'referential.json')
    referential = AICertificationReferential(referential_path)

    assert referential.search_exact('SVM')['id'] == 'B1-A1-C1'
    assert referential.search_alias('RandomForest')['id'] == 'B1-A1-C2'
    assert referential.get_skill_by_id('B1-A1-C1')['label'] == 'SVM'
    assert referential.get_skill_by_id('B3-A1-C1')['label'] == 'SVM avancé'
    assert referential.normalize_label('Tensor Flow') == 'tensor flow'


def test_extractor_returns_strict_root_keys_and_detects_explicit_alias_and_implicit_matches(tmp_path):
    referential_path = _make_referential(tmp_path / 'referential.json')
    extractor = AICertificationSkillExtractor(
        referential_path=referential_path,
        semantic_threshold=0.15,
        implicit_threshold=0.2,
    )

    result = extractor.extract(
        title='Ingénieur Machine Learning',
        description=(
            'Nous réalisons le nettoyage et la normalisation des corpus textuels avant entraînement. '
            'Nous comparons plusieurs modèles avec des métriques de précision, rappel et F-mesure. '
            'Nous utilisons SVM pour classer les données. '
            'Le RandomForest est également utilisé.'
        ),
    )

    assert set(result) == {'intitule_poste', 'competences'}
    assert result['intitule_poste'] == 'Ingénieur Machine Learning'
    assert any(item['referential_id'] == 'B1-A1-C1' for item in result['competences'])
    assert any(item['referential_id'] == 'B1-A1-C2' for item in result['competences'])
    assert any(item['referential_id'] == 'B2-A2-C3' for item in result['competences'])
    assert any(item['referential_id'] == 'B2-A5-C1' and item['match_type'] in {'semantic', 'implicit'} for item in result['competences'])
    assert all('evidence' in item and item['evidence'] for item in result['competences'])


def test_extractor_title_fallback_and_negative_cases(tmp_path):
    referential_path = _make_referential(tmp_path / 'referential.json')
    extractor = AICertificationSkillExtractor(referential_path=referential_path, semantic_threshold=0.15, implicit_threshold=0.2)

    title_result = extractor.extract(title=None, description='Intitulé : Lead Data Scientist\nNous travaillons sur des modèles.')
    assert title_result['intitule_poste'] == 'Lead Data Scientist'

    negative_result = extractor.extract(
        title='Data Scientist',
        description='Bac+5, trois ans d’expérience, personne dynamique, télétravail deux jours par semaine.',
    )
    assert negative_result['intitule_poste'] == 'Data Scientist'
    assert negative_result['competences'] == []

    rejected_result = extractor.extract(
        title='Data Scientist',
        description='Modalités d’évaluation, critères d’évaluation, mise en situation professionnelle, jeux de rôle.',
    )
    assert rejected_result['competences'] == []


def test_duplicate_competence_is_deduplicated(tmp_path):
    referential_path = _make_referential(tmp_path / 'referential.json')
    extractor = AICertificationSkillExtractor(referential_path=referential_path, semantic_threshold=0.15, implicit_threshold=0.2)

    result = extractor.extract(
        title='Ingénieur IA',
        description='SVM est utilisé. Nous utilisons aussi SVM pour les tests. Support Vector Machine est présent.',
    )
    ids = [item['referential_id'] for item in result['competences']]
    assert ids.count('B1-A1-C1') == 1


def test_store_update_preserves_other_columns(tmp_path):
    store = ContinualLearningStore(tmp_path / 'cl.sqlite3')
    offer = store.upsert_offer(
        offer_id='offer-1',
        title='Data Scientist',
        description_original='Analyse de données',
        collected_at='2026-07-02T00:00:00+00:00',
        location_label='Paris',
        territory='75',
        job_family='Data',
        structured_skills=[{'label': 'Python'}],
        predicted_skills=[],
        detected_forms=[],
        offsets=[],
        confidence={},
        sources=[],
        model_version='m1',
        raw_payload={'offer_id': 'offer-1'},
    )
    store.update_offer_title_and_competences(
        offer.offer_row_id,
        title='Lead Data Scientist',
        competences=[{'referential_id': 'B1-A1-C1', 'code': 'A1-C1', 'libelle': 'SVM'}],
    )
    row = store.get_offer(offer.offer_row_id)
    assert row['title'] == 'Lead Data Scientist'
    assert row['location_label'] == 'Paris'
    assert row['territory'] == '75'
    assert json.loads(row['raw_payload_json'])['competences'][0]['referential_id'] == 'B1-A1-C1'
    assert json.loads(row['raw_payload_json'])['title'] == 'Lead Data Scientist'


def test_builder_excludes_evaluation_column(tmp_path):
    import fitz  # type: ignore

    pdf_path = tmp_path / 'sample.pdf'
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), 'BLOC 1', fontsize=12)
    page.insert_text((180, 100), 'A1-C1) Préparer le texte pour l’apprentissage', fontsize=11)
    page.insert_text((620, 100), 'A1-C1-E1 Modalités d’évaluation et critères d’évaluation', fontsize=11)
    doc.save(pdf_path)
    doc.close()

    referential = build_referential(pdf_path)
    assert referential['skills']
    assert any(skill['code'] == 'A1-C1' for skill in referential['skills'])
    assert all('évaluation' not in skill['official_description'].lower() for skill in referential['skills'])
