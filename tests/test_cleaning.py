from pathlib import Path

import pandas as pd

from scripts.clean_and_merge_datasets import (
    build_group_key,
    build_text_modele,
    canonicalize_competence,
    classify_ai_status,
    normalize_competence_field,
)


def test_canonicalize_competence_variants():
    assert canonicalize_competence('Machine learning') == 'Machine Learning'
    assert canonicalize_competence('IA Generative') == 'IA générative'
    assert canonicalize_competence('Ethique IA & RGPD') == 'Éthique de l’IA'


def test_normalize_competence_field_preserves_known_labels():
    values = normalize_competence_field('Prompt Engineering | IA Générative | No-code / Low-code')
    assert values == ['Prompt Engineering', 'IA générative', 'No-code / Low-code']


def test_build_text_modele_excludes_targets():
    row = pd.Series(
        {
            'intitule': 'Formation pour les RH',
            'description': 'Apprendre à utiliser des modèles',
            'objectif': '',
            'objectifs': 'Mettre en pratique',
            'programme': '',
            'public_cible': 'Managers',
            'prerequis': '',
            'niveau': 'Débutant',
            'modalite': 'À distance',
            'duree': '2 h',
            'certification': 'RS 1234',
            'codes_rome': 'M1805',
            'organisme': 'X',
            'competences_ia': 'Machine Learning | IA générative',
        }
    )
    text = build_text_modele(row)
    assert 'Machine Learning' not in text
    assert 'Intitulé : Formation pour les RH' in text


def test_classify_ai_status_with_annotations():
    row = pd.Series(
        {
            'intitule': 'Formation pour les RH',
            'description': '',
            'objectifs': '',
            'programme': '',
            'certification': '',
            'codes_rome': '',
            'competences_ia': 'Machine Learning | IA générative',
        }
    )
    statut, est_lie_ia, matches = classify_ai_status(row)
    assert statut == 'ia_confirmee'
    assert est_lie_ia == 1


def test_classify_non_ai_course():
    row = pd.Series(
        {
            'intitule': 'Anglais professionnel',
            'description': 'Améliorer son niveau de langue',
            'objectifs': '',
            'programme': '',
            'certification': '',
            'codes_rome': '',
            'competences_ia': '',
        }
    )
    statut, est_lie_ia, matches = classify_ai_status(row)
    assert statut == 'non_ia_confirmee'
    assert est_lie_ia == 0
    assert matches == []


def test_group_key_stable_for_same_title():
    key1 = build_group_key('IA générative pour RH', 'Organisme X', 'RS 1', 'À distance', '2 h')
    key2 = build_group_key('IA générative pour RH', 'Organisme X', 'RS 1', 'À distance', '2 h')
    assert key1 == key2
