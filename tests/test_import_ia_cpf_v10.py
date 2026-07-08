"""Tests pour le pipeline d'import Dataset_IA_V10_CPF."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.cpf_ia_v10 import (
    CPFIAFormation,
    ParsedSkill,
    QualityReport,
    detect_skill_type,
    determine_quality_status,
    export_training_candidates,
    inspect_excel,
    match_referential_skills_to_cpf_ia_formations,
    normalize_certification_code,
    normalize_skill_text,
    parse_certification_type,
    parse_formation,
    parse_price,
    parse_reviewed,
    parse_rome_codes,
    parse_tags,
    run_pipeline,
    split_extracted_ai_skills,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_xlsx(tmp_path):
    """Crée un fichier Excel minimal avec 3 lignes de test."""
    data = {
        '#': [1, 2, 3],
        'Secteur': ['Numérique & Informatique', 'Comptabilité & Finance', 'Langues'],
        'Organisme de formation': ['ORG IA', 'COMPTA PRO', 'LINGUA'],
        'Intitulé de la formation': [
            'Data Scientist Expert',
            'Expert Comptable IA',
            'Traducteur IA',
        ],
        'Type de certification': ['RNCP', 'RNCP', 'RS'],
        'Code certification': ['RNCP12345', 'RNCP67890', 'RS99999'],
        'Niveau': ['BAC+5 (NIVEAU 7)', 'BAC+3 (NIVEAU 6)', ''],
        'Codes ROME': ['M1805, M1806', '', ''],
        'Compétences IA extraites': [
            'Machine Learning | Deep Learning | Python',
            'Formation en comptabilité',
            'un texte très court',
        ],
        'Modalité': ['Présentiel', 'À distance', 'Hybride'],
        'Durée': ['500h', '300h', ''],
        'Prix TTC (€)': [8000, 4500, 1200],
        'Tags TrendRadar': [
            'data | ia | machine-learning',
            'comptabilité | finance | ia',
            'langues | traduction',
        ],
        '✅ Relu / Validé (oui/non)': ['oui', '', 'non'],
        '🗒 Corrections / Remarques': ['', 'Niveau à vérifier', ''],
    }
    df = pd.DataFrame(data)
    path = tmp_path / 'test_dataset.xlsx'
    df.to_excel(path, sheet_name='Dataset_IA_V10', index=False)
    return path


@pytest.fixture
def empty_xlsx(tmp_path):
    path = tmp_path / 'empty.xlsx'
    df = pd.DataFrame({'col': []})
    df.to_excel(path, sheet_name='WrongSheet', index=False)
    return path


# ---------------------------------------------------------------------------
# Tests d'inspection
# ---------------------------------------------------------------------------

def test_inspect_excel_ok(sample_xlsx):
    result = inspect_excel(sample_xlsx)
    assert result['row_count'] == 3
    assert result['selected_sheet'] == 'Dataset_IA_V10'
    assert result['missing_required_columns'] == []


def test_inspect_excel_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        inspect_excel(tmp_path / 'nope.xlsx')


def test_inspect_excel_wrong_sheet(empty_xlsx):
    with pytest.raises(ValueError, match='Feuille "Dataset_IA_V10" introuvable'):
        inspect_excel(empty_xlsx)


def test_inspect_excel_missing_columns(tmp_path):
    df = pd.DataFrame({'A': [1]})
    path = tmp_path / 'bad.xlsx'
    df.to_excel(path, sheet_name='Dataset_IA_V10', index=False)
    result = inspect_excel(path)
    assert 'Secteur' in result['missing_required_columns']


# ---------------------------------------------------------------------------
# Tests de parsing des codes certification
# ---------------------------------------------------------------------------

def test_normalize_certification_code():
    assert normalize_certification_code('  rncp12345 ') == 'RNCP12345'
    assert normalize_certification_code('RS 7318') == 'RS7318'
    assert normalize_certification_code('rncp 40866 ') == 'RNCP40866'


def test_parse_certification_type():
    assert parse_certification_type('RNCP12345') == 'RNCP'
    assert parse_certification_type('RS99999') == 'RS'
    assert parse_certification_type('UNKNOWN') == 'UNKNOWN'
    assert parse_certification_type('rncp123') == 'RNCP'


# ---------------------------------------------------------------------------
# Tests de parsing des codes ROME
# ---------------------------------------------------------------------------

def test_parse_rome_codes_single():
    assert parse_rome_codes('M1805') == ['M1805']


def test_parse_rome_codes_comma():
    assert parse_rome_codes('M1805, M1806, M1807') == ['M1805', 'M1806', 'M1807']


def test_parse_rome_codes_semicolon():
    assert parse_rome_codes('M1805; M1806') == ['M1805', 'M1806']


def test_parse_rome_codes_mixed():
    assert parse_rome_codes('M1805, M1806 | M1807') == ['M1805', 'M1806', 'M1807']


def test_parse_rome_codes_deduplicate():
    assert parse_rome_codes('M1805, M1805, M1806') == ['M1805', 'M1806']


def test_parse_rome_codes_empty():
    assert parse_rome_codes('') == []
    assert parse_rome_codes(None) == []
    assert parse_rome_codes(float('nan')) == []


def test_parse_rome_codes_invalid():
    assert parse_rome_codes('M1805, INVALID, M1806') == ['M1805', 'M1806']


# ---------------------------------------------------------------------------
# Tests de parsing des prix
# ---------------------------------------------------------------------------

def test_parse_price_int():
    assert parse_price(8000) == 8000.0


def test_parse_price_float():
    assert parse_price(4500.50) == 4500.50


def test_parse_price_string():
    assert parse_price('7500') == 7500.0
    assert parse_price('1 200 €') == 1200.0


def test_parse_price_nan():
    assert parse_price(None) is None
    assert parse_price(float('nan')) is None


# ---------------------------------------------------------------------------
# Tests de parsing des tags TrendRadar
# ---------------------------------------------------------------------------

def test_parse_tags():
    assert parse_tags('data | ia | machine-learning') == ['data', 'ia', 'machine-learning']


def test_parse_tags_empty():
    assert parse_tags('') == []
    assert parse_tags(None) == []
    assert parse_tags(float('nan')) == []


# ---------------------------------------------------------------------------
# Tests de parsing du champ review
# ---------------------------------------------------------------------------

def test_parse_reviewed():
    assert parse_reviewed('oui') is True
    assert parse_reviewed('non') is False
    assert parse_reviewed(None) is None
    assert parse_reviewed(float('nan')) is None


# ---------------------------------------------------------------------------
# Tests de découpage des compétences IA
# ---------------------------------------------------------------------------

def test_split_extracted_skills_pipe():
    result = split_extracted_ai_skills('Machine Learning | Deep Learning | Python')
    assert result == ['Machine Learning', 'Deep Learning', 'Python']


def test_split_extracted_skills_newline():
    result = split_extracted_ai_skills('Skill A\nSkill B\nSkill C')
    assert result == ['Skill A', 'Skill B', 'Skill C']


def test_split_extracted_skills_semicolon():
    result = split_extracted_ai_skills('Skill A; Skill B; Skill C')
    assert result == ['Skill A', 'Skill B', 'Skill C']


def test_split_extracted_skills_single():
    result = split_extracted_ai_skills('Compétence unique')
    assert result == ['Compétence unique']


def test_split_extracted_skills_empty():
    assert split_extracted_ai_skills('') == []
    assert split_extracted_ai_skills(None) == []
    assert split_extracted_ai_skills(float('nan')) == []


# ---------------------------------------------------------------------------
# Tests de normalisation des compétences
# ---------------------------------------------------------------------------

def test_normalize_skill_text():
    assert normalize_skill_text('  Hello   World  ') == 'Hello World'
    assert normalize_skill_text('Single') == 'Single'


# ---------------------------------------------------------------------------
# Tests de détection des types de compétence
# ---------------------------------------------------------------------------

def test_detect_skill_type_ai():
    assert detect_skill_type('Machine Learning avancé') == 'SKILL'
    assert detect_skill_type('Deep Learning pour la vision') == 'SKILL'
    assert detect_skill_type('IA et NLP') == 'SKILL'


def test_detect_skill_type_tool():
    assert detect_skill_type('Python avancé') == 'TOOL'
    assert detect_skill_type('TensorFlow') == 'TOOL'
    assert detect_skill_type('Docker et Kubernetes') == 'TOOL'


def test_detect_skill_type_course():
    assert detect_skill_type('Formation en comptabilité') == 'COURSE_CONTENT'
    assert detect_skill_type('Formation à la data science') == 'COURSE_CONTENT'


def test_detect_skill_type_too_short():
    assert detect_skill_type('abc') == 'TO_REVIEW'


# ---------------------------------------------------------------------------
# Tests de détermination de la qualité
# ---------------------------------------------------------------------------

def test_quality_status_ok():
    assert determine_quality_status('Analyser des données avec Python') == 'OK'


def test_quality_status_too_short():
    assert determine_quality_status('abc') == 'TOO_SHORT'


def test_quality_status_truncated():
    assert determine_quality_status('Compétence tronquée…') == 'TRUNCATED'
    assert determine_quality_status('constitu') == 'TRUNCATED'


def test_quality_status_vague():
    assert determine_quality_status('Compétence') == 'VAGUE'


def test_quality_status_not_a_skill():
    assert determine_quality_status('Formation en comptabilité') == 'NOT_A_SKILL'
    assert determine_quality_status('Nos formateurs experts accompagnent nos apprenants') == 'NOT_A_SKILL'


# ---------------------------------------------------------------------------
# Test de parse_formation
# ---------------------------------------------------------------------------

def test_parse_formation(sample_xlsx):
    df = pd.read_excel(sample_xlsx, sheet_name='Dataset_IA_V10')
    row = df.iloc[0].to_dict()
    formation = parse_formation(row, 1)
    assert isinstance(formation, CPFIAFormation)
    assert formation.source_row_id == 1
    assert formation.certification_code == 'RNCP12345'
    assert formation.certification_type == 'RNCP'
    assert formation.rome_codes == ['M1805', 'M1806']
    assert formation.price_ttc == 8000.0
    assert formation.trendradar_tags == ['data', 'ia', 'machine-learning']
    assert formation.reviewed is True
    assert formation.duration == '500h'


# ---------------------------------------------------------------------------
# Test du pipeline complet
# ---------------------------------------------------------------------------

def test_run_pipeline(sample_xlsx, tmp_path):
    output = tmp_path / 'output'
    report = run_pipeline(sample_xlsx, output)
    assert isinstance(report, QualityReport)
    assert report.rows_total == 3
    assert report.rncp_count == 2
    assert report.rs_count == 1
    assert report.missing_rome_count == 2
    assert report.skills_total == 5

    formations = pd.read_parquet(output / 'formations.parquet')
    assert len(formations) == 3
    assert 'certification_code' in formations.columns
    assert 'rome_codes' in formations.columns
    assert 'price_ttc' in formations.columns

    skills = pd.read_csv(output / 'skills_to_review.csv')
    assert len(skills) == 5
    assert 'quality_status' in skills.columns
    assert 'detected_type' in skills.columns

    rome_missing = pd.read_csv(output / 'rome_missing.csv')
    assert len(rome_missing) == 2

    with open(output / 'quality_report.json') as f:
        qr = json.load(f)
    assert qr['rows_total'] == 3
    assert qr['rncp_count'] == 2
    assert qr['duplicate_certification_codes'] == []


def test_run_pipeline_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_pipeline(tmp_path / 'nope.xlsx', tmp_path / 'out')


def test_run_pipeline_missing_columns(tmp_path):
    df = pd.DataFrame({'A': [1]})
    path = tmp_path / 'bad.xlsx'
    df.to_excel(path, sheet_name='Dataset_IA_V10', index=False)
    with pytest.raises(ValueError, match='Colonnes obligatoires manquantes'):
        run_pipeline(path, tmp_path / 'out')


# ---------------------------------------------------------------------------
# Tests d'export des candidats à l'entraînement
# ---------------------------------------------------------------------------

def test_export_training_candidates(tmp_path):
    skills = [
        ParsedSkill(1, 'RNCP001', 'F1', 'Secteur', 'Org', 'Machine Learning', 'Machine Learning', 'SKILL', 'OK'),
        ParsedSkill(2, 'RS001', 'F2', 'Secteur', 'Org', 'Formation en compta', 'Formation en compta', 'COURSE_CONTENT', 'NOT_A_SKILL'),
        ParsedSkill(3, 'RNCP002', 'F3', 'Secteur', 'Org', 'abc', 'abc', 'TO_REVIEW', 'TOO_SHORT'),
    ]
    out = tmp_path / 'candidates.jsonl'
    n = export_training_candidates(skills, out)
    assert n == 1
    lines = out.read_text().strip().split('\n')
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry['text'] == 'Machine Learning'
    assert entry['detected_type'] == 'SKILL'
    assert entry['source'] == 'Dataset_IA_V10_CPF'


def test_export_training_candidates_empty(tmp_path):
    n = export_training_candidates([], tmp_path / 'empty.jsonl')
    assert n == 0


# ---------------------------------------------------------------------------
# Tests du matching référentiel ↔ formations
# ---------------------------------------------------------------------------

def test_match_referential_skills_to_cpf_ia_formations():
    formations = [
        CPFIAFormation(
            source_row_id=1, sector='IT', provider_name='School',
            title='Data IA', certification_type='RNCP',
            certification_code='RNCP123',
            extracted_ai_skills_raw='Machine Learning | Deep Learning | Python',
            modality='Présentiel', duration='500h', price_ttc=5000.0,
            trendradar_tags=['ia', 'data'],
            rome_codes=['M1805'],
        ),
    ]
    skills = [{'label': 'Machine Learning', 'rome_codes': ['M1805'], 'tags': ['ia']}]
    results = match_referential_skills_to_cpf_ia_formations(skills, formations, min_similarity=0.5)
    assert len(results) >= 1
    assert results[0]['formation_title'] == 'Data IA'
    assert results[0]['similarity_score'] > 0.5


def test_match_referential_skills_no_match():
    formations = [
        CPFIAFormation(
            source_row_id=1, sector='IT', provider_name='School',
            title='Comptabilité', certification_type='RNCP',
            certification_code='RNCP999',
            extracted_ai_skills_raw='Comptabilité | Bilan',
            modality='Présentiel',
            trendradar_tags=['finance'],
        ),
    ]
    skills = [{'label': 'Deep Learning', 'rome_codes': [], 'tags': []}]
    results = match_referential_skills_to_cpf_ia_formations(skills, formations, min_similarity=0.9)
    assert len(results) == 0
