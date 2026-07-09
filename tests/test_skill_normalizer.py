from __future__ import annotations

import json
from pathlib import Path

import pytest

from skills.skill_normalizer import (
    DEFAULT_SKILLS_PATH,
    LoadReport,
    SkillNormalizer,
    SkillReferentialNotFoundError,
    load_referential,
    normalize_skill_entry,
)
from skills.merge_offer_skills import get_default_normalizer


# ── normalize_skill_entry ────────────────────────────────────────────────


def test_normalize_entry_from_string():
    result = normalize_skill_entry('Python')
    assert result is not None
    assert result['label'] == 'Python'
    assert result['aliases'] == []
    assert result['skill_id'] == 'Python'


def test_normalize_entry_from_dict_with_label():
    result = normalize_skill_entry({'label': 'Python', 'category': 'langage'})
    assert result is not None
    assert result['label'] == 'Python'
    assert result['category'] == 'langage'
    assert result['aliases'] == []


def test_normalize_entry_from_dict_with_name():
    result = normalize_skill_entry({'name': 'Machine Learning', 'category': 'ML'})
    assert result is not None
    assert result['label'] == 'Machine Learning'
    assert result['category'] == 'ML'


def test_normalize_entry_from_dict_with_skill():
    result = normalize_skill_entry({'skill': 'TensorFlow'})
    assert result is not None
    assert result['label'] == 'TensorFlow'


def test_normalize_entry_from_dict_with_competence():
    result = normalize_skill_entry({'competence': 'Analyse de données'})
    assert result is not None
    assert result['label'] == 'Analyse de données'


def test_normalize_entry_from_dict_with_title():
    result = normalize_skill_entry({'title': 'Data Science'})
    assert result is not None
    assert result['label'] == 'Data Science'


def test_normalize_entry_empty_string_returns_none():
    assert normalize_skill_entry('') is None
    assert normalize_skill_entry('   ') is None


def test_normalize_entry_dict_without_label_returns_none():
    result = normalize_skill_entry({'code': 'C1', 'category': 'tech'})
    assert result is None


def test_normalize_entry_invalid_type_returns_none():
    assert normalize_skill_entry(42) is None
    assert normalize_skill_entry(None) is None
    assert normalize_skill_entry([]) is None


def test_normalize_entry_preserves_aliases():
    result = normalize_skill_entry({'label': 'ML', 'aliases': ['machine learning', 'apprentissage automatique']})
    assert result is not None
    assert result['aliases'] == ['machine learning', 'apprentissage automatique']


def test_normalize_entry_sets_missing_skill_id():
    result = normalize_skill_entry({'label': 'Python'})
    assert result is not None
    assert result['skill_id'] == 'Python'


def test_normalize_entry_preserves_existing_skill_id():
    result = normalize_skill_entry({'label': 'Python', 'skill_id': 'py01'})
    assert result is not None
    assert result['skill_id'] == 'py01'


# ── load_referential ─────────────────────────────────────────────────────


def test_load_referential_list_format(tmp_path):
    p = tmp_path / 'skills.json'
    p.write_text(json.dumps(['Python', 'Java', 'SQL']), encoding='utf-8')
    result = load_referential(p)
    assert result == ['Python', 'Java', 'SQL']


def test_load_referential_dict_with_skills(tmp_path):
    p = tmp_path / 'skills.json'
    p.write_text(json.dumps({'skills': ['Python', 'Java'], 'version': '1'}), encoding='utf-8')
    result = load_referential(p)
    assert result == ['Python', 'Java']


def test_load_referential_dict_with_competencies(tmp_path):
    p = tmp_path / 'skills.json'
    p.write_text(json.dumps({'competencies': ['C1', 'C2']}), encoding='utf-8')
    result = load_referential(p)
    assert result == ['C1', 'C2']


def test_load_referential_dict_without_skill_list_returns_empty(tmp_path):
    p = tmp_path / 'skills.json'
    p.write_text(json.dumps({'version': '1', 'author': 'test'}), encoding='utf-8')
    result = load_referential(p)
    assert result == []


def test_load_referential_missing_file_raises(tmp_path):
    p = tmp_path / 'nonexistent.json'
    with pytest.raises(SkillReferentialNotFoundError):
        load_referential(p)


# ── SkillNormalizer ──────────────────────────────────────────────────────


def test_normalizer_with_string_list(tmp_path):
    p = tmp_path / 'skills.json'
    p.write_text(json.dumps(['Python', 'Machine Learning']), encoding='utf-8')
    n = SkillNormalizer(p)
    assert len(n.reference) == 2
    assert n.reference[0]['label'] == 'Python'
    assert n.reference[1]['label'] == 'Machine Learning'
    assert n.load_report is not None
    assert n.load_report.valid == 2
    assert n.load_report.total == 2


def test_normalizer_with_dict_list(tmp_path):
    p = tmp_path / 'skills.json'
    p.write_text(json.dumps([
        {'label': 'Python', 'category': 'langage'},
        {'label': 'Machine Learning', 'category': 'ML'},
    ]), encoding='utf-8')
    n = SkillNormalizer(p)
    assert len(n.reference) == 2
    assert n.reference[0]['label'] == 'Python'
    assert n.reference[0]['category'] == 'langage'


def test_normalizer_with_mixed_list(tmp_path):
    p = tmp_path / 'skills.json'
    p.write_text(json.dumps([
        'Python',
        {'label': 'Machine Learning', 'category': 'ML'},
        'Java',
        {'name': 'TensorFlow'},
    ]), encoding='utf-8')
    n = SkillNormalizer(p)
    assert len(n.reference) == 4
    assert n.reference[0]['label'] == 'Python'
    assert n.reference[1]['label'] == 'Machine Learning'
    assert n.reference[2]['label'] == 'Java'
    assert n.reference[3]['label'] == 'TensorFlow'


def test_normalizer_with_empty_list(tmp_path):
    p = tmp_path / 'skills.json'
    p.write_text(json.dumps([]), encoding='utf-8')
    n = SkillNormalizer(p)
    assert n.reference == []
    assert n.load_report is not None
    assert n.load_report.valid == 0


def test_normalizer_ignores_invalid_entries(tmp_path):
    p = tmp_path / 'skills.json'
    p.write_text(json.dumps([
        'Python',
        {'code': 'no_label'},
        42,
        None,
        '',
        {'label': 'Valid'},
    ]), encoding='utf-8')
    n = SkillNormalizer(p)
    assert len(n.reference) == 2
    assert n.reference[0]['label'] == 'Python'
    assert n.reference[1]['label'] == 'Valid'
    assert n.load_report is not None
    assert n.load_report.total == 6
    assert n.load_report.valid == 2
    assert n.load_report.ignored == 4


def test_normalizer_missing_file_creates_empty_normalizer(tmp_path):
    p = tmp_path / 'nonexistent.json'
    n = SkillNormalizer(p)
    assert n.reference == []
    assert n._index == []
    assert n.load_report is not None
    assert len(n.load_report.errors) == 1


def test_normalizer_normalize_after_loading(tmp_path):
    p = tmp_path / 'skills.json'
    p.write_text(json.dumps([
        {'label': 'Python', 'aliases': ['py', 'python3']},
        {'label': 'Machine Learning', 'aliases': ['ML']},
    ]), encoding='utf-8')
    n = SkillNormalizer(p)
    label, conf, sid = n.normalize('Python')
    assert label == 'Python'
    assert conf == 1.0
    label, conf, sid = n.normalize('ML')
    assert label == 'Machine Learning'
    assert conf == 1.0
    label, conf, sid = n.normalize('UnknownSkill')
    assert label is None


def test_normalizer_normalize_many(tmp_path):
    p = tmp_path / 'skills.json'
    p.write_text(json.dumps([
        {'label': 'Python'},
        {'label': 'Java'},
    ]), encoding='utf-8')
    n = SkillNormalizer(p)
    normalized, unknowns = n.normalize_many(['Python', 'Ruby', 'Java', ''])
    assert len(normalized) == 2
    labels = {item['label'] for item in normalized}
    assert labels == {'Python', 'Java'}
    assert unknowns == ['Ruby']


# ── get_default_normalizer ───────────────────────────────────────────────


def test_get_default_normalizer_is_lazy():
    n1 = get_default_normalizer()
    n2 = get_default_normalizer()
    assert n1 is n2


# ── web_app import safeguard ─────────────────────────────────────────────


def test_import_web_app_does_not_crash():
    from web_app import create_app
    app = create_app()
    assert app is not None
