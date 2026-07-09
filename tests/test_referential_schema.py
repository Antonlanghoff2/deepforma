from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from referentials.referential_registry import (
    count_referential_skills,
    count_referential_skills_detail,
    convert_imported_to_skills_format,
    flatten_referential_skills,
    normalize_referential_payload,
    normalize_skill_entry,
)


def test_official_skills_only():
    payload = {
        'skills': [
            {'id': 's1', 'label': 'Machine Learning', 'type': 'official_skill'},
            {'id': 's2', 'label': 'Deep Learning', 'type': 'official_skill'},
        ],
    }
    counts = count_referential_skills_detail(payload)
    assert counts['official_skills_count'] == 2
    assert counts['subskills_count'] == 0
    assert counts['exploitable_skills_count'] == 2
    assert counts['skills_count'] == 2


def test_children_are_counted():
    payload = {
        'skills': [
            {
                'id': 'p1',
                'label': 'Bloc IA',
                'type': 'official_skill',
                'children': [
                    {'id': 'c1', 'label': 'Identifier un cas d\'usage IA'},
                    {'id': 'c2', 'label': 'Déployer un modèle ML'},
                ],
            },
        ],
    }
    counts = count_referential_skills_detail(payload)
    assert counts['official_skills_count'] == 1
    assert counts['subskills_count'] == 2
    assert counts['exploitable_skills_count'] == 3
    assert counts['skills_count'] == 3


def test_subskills_key():
    payload = {
        'skills': [
            {
                'id': 'p1',
                'label': 'Bloc Data',
                'subskills': [
                    {'id': 'c1', 'label': 'Nettoyer les données'},
                    {'id': 'c2', 'label': 'Feature engineering'},
                ],
            },
        ],
    }
    normalized = normalize_referential_payload(payload)
    assert len(normalized['skills'][0]['children']) == 2
    counts = count_referential_skills_detail(normalized)
    assert counts['subskills_count'] == 2


def test_sous_competences_key():
    payload = {
        'skills': [
            {
                'id': 'p1',
                'label': 'Bloc NLP',
                'sous_competences': [
                    {'label': 'Tokenisation'},
                    {'label': 'Embeddings'},
                ],
            },
        ],
    }
    normalized = normalize_referential_payload(payload)
    assert len(normalized['skills'][0]['children']) == 2


def test_only_subskills_creates_generated_group():
    payload = {
        'derived_skills': [
            {'label': 'Compétence A', 'canonical_label': 'Compétence A', 'confidence': 0.9},
            {'label': 'Compétence B', 'canonical_label': 'Compétence B', 'confidence': 0.8},
            {'label': 'Compétence C', 'canonical_label': 'Compétence C', 'confidence': 0.7},
        ],
    }
    normalized = normalize_referential_payload(payload)
    assert len(normalized['skills']) == 1
    group = normalized['skills'][0]
    assert group['type'] == 'generated_group'
    assert group['label'] == 'Compétences extraites du référentiel'
    assert len(group['children']) == 3
    counts = count_referential_skills_detail(normalized)
    assert counts['exploitable_skills_count'] == 3


def test_generated_group_via_convert():
    payload = {
        'document': {'title': 'Test PDF', 'sha256': 'abc123', 'file_name': 'test.pdf', 'source_path': '/tmp/test.pdf'},
        'competencies': [],
        'derived_skills': [
            {'label': 'Skill A', 'canonical_label': 'Skill A', 'confidence': 0.9, 'page_start': 1},
            {'label': 'Skill B', 'canonical_label': 'Skill B', 'confidence': 0.8, 'page_start': 2},
        ],
    }
    converted = convert_imported_to_skills_format(payload)
    assert len(converted['skills']) == 1
    group = converted['skills'][0]
    assert group['type'] == 'generated_group'
    assert len(group['children']) == 2
    assert converted['metadata']['exploitable_skills_count'] == 2


def test_count_referential_skills_recursive():
    payload = {
        'skills': [
            {
                'id': 'p1',
                'label': 'Parent',
                'type': 'official_skill',
                'children': [
                    {
                        'id': 'c1',
                        'label': 'Child 1',
                        'children': [
                            {'id': 'gc1', 'label': 'Grandchild 1'},
                        ],
                    },
                    {'id': 'c2', 'label': 'Child 2'},
                ],
            },
        ],
    }
    counts = count_referential_skills_detail(payload)
    assert counts['official_skills_count'] == 1
    assert counts['subskills_count'] == 3
    assert counts['exploitable_skills_count'] == 4


def test_flatten_returns_parents_and_children():
    payload = {
        'skills': [
            {
                'id': 'p1',
                'label': 'Parent',
                'type': 'official_skill',
                'block': 'B1',
                'children': [
                    {'id': 'c1', 'label': 'Child 1'},
                    {'id': 'c2', 'label': 'Child 2'},
                ],
            },
        ],
    }
    flat = flatten_referential_skills(payload)
    assert len(flat) == 3
    labels = [item['label'] for item in flat]
    assert 'Parent' in labels
    assert 'Child 1' in labels
    assert 'Child 2' in labels
    child = next(item for item in flat if item['label'] == 'Child 1')
    assert child['parent_id'] == 'p1'
    assert 'Parent > Child 1' in child['path']


def test_non_regression_children_count():
    payload = {
        'skills': [
            {
                'label': 'Bloc IA',
                'children': [
                    {'label': 'Identifier un cas d\'usage IA'},
                    {'label': 'Déployer un modèle ML'},
                ],
            },
        ],
    }
    counts = count_referential_skills_detail(payload)
    assert counts['official_skills_count'] == 1
    assert counts['subskills_count'] == 2
    assert counts['exploitable_skills_count'] == 3
    assert counts['skills_count'] == 3


def test_normalize_skill_entry_string():
    result = normalize_skill_entry('Some skill label', index=0)
    assert result is not None
    assert result['label'] == 'Some skill label'
    assert result['type'] == 'subskill'


def test_normalize_skill_entry_empty_string():
    result = normalize_skill_entry('', index=0)
    assert result is None


def test_normalize_skill_entry_dict():
    result = normalize_skill_entry({'label': 'Test Skill', 'id': 'sk_1'}, index=0)
    assert result is not None
    assert result['label'] == 'Test Skill'
    assert result['id'] == 'sk_1'


def test_normalize_referential_payload_empty():
    result = normalize_referential_payload({})
    assert result['skills'] == []
    assert result['metadata']['skills_count'] == 0


def test_normalize_referential_payload_not_dict():
    result = normalize_referential_payload('not a dict')
    assert result['skills'] == []


def test_convert_imported_with_competencies_and_derived():
    payload = {
        'document': {'title': 'Test', 'sha256': 'abc', 'file_name': 'test.pdf', 'source_path': '/tmp/test.pdf'},
        'competencies': [
            {
                'code': 'C1',
                'official_label': 'Competence 1',
                'block_code': 'B1',
                'source_pages': [1],
                'derived_skills': [
                    {'label': 'Sub 1', 'canonical_label': 'Sub 1', 'confidence': 0.9},
                    {'label': 'Sub 2', 'canonical_label': 'Sub 2', 'confidence': 0.8},
                ],
            },
        ],
    }
    converted = convert_imported_to_skills_format(payload)
    assert len(converted['skills']) == 1
    skill = converted['skills'][0]
    assert skill['type'] == 'official_skill'
    assert len(skill['children']) == 2
    assert converted['metadata']['official_skills_count'] == 1
    assert converted['metadata']['subskills_count'] == 2
    assert converted['metadata']['exploitable_skills_count'] == 3


def test_count_with_derived_skills_top_level():
    payload = {
        'derived_skills': [
            {'label': 'Skill A'},
            {'label': 'Skill B'},
        ],
    }
    count = count_referential_skills(payload)
    assert count == 2


def test_count_with_competencies_top_level():
    payload = {
        'competencies': [
            {'label': 'Comp A'},
            {'label': 'Comp B'},
            {'label': 'Comp C'},
        ],
    }
    count = count_referential_skills(payload)
    assert count == 3


def test_count_with_sous_competences_in_skill():
    payload = {
        'skills': [
            {
                'label': 'Parent',
                'sous_compétences': [
                    {'label': 'Sous 1'},
                    {'label': 'Sous 2'},
                ],
            },
        ],
    }
    normalized = normalize_referential_payload(payload)
    assert len(normalized['skills'][0]['children']) == 2
    counts = count_referential_skills_detail(normalized)
    assert counts['subskills_count'] == 2


def test_flatten_with_no_children():
    payload = {
        'skills': [
            {'id': 's1', 'label': 'Skill 1', 'type': 'official_skill'},
        ],
    }
    flat = flatten_referential_skills(payload)
    assert len(flat) == 1
    assert flat[0]['label'] == 'Skill 1'


def test_migration_script_validate(tmp_path: Path):
    from scripts.migrate_referentials_to_canonical_schema import validate_file

    valid_payload = {
        'schema_version': '1.0',
        'skills': [
            {
                'id': 's1',
                'label': 'Test',
                'type': 'official_skill',
                'children': [
                    {'id': 'c1', 'label': 'Child', 'type': 'subskill'},
                ],
            },
        ],
        'metadata': {
            'official_skills_count': 1,
            'subskills_count': 1,
            'exploitable_skills_count': 2,
            'skills_count': 2,
        },
    }
    path = tmp_path / 'valid.json'
    path.write_text(json.dumps(valid_payload, ensure_ascii=False), encoding='utf-8')
    result = validate_file(path)
    assert result['valid'] is True


def test_migration_script_migrate(tmp_path: Path):
    from scripts.migrate_referentials_to_canonical_schema import migrate_file

    old_payload = {
        'derived_skills': [
            {'label': 'Skill A', 'canonical_label': 'Skill A'},
            {'label': 'Skill B', 'canonical_label': 'Skill B'},
        ],
    }
    path = tmp_path / 'old.json'
    path.write_text(json.dumps(old_payload, ensure_ascii=False), encoding='utf-8')
    result = migrate_file(path)
    assert result['status'] == 'migrated'
    migrated = json.loads(path.read_text(encoding='utf-8'))
    assert 'skills' in migrated
    assert len(migrated['skills']) == 1
    assert migrated['skills'][0]['type'] == 'generated_group'
    assert len(migrated['skills'][0]['children']) == 2
    bak = path.with_suffix('.json.bak')
    assert bak.exists()


def test_referential_option_has_new_fields():
    from referentials.referential_registry import ReferentialOption
    opt = ReferentialOption(
        id='test',
        label='Test',
        type='certification',
        path=None,
        record_id=None,
        status='active',
        source='json_file',
        skill_count=10,
        is_selectable=True,
        official_skills_count=3,
        subskills_count=7,
        exploitable_skills_count=10,
    )
    d = opt.to_dict()
    assert d['official_skills_count'] == 3
    assert d['subskills_count'] == 7
    assert d['exploitable_skills_count'] == 10
