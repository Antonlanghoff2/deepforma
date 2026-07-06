from __future__ import annotations

import json
from pathlib import Path

import pytest

from referentials.rome_referential import RomeJob, RomeService, validate_rome_code


def _write_rome_fixture(tmp_path: Path) -> Path:
    payload = [
        {
            'rome_code': 'M1805',
            'label': 'Data Scientist',
            'definition': 'Concevoir et exploiter des modèles de données.',
            'alternative_titles': ['Machine Learning Engineer', 'Ingénieur IA'],
            'activity_ids': ['ACT1'],
            'skill_ids': ['SK1'],
            'domain': 'Data',
        }
    ]
    path = tmp_path / 'jobs.jsonl'
    path.write_text('\n'.join(json.dumps(row, ensure_ascii=False) for row in payload), encoding='utf-8')
    return path


def test_rome_service_search_and_validate(tmp_path: Path):
    service = RomeService(_write_rome_fixture(tmp_path))
    results = service.search('machine learning', limit=5)
    assert results
    assert results[0].rome_code == 'M1805'
    assert results[0].alternative_labels == ['Machine Learning Engineer', 'Ingénieur IA']
    assert service.get('M1805').label == 'Data Scientist'
    assert validate_rome_code(' m1805 ', service=service) == 'M1805'


def test_validate_rome_code_rejects_bad_format(tmp_path: Path):
    service = RomeService(_write_rome_fixture(tmp_path))
    with pytest.raises(ValueError, match='Format de code ROME invalide'):
        validate_rome_code('1234', service=service)


def test_validate_rome_code_rejects_unknown_when_reference_loaded(tmp_path: Path):
    service = RomeService(_write_rome_fixture(tmp_path))
    with pytest.raises(ValueError, match='Code ROME inconnu dans le référentiel chargé'):
        validate_rome_code('M1806', service=service)
