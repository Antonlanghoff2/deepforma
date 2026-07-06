from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from jobs import collect_france_travail_offers as collector
from referentials.offer_skill_enricher import RNCPROMEOfferEnricher


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    path.write_text(payload + "\n", encoding='utf-8')


def test_rncp_rome_enricher_matches_evidence(tmp_path: Path) -> None:
    unified = tmp_path / 'skills.jsonl'
    mappings = tmp_path / 'rncp_rome_links.jsonl'
    _write_jsonl(
        unified,
        [
            {
                'canonical_skill_id': 'skill_prepare_text',
                'canonical_label': 'Préparer des données textuelles',
                'aliases': ['normaliser les données textuelles'],
                'sources': [
                    {'source': 'france_competences', 'source_id': 'RNCP1BC1-C001'},
                    {'source': 'rome', 'source_id': 'ROME1'},
                ],
            }
        ],
    )
    _write_jsonl(
        mappings,
        [
            {'rncp_code': 'RNCP1', 'rome_code': 'M1805', 'score': 0.9, 'match_method': 'hybrid', 'validated': True, 'evidence': {}},
        ],
    )
    enricher = RNCPROMEOfferEnricher(unified, mappings)
    result = enricher.extract(
        title='Data Scientist',
        description='Vous préparerez et normaliserez les données textuelles avant l entraînement des modèles.',
        rome_code='M1805',
        rome_label='Études et développement informatique',
    )
    assert result['rncp_candidates'] == [{'rncp_code': 'RNCP1', 'mapping_score': 0.9}]
    assert result['competences']
    assert result['competences'][0]['canonical_skill_id'] == 'skill_prepare_text'
    assert result['competences'][0]['evidence']
    assert result['competences'][0]['match_type'] in {'exact', 'alias', 'semantic', 'implicit'}


class _FakeClient:
    def iter_offers(self, *args, **kwargs):
        yield {
            'id': 'offer-1',
            'title': 'Data Scientist',
            'description': 'Vous préparerez et normaliserez les données textuelles avant l entraînement des modèles.',
            'rome': {'code': 'M1805', 'libelle': 'Data scientist'},
        }


class _FakeStore:
    last_instance: '_FakeStore' | None = None

    def __init__(self, *args, **kwargs):
        self.updated = []
        self.upserts = []
        _FakeStore.last_instance = self

    def normalize_content_version(self, title, description, structured_skills):
        return 'content-version'

    def upsert_offer(self, **kwargs):
        self.upserts.append(kwargs)
        return Namespace(offer_row_id=1, inserted=True, content_version='content-version')

    def update_offer_title_and_competences(self, offer_row_id, *, title=None, competences=None):
        self.updated.append({'offer_row_id': offer_row_id, 'title': title, 'competences': competences})

    def upsert_annotation(self, **kwargs):
        return None


def test_collect_pipeline_writes_rncp_rome_competences(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    unified = tmp_path / 'skills.jsonl'
    mappings = tmp_path / 'rncp_rome_links.jsonl'
    _write_jsonl(
        unified,
        [
            {
                'canonical_skill_id': 'skill_prepare_text',
                'canonical_label': 'Préparer des données textuelles',
                'aliases': ['normaliser les données textuelles'],
                'sources': [
                    {'source': 'france_competences', 'source_id': 'RNCP1BC1-C001'},
                    {'source': 'rome', 'source_id': 'ROME1'},
                ],
            }
        ],
    )
    _write_jsonl(
        mappings,
        [
            {'rncp_code': 'RNCP1', 'rome_code': 'M1805', 'score': 0.9, 'match_method': 'hybrid', 'validated': True, 'evidence': {}},
        ],
    )
    monkeypatch.setenv('RNCP_ROME_EXTRACTION_ENABLED', 'true')
    monkeypatch.setenv('RNCP_ROME_UNIFIED_REFERENTIAL_PATH', str(unified))
    monkeypatch.setenv('RNCP_ROME_MAPPINGS_PATH', str(mappings))
    monkeypatch.setenv('AI_CERTIFICATION_EXTRACTION_ENABLED', 'false')
    monkeypatch.setattr(collector, 'FranceTravailClient', lambda *args, **kwargs: _FakeClient())
    monkeypatch.setattr(collector, 'ContinualLearningStore', _FakeStore)

    output = tmp_path / 'offers.jsonl'
    args = Namespace(
        departement=None,
        commune=None,
        rome_code=None,
        keywords=None,
        contract=None,
        max_pages=1,
        max_offers=1,
        output=output,
        run_model=False,
        keep_raw=False,
        overwrite=True,
        page_size=1,
        pause_seconds=0.0,
        save_prefix=None,
    )
    report = collector.run_collection(args)
    assert report['offers_deduplicated'] == 1
    payload = [json.loads(line) for line in output.read_text(encoding='utf-8').splitlines() if line.strip()]
    assert payload[0]['competences']
    assert payload[0]['competences'][0]['canonical_skill_id'] == 'skill_prepare_text'
    assert _FakeStore.last_instance is not None
    assert _FakeStore.last_instance.updated
    assert _FakeStore.last_instance.updated[0]['competences']
