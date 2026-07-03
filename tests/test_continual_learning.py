from __future__ import annotations

import json
from pathlib import Path

import pytest

from continual_learning.review_selector import ReviewCandidate, ReviewQueueSelector, build_review_candidates_from_offers
from continual_learning.store import ContinualLearningStore
from continual_learning.dataset_export import build_export_record
from scripts.export_continual_training_dataset import _keep_annotation
from scripts.compare_model_versions import compare_models
from scripts.train_continual_skill_extractor import Example, deduplicate_examples, split_rehearsal
from continual_learning.model_registry import promote_model_version


def test_offer_deduplication_by_content_version(tmp_path):
    store = ContinualLearningStore(tmp_path / 'cl.sqlite3')
    result1 = store.upsert_offer(
        offer_id='offer-1',
        title='Python developer',
        description_original='Develop Python services',
        collected_at='2026-07-02T00:00:00+00:00',
        location_label='Paris',
        territory='75',
        job_family='Informatique',
        structured_skills=[], predicted_skills=[], detected_forms=[], offsets=[], confidence={}, sources=[], model_version='m1', raw_payload={'id': 'offer-1'},
    )
    result2 = store.upsert_offer(
        offer_id='offer-1',
        title='Python developer',
        description_original='Develop Python services',
        collected_at='2026-07-02T00:00:00+00:00',
        location_label='Paris',
        territory='75',
        job_family='Informatique',
        structured_skills=[], predicted_skills=[], detected_forms=[], offsets=[], confidence={}, sources=[], model_version='m1', raw_payload={'id': 'offer-1'},
    )
    assert result1.content_version == result2.content_version
    assert result1.offer_row_id == result2.offer_row_id
    assert result2.inserted is False


def test_provenance_is_preserved(tmp_path):
    store = ContinualLearningStore(tmp_path / 'cl.sqlite3')
    offer = store.upsert_offer(
        offer_id='offer-2',
        title='Data scientist',
        description_original='Analyse de données avec Python',
        collected_at='2026-07-02T00:00:00+00:00',
        location_label='Lyon',
        territory='69',
        job_family='Data',
        structured_skills=[], predicted_skills=[], detected_forms=[], offsets=[], confidence={}, sources=[], model_version='m1', raw_payload={'id': 'offer-2'},
    )
    store.upsert_annotation(
        offer_row_id=offer.offer_row_id,
        offer_id='offer-2',
        content_version=offer.content_version,
        canonical_name='Python',
        surface_form='Python',
        normalized_name='Python',
        label='SKILL',
        start=18,
        end=24,
        confidence=0.9,
        source='human_review',
        provenance='human_review',
        is_explicit=True,
        validation_status='approved',
        validated_at='2026-07-02T00:00:00+00:00',
        validated_by='tester',
    )
    annotation = store.get_annotation(1)
    assert annotation['provenance'] == 'human_review'
    assert annotation['validation_status'] == 'approved'


def test_model_predictions_are_excluded_by_default():
    assert _keep_annotation({'provenance': 'model_prediction', 'validation_status': 'approved'}, 'semantic_match', False, False, True) is False
    assert _keep_annotation({'provenance': 'human_review', 'validation_status': 'approved'}, 'semantic_match', False, True, True) is True


def test_review_queue_selection_keeps_diversity():
    candidates = [
        ReviewCandidate(1, '1', 'v1', 'A', 'text A', '75', 'IT', 'm', 0.1, 0.9, 1, 0, 0, 0.8, 0.0, False, False),
        ReviewCandidate(2, '2', 'v1', 'B', 'text B', '75', 'RH', 'm', 0.2, 0.8, 0, 1, 0, 0.7, 0.1, False, False),
        ReviewCandidate(3, '3', 'v1', 'C', 'text C', '69', 'IT', 'm', 0.3, 0.7, 0, 0, 1, 0.6, 0.2, False, False),
        ReviewCandidate(4, '4', 'v1', 'D', 'text D', '69', 'RH', 'm', 0.4, 0.6, 0, 0, 0, 0.5, 0.3, False, False),
    ]
    selected = ReviewQueueSelector().select(candidates, limit=2)
    assert len(selected) == 2
    assert {item['territory'] for item in selected} == {'75', '69'}


def test_export_record_keeps_document_labels_without_offsets():
    offer = {'offer_id': 'o1', 'content_version': 'v1', 'description_original': 'Python et SQL', 'territory': '75', 'job_family': 'IT', 'collected_at': '2026-07-02T00:00:00+00:00'}
    annotations = [
        {'start': 0, 'end': 6, 'label': 'SKILL', 'canonical_name': 'Python', 'surface_form': 'Python', 'provenance': 'human_review'},
        {'start': None, 'end': None, 'label': 'SKILL', 'canonical_name': 'Cloud', 'surface_form': 'Cloud', 'provenance': 'france_travail_api'},
    ]
    record = build_export_record(offer, annotations).to_dict()
    assert len(record['entities']) == 1
    assert len(record['document_skills']) == 1
    assert record['document_skills'][0]['canonical_name'] == 'Cloud'


def test_review_candidate_dedup_and_rehearsal_seed():
    base = [Example(id='1', text='alpha', entities=[{'start': 0, 'end': 5, 'canonical_name': 'A', 'surface_form': 'A', 'label': 'SKILL'}], document_skills=[], metadata={}, source_path='base'), Example(id='2', text='beta', entities=[], document_skills=[], metadata={}, source_path='base')]
    incremental = [Example(id='3', text='alpha', entities=[], document_skills=[], metadata={}, source_path='inc'), Example(id='4', text='gamma', entities=[], document_skills=[], metadata={}, source_path='inc')]
    assert len(deduplicate_examples(base + incremental)) == 3
    selected1 = split_rehearsal(base, incremental, 3, 42)
    selected2 = split_rehearsal(base, incremental, 3, 42)
    assert [e.id for e in selected1] == [e.id for e in selected2]


def test_compare_model_refuses_regression():
    candidate = {'exact': {'f1': 0.61, 'precision': 0.70, 'recall': 0.55}, 'normalized': {'f1': 0.61}, 'false_positive_rate': 0.12, 'false_negative_rate': 0.10, 'no_textual_justification_rate': 0.01}
    production = {'exact': {'f1': 0.65, 'precision': 0.72, 'recall': 0.60}, 'normalized': {'f1': 0.64}, 'false_positive_rate': 0.09, 'false_negative_rate': 0.09, 'no_textual_justification_rate': 0.01}
    result = compare_models(candidate, production)
    assert result['promotion']['eligible'] is False


def test_promote_model_version_updates_production_symlink(tmp_path):
    version_dir = tmp_path / 'versions' / 'v1'
    version_dir.mkdir(parents=True)
    (version_dir / 'config.json').write_text('{}', encoding='utf-8')
    production = tmp_path / 'production'
    promote_model_version(version_dir=version_dir, production_link=production)
    assert production.is_symlink()
    assert production.resolve() == version_dir.resolve()
