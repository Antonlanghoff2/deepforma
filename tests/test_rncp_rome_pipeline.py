from __future__ import annotations

import json
from pathlib import Path

from referentials.france_competences import FranceCompetenceCertification, FranceCompetenceSkill, FranceCompetencesOpenDataImporter
from referentials.rncp_rome_mapper import RNCPRomeMapper
from referentials.rome_referential import RomeJob, RomeSkill
from referentials.unified_skill_referential import build_unified_skill_referential


def test_rncp_import_supports_jsonl(tmp_path: Path) -> None:
    source = tmp_path / 'france_competences'
    source.mkdir()
    (source / 'skills.jsonl').write_text(
        '\n'.join([
            json.dumps({'skill_id': 'RNCP1BC1-C001', 'rncp_code': 'RNCP1', 'block_id': 'RNCP1BC1', 'official_label': 'Préparer des données', 'aliases': ['préparation des données'], 'active': True}),
        ]),
        encoding='utf-8',
    )
    (source / 'certifications.jsonl').write_text(
        json.dumps({'rncp_code': 'RNCP1', 'title': 'Ingénieur IA', 'status': 'active', 'level': 7, 'activities': ['concevoir des modèles'], 'target_jobs': ['ingénieur ia']}),
        encoding='utf-8',
    )
    payload = FranceCompetencesOpenDataImporter(source).load(active_only=True)
    assert len(payload['skills']) == 1
    assert len(payload['certifications']) == 1


def test_mapper_scores_hybrid_match() -> None:
    cert = FranceCompetenceCertification('RNCP1', 'RNCP', 'Ingénieur IA', 'active', 7, None, ['préparer et normaliser les données'], ['ingénieur ia'], [], [], None, None)
    job = RomeJob('M1805', 'Ingénieur IA', 'Concevoir des solutions IA', ['data scientist'], ['data'], ['préparation des données'])
    match = RNCPRomeMapper().score(cert, job, cert_skill_labels=['Préparer des données'], rome_skill_labels=['Préparation des données'])
    assert match.score > 0
    assert match.match_method in {'title', 'skills', 'semantic', 'hybrid'}


def test_mapper_can_validate_official_match() -> None:
    cert = FranceCompetenceCertification('RNCP1', 'RNCP', 'Ingénieur IA', 'active', 7, None, [], [], [], [], None, None)
    job = RomeJob('M1805', 'Ingénieur IA', '', [], [], [])
    match = RNCPRomeMapper().score(cert, job, official=True)
    assert match.score == 1.0
    assert match.validated is True
    assert match.match_method == 'official'


def test_unified_referential_keeps_distinct_sources() -> None:
    rncp = {
        'skills': [
            {'skill_id': 'RNCP1BC1-C001', 'rncp_code': 'RNCP1', 'block_id': 'RNCP1BC1', 'official_label': 'Préparer des données', 'aliases': []},
        ],
    }
    rome = {
        'jobs': [{'rome_code': 'M1805', 'label': 'Data Scientist', 'definition': '', 'alternative_titles': [], 'activity_ids': [], 'skill_ids': []}],
        'skills': [{'rome_skill_id': 'ROME1', 'official_label': 'Préparation des données', 'normalized_label': 'preparation des donnees', 'skill_type': 'competence'}],
    }
    unified, source_links = build_unified_skill_referential(france_competences=rncp, rome=rome, mappings=[])
    assert unified
    assert source_links == []
    assert any(item['canonical_skill_id'] == 'RNCP1BC1-C001' for item in unified)


def test_structures_are_serializable() -> None:
    skill = FranceCompetenceSkill('RNCP1BC1-C001', 'RNCP1', 'RNCP1BC1', 'Préparer des données', 'preparer des donnees', ['préparation des données'])
    job_skill = RomeSkill('ROME1', 'Préparation des données', 'preparation des donnees', 'competence')
    assert skill.to_dict()['skill_id'] == 'RNCP1BC1-C001'
    assert job_skill.to_dict()['rome_skill_id'] == 'ROME1'
