from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from deepforma.cpf.cleaning import CPFDeduper, build_formation_uid, build_search_text, normalize_department_code, normalize_referential_type, normalize_region_code, normalize_row, normalize_siret, row_hash, strip_html
from deepforma.cpf.columns import detect_columns, load_column_aliases
from deepforma.cpf.prepare import prepare_catalog
from deepforma.cpf.schema import inspect_catalog
from deepforma.cpf.skill_extractor import CPFSkillExtractor
from deepforma.recommendation.training_recommender import RecommenderConfig, TrainingRecommender
from deepforma.skills.normalizer import SkillTaxonomyNormalizer


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame.from_records(rows).to_csv(path, index=False, encoding="utf-8")


def test_detect_columns_alias_tolerance():
    aliases = load_column_aliases()
    detection = detect_columns(
        ["Intitulé de la formation", "Type de certification", "Nom région", "SIRET organisme"],
        aliases,
    )
    assert detection.resolved["title"] == "Intitulé de la formation"
    assert detection.resolved["certification"] == "Type de certification"
    assert detection.resolved["region"] == "Nom région"
    assert detection.resolved["organization_siret"] == "SIRET organisme"


def test_strip_html_and_unicode():
    value = strip_html("<p>Formación&nbsp;Python<br/>et&nbsp;IA</p>")
    assert "Python" in value
    assert "<" not in value
    assert "  " not in value


def test_search_text_never_contains_nan():
    row = {
        "title": "Python",
        "certification": None,
        "description": "nan",
        "objectives": "null",
        "nsf": "None",
        "exit_level": "",
        "referential_type": "RNCP",
    }
    search_text = build_search_text(row)
    assert "nan" not in search_text.lower()
    assert "null" not in search_text.lower()


def test_identifiant_stable():
    row = {
        "source_id": "A1",
        "certification": "RS 1234",
        "organization_siret": "12345678901234",
        "title": "python data",
        "department_code": "93",
    }
    assert build_formation_uid(row) == build_formation_uid(row)
    assert row_hash({"title": "Python", "certification": "RS 1", "organization": "X", "organization_siret": "12345678901234", "department_code": "93", "region_code": "11", "search_text": "Python"}) == row_hash({"title": "Python", "certification": "RS 1", "organization": "X", "organization_siret": "12345678901234", "department_code": "93", "region_code": "11", "search_text": "Python"})


def test_deduplication_exact_and_territory():
    deduper = CPFDeduper()
    base = {
        "formation_uid": "uid-1",
        "normalized_certification": "rs 1",
        "normalized_organization": "orga",
        "department_code": "93",
        "region_code": "11",
        "normalized_title": "python data",
    }
    assert deduper.is_duplicate(base) is False
    deduper.register(base)
    assert deduper.is_duplicate(base) is True
    other_territory = dict(base, formation_uid="uid-2", department_code="94", region_code="11")
    assert deduper.is_duplicate(other_territory) is False


def test_referential_and_skill_extraction_distinguish_java_java_script(tmp_path):
    referential = [
        {"skill_id": "java", "label": "Java", "aliases": ["java programming"], "category": "it"},
        {"skill_id": "javascript", "label": "JavaScript", "aliases": ["js"], "category": "it"},
        {"skill_id": "python", "label": "Python", "aliases": ["python 3"], "category": "it"},
    ]
    ref_path = tmp_path / "skills.json"
    ref_path.write_text(json.dumps(referential, ensure_ascii=False), encoding="utf-8")
    normalizer = SkillTaxonomyNormalizer(ref_path)
    assert normalizer.normalize("Java").canonical_label == "Java"
    assert normalizer.normalize("JavaScript").canonical_label == "JavaScript"
    assert normalizer.normalize("JavaScript").canonical_id != normalizer.normalize("Java").canonical_id

    extractor = CPFSkillExtractor(normalizer=normalizer, confidence_threshold=0.5)
    result = extractor.extract(
        {
            "title": "Développement JavaScript et Python",
            "certification": "",
            "description": "Maîtrise de JavaScript.",
            "objectives": "",
            "nsf": "",
        }
    )
    labels = {item["canonical_label"] for item in result.skills_normalized}
    assert "JavaScript" in labels
    assert "Java" not in labels


def test_normalize_rncp_rs():
    assert normalize_referential_type("RNCP") == "RNCP"
    assert normalize_referential_type("Répertoire spécifique") == "RS"


def test_normalize_codes():
    assert normalize_siret("123 456 789 01234") == "12345678901234"
    assert normalize_department_code("93") == "93"
    assert normalize_department_code("2A") == "2A"
    assert normalize_region_code("11") == "11"


def test_coverage_and_low_score_when_no_skill_covered():
    metadata = pd.DataFrame(
        [
            {
                "formation_uid": "f1",
                "search_text": "Formation Python et data",
                "title": "Python data",
                "organization": "Org A",
                "certification": "RS 1",
                "referential_type": "RS",
                "region_code": "11",
                "department_code": "93",
                "exit_level": "5",
                "skills_normalized": ["Python"],
            },
            {
                "formation_uid": "f2",
                "search_text": "Formation SQL",
                "title": "SQL avancé",
                "organization": "Org B",
                "certification": "RS 2",
                "referential_type": "RS",
                "region_code": "11",
                "department_code": "93",
                "exit_level": "5",
                "skills_normalized": ["SQL"],
            },
        ]
    )

    class FixedIndex:
        def search(self, vector, top_k=10):
            return [("f1", 0.91), ("f2", 0.89)]

    recommender = TrainingRecommender(metadata, index=FixedIndex(), config=RecommenderConfig(limit=2))
    results = recommender.recommend(
        {
            "target_job": "Développeur IA",
            "user_skills": ["Python"],
            "missing_skills": ["Machine Learning"],
            "desired_skills": [],
            "region_code": "11",
            "department_code": "93",
            "remote_allowed": False,
            "limit": 2,
        }
    )
    assert results
    assert results[0]["global_score"] <= 45.0
    assert results[0]["skill_coverage_score"] == 0.0


def test_region_department_filter_and_determinism():
    metadata = pd.DataFrame(
        [
            {
                "formation_uid": "a",
                "search_text": "Formation Python",
                "title": "Python A",
                "organization": "Org A",
                "certification": "RS 1",
                "referential_type": "RS",
                "region_code": "11",
                "department_code": "93",
                "exit_level": "5",
                "skills_normalized": ["Python"],
            },
            {
                "formation_uid": "b",
                "search_text": "Formation Python",
                "title": "Python B",
                "organization": "Org A",
                "certification": "RS 1",
                "referential_type": "RS",
                "region_code": "84",
                "department_code": "69",
                "exit_level": "5",
                "skills_normalized": ["Python"],
            },
        ]
    )

    class FixedIndex:
        def search(self, vector, top_k=10):
            return [("a", 0.95), ("b", 0.94)]

    recommender = TrainingRecommender(metadata, index=FixedIndex())
    first = recommender.recommend(
        {
            "target_job": "Développeur Python",
            "user_skills": [],
            "missing_skills": ["Python"],
            "desired_skills": [],
            "region_code": "11",
            "department_code": "93",
            "remote_allowed": False,
            "limit": 10,
        }
    )
    second = recommender.recommend(
        {
            "target_job": "Développeur Python",
            "user_skills": [],
            "missing_skills": ["Python"],
            "desired_skills": [],
            "region_code": "11",
            "department_code": "93",
            "remote_allowed": False,
            "limit": 10,
        }
    )
    assert first == second
    assert all(item["department"] == "93" for item in first)



@pytest.mark.parametrize(
    "script_name",
    [
        "download_cpf_catalog.py",
        "inspect_cpf_catalog.py",
        "prepare_cpf_catalog.py",
        "build_cpf_embeddings.py",
    ],
)
def test_cpf_scripts_help_commands(script_name):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / script_name
    result = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "usage" in result.stdout.lower()


def test_inspect_and_prepare_with_chunks_and_small_fixtures(tmp_path, monkeypatch):
    csv_path = tmp_path / "cpf.csv"
    rows = [
        {
            "id_action": "1",
            "intitulé": "Python data",
            "code certification": "RS 1",
            "type de certification": "RS",
            "description": "<p>Apprendre Python</p>",
            "objectifs": "Data",
            "niveau": "5",
            "siret_organisme": "12345678901234",
            "nom_region": "Île-de-France",
            "code_departement": "93",
            "organisme": "Org A",
        },
        {
            "id_action": "1",
            "intitulé": "Python data",
            "code certification": "RS 1",
            "type de certification": "RS",
            "description": "<p>Apprendre Python</p>",
            "objectifs": "Data",
            "niveau": "5",
            "siret_organisme": "12345678901234",
            "nom_region": "Île-de-France",
            "code_departement": "93",
            "organisme": "Org A",
        },
        {
            "id_action": "2",
            "intitulé": "Python data",
            "code certification": "RS 1",
            "type de certification": "RS",
            "description": "<p>Apprendre Python</p>",
            "objectifs": "Data",
            "niveau": "5",
            "siret_organisme": "12345678901234",
            "nom_region": "Île-de-France",
            "code_departement": "94",
            "organisme": "Org A",
        },
    ]
    _write_csv(csv_path, rows)

    report = inspect_catalog(csv_path, sample_limit=10, chunksize=1)
    assert report["column_count"] >= 10
    assert report["resolved_columns"]["title"] in {"intitulé", "Intitulé de la formation"}

    from deepforma.cpf import prepare as prepare_module

    def fake_write_parquet(records, output_path):
        output_path.write_text(f"rows={len(records)}", encoding="utf-8")

    monkeypatch.setattr(prepare_module, "_write_parquet", fake_write_parquet)
    result = prepare_catalog(csv_path, tmp_path / "out", chunksize=1, sample_limit=10)
    assert result["stats"].rows_read == 3
    assert result["stats"].rows_kept >= 2
    assert (tmp_path / "out" / "processed" / "cpf" / "formations_sample.csv").exists()
    assert (tmp_path / "out" / "reports" / "cpf_cleaning_report.json").exists()

