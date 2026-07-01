"""Tests pour la construction des paires d'entrainement CPF."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_cpf_training_pairs import (
    build_text_representation,
    jaccard_similarity,
    load_formations,
)


def test_build_text_representation():
    rec = {
        "title": "Formation Python",
        "sector": "Tech",
        "source_text": "programmation python, data science",
        "skills": ["Python", "Machine Learning"],
        "tags": ["python", "ml"],
        "rome_codes": ["M1805"],
        "modality": "Distanciel",
    }
    text = build_text_representation(rec)
    assert "[TITRE] Formation Python" in text
    assert "[SECTEUR] Tech" in text
    assert "[COMPETENCES] programmation python, data science" in text
    assert "Python | Machine Learning" in text
    assert "[TAGS] python | ml" in text
    assert "[ROME] M1805" in text
    assert "[MODALITE] Distanciel" in text


def test_build_text_representation_minimal():
    rec = {
        "title": "Formation",
        "skills": [],
        "tags": [],
    }
    text = build_text_representation(rec)
    assert "[TITRE] Formation" in text
    assert "[COMPETENCES_STRUCTUREES]" not in text


def test_jaccard_similarity():
    a = {"Python", "ML", "Data"}
    b = {"Python", "DL", "NLP"}
    sim = jaccard_similarity(a, b)
    # intersection = {Python}, union = {Python, ML, Data, DL, NLP}
    assert sim == pytest.approx(1 / 5)


def test_jaccard_similarity_empty():
    assert jaccard_similarity(set(), set()) == 0.0
    assert jaccard_similarity({"A"}, set()) == 0.0


def test_jaccard_similarity_identical():
    a = {"Python", "ML"}
    assert jaccard_similarity(a, a) == 1.0


def test_load_formations(tmp_path):
    records = [
        {"formation_id": "CPF-1", "title": "Formation A", "skills": ["Python"]},
        {"formation_id": "CPF-2", "title": "Formation B", "skills": ["ML"]},
    ]
    jsonl_path = tmp_path / "test_formations.jsonl"
    with open(jsonl_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    loaded = load_formations(str(jsonl_path))
    assert len(loaded) == 2
    assert loaded[0]["formation_id"] == "CPF-1"
    assert loaded[1]["formation_id"] == "CPF-2"


def test_build_pairs_structure(tmp_path):
    """Test minimal de build_pairs avec 3 formations."""
    from scripts.build_cpf_training_pairs import build_pairs

    records = [
        {
            "formation_id": "CPF-1",
            "group_id": "cert:RNCP1",
            "title": "Formation Python",
            "sector": "Tech",
            "source_text": "programmation python",
            "skills": ["Python", "Machine Learning"],
            "tags": ["python"],
            "rome_codes": ["M1805"],
            "modality": "",
        },
        {
            "formation_id": "CPF-2",
            "group_id": "cert:RNCP1",
            "title": "Python Approfondi",
            "sector": "Tech",
            "source_text": "python avance, deep learning",
            "skills": ["Python", "Deep Learning"],
            "tags": ["python"],
            "rome_codes": ["M1805"],
            "modality": "",
        },
        {
            "formation_id": "CPF-3",
            "group_id": "cert:RNCP2",
            "title": "Comptabilite",
            "sector": "Finance",
            "source_text": "comptabilite, gestion",
            "skills": ["Comptabilite"],
            "tags": ["finance"],
            "rome_codes": ["M1201"],
            "modality": "",
        },
    ]

    jsonl_path = tmp_path / "test_formations.jsonl"
    with open(jsonl_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    class Args:
        input = str(jsonl_path)
        output_dir = str(tmp_path / "output")
        output_pairs = "pairs_test.jsonl"
        seed = 42

    summary = build_pairs(Args())

    assert summary["total_formations"] == 3
    # Same cert group (RNCP1) -> 1 positive pair
    assert summary["positive_pairs"] >= 1
    assert summary["total_pairs"] >= 1
    assert (tmp_path / "output" / "pairs_test.jsonl").exists()
