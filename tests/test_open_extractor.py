"""Tests for the open skill extractor."""

from src.skills.open_extractor import extract_skills, ExtractedSkill


def test_empty_text():
    assert extract_skills("") == []
    assert extract_skills("   ") == []
    assert extract_skills("\n\n") == []


def test_maitriser():
    results = extract_skills("Le candidat maîtrise Python et TensorFlow.")
    assert len(results) >= 1
    # Should contain Python and TensorFlow
    labels = [r.source_label.lower() for r in results]
    assert any("python" in l for l in labels)


def test_savoir():
    results = extract_skills("Savoir utiliser Docker pour le déploiement.")
    assert len(results) >= 1
    assert any("docker" in r.source_label.lower() for r in results)


def test_etre_capable():
    results = extract_skills(
        "Être capable de concevoir une architecture de données."
    )
    assert len(results) >= 1
    labels_text = " ".join(r.source_label.lower() for r in results)
    assert "architecture" in labels_text or "concevoir" in labels_text


def test_tool_only():
    results = extract_skills("Utilisation avancée de PostgreSQL et MongoDB.")
    assert len(results) >= 1
    labels = [r.source_label.lower() for r in results]
    assert any("postgresql" in l for l in labels) or any("mongodb" in l for l in labels)


def test_knowledge():
    results = extract_skills("Bonnes connaissances en deep learning.")
    assert len(results) >= 1
    labels = [r.source_label.lower() for r in results]
    assert any("deep learning" in l for l in labels)


def test_multi_sentence():
    text = (
        "Le programme permet d'acquérir les compétences suivantes : "
        "Maîtriser les techniques de machine learning. "
        "Savoir développer des pipelines de données avec Python. "
        "Connaissances en Cloud Computing (AWS)."
    )
    results = extract_skills(text)
    assert len(results) >= 3
    all_text = " ".join(r.source_label.lower() for r in results)
    assert "machine learning" in all_text
    assert "python" in all_text or "pipelines" in all_text
    assert "cloud" in all_text or "aws" in all_text


def test_deduplication():
    text = "Maîtrise Python. Savoir programmer en Python."
    results = extract_skills(text)
    # Python standalone (from tool pattern) should appear at most once
    exact_python = sum(1 for r in results if r.source_label.lower() == "python")
    assert exact_python <= 1
    # "programmer en Python" is a different extraction
    prog_python = sum(1 for r in results if "programmer" in r.source_label.lower())
    assert prog_python <= 1


def test_soft_skill():
    results = extract_skills("Capable de travailler en équipe.")
    assert len(results) >= 1
    # Should detect teamwork
    all_text = " ".join(r.source_label.lower() for r in results)
    assert "équipe" in all_text or "equipe" in all_text


def test_no_false_positive():
    """Ensure very short or stop-word-only text yields no results."""
    results = extract_skills("Le programme est intéressant.")
    assert len(results) == 0


def test_long_phrase_truncated():
    """Very long phrases after verb should be excluded."""
    long = "Savoir " + "bla bla " * 50
    results = extract_skills(long)
    assert len(results) == 0


def test_ia_domain_tagging():
    results = extract_skills("Savoir utiliser PyTorch pour le deep learning.")
    assert len(results) >= 1
    for r in results:
        if "deep learning" in r.source_label.lower():
            assert r.type == "technical_skill"


def test_mettre_en_oeuvre():
    results = extract_skills("Mettre en œuvre une solution de cloud hybride.")
    assert len(results) >= 1
    labels_text = " ".join(r.source_label.lower() for r in results)
    assert "cloud" in labels_text or "solution" in labels_text


def test_dedup_conflict():
    """Lots of overlapping patterns should not explode."""
    text = "Maîtrise Python. Python est utilisé. Utilisation de Python."
    results = extract_skills(text)
    total = len(results)
    # Should not have 3+ identical Python entries
    python_count = sum(1 for r in results if r.source_label.lower() == "python")
    assert python_count <= 2, f"Python dedup failed: {python_count} entries"


def test_span_markers():
    """Start/end positions should be valid."""
    text = "Maîtrise Python et SQL."
    results = extract_skills(text)
    for r in results:
        assert 0 <= r.start < r.end <= len(text)
        assert text[r.start:r.end] == r.source_text
