from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from data_sources.ia_recommendations import (
    load_ia_recommendations_csv,
    normalize_recommendation_keyword,
    robust_parse_line,
    validate_ia_recommendations,
)
from domain.ia_recommendation_matching import (
    FRENCH_STOP_WORDS,
    _get_significant_words,
    _is_significant,
    _phrase_in_text,
    _significant_word_overlap_ratio,
    build_recommendation_index,
    match_ia_recommendations,
)


# ---------------------------------------------------------------------------
# Normalizer
# ---------------------------------------------------------------------------

def test_normalize_recommendation_keyword_lowercase() -> None:
    assert normalize_recommendation_keyword("Relation Client") == "relation client"


def test_normalize_recommendation_keyword_apostrophe() -> None:
    assert normalize_recommendation_keyword("plan d'actions commerciales") == "plan d actions commerciales"


def test_normalize_recommendation_keyword_accents() -> None:
    assert normalize_recommendation_keyword("Études de marché") == "etudes de marche"


def test_normalize_recommendation_keyword_punctuation() -> None:
    assert normalize_recommendation_keyword("vente-conseil !") == "vente conseil"


def test_normalize_recommendation_keyword_empty() -> None:
    assert normalize_recommendation_keyword("") == ""
    assert normalize_recommendation_keyword(None) == ""


# ---------------------------------------------------------------------------
# Robust CSV line parser
# ---------------------------------------------------------------------------

def test_robust_parse_line_quoted() -> None:
    kw, rec = robust_parse_line('"keyword","recommendation text"')
    assert kw == "keyword"
    assert rec == "recommendation text"


def test_robust_parse_line_quoted_with_commas() -> None:
    kw, rec = robust_parse_line('"keyword","rec, with, commas"')
    assert kw == "keyword"
    assert rec == "rec, with, commas"


def test_robust_parse_line_quoted_with_semicolons() -> None:
    kw, rec = robust_parse_line('"keyword","rec; with; semicolons"')
    assert kw == "keyword"
    assert rec == "rec; with; semicolons"


def test_robust_parse_line_quoted_trailing_semicolons() -> None:
    kw, rec = robust_parse_line('"keyword","recommandation text";;')
    assert kw == "keyword"
    assert rec == "recommandation text"


def test_robust_parse_line_escaped_quotes() -> None:
    kw, rec = robust_parse_line('"key""word","rec with ""quotes"""')
    assert kw == 'key"word'
    assert rec == 'rec with "quotes"'


def test_robust_parse_line_unquoted() -> None:
    kw, rec = robust_parse_line("keyword,recommendation text")
    assert kw == "keyword"
    assert rec == "recommendation text"


def test_robust_parse_line_unquoted_trailing_semicolons() -> None:
    kw, rec = robust_parse_line("keyword,recommendation text;;")
    assert kw == "keyword"
    assert rec == "recommendation text"


def test_robust_parse_line_unquoted_with_commas() -> None:
    kw, rec = robust_parse_line("keyword,rec with, commas inside")
    assert kw == "keyword"
    assert rec == "rec with, commas inside"


def test_robust_parse_line_empty_line() -> None:
    assert robust_parse_line("") is None
    assert robust_parse_line("  ") is None


# ---------------------------------------------------------------------------
# CSV loader integration
# ---------------------------------------------------------------------------

CSV_SAMPLE = """keyword,recommandation
relation client,"Decouvrir les chatbots de service client et les CRM a IA integree"
vente conseil,"S'entrainer a l'entretien de vente avec un agent conversationnel"
etudes commerciales,Utiliser ChatGPT ou Claude pour analyser des donnees
plan d'actions commerciales,"Co-construire le PAC avec un LLM"
defaut,Recommandation par defaut
actif,Recommandation active
"""

CSV_WITH_BOM = "\ufeff" + CSV_SAMPLE


def _write_csv(tmpdir: Path, content: str = CSV_SAMPLE) -> Path:
    p = tmpdir / "test.csv"
    p.write_bytes(content.encode("utf-8-sig"))
    return p


def test_load_ia_recommendations_csv_basic() -> None:
    with tempfile.TemporaryDirectory() as td:
        path = _write_csv(Path(td))
        records, report = load_ia_recommendations_csv(path)
    assert report.total_lines == 7
    assert report.valid_lines == 6
    assert report.rejected_lines == 0
    assert len(records) == 6


def test_load_ia_recommendations_csv_rejects_empty_keyword() -> None:
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "test.csv"
        path.write_bytes(b'\xef\xbb\xbfkeyword,recommandation\n,rec')
        records, report = load_ia_recommendations_csv(path)
    assert report.rejected_lines == 1
    assert report.empty_keyword == 1


def test_load_ia_recommendations_csv_rejects_empty_recommendation() -> None:
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "test.csv"
        path.write_bytes(b'\xef\xbb\xbfkeyword,recommandation\nkw,')
        records, report = load_ia_recommendations_csv(path)
    assert report.rejected_lines == 1
    assert report.empty_recommendation == 1


def test_load_ia_recommendations_csv_detects_default() -> None:
    csv = "keyword,recommandation,is_default\n"
    csv += "defaut,Rec par defaut,true\n"
    csv += "actif,Rec active,\n"
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "test.csv"
        path.write_bytes(("\ufeff" + csv).encode("utf-8"))
        records, report = load_ia_recommendations_csv(path)
    assert report.default_rules == 1
    defaults = [r for r in records if r.get("is_default")]
    assert len(defaults) == 1


def test_load_ia_recommendations_csv_with_bom() -> None:
    with tempfile.TemporaryDirectory() as td:
        path = _write_csv(Path(td), CSV_WITH_BOM)
        records, report = load_ia_recommendations_csv(path)
    assert report.valid_lines == 6
    assert len(records) == 6


def test_load_ia_recommendations_csv_produces_normalized_keyword() -> None:
    with tempfile.TemporaryDirectory() as td:
        path = _write_csv(Path(td))
        records, _ = load_ia_recommendations_csv(path)
    rec_by_kw = {r["keyword"]: r for r in records}
    assert rec_by_kw["plan d'actions commerciales"]["keyword_normalized"] == "plan d actions commerciales"
    assert rec_by_kw["relation client"]["keyword_normalized"] == "relation client"


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

def test_validate_ia_recommendations_passes() -> None:
    records = [
        {"keyword": "test", "keyword_normalized": "test", "recommendation": "rec", "is_active": True},
    ]
    report = validate_ia_recommendations(records)
    assert report.valid_lines == 1
    assert report.rejected_lines == 0


def test_validate_ia_recommendations_empty_keyword() -> None:
    records = [
        {"keyword_normalized": "", "recommendation": "rec", "is_active": True},
    ]
    report = validate_ia_recommendations(records)
    assert report.valid_lines == 0
    assert report.empty_keyword == 1


def test_validate_ia_recommendations_missing_recommendation() -> None:
    records = [
        {"keyword": "test", "keyword_normalized": "test", "is_active": True},
    ]
    report = validate_ia_recommendations(records)
    assert report.valid_lines == 0
    assert report.empty_recommendation == 1


# ---------------------------------------------------------------------------
# Matching utilities
# ---------------------------------------------------------------------------

def test_phrase_in_text() -> None:
    assert _phrase_in_text("relation client", "developper la relation client et la relation partenariale")
    assert _phrase_in_text("plan d actions", "elaborer et piloter le plan d actions commerciales")
    assert not _phrase_in_text("client", "developper la relation")
    assert not _phrase_in_text("", "some text")
    assert _phrase_in_text("abc", "abc")


def test_phrase_in_text_edge_cases() -> None:
    assert _phrase_in_text("x", "a x b")
    assert not _phrase_in_text("x y", "a xy b")
    assert not _phrase_in_text("abc def", "abcdef")


def test_is_significant() -> None:
    assert not _is_significant("les")
    assert not _is_significant("le")
    assert not _is_significant("de")
    assert not _is_significant("et")
    assert not _is_significant("un")
    assert not _is_significant("une")
    assert not _is_significant("la")
    assert not _is_significant("x")
    assert not _is_significant("ab")
    assert _is_significant("client")
    assert _is_significant("relation")
    assert _is_significant("commercial")
    assert _is_significant("etude")


def test_get_significant_words() -> None:
    result = _get_significant_words("developper la relation client et la relation partenariale")
    assert result == ["developper", "relation", "client", "relation", "partenariale"]
    result2 = _get_significant_words("le un et x ab")
    assert result2 == []


def test_significant_word_overlap_ratio() -> None:
    kw = ["relation", "client"]
    skill = ["developper", "relation", "client", "partenariale"]
    assert _significant_word_overlap_ratio(kw, skill) == 1.0
    kw2 = ["relation", "client", "commercial"]
    skill2 = ["developper", "relation", "partenariale"]
    assert _significant_word_overlap_ratio(kw2, skill2) == 1 / 3


def test_significant_word_overlap_ratio_empty() -> None:
    assert _significant_word_overlap_ratio([], ["a"]) == 0.0
    assert _significant_word_overlap_ratio(["a"], []) == 0.0


# ---------------------------------------------------------------------------
# Index building
# ---------------------------------------------------------------------------

def _make_rec(keyword: str, rec_text: str = "Rec", **overrides) -> dict[str, Any]:
    return {
        "keyword": keyword,
        "keyword_normalized": normalize_recommendation_keyword(keyword),
        "aliases": [],
        "aliases_normalized": [],
        "recommendation": rec_text,
        "recommendation_id": f"id_{keyword}",
        "is_active": True,
        **overrides,
    }


def test_build_recommendation_index_basic() -> None:
    recs = [_make_rec("relation client"), _make_rec("vente conseil")]
    idx = build_recommendation_index(recs)
    assert "relation client" in idx.phrase_index
    assert "vente conseil" in idx.phrase_index
    assert "relation" in idx.significant_word_index
    assert "vente" in idx.significant_word_index
    assert len(idx.default_rules) == 0


def test_build_recommendation_index_skips_inactive() -> None:
    recs = [_make_rec("active"), _make_rec("inactive", is_active=False)]
    idx = build_recommendation_index(recs)
    assert len(idx.all_recommendations) == 1


def test_build_recommendation_index_default_rules() -> None:
    recs = [_make_rec("defaut", is_default=True)]
    idx = build_recommendation_index(recs)
    assert len(idx.default_rules) == 1
    assert "defaut" not in idx.phrase_index


def test_build_recommendation_index_with_aliases() -> None:
    recs = [_make_rec("relation client", aliases=["customer relation"], aliases_normalized=["customer relation"])]
    idx = build_recommendation_index(recs)
    assert "customer relation" in idx.phrase_index


# ---------------------------------------------------------------------------
# Matching: exact / alias
# ---------------------------------------------------------------------------

def test_match_ia_recommendations_exact() -> None:
    recs = [_make_rec("relation client")]
    skills = [{"name": "Developper la relation client et la relation partenariale"}]
    matches = match_ia_recommendations(skills, recs)
    assert len(matches) == 1
    assert matches[0].match_method == "EXACT"
    assert matches[0].score == 1.0
    assert matches[0].confidence_label == "HIGH"
    assert matches[0].matched_keyword == "relation client"


def test_match_ia_recommendations_alias() -> None:
    recs = [_make_rec("relation client", aliases=["customer relation"], aliases_normalized=["customer relation"])]
    skills = [{"name": "Gerer la customer relation"}]
    matches = match_ia_recommendations(skills, recs)
    assert len(matches) == 1
    assert matches[0].match_method == "ALIAS"
    assert matches[0].score == 0.85


def test_match_ia_recommendations_exact_over_alias() -> None:
    recs = [_make_rec("relation client", aliases=["relation clientale"], aliases_normalized=["relation clientale"])]
    skills = [{"name": "Developper la relation client et la relation clientale"}]
    matches = match_ia_recommendations(skills, recs)
    methods = {m.match_method for m in matches}
    assert "EXACT" in methods
    exacts = [m for m in matches if m.match_method == "EXACT"]
    assert len(exacts) == 1
    assert exacts[0].score == 1.0


# ---------------------------------------------------------------------------
# Matching: inclusion
# ---------------------------------------------------------------------------

def test_match_ia_recommendations_inclusion() -> None:
    recs = [_make_rec("relation client")]
    skills = [{"name": "Animer la relation de partenariat"}]
    matches = match_ia_recommendations(skills, recs)
    assert len(matches) == 1
    assert matches[0].match_method == "INCLUSION"
    assert 0.80 <= matches[0].score < 1.0


def test_match_ia_recommendations_inclusion_no_stop_word_match() -> None:
    recs = [_make_rec("les et un")]
    skills = [{"name": "Developper les relations et un partenariat"}]
    matches = match_ia_recommendations(skills, recs)
    assert len(matches) == 0  # only stop words have 0 overlap


# ---------------------------------------------------------------------------
# Matching: default
# ---------------------------------------------------------------------------

def test_match_ia_recommendations_default() -> None:
    recs = [_make_rec("defaut", "Default rec", is_default=True)]
    skills = [{"name": "Competence inconnue"}]
    matches = match_ia_recommendations(skills, recs)
    assert len(matches) == 1
    assert matches[0].match_method == "DEFAULT"


def test_match_ia_recommendations_default_not_used_when_match_exists() -> None:
    recs = [
        _make_rec("relation client"),
        _make_rec("defaut", "Default rec", is_default=True),
    ]
    skills = [{"name": "Developper la relation client"}]
    matches = match_ia_recommendations(skills, recs)
    assert len(matches) == 1
    assert matches[0].match_method == "EXACT"


# ---------------------------------------------------------------------------
# Matching: edges
# ---------------------------------------------------------------------------

def test_match_ia_recommendations_empty_skills() -> None:
    assert match_ia_recommendations([], [_make_rec("test")]) == []


def test_match_ia_recommendations_empty_recommendations() -> None:
    assert match_ia_recommendations([{"name": "test"}], []) == []


def test_match_ia_recommendations_no_match() -> None:
    recs = [_make_rec("python")]
    skills = [{"name": "Jardinage"}]
    matches = match_ia_recommendations(skills, recs)
    assert len(matches) == 0


def test_match_ia_recommendations_max_per_skill() -> None:
    recs = [
        _make_rec("relation client", "Rec relation"),
        _make_rec("vente conseil", "Rec vente"),
        _make_rec("communication", "Rec communication"),
        _make_rec("etudes marche", "Rec etudes"),
    ]
    skills = [{"name": "Relation client vente conseil communication etudes marche"}]
    matches = match_ia_recommendations(skills, recs, max_recommendations_per_skill=2)
    assert len(matches) == 2


def test_match_ia_recommendations_dedup() -> None:
    recs = [
        _make_rec("relation client", "Same recommendation"),
        _make_rec("relation client", "Same recommendation"),
    ]
    skills = [{"name": "Developper la relation client"}]
    matches = match_ia_recommendations(skills, recs)
    assert len(matches) == 1


# ---------------------------------------------------------------------------
# Matching: embedding (mocked)
# ---------------------------------------------------------------------------

class _MockEmbeddingModel:
    def encode(self, texts, convert_to_numpy=True):
        import numpy as np
        rng = np.random.RandomState(42)
        return rng.randn(len(texts), 4).astype(np.float32)


def test_match_ia_recommendations_embedding() -> None:
    recs = [_make_rec("management commercial")]
    skills = [{"name": "Gestion d equipe commerciale"}]
    matches = match_ia_recommendations(skills, recs, embedding_model=_MockEmbeddingModel())
    # May or may not match depending on random embeddings; just verify it runs
    # and returns IARecommendationMatch instances if it does match
    for m in matches:
        assert hasattr(m, "match_method")
        assert hasattr(m, "score")
        assert hasattr(m, "confidence_label")


# ---------------------------------------------------------------------------
# Demo scenario: RNCP41966-like
# ---------------------------------------------------------------------------

def _make_records() -> list[dict[str, Any]]:
    return [
        _make_rec("relation client", "Decouvrir les chatbots de service client"),
        _make_rec("vente conseil", "S'entrainer a l'entretien de vente"),
        _make_rec("communication commerciale", "Generer des contenus publicitaires"),
        _make_rec("etudes commerciales", "Utiliser ChatGPT pour analyser des donnees"),
        _make_rec("plan d'actions commerciales", "Co-construire le PAC avec un LLM"),
        _make_rec("evaluation commerciale", "Construire des tableaux de bord"),
        _make_rec("veille informationnelle", "Mettre en place une veille avec des alertes"),
        _make_rec("negociation commerciale", "Simuler des negociations avec un LLM"),
        _make_rec("outils CRM", "Utiliser un CRM pour le suivi"),
    ]


def test_demo_scenario_rncp41966() -> None:
    skills_rncp41966 = [
        "Developper la relation client et la relation partenariale",
        "Conduire les negociations commerciales avec les clients et les partenaires",
        "Animer la relation de partenariat avec les acteurs de la filiere",
        "Realiser des etudes de marche",
        "Elaborer et piloter le plan d'actions commerciales",
        "Evaluer et optimiser la performance commerciale",
        "Mettre en place la veille informationnelle",
        "Utiliser un CRM pour le suivi de l'activite commerciale",
    ]
    recs = _make_records()
    skills = [{"name": s} for s in skills_rncp41966]
    matches = match_ia_recommendations(skills, recs)

    match_by_skill = {m.skill_original: m for m in matches}
    assert "Developper la relation client et la relation partenariale" in match_by_skill
    assert match_by_skill["Developper la relation client et la relation partenariale"].match_method == "EXACT"
    assert match_by_skill["Developper la relation client et la relation partenariale"].score == 1.0

    assert "Elaborer et piloter le plan d'actions commerciales" in match_by_skill
    assert match_by_skill["Elaborer et piloter le plan d'actions commerciales"].match_method == "EXACT"
    assert match_by_skill["Elaborer et piloter le plan d'actions commerciales"].score == 1.0

    assert "Utiliser un CRM pour le suivi de l'activite commerciale" in match_by_skill

    assert "Mettre en place la veille informationnelle" in match_by_skill

    assert len(matches) > 0
    assert all(m.skill_original in skills_rncp41966 for m in matches)


def test_demo_scenario_all_skills_get_at_least_one() -> None:
    skills_rncp41966 = [
        "Developper la relation client et la relation partenariale",
        "Conduire les negociations commerciales avec les clients et les partenaires",
        "Animer la relation de partenariat avec les acteurs de la filiere",
    ]
    recs = _make_records()
    skills = [{"name": s} for s in skills_rncp41966]
    matches = match_ia_recommendations(skills, recs)

    matched_skills = set(m.skill_original for m in matches)
    assert len(matched_skills) == len(skills_rncp41966)


# ---------------------------------------------------------------------------
# Real CSV integration (only if CSV file exists)
# ---------------------------------------------------------------------------

def test_real_csv_load() -> None:
    csv_path = Path("data/raw/recommandations_IA_consolide.csv")
    if not csv_path.exists():
        return
    records, report = load_ia_recommendations_csv(csv_path)
    assert report.valid_lines > 0
    assert report.rejected_lines == 0
    assert len(records) == report.valid_lines
    assert all(r.get("keyword") for r in records)
    assert all(r.get("recommendation") for r in records)
    assert all(r.get("keyword_normalized") for r in records)
