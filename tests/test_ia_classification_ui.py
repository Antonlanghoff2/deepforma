"""Tests pour l'interface de classification IA (non-duplication, métadonnées, arrondi)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from flask import Flask

from models.analysis_result import (
    AnalysisResult,
    IAClassificationInfo,
    ModelMetadata,
)
from web_app import create_app


# ---------------------------------------------------------------------------
#  Helpers: render result.html with controlled data via test_request_context
# ---------------------------------------------------------------------------

TEMPLATE_DIR = Path(__file__).resolve().parents[1] / "templates"


def _make_families(n_skills: int) -> list[dict]:
    families = [
        {
            "family_id": "ml",
            "family_label": "Machine Learning",
            "skills": [
                {"skill_id": f"ml.{i}", "label": f"ML Skill {i}",
                 "probability": round(0.3 + (i * 0.01) % 0.6, 4),
                 "above_threshold": False}
                for i in range(min(n_skills, 5))
            ],
        },
        {
            "family_id": "nlp",
            "family_label": "NLP",
            "skills": [
                {"skill_id": f"nlp.{i}", "label": f"NLP Skill {i}",
                 "probability": round(0.4 + (i * 0.01) % 0.5, 4),
                 "above_threshold": False}
                for i in range(min(max(n_skills - 5, 0), 5))
            ],
        },
    ]
    if n_skills > 10:
        families.append({
            "family_id": "cv",
            "family_label": "Computer Vision",
            "skills": [
                {"skill_id": f"cv.{i}", "label": f"CV Skill {i}",
                 "probability": round(0.2 + (i * 0.005) % 0.7, 4),
                 "above_threshold": False}
                for i in range(n_skills - 10)
            ],
        })
    return [f for f in families if f["skills"]]


def _make_scores(n: int) -> list[float]:
    return [round(0.3 + (i * 0.02) % 0.6, 4) for i in range(n)]


def _make_labels(n: int) -> list[str]:
    return [f"Label_{i}" for i in range(n)]


def _make_ia_info(
    n_labels: int = 18,
    status: str = "unreliable",
    discriminating: bool = False,
) -> IAClassificationInfo:
    scores = _make_scores(n_labels)
    labels = _make_labels(n_labels)
    return IAClassificationInfo(
        status=status,
        categories=[
            {"label": labels[i], "probability": scores[i]}
            for i in range(n_labels) if scores[i] >= 0.35
        ],
        families=_make_families(n_labels),
        scores=scores,
        score_min=min(scores),
        score_max=max(scores),
        score_mean=sum(scores) / len(scores),
        score_std=0.08,
        discriminating=discriminating,
        threshold_applied=0.35,
    )


def _make_result(
    n_labels: int = 18,
    status: str = "unreliable",
    discriminating: bool = False,
    taxonomy_version: str = "1.0",
) -> AnalysisResult:
    result = AnalysisResult()
    result.ia_classification = _make_ia_info(n_labels, status, discriminating)
    result.model_metadata = ModelMetadata(
        model_name=f"Classifieur IA v{taxonomy_version}" if taxonomy_version else "Classifieur IA",
        taxonomy_version=taxonomy_version,
        validation_status="non validé",
        num_labels=n_labels,
        labels=_make_labels(n_labels),
        thresholds={"multilabel": 0.35, "binary": None},
    )
    result.summary = {
        "total_skills_extracted": 5,
        "total_tools_detected": 2,
        "total_offers_analyzed": 50,
        "inference_time_ms": 150.0,
    }
    return result


def _render_result(
    app: Flask,
    n_labels: int = 18,
    status: str = "unreliable",
    discriminating: bool = False,
    taxonomy_version: str = "1.0",
) -> str:
    """Render result.html inside a test request context so url_for works."""
    from flask import render_template
    result = _make_result(
        n_labels=n_labels,
        status=status,
        discriminating=discriminating,
        taxonomy_version=taxonomy_version,
    )
    with app.test_request_context():
        return render_template(
            "result.html",
            result_dict=result.to_dict(),
            context={"market_status": "unavailable"},
            model_only=True,
        )


@pytest.fixture
def app():
    return create_app()


# ---------------------------------------------------------------------------
#  Tests
# ---------------------------------------------------------------------------

class TestNoDuplication:
    """Vérifie qu'il n'y a pas de listes de labels dupliquées dans le HTML."""

    def _count_chip_labels(self, html_section: str) -> list[str]:
        return re.findall(
            r'<div class="chip[^"]*"[^>]*>\s*<span>([^<]+)</span>',
            html_section,
        )

    def test_single_score_list_render(self, app):
        """Un seul bloc de scores techniques — pas de duplication."""
        html = _render_result(app, n_labels=18, status="unreliable")
        assert "Scores bruts" not in html, "Terme 'Scores bruts' encore présent"
        assert "Scores detailles" not in html, "Terme 'Scores detailles' encore présent"
        score_sections = re.findall(r'Scores techniques du classifieur IA', html)
        assert len(score_sections) == 1, f"Attendu 1 bloc scores, trouvé {len(score_sections)}"

    def test_no_duplicate_labels_in_html(self, app):
        """Aucun label ne doit apparaître deux fois dans la section IA."""
        html = _render_result(app, n_labels=18, status="unreliable")
        ia_section = _extract_ia_section(html)
        labels = self._count_chip_labels(ia_section)
        assert len(labels) == len(set(labels)), f"Labels dupliqués: {labels}"


def _extract_ia_section(html: str) -> str:
    parts = html.split("Categorisation IA")
    if len(parts) < 2:
        return ""
    section = parts[1]
    parts2 = section.split("Fiabilite et methodologie")
    return parts2[0] if len(parts2) > 1 else section


class TestDynamicMetadata:
    """Vérifie que les métadonnées sont dynamiques."""

    def test_model_name_dynamic(self, app):
        html = _render_result(app, taxonomy_version="1.5")
        assert "Classifieur IA v1.5" in html

    def test_taxonomy_version_displayed(self, app):
        html = _render_result(app, taxonomy_version="2.0")
        assert "2.0" in html

    def test_num_labels_displayed_18(self, app):
        html = _render_result(app, n_labels=18)
        assert "Nombre de labels" in html

    def test_num_labels_displayed_80(self, app):
        html = _render_result(app, n_labels=80)
        assert "Nombre de labels" in html

    def test_validation_status_displayed(self, app):
        html = _render_result(app)
        assert "non validé" in html


class TestCollapsibleBehavior:
    """Vérifie l'état replié/déplié du bloc technique."""

    def test_collapsed_when_unreliable(self, app):
        """Bloc technique replié par défaut si status == 'unreliable'."""
        html = _render_result(app, status="unreliable", discriminating=False)
        assert "Classification IA non disponible" in html
        assert "Scores techniques" in html

    def test_families_rendered_when_success(self, app):
        """Familles affichées si le modèle discrimine."""
        html = _render_result(app, status="success", discriminating=True, n_labels=10)
        assert "Machine Learning" in html
        assert "NLP" in html


class TestRoundingConsistency:
    """Vérifie que l'arrondi est cohérent via sigmoid_pct."""

    def test_sigmoid_pct_filter(self, app):
        """Test du filtre sigmoid_pct directement."""
        from flask import render_template_string
        with app.test_request_context():
            result = render_template_string("{{ 0.5234|sigmoid_pct }}")
            assert result == "52.3%"
            result = render_template_string("{{ 0.9999|sigmoid_pct }}")
            assert result == "100.0%"
            result = render_template_string("{{ 0.0001|sigmoid_pct }}")
            assert result == "0.0%"


class TestTerminology:
    """Vérifie la terminologie correcte."""

    def test_no_scores_bruts(self, app):
        html = _render_result(app)
        assert "scores bruts" not in html.lower()
        assert "scores detailles" not in html.lower()

    def test_scores_techniques_present(self, app):
        html = _render_result(app)
        assert "Scores techniques" in html


class TestModelNameNotHardcoded:
    """Vérifie que 'modèle 18 labels' n'est plus en dur."""

    def test_no_18_labels_hardcoded_in_ia_message(self, app):
        """Le message d'alerte ne doit pas contenir '18 labels' en dur."""
        html = _render_result(app, status="unreliable")
        assert "18 labels" not in html


class TestManyLabels:
    """Vérifie le rendu avec un grand nombre de labels."""

    def test_80_labels_no_family_crash(self, app):
        html = _render_result(app, n_labels=80)
        assert "Rechercher un label" in html or "Tous les labels" in html

    def test_120_labels_no_family_crash(self, app):
        html = _render_result(app, n_labels=120)
        assert "Rechercher un label" in html or "Tous les labels" in html
