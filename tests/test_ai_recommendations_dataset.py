from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np

from ai_recommendations import (
    AIRecommendationSourceScore,
    build_or_load_index,
    fuse_ai_recommendation_scores,
    import_ai_recommendation_dataset,
    load_ai_recommendation_rules,
    load_ai_recommendation_rules_csv,
    load_index,
    match_ai_recommendations,
    normalize_ai_keyword,
    rebuild_index,
)


def _auth_headers(username: str = "anton", password: str = "deepforma") -> dict[str, str]:
    token = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
    return {"Authorization": f"Basic {token}"}


def _write_raw_csv(path: Path) -> None:
    lines = [
        "﻿mot clé dans le référentiel,recommandation IA",
        "(aucune mention IA dans le référentiel),Règle par défaut : si le référentiel ne mentionne ni IA ni intelligence artificielle",
        "présentation,Créer des présentations assistées par IA",
        '"veille informationnelle,Automatiser la veille avec Perplexity ou Feedly AI " , créer une veille automatisée récurrente avec les routines planifiées',
        "vente,Gestion de l'inventaire",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_normalize_ai_keyword_handles_accents_and_quotes() -> None:
    assert normalize_ai_keyword("  Véille informationnelle  ") == "veille informationnelle"
    assert normalize_ai_keyword('"Présentation"') == "presentation"


def test_load_csv_utf8_bom_and_corrects_misaligned_row(tmp_path):
    csv_path = tmp_path / "dataset.csv"
    _write_raw_csv(csv_path)
    rules, report, review_rows = load_ai_recommendation_rules_csv(csv_path)
    assert report.total_lines == 5
    assert len(rules) == 4
    corrected = next(rule for rule in rules if rule["keyword"] == "veille informationnelle")
    assert "Automatiser la veille" in corrected["recommendation"]
    assert report.review_lines >= 1
    assert review_rows
    assert any(row["anomaly_type"] == "misaligned_keyword_fragment" for row in review_rows)


def test_import_dataset_writes_outputs(tmp_path):
    input_path = tmp_path / "dataset.csv"
    _write_raw_csv(input_path)
    output_csv = tmp_path / "rules.csv"
    output_json = tmp_path / "rules.json"
    review_csv = tmp_path / "review.csv"
    result = import_ai_recommendation_dataset(input_path, output_csv, output_json, review_csv)
    assert result["rules_count"] == 4
    assert output_csv.exists()
    assert output_json.exists()
    assert review_csv.exists()
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["rules"]
    assert payload["rules"][0]["normalized_keyword"]


def test_load_rules_json_roundtrip(tmp_path):
    rules = [{
        "id": "airule-1",
        "keyword": "veille informationnelle",
        "normalized_keyword": "veille informationnelle",
        "categories": [],
        "recommendation": "Créer une veille automatisée",
        "match_type": "hybrid",
        "priority": 50,
        "enabled": True,
        "is_default": False,
        "source": "dataset_recommandations_IA_complet.csv",
        "source_line": 2,
        "review_status": "accepted",
        "metadata": {},
    }]
    path = tmp_path / "rules.json"
    path.write_text(json.dumps({"rules": rules}, ensure_ascii=False), encoding="utf-8")
    loaded = load_ai_recommendation_rules(path)
    assert loaded[0]["keyword"] == "veille informationnelle"


def test_matching_exact_retrieves_recommendation(tmp_path):
    rules = [{
        "id": "airule-1",
        "keyword": "veille informationnelle",
        "normalized_keyword": "veille informationnelle",
        "categories": [{"label": "IA générative", "score": 1.0, "method": "exact", "status": "accepted"}],
        "recommendation": "Automatiser la veille avec Perplexity ou Feedly AI",
        "match_type": "hybrid",
        "priority": 50,
        "enabled": True,
        "is_default": False,
        "source": "dataset_recommandations_IA_complet.csv",
        "source_line": 2,
        "review_status": "accepted",
        "metadata": {},
    }, {
        "id": "airule-default",
        "keyword": "(aucune mention IA dans le référentiel)",
        "normalized_keyword": "(aucune mention ia dans le referentiel)",
        "categories": [],
        "recommendation": "Acculturation à l'IA et découverte de l'IA agentique",
        "match_type": "hybrid",
        "priority": 50,
        "enabled": True,
        "is_default": True,
        "source": "dataset_recommandations_IA_complet.csv",
        "source_line": 1,
        "review_status": "accepted",
        "metadata": {},
    }]
    rules_path = tmp_path / "rules.json"
    rules_path.write_text(json.dumps({"rules": rules}, ensure_ascii=False), encoding="utf-8")
    result = match_ai_recommendations(
        referential_title="Le titulaire réalise une veille informationnelle sur son secteur.",
        full_text="Le titulaire réalise une veille informationnelle sur son secteur.",
        rules_path=rules_path,
    )
    assert result["default_recommendation_applied"] is False
    assert result["recommendations"]
    assert result["recommendations"][0]["match_method"] == "exact"
    assert result["recommendations"][0]["keyword"] == "veille informationnelle"


def test_matching_negative_no_substring_false_positive(tmp_path):
    rules = [{
        "id": "airule-1",
        "keyword": "vente",
        "normalized_keyword": "vente",
        "categories": [],
        "recommendation": "Recommandation vente",
        "match_type": "hybrid",
        "priority": 50,
        "enabled": True,
        "is_default": False,
        "source": "dataset_recommandations_IA_complet.csv",
        "source_line": 2,
        "review_status": "accepted",
        "metadata": {},
    }]
    rules_path = tmp_path / "rules.json"
    rules_path.write_text(json.dumps({"rules": rules}, ensure_ascii=False), encoding="utf-8")
    result = match_ai_recommendations(
        referential_title="Gestion de l’inventaire",
        full_text="Gestion de l’inventaire",
        rules_path=rules_path,
    )
    assert result["recommendations"] == []
    assert result["detected_categories"] == []


def test_matching_semantic_with_fake_encoder(monkeypatch, tmp_path):
    rules = [{
        "id": "airule-1",
        "keyword": "veille informationnelle",
        "normalized_keyword": "veille informationnelle",
        "categories": [{"label": "Prompt Engineering", "score": 0.9, "method": "semantic", "status": "to_review"}],
        "recommendation": "Automatiser la veille avec Perplexity ou Feedly AI",
        "match_type": "hybrid",
        "priority": 50,
        "enabled": True,
        "is_default": False,
        "source": "dataset_recommandations_IA_complet.csv",
        "source_line": 2,
        "review_status": "accepted",
        "metadata": {},
    }]
    rules_path = tmp_path / "rules.json"
    rules_path.write_text(json.dumps({"rules": rules}, ensure_ascii=False), encoding="utf-8")
    model_dir = tmp_path / "fake-model"
    model_dir.mkdir()

    class FakeEncoder:
        def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True, **kwargs):
            vectors = []
            for text in texts:
                normalized = text.lower()
                if any(token in normalized for token in ("veille informationnelle", "veille", "surveiller", "actualité", "actualite")):
                    vectors.append(np.array([1.0, 0.0], dtype=float))
                else:
                    vectors.append(np.array([0.0, 1.0], dtype=float))
            return np.vstack(vectors)

    monkeypatch.setattr("deepforma.cpf.embeddings.build_encoder", lambda *_args, **_kwargs: FakeEncoder())
    result = match_ai_recommendations(
        referential_title="Surveiller l'actualité du secteur",
        full_text="Surveiller l'actualité du secteur",
        rules_path=rules_path,
        embedding_model=model_dir,
    )
    assert result["recommendations"]
    assert result["recommendations"][0]["match_method"] in {"semantic", "exact", "lexical"}


def test_fusion_reduces_non_discriminant_model_weight() -> None:
    fused = fuse_ai_recommendation_scores(
        [
            AIRecommendationSourceScore(source="exact_rule", score=1.0),
            AIRecommendationSourceScore(source="multilabel_model", score=0.51),
        ],
        model_score_std=0.01,
        model_mean_score=0.5,
        model_non_discriminant=True,
    )
    assert fused["weights"]["multilabel_model"] == 0.0
    assert fused["score"] == 1.0


def test_default_rule_applies_only_when_no_specific_match(tmp_path):
    rules = [{
        "id": "airule-default",
        "keyword": "(aucune mention IA dans le référentiel)",
        "normalized_keyword": "(aucune mention ia dans le referentiel)",
        "categories": [],
        "recommendation": "Acculturation à l'IA et découverte de l'IA agentique",
        "match_type": "hybrid",
        "priority": 50,
        "enabled": True,
        "is_default": True,
        "source": "dataset_recommandations_IA_complet.csv",
        "source_line": 1,
        "review_status": "accepted",
        "metadata": {},
    }]
    rules_path = tmp_path / "rules.json"
    rules_path.write_text(json.dumps({"rules": rules}, ensure_ascii=False), encoding="utf-8")
    result = match_ai_recommendations(full_text="Aucun contenu IA explicite", rules_path=rules_path)
    assert result["default_recommendation_applied"] is True
    assert result["recommendations"][0]["match_method"] == "default"


def test_default_rule_not_used_when_specific_rule_exists(tmp_path):
    rules = [{
        "id": "airule-default",
        "keyword": "(aucune mention IA dans le référentiel)",
        "normalized_keyword": "(aucune mention ia dans le referentiel)",
        "categories": [],
        "recommendation": "Acculturation à l'IA et découverte de l'IA agentique",
        "match_type": "hybrid",
        "priority": 50,
        "enabled": True,
        "is_default": True,
        "source": "dataset_recommandations_IA_complet.csv",
        "source_line": 1,
        "review_status": "accepted",
        "metadata": {},
    }, {
        "id": "airule-veille",
        "keyword": "veille informationnelle",
        "normalized_keyword": "veille informationnelle",
        "categories": [],
        "recommendation": "Automatiser la veille",
        "match_type": "hybrid",
        "priority": 50,
        "enabled": True,
        "is_default": False,
        "source": "dataset_recommandations_IA_complet.csv",
        "source_line": 2,
        "review_status": "accepted",
        "metadata": {},
    }]
    rules_path = tmp_path / "rules.json"
    rules_path.write_text(json.dumps({"rules": rules}, ensure_ascii=False), encoding="utf-8")
    result = match_ai_recommendations(full_text="veille informationnelle", rules_path=rules_path)
    assert result["default_recommendation_applied"] is False
    assert any(rec["keyword"] == "veille informationnelle" for rec in result["recommendations"])


def test_disabled_rule_is_ignored(tmp_path):
    rules = [{
        "id": "airule-1",
        "keyword": "veille informationnelle",
        "normalized_keyword": "veille informationnelle",
        "categories": [],
        "recommendation": "Automatiser la veille",
        "match_type": "hybrid",
        "priority": 50,
        "enabled": False,
        "is_default": False,
        "source": "dataset_recommandations_IA_complet.csv",
        "source_line": 2,
        "review_status": "accepted",
        "metadata": {},
    }]
    rules_path = tmp_path / "rules.json"
    rules_path.write_text(json.dumps({"rules": rules}, ensure_ascii=False), encoding="utf-8")
    result = match_ai_recommendations(full_text="veille informationnelle", rules_path=rules_path)
    assert result["recommendations"] == []


def test_index_rebuild_and_load(tmp_path, monkeypatch):
    rules = [{
        "id": "airule-1",
        "keyword": "veille informationnelle",
        "normalized_keyword": "veille informationnelle",
        "categories": [],
        "recommendation": "Automatiser la veille",
        "match_type": "hybrid",
        "priority": 50,
        "enabled": True,
        "is_default": False,
        "source": "dataset_recommandations_IA_complet.csv",
        "source_line": 2,
        "review_status": "accepted",
        "metadata": {},
    }]
    rules_path = tmp_path / "rules.json"
    rules_path.write_text(json.dumps({"rules": rules}, ensure_ascii=False), encoding="utf-8")
    index_dir = tmp_path / "index"
    model_dir = tmp_path / "fake-model"
    model_dir.mkdir()

    class FakeEncoder:
        def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True, **kwargs):
            return np.vstack([np.array([1.0, 0.0], dtype=float) for _ in texts])

    monkeypatch.setattr("ai_recommendations.semantic_index.build_encoder", lambda *_args, **_kwargs: FakeEncoder())
    index = rebuild_index(rules_path, index_dir=index_dir, embedding_model=model_dir)
    assert index.rule_ids == ["airule-1"]
    loaded = load_index(index_dir)
    assert loaded is not None
    rebuilt = build_or_load_index(rules_path, index_dir=index_dir, embedding_model=model_dir)
    assert rebuilt.rules_hash == index.rules_hash


def test_admin_route(monkeypatch, tmp_path):
    import web_app as web_app_module
    from web_app import create_app

    monkeypatch.setenv("DEEPFORMA_ADMIN_USER", "anton")
    monkeypatch.setenv("DEEPFORMA_ADMIN_PASSWORD", "deepforma")
    monkeypatch.setattr(web_app_module, "PROJECT_ROOT", tmp_path)

    rules_dir = tmp_path / "data" / "referentials"
    rules_dir.mkdir(parents=True)
    (rules_dir / "ai_recommendation_rules.json").write_text(json.dumps({"rules": [{
        "id": "airule-1",
        "keyword": "veille informationnelle",
        "normalized_keyword": "veille informationnelle",
        "categories": [{"label": "IA générative", "score": 1.0, "method": "exact", "status": "accepted"}],
        "recommendation": "Automatiser la veille",
        "match_type": "hybrid",
        "priority": 50,
        "enabled": True,
        "is_default": False,
        "source": "dataset_recommandations_IA_complet.csv",
        "source_line": 2,
        "review_status": "accepted",
        "metadata": {"anomalies": []},
    }]}, ensure_ascii=False), encoding="utf-8")

    app = create_app(predictor=object(), france_travail_client_factory=lambda: object())
    app.testing = True
    client = app.test_client()
    response = client.get('/admin/ai-recommendation-rules', headers=_auth_headers())
    assert response.status_code == 200
    assert 'veille informationnelle' in response.get_data(as_text=True)
