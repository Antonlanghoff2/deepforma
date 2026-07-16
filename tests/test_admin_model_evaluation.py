from __future__ import annotations

import base64
import json
from types import SimpleNamespace

import pytest

from web_app import _normalize_per_label_rows, create_app


def _auth_headers(username: str = "anton", password: str = "deepforma") -> dict[str, str]:
    token = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
    return {"Authorization": f"Basic {token}"}


@pytest.mark.parametrize(
    ("per_label", "expected"),
    [
        (
            {
                "Machine Learning": {
                    "precision": 0.8,
                    "recall": 0.7,
                    "f1": 0.74,
                }
            },
            [
                {
                    "label": "Machine Learning",
                    "precision": 0.8,
                    "recall": 0.7,
                    "f1": 0.74,
                }
            ],
        ),
        (
            [
                {"label": "Machine Learning", "precision": 0.8},
                {"label": "Vision", "precision": 0.7},
            ],
            [
                {"label": "Machine Learning", "precision": 0.8},
                {"label": "Vision", "precision": 0.7},
            ],
        ),
        (
            [SimpleNamespace(label="Machine Learning", precision=0.8), SimpleNamespace(label="Vision", precision=0.7)],
            [
                {"label": "Machine Learning", "precision": 0.8},
                {"label": "Vision", "precision": 0.7},
            ],
        ),
        (
            ["Machine Learning", "Vision"],
            [{"label": "Machine Learning"}, {"label": "Vision"}],
        ),
        (
            {"Machine Learning": 0.8},
            [{"label": "Machine Learning", "value": 0.8}],
        ),
        (None, []),
        ([], []),
        ({}, []),
    ],
)
def test_normalize_per_label_rows(per_label, expected):
    assert _normalize_per_label_rows(per_label) == expected


def test_admin_model_evaluation_reads_artifacts(monkeypatch, tmp_path):
    import web_app as web_app_module

    monkeypatch.setenv("DEEPFORMA_ADMIN_USER", "anton")
    monkeypatch.setenv("DEEPFORMA_ADMIN_PASSWORD", "deepforma")
    monkeypatch.setattr(web_app_module, "PROJECT_ROOT", tmp_path)

    binary_run = tmp_path / "artifacts" / "evaluations" / "binary_ia" / "20260714T120000Z"
    binary_run.mkdir(parents=True)
    (binary_run / "report.json").write_text(json.dumps({
        "task": "binary",
        "generated_at": "2026-07-14T12:00:00+00:00",
        "metrics": {"accuracy": 1.0, "loss": 0.12, "f1": 1.0},
        "warnings": [],
        "per_label": {
            "Machine Learning": {
                "precision": 0.8,
                "recall": 0.7,
                "f1": 0.74,
            }
        },
        "thresholds": {"positive": 0.5},
    }, ensure_ascii=False), encoding="utf-8")
    (binary_run / "summary.csv").write_text("accuracy,loss,f1\n1.0,0.12,1.0\n", encoding="utf-8")
    (binary_run / "report.html").write_text("<html><body>ok</body></html>", encoding="utf-8")

    ml_run = tmp_path / "artifacts" / "evaluations" / "ia_multilabel" / "20260714T120500Z"
    ml_run.mkdir(parents=True)
    (ml_run / "report.json").write_text(json.dumps({
        "task": "multilabel",
        "generated_at": "2026-07-14T12:05:00+00:00",
        "metrics": {"micro_f1": 0.82, "loss": 0.31, "macro_f1": 0.79},
        "warnings": [],
        "per_label": [
            {"label": "Vision", "f1": 0.8},
        ],
    }, ensure_ascii=False), encoding="utf-8")
    (ml_run / "summary.csv").write_text("micro_f1,loss,macro_f1\n0.82,0.31,0.79\n", encoding="utf-8")
    (ml_run / "report.html").write_text("<html><body>ok</body></html>", encoding="utf-8")

    app = create_app(predictor=object(), france_travail_client_factory=lambda: object())
    app.testing = True
    client = app.test_client()
    response = client.get('/admin/model-evaluation', headers=_auth_headers())
    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert 'Aperçu rapide des modèles' in body
    assert 'binary_ia' in body
    assert 'ia_multilabel' in body
    assert 'accuracy' in body
    assert 'loss' in body
    assert 'report.json' in body
    assert 'Machine Learning' in body
