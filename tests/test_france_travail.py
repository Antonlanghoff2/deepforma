from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.analytics.territorial_skills import compute_territorial_stats
from src.france_travail.client import FranceTravailClient, FranceTravailError, FranceTravailRateLimitError, FranceTravailTimeoutError, SearchCriteria
from src.france_travail.normalizer import normalize_offer
from src.france_travail.skill_extractor import extract_structured_skills
from src.inference.skill_model import load_label_classes, load_thresholds
from src.jobs.collect_france_travail_offers import deduplicate_offers, save_jsonl
from src.skills.skill_normalizer import SkillNormalizer


class FakeResponse:
    def __init__(self, status_code=200, json_data=None, headers=None):
        self.status_code = status_code
        self._json_data = json_data if json_data is not None else {}
        self.headers = headers or {}

    def json(self):
        return self._json_data


class FakeSession:
    def __init__(self, post_responses=None, request_responses=None):
        self.post_responses = list(post_responses or [])
        self.request_responses = list(request_responses or [])
        self.post_calls = 0
        self.request_calls = 0
        self.last_request = None

    def post(self, *args, **kwargs):
        self.post_calls += 1
        if not self.post_responses:
            raise AssertionError("No post response queued")
        return self.post_responses.pop(0)

    def request(self, method, url, params=None, headers=None, timeout=None):
        self.request_calls += 1
        self.last_request = {"method": method, "url": url, "params": params, "headers": headers, "timeout": timeout}
        if not self.request_responses:
            raise AssertionError("No request response queued")
        return self.request_responses.pop(0)


def test_authentication_and_token_cache(monkeypatch):
    session = FakeSession(post_responses=[FakeResponse(json_data={"access_token": "tok-1", "expires_in": 3600})])
    client = FranceTravailClient(
        client_id="id",
        client_secret="secret",
        session=session,
        load_env=False,
    )
    token1 = client.authenticate()
    token2 = client.authenticate()
    assert token1 == token2 == "tok-1"
    assert session.post_calls == 1


def test_token_renewal(monkeypatch):
    session = FakeSession(
        post_responses=[
            FakeResponse(json_data={"access_token": "tok-1", "expires_in": 1}),
            FakeResponse(json_data={"access_token": "tok-2", "expires_in": 3600}),
        ]
    )
    client = FranceTravailClient(client_id="id", client_secret="secret", session=session, load_env=False)
    token1 = client.authenticate(force_refresh=True)
    client._token_expires_at = client._token_expires_at.replace(year=2000)
    token2 = client.authenticate()
    assert token1 == "tok-1"
    assert token2 == "tok-2"
    assert session.post_calls == 2


def test_search_offers_uses_range_and_ignores_none():
    session = FakeSession(
        post_responses=[FakeResponse(json_data={"access_token": "tok", "expires_in": 3600})],
        request_responses=[FakeResponse(json_data={"resultats": []}, headers={"Content-Range": "items 0-0/0"})],
    )
    client = FranceTravailClient(client_id="id", client_secret="secret", session=session, load_env=False)
    result = client.search_offers(
        SearchCriteria(keywords="python", rome_code=None, commune="93001", offset=0, size=20)
    )
    assert session.last_request["headers"]["Range"] == "items=0-19"
    assert "romeCode" not in session.last_request["params"]
    assert result.content_range == "items 0-0/0"


def test_error_401_raises(monkeypatch):
    session = FakeSession(
        post_responses=[
            FakeResponse(json_data={"access_token": "tok-1", "expires_in": 3600}),
            FakeResponse(json_data={"access_token": "tok-2", "expires_in": 3600}),
            FakeResponse(json_data={"access_token": "tok-3", "expires_in": 3600}),
        ],
        request_responses=[
            FakeResponse(status_code=401, json_data={}),
            FakeResponse(status_code=401, json_data={}),
            FakeResponse(status_code=401, json_data={}),
        ],
    )
    client = FranceTravailClient(client_id="id", client_secret="secret", session=session, load_env=False)
    with pytest.raises(FranceTravailError):
        client.get_offer("x")


def test_error_429_retries(monkeypatch):
    session = FakeSession(
        post_responses=[FakeResponse(json_data={"access_token": "tok", "expires_in": 3600})],
        request_responses=[
            FakeResponse(status_code=429, json_data={}),
            FakeResponse(status_code=200, json_data={"id": "1"}),
        ],
    )
    sleeps = []
    monkeypatch.setattr("src.france_travail.client.time.sleep", lambda s: sleeps.append(s))
    client = FranceTravailClient(client_id="id", client_secret="secret", session=session, load_env=False)
    payload = client.get_offer("1")
    assert payload["id"] == "1"
    assert sleeps == [1]


def test_error_429_final_raises_specific_error(monkeypatch):
    session = FakeSession(
        post_responses=[FakeResponse(json_data={"access_token": "tok", "expires_in": 3600})],
        request_responses=[
            FakeResponse(status_code=429, json_data={}),
            FakeResponse(status_code=429, json_data={}),
            FakeResponse(status_code=429, json_data={}),
        ],
    )
    sleeps = []
    monkeypatch.setattr("src.france_travail.client.time.sleep", lambda s: sleeps.append(s))
    client = FranceTravailClient(client_id="id", client_secret="secret", session=session, load_env=False)
    with pytest.raises(FranceTravailRateLimitError):
        client.get_offer("1")
    assert sleeps == [1, 2]


def test_timeout_is_wrapped(monkeypatch):
    class TimeoutSession(FakeSession):
        def request(self, *args, **kwargs):
            raise __import__('requests').Timeout('boom')

    session = TimeoutSession(post_responses=[FakeResponse(json_data={"access_token": "tok", "expires_in": 3600})])
    client = FranceTravailClient(client_id="id", client_secret="secret", session=session, load_env=False)
    with pytest.raises(FranceTravailTimeoutError):
        client.get_offer("1")


def test_pagination_and_deduplication():
    session = FakeSession(
        post_responses=[FakeResponse(json_data={"access_token": "tok", "expires_in": 3600})],
        request_responses=[
            FakeResponse(json_data={"resultats": [{"id": "1"}, {"id": "2"}]}),
            FakeResponse(json_data={"resultats": [{"id": "2"}, {"id": "3"}]}),
        ],
    )
    client = FranceTravailClient(client_id="id", client_secret="secret", session=session, load_env=False)
    offers = list(client.iter_offers(max_pages=2, page_size=2))
    assert [offer["id"] for offer in offers] == ["1", "2", "3"]


def test_structured_skill_extraction_and_missing_property():
    assert extract_structured_skills({}) == []
    skills = extract_structured_skills(
        {
            "competences": [
                {"code": "S1", "label": "Python", "requirement": "souhaité"},
                {"label": "Machine Learning"},
                "",
            ]
        }
    )
    assert len(skills) == 2
    assert skills[0]["source"] == "france_travail_structured"


def test_skill_normalizer_alignment():
    normalizer = SkillNormalizer()
    label, confidence, skill_id = normalizer.normalize("project management")
    assert label == "Gestion de projet"
    assert confidence == 1.0
    assert skill_id == "gestion_de_projet"


def test_normalize_offer_and_stats():
    raw = {
        "id": "1",
        "title": "<b>Python</b> pour data",
        "description": "Découverte de Python",
        "competences": [{"label": "Python"}],
        "rome": {"code": "M1805", "libelle": "Études et développement informatique"},
        "lieuTravail": {"libelle": "Seine-Saint-Denis", "codePostal": "93000"},
    }
    normalized = normalize_offer(raw)
    assert normalized.offer_id == "1"
    assert normalized.structured_skills[0]["label"] == "Python"
    stats = compute_territorial_stats([normalized.to_dict()], "93")
    assert stats.offer_count == 1
    assert "Python" in stats.skill_counts


def test_save_jsonl(tmp_path):
    path = tmp_path / "offers.jsonl"
    save_jsonl(path, [{"a": 1}, {"b": 2}])
    assert path.exists()
    content = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(content) == 2


def test_no_secret_in_logs(caplog):
    session = FakeSession(post_responses=[FakeResponse(json_data={"access_token": "tok", "expires_in": 3600})])
    caplog.set_level(logging.INFO)
    client = FranceTravailClient(client_id="id", client_secret="super-secret", session=session, load_env=False)
    client.authenticate()
    assert "super-secret" not in caplog.text

