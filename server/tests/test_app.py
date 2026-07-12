"""API tests for the /transliterate service.

Uses a stub engine and a tiny in-memory dictionary so no model is loaded, so
these run fast and in CI. Model-backed behavior is covered in test_engines.py.
"""

import json
from typing import List

import pytest
from fastapi.testclient import TestClient

from server import app as app_module


class StubEngine:
    """Deterministic fake engine returning `topk` synthetic candidates."""

    def transliterate(self, word: str, topk: int = 5) -> List[str]:
        return [f"{word}-{i}" for i in range(topk)]

    def transliterate_batch(self, words, topk=5):
        return [self.transliterate(w, topk) for w in words]


@pytest.fixture
def client(tmp_path, monkeypatch):
    dict_path = tmp_path / "dict.json"
    dict_path.write_text(json.dumps({"mera": ["मेरा", "मीरा", "मैरा", "मेर", "मरा"]}),
                         encoding="utf-8")
    monkeypatch.setattr(app_module.settings, "dict_path", str(dict_path))
    monkeypatch.setattr(app_module.settings, "lru_cache_size", 10)
    monkeypatch.setattr(app_module.settings, "lang", "hi")
    monkeypatch.setattr(app_module.settings, "topk", 5)
    monkeypatch.setattr(app_module, "_build_engine", lambda: StubEngine())
    with TestClient(app_module.app) as c:
        yield c


def test_dictionary_hit(client):
    r = client.get("/transliterate", params={"word": "mera"})
    assert r.status_code == 200
    body = r.json()
    assert body["source"] == "dict"
    assert body["candidates"][0] == "मेरा"
    assert "latency_ms" in body


def test_model_then_cache(client):
    r1 = client.get("/transliterate", params={"word": "zzq"})
    assert r1.json()["source"] == "model"
    r2 = client.get("/transliterate", params={"word": "zzq"})
    assert r2.json()["source"] == "cache"


def test_topk_no_cache_contamination(client):
    """The regression: a small-topk request must not poison a later larger one."""
    small = client.get("/transliterate", params={"word": "novel", "topk": 1})
    assert len(small.json()["candidates"]) == 1
    big = client.get("/transliterate", params={"word": "novel", "topk": 5})
    assert len(big.json()["candidates"]) == 5   # full list, not the cached 1


def test_topk_above_max_is_422(client):
    assert client.get("/transliterate", params={"word": "mera", "topk": 6}).status_code == 422


def test_unsupported_lang_is_422(client):
    assert client.get("/transliterate", params={"word": "mera", "lang": "ta"}).status_code == 422
    assert client.get("/transliterate", params={"word": "mera", "lang": "hi"}).status_code == 200


def test_non_latin_word_is_422(client):
    assert client.get("/transliterate", params={"word": "me3ra"}).status_code == 422
    assert client.get("/transliterate", params={"word": "मेरा"}).status_code == 422


def test_overlong_word_is_422(client):
    assert client.get("/transliterate", params={"word": "a" * 65}).status_code == 422


def test_healthz(client):
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["dict_size"] == 1


def test_metrics_shape(client):
    client.get("/transliterate", params={"word": "mera"})
    body = client.get("/metrics").json()
    assert "counts" in body and "hit_ratio" in body and "latency_ms" in body
