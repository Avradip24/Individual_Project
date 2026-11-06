"""Tests for the demo API. Unit tests use a stub predictor (no model files needed);
the integration test runs the real models and is skipped when `models/` is not present."""
# pylint: disable=import-error,missing-function-docstring,redefined-outer-name
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.demo import api

LABELS = ["Education", "Entertainment", "Groceries", "Healthcare", "Housing", "Income",
          "Insurance", "Other", "Restaurants", "Shopping", "Transportation", "Travel",
          "Utilities"]


def make_stub(seen):
    def stub_loader(_name):
        def predict(text):
            seen.append(text)
            p = np.full(len(LABELS), 0.01)
            p[LABELS.index("Groceries")] = 0.7
            p[LABELS.index("Restaurants")] = 0.1
            p[LABELS.index("Shopping")] = 0.05
            return p / p.sum()
        return predict
    return stub_loader


@pytest.fixture
def client(monkeypatch):
    seen = []
    monkeypatch.setattr(api, "get_predictor", make_stub(seen))
    monkeypatch.setattr(api, "get_labels", lambda: LABELS)
    c = TestClient(api.app)
    c.seen = seen
    return c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_models_lists_both(client):
    assert set(client.get("/models").json()) == {"distilbert", "logreg"}


def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "SmartSpend" in r.text


def test_predict_returns_sorted_top_k(client):
    r = client.post("/predict", json={"text": "REWE SAG 7342", "top_k": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["category"] == "Groceries"
    probs = [c["probability"] for c in body["top"]]
    assert len(probs) == 3
    assert probs == sorted(probs, reverse=True)
    assert body["confidence"] == probs[0]
    assert body["latency_ms"] >= 0


def test_direction_prefix_is_added(client):
    client.post(
        "/predict", json={"text": "Zelle payment", "direction": "credit"})
    assert client.seen[-1] == "[credit] Zelle payment"


def test_existing_tag_is_not_duplicated(client):
    client.post(
        "/predict", json={"text": "[debit] REWE", "direction": "credit"})
    assert client.seen[-1] == "[debit] REWE"


def test_whitespace_is_collapsed(client):
    client.post("/predict", json={"text": "  REWE    SAG   "})
    assert client.seen[-1] == "REWE SAG"


@pytest.mark.parametrize("payload", [
    {"text": ""},
    {"text": "   "},
    {"text": "x" * 201},
    {"text": "REWE", "model": "xgboost"},
    {"text": "REWE", "top_k": 0},
    {"text": "REWE", "top_k": 6},
    {"text": "REWE", "direction": "sideways"},
    {},
])
def test_invalid_input_is_rejected(client, payload):
    assert client.post("/predict", json=payload).status_code == 422


def test_missing_model_files_give_503(monkeypatch):
    def broken(_name):
        raise FileNotFoundError("no such model")
    monkeypatch.setattr(api, "get_predictor", broken)
    r = TestClient(api.app).post("/predict", json={"text": "REWE"})
    assert r.status_code == 503
    assert "could not be loaded" in r.json()["detail"]


def test_preload_warms_every_model(monkeypatch):
    seen = []
    monkeypatch.setenv("PRELOAD_MODELS", "1")
    monkeypatch.setattr(api, "get_predictor", make_stub(seen))
    with TestClient(api.app):
        pass
    assert len(seen) == 2


def test_preload_is_off_by_default(monkeypatch):
    seen = []
    monkeypatch.delenv("PRELOAD_MODELS", raising=False)
    monkeypatch.setattr(api, "get_predictor", make_stub(seen))
    with TestClient(api.app):
        pass
    assert not seen


def test_preload_failure_does_not_stop_startup(monkeypatch):
    def broken(_name):
        raise FileNotFoundError("no such model")
    monkeypatch.setenv("PRELOAD_MODELS", "1")
    monkeypatch.setattr(api, "get_predictor", broken)
    with TestClient(api.app) as c:
        assert c.get("/health").status_code == 200


MODELS_PRESENT = (Path(api.MODELS_DIR) / "distilbert_finetuned" / "final").exists() and (
    Path(api.MODELS_DIR) / "classical_v2" / "logistic_regression.joblib").exists()


@pytest.mark.skipif(not MODELS_PRESENT, reason="model files not present in MODELS_DIR")
@pytest.mark.parametrize("model", ["distilbert", "logreg"])
def test_real_models_return_valid_probabilities(model):
    r = TestClient(api.app).post(
        "/predict", json={"text": "[debit] Starbucks #1234", "model": model, "top_k": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["category"] in api.get_labels()
    assert 0 < body["confidence"] <= 1
    assert sum(c["probability"] for c in body["top"]) <= 1.0001
