"""
SmartSpend demonstrator -- a minimal FastAPI service around two of the trained classifiers.

    GET  /          the web interface (src/demo/static/index.html)
    GET  /health    liveness plus which models are loaded in memory
    GET  /models    the models the service can serve
    POST /predict   classify ONE transaction description

Models are loaded lazily on first use, so the service starts instantly and a missing
model file only affects the model that needs it (HTTP 503), not the whole service.
Model files are NOT part of the repository: they are read from MODELS_DIR (default
`models/`, `/models` in the Docker image) and must be present there; see the README.

Run locally from the project root:
    uv run uvicorn src.demo.api:app --reload
"""
# Heavy libraries (torch, transformers, joblib) are imported inside the loaders on purpose,
# so the service starts fast and a missing package only affects the model that needs it.
# pylint: disable=import-outside-toplevel
from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

MODELS_DIR = Path(os.environ.get("MODELS_DIR", "models"))
STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_LENGTH = 32  # tokens; the same limit the transformer was trained and evaluated with

MODEL_INFO = {
    "distilbert": {
        "name": "Fine-tuned DistilBERT",
        "description": "Most accurate model in the study (OOD macro F1 0.933). Slower on CPU.",
    },
    "logreg": {
        "name": "Logistic Regression (TF-IDF + hand-crafted features)",
        "description": "Best classical model (OOD macro F1 0.881). Much cheaper to run.",
    },
}

# text -> probability per class (alphabetical order)
Predictor = Callable[[str], np.ndarray]


# --------------------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------------------
_LABELS: list[str] | None = None


def get_labels() -> list[str]:
    """Class names in the order the models output them (alphabetical)."""
    global _LABELS  # pylint: disable=global-statement
    if _LABELS is None:
        _LABELS = json.loads(
            (MODELS_DIR / "classical_v2" / "labels.json").read_text())
    return _LABELS


def _load_distilbert() -> Predictor:
    import torch  # imported here so the service can start without loading torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    path = MODELS_DIR / "distilbert_finetuned" / "final"
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).eval()

    def predict_distilbert(text: str) -> np.ndarray:
        with torch.no_grad():
            enc = tok([text], return_tensors="pt",
                      truncation=True, max_length=MAX_LENGTH)
            return torch.softmax(model(**enc).logits, dim=-1)[0].numpy()

    return predict_distilbert


def _load_logreg() -> Predictor:
    import joblib

    folder = MODELS_DIR / "classical_v2"
    union = joblib.load(folder / "feature_union.joblib")
    clf = joblib.load(folder / "logistic_regression.joblib")

    def predict_logreg(text: str) -> np.ndarray:
        return clf.predict_proba(union.transform([text]))[0]

    return predict_logreg


_LOADERS: dict[str, Callable[[], Predictor]] = {
    "distilbert": _load_distilbert,
    "logreg": _load_logreg,
}
_CACHE: dict[str, Predictor] = {}
_LOCK = threading.Lock()


def get_predictor(name: str) -> Predictor:
    """Return the (lazily loaded, cached) predictor for a model key."""
    with _LOCK:
        if name not in _CACHE:
            _CACHE[name] = _LOADERS[name]()
        return _CACHE[name]


# --------------------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Optionally load and warm up every model at start-up (PRELOAD_MODELS=1), so the first
    request is not slow. A model that cannot be loaded is logged and skipped."""
    if os.environ.get("PRELOAD_MODELS") == "1":
        log = logging.getLogger("uvicorn.error")
        for name in _LOADERS:
            try:
                get_predictor(name)("warm up")
            except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-exception-caught
                log.warning("Could not preload model '%s': %s", name, exc)
    yield


app = FastAPI(title="SmartSpend transaction categoriser",
              version="1.0", lifespan=lifespan)


class PredictRequest(BaseModel):
    """Body of POST /predict."""

    text: str = Field(min_length=1, max_length=200,
                      description="Transaction description")
    model: Literal["distilbert", "logreg"] = "distilbert"
    direction: Literal["debit", "credit", "none"] = Field(
        "none",
        description="Optional: prefix the text with the [debit]/[credit] tag used in the data")
    top_k: int = Field(3, ge=1, le=5)

    @field_validator("text")
    @classmethod
    def not_blank(cls, v: str) -> str:
        """Collapse whitespace and reject blank text."""
        v = " ".join(v.split())
        if not v:
            raise ValueError("text must not be blank")
        return v


class Candidate(BaseModel):
    """One category with its predicted probability."""

    category: str
    probability: float


class PredictResponse(BaseModel):
    """Result of POST /predict."""

    model: str
    model_input: str
    category: str
    confidence: float
    top: list[Candidate]
    latency_ms: float
    note: str


CONFIDENCE_NOTE = (
    "Confidence is the model's own probability. In the study it is not a reliable error "
    "signal: wrong predictions can carry very high confidence, especially on merchants "
    "the model has not seen."
)


@app.get("/health")
def health() -> dict:
    """Liveness check; also lists which models are currently loaded."""
    return {"status": "ok", "loaded_models": sorted(_CACHE)}


@app.get("/models")
def models() -> dict:
    """The models the service can serve."""
    return dict(MODEL_INFO)


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    """Classify one transaction description."""
    text = req.text
    if req.direction != "none" and not text.startswith("["):
        text = f"[{req.direction}] {text}"
    try:
        predictor = get_predictor(req.model)
        labels = get_labels()
    except (OSError, ImportError) as exc:  # missing files, missing packages
        raise HTTPException(
            status_code=503,
            detail=f"Model '{req.model}' could not be loaded ({exc}). "
            "Check that the model files are present in MODELS_DIR.") from exc

    start = time.perf_counter()
    probs = np.asarray(predictor(text), dtype=float)
    latency = (time.perf_counter() - start) * 1000
    if len(probs) != len(labels):
        raise HTTPException(
            status_code=500, detail="Model output does not match the label set.")

    order = np.argsort(probs)[::-1][:req.top_k]
    top = [Candidate(category=labels[i], probability=round(
        float(probs[i]), 4)) for i in order]
    return PredictResponse(
        model=MODEL_INFO[req.model]["name"], model_input=text, category=top[0].category,
        confidence=top[0].probability, top=top, latency_ms=round(latency, 1),
        note=CONFIDENCE_NOTE)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """Serve the web interface."""
    return FileResponse(STATIC_DIR / "index.html")
