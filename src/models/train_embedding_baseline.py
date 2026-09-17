"""
SmartSpend — Transformer approach 1: multilingual sentence-embedding model
+ lightweight classifier.

Uses a FROZEN, pretrained multilingual sentence-transformer
(paraphrase-multilingual-MiniLM-L12-v2) to produce fixed-dimensional
embeddings for each transaction description, then trains a simple
Logistic Regression classifier on top of those embeddings. The transformer
itself is never fine-tuned here — only the small classifier head is trained,
which is what makes this the "embedding-based" approach rather than the
"fine-tuned transformer" approach (that's the second transformer model).

Why this specific model: it's pretrained on massive multilingual web text,
so it likely already encodes that brands like ALDI, REWE, or Aldi Sued are
supermarkets -- purely from general world knowledge baked in during
pretraining, not from anything in THIS project's training data (which is
entirely English). This is the mechanism that could let it generalize to
unseen, and even non-English, merchants in a way the classical TF-IDF
models structurally cannot -- directly testing the project's central
hypothesis.

Requires: uv add sentence-transformers
"""

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score

DATA_DIR = Path("data/processed")
MODELS_DIR = Path("models")
EMBEDDINGS_DIR = Path("data/embeddings")
MODELS_DIR.mkdir(exist_ok=True)
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


def load_split(name: str):
    df = pd.read_csv(DATA_DIR / f"{name}.csv")
    return df["transaction_description"], df["final_category"]


X_train, y_train = load_split("train")
X_val, y_val = load_split("val")
X_test, y_test = load_split("test_in_distribution")
X_ood, y_ood = load_split("test_ood")

print(
    f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)} | OOD: {len(X_ood)}")

print(f"\nLoading sentence-transformer: {MODEL_NAME} ...")
embedding_model = SentenceTransformer(MODEL_NAME)


def embed(name: str, texts) -> np.ndarray:
    """Encode texts to embeddings, caching to disk so re-runs are instant."""
    cache_path = EMBEDDINGS_DIR / f"{name}_embeddings.npy"
    if cache_path.exists():
        print(f"Loading cached embeddings for {name}")
        return np.load(cache_path)

    print(f"Encoding {len(texts)} texts for {name} ...")
    start = time.time()
    embeddings = embedding_model.encode(
        texts.tolist(),
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    print(f"  done in {time.time() - start:.1f}s")
    np.save(cache_path, embeddings)
    return embeddings


X_train_emb = embed("train", X_train)
X_val_emb = embed("val", X_val)
X_test_emb = embed("test_in_distribution", X_test)
X_ood_emb = embed("test_ood", X_ood)

print(f"\nEmbedding dimension: {X_train_emb.shape[1]}")

# ---------------------------------------------------------------------------
# Lightweight classifier on top of the frozen embeddings
# ---------------------------------------------------------------------------
clf = LogisticRegression(
    max_iter=2000, class_weight="balanced", random_state=RANDOM_SEED)

print("\nTraining Logistic Regression on multilingual embeddings...")
clf.fit(X_train_emb, y_train)


def evaluate(name: str, X, y):
    preds = clf.predict(X)
    acc = accuracy_score(y, preds)
    macro_f1 = f1_score(y, preds, average="macro")
    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.4f} | Macro F1: {macro_f1:.4f}")
    print(classification_report(y, preds, zero_division=0))
    return preds


evaluate("Validation (in-distribution)", X_val_emb, y_val)
evaluate("Test (in-distribution)", X_test_emb, y_test)
evaluate("Test (Out-of-Distribution / unseen merchants)", X_ood_emb, y_ood)

joblib.dump(clf, MODELS_DIR / "embedding_logreg_classifier.joblib")
print(
    f"\nSaved classifier to {MODELS_DIR / 'embedding_logreg_classifier.joblib'}")
print(f"Embedding model used: {MODEL_NAME} (frozen, not fine-tuned)")
