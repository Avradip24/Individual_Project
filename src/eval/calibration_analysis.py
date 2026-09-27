"""
SmartSpend — Confidence calibration / Expected Calibration Error (ECE).

A model is well-calibrated if, among all predictions where it states X%
confidence, it is actually correct about X% of the time. ECE measures the
average gap between stated confidence and actual accuracy, across confidence
buckets. Evaluated on the OOD test set specifically, since that's where the
project's live German-merchant probe already showed a concrete calibration
failure (the model was 80-100% confident while wrong on most German
strings) -- this quantifies that failure mode properly, at scale.

Compares:
  - Logistic Regression (classical) -- has predict_proba natively
  - Frozen multilingual embedding + LogReg -- has predict_proba natively
  - Fine-tuned DistilBERT -- softmax over logits gives per-class probability

NOT included: Linear SVM (your best classical model) -- LinearSVC has no
native probability output. This is a real, worth-stating limitation: your
strongest classical model can't be calibration-tested without wrapping it
in CalibratedClassifierCV (which would require retraining, a small
follow-up if calibration comparison for the classical winner specifically
turns out to matter for the report).
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DATA_DIR = Path("data/processed")
MODELS_DIR = Path("models")
EMBEDDINGS_DIR = Path("data/embeddings")
N_BINS = 10

LABELS = [
    "Education", "Entertainment", "Groceries", "Healthcare", "Housing",
    "Income", "Insurance", "Other", "Restaurants", "Shopping",
    "Transportation", "Travel", "Utilities",
]


def compute_ece(confidences: np.ndarray, correct: np.ndarray, n_bins: int = N_BINS):
    """Returns (ece, bin_table) where bin_table has one row per confidence bucket."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(confidences, bin_edges[1:-1])

    rows = []
    ece = 0.0
    n_total = len(confidences)

    for b in range(n_bins):
        mask = bin_ids == b
        count = mask.sum()
        if count == 0:
            continue
        bin_confidence = confidences[mask].mean()
        bin_accuracy = correct[mask].mean()
        ece += (count / n_total) * abs(bin_accuracy - bin_confidence)
        rows.append({
            "bin_range": f"{bin_edges[b]:.1f}-{bin_edges[b+1]:.1f}",
            "count": count,
            "avg_confidence": bin_confidence,
            "accuracy": bin_accuracy,
            "gap": bin_accuracy - bin_confidence,
        })

    return ece, pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Load OOD test set (the split where calibration failures matter most)
# ---------------------------------------------------------------------------
df_ood = pd.read_csv(DATA_DIR / "test_ood.csv")
X_ood_text = df_ood["transaction_description"]
y_ood = df_ood["final_category"].values

results_summary = []

# ---------------------------------------------------------------------------
# 1. Logistic Regression (classical, shared TF-IDF vectorizer)
# ---------------------------------------------------------------------------
print("=" * 60)
print("Logistic Regression (classical)")
print("=" * 60)
vectorizer = joblib.load(MODELS_DIR / "tfidf_vectorizer_shared.joblib")
logreg = joblib.load(MODELS_DIR / "logistic_regression.joblib")

X_ood_vec = vectorizer.transform(X_ood_text)
proba = logreg.predict_proba(X_ood_vec)
preds = logreg.classes_[np.argmax(proba, axis=1)]
confidences = np.max(proba, axis=1)
correct = (preds == y_ood).astype(int)

ece, bin_table = compute_ece(confidences, correct)
print(f"ECE: {ece:.4f}")
print(bin_table.to_string(index=False))
results_summary.append({"model": "logistic_regression",
                       "ece": ece, "overall_accuracy": correct.mean()})

# ---------------------------------------------------------------------------
# 2. Frozen multilingual embedding + Logistic Regression
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Frozen multilingual embedding + Logistic Regression")
print("=" * 60)
embedding_clf = joblib.load(MODELS_DIR / "embedding_logreg_classifier.joblib")
X_ood_emb = np.load(EMBEDDINGS_DIR / "test_ood_embeddings.npy")

proba = embedding_clf.predict_proba(X_ood_emb)
preds = embedding_clf.classes_[np.argmax(proba, axis=1)]
confidences = np.max(proba, axis=1)
correct = (preds == y_ood).astype(int)

ece, bin_table = compute_ece(confidences, correct)
print(f"ECE: {ece:.4f}")
print(bin_table.to_string(index=False))
results_summary.append({"model": "frozen_embedding_logreg",
                       "ece": ece, "overall_accuracy": correct.mean()})

# ---------------------------------------------------------------------------
# 3. Fine-tuned DistilBERT
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Fine-tuned DistilBERT")
print("=" * 60)
MODEL_PATH = "models/distilbert_finetuned/final"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

all_confidences = []
all_preds = []
batch_size = 64
texts = X_ood_text.tolist()

print(
    f"Running inference on {len(texts)} OOD examples (CPU — this may take several minutes)...")
with torch.no_grad():
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt",
                           truncation=True, max_length=32, padding=True)
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        batch_conf, batch_idx = torch.max(probs, dim=-1)
        all_confidences.extend(batch_conf.tolist())
        all_preds.extend([LABELS[i] for i in batch_idx.tolist()])
        if i % (batch_size * 20) == 0:
            print(f"  {i}/{len(texts)}...")

confidences = np.array(all_confidences)
preds = np.array(all_preds)
correct = (preds == y_ood).astype(int)

ece, bin_table = compute_ece(confidences, correct)
print(f"ECE: {ece:.4f}")
print(bin_table.to_string(index=False))
results_summary.append({"model": "finetuned_distilbert",
                       "ece": ece, "overall_accuracy": correct.mean()})

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n\n" + "=" * 60)
print("SUMMARY — lower ECE is better calibrated")
print("=" * 60)
summary_df = pd.DataFrame(results_summary)
print(summary_df.to_string(index=False))
summary_df.to_csv(MODELS_DIR / "calibration_summary.csv", index=False)
print(f"\nSaved to {MODELS_DIR / 'calibration_summary.csv'}")
print("\nNOTE: Linear SVM (best classical model overall) excluded — LinearSVC "
      "has no native predict_proba. See docstring for why.")
