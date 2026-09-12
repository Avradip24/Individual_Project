"""
SmartSpend — first classical baseline: Logistic Regression + TF-IDF.

Purpose: validate the full pipeline end-to-end (data -> features -> model ->
metrics) before building the remaining classical baselines (Naive Bayes,
Random Forest, Gradient Boosting, Linear SVM) and the transformer models.
This is intentionally the simplest model in the comparison, and becomes the
reference point everything else is measured against.

Evaluates on all three held-out sets, since the whole point of this project
is comparing behavior across them, not just reporting one accuracy number:
  - val / test (in-distribution): "how good is it on data like it trained on"
  - test_ood (unseen merchants):   "how good is it on merchants it never saw"
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline

DATA_DIR = Path("data/processed")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

RANDOM_SEED = 42


def load_split(name: str):
    df = pd.read_csv(DATA_DIR / f"{name}.csv")
    return df["transaction_description"], df["final_category"]


X_train, y_train = load_split("train")
X_val, y_val = load_split("val")
X_test, y_test = load_split("test_in_distribution")
X_ood, y_ood = load_split("test_ood")

print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)} | OOD: {len(X_ood)}")

# ---------------------------------------------------------------------------
# Pipeline: TF-IDF (word unigrams+bigrams) -> Logistic Regression
# class_weight="balanced" matters here — your categories range from 4,000
# (Insurance) to 12,000 (Entertainment/Shopping) rows, and an unweighted
# model would just learn to favor the bigger classes.
# ---------------------------------------------------------------------------
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_features=20_000,
        lowercase=True,
    )),
    ("clf", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )),
])

print("\nTraining Logistic Regression + TF-IDF...")
pipeline.fit(X_train, y_train)


def evaluate(name: str, X, y):
    preds = pipeline.predict(X)
    acc = accuracy_score(y, preds)
    macro_f1 = f1_score(y, preds, average="macro")
    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.4f} | Macro F1: {macro_f1:.4f}")
    print(classification_report(y, preds, zero_division=0))
    return preds


evaluate("Validation (in-distribution)", X_val, y_val)
evaluate("Test (in-distribution)", X_test, y_test)
evaluate("Test (Out-of-Distribution / unseen merchants)", X_ood, y_ood)

joblib.dump(pipeline, MODELS_DIR / "logreg_tfidf_baseline.joblib")
print(f"\nSaved model to {MODELS_DIR / 'logreg_tfidf_baseline.joblib'}")