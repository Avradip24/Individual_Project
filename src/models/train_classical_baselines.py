"""
SmartSpend — classical ML baselines under a consistent protocol.

Trains Logistic Regression, Multinomial Naive Bayes, and Linear SVM on the
SAME TF-IDF feature space (fit once, shared across all three), so any
performance difference you see reflects the algorithm, not different
features. This directly matches the expose's requirement for "a consistent
feature engineering and hyperparameter tuning protocol across all classical
models."

Note: hyperparameters here are reasonable defaults, not yet tuned. A
validation-set hyperparameter search comes later, once all three algorithms
have a working baseline — no point tuning a model you might not keep.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

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

print(
    f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)} | OOD: {len(X_ood)}")

# ---------------------------------------------------------------------------
# Fit TF-IDF ONCE, shared across all three models
# ---------------------------------------------------------------------------
vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 2),
    min_df=2,
    max_features=20_000,
    lowercase=True,
)
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)
X_test_vec = vectorizer.transform(X_test)
X_ood_vec = vectorizer.transform(X_ood)

joblib.dump(vectorizer, MODELS_DIR / "tfidf_vectorizer_shared.joblib")

# XGBoost needs numeric class labels internally — the other four models
# accept the string category names directly, so this encoder is only
# used for XGBoost's fit/predict, and we decode its predictions straight
# back to string labels before computing any metric, so every model's
# results stay directly comparable.
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

# ---------------------------------------------------------------------------
# Five classical models, same features, same class-imbalance handling
# where the algorithm supports it
# ---------------------------------------------------------------------------
models = {
    "logistic_regression": LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED
    ),
    # no class_weight param — MNB doesn't support it directly
    "multinomial_nb": MultinomialNB(),
    "linear_svm": LinearSVC(
        class_weight="balanced", random_state=RANDOM_SEED, max_iter=5000
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    ),
    "gradient_boosting": XGBClassifier(
        n_estimators=300,
        tree_method="hist",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        eval_metric="mlogloss",
    ),
}

results = []

for name, model in models.items():
    print(f"\n{'=' * 60}\nTraining {name}\n{'=' * 60}")

    if name == "gradient_boosting":
        model.fit(X_train_vec, y_train_encoded)
    else:
        model.fit(X_train_vec, y_train)
    joblib.dump(model, MODELS_DIR / f"{name}.joblib")

    for split_name, X_vec, y in [
        ("val", X_val_vec, y_val),
        ("test_in_distribution", X_test_vec, y_test),
        ("test_ood", X_ood_vec, y_ood),
    ]:
        if name == "gradient_boosting":
            preds_encoded = model.predict(X_vec)
            preds = label_encoder.inverse_transform(preds_encoded)
        else:
            preds = model.predict(X_vec)

        acc = accuracy_score(y, preds)
        macro_f1 = f1_score(y, preds, average="macro")
        results.append({"model": name, "split": split_name,
                       "accuracy": acc, "macro_f1": macro_f1})
        print(f"{split_name:25s} accuracy={acc:.4f}  macro_f1={macro_f1:.4f}")

# ---------------------------------------------------------------------------
# Summary table across all models and splits — this is the comparison
# table you'll build on for the rest of the project.
# ---------------------------------------------------------------------------
results_df = pd.DataFrame(results)
print("\n\n=== SUMMARY: accuracy across all models and splits ===")
print(results_df.pivot(index="model", columns="split", values="accuracy").round(4))
print("\n=== SUMMARY: macro F1 across all models and splits ===")
print(results_df.pivot(index="model", columns="split", values="macro_f1").round(4))

results_df.to_csv(MODELS_DIR / "classical_baselines_summary.csv", index=False)
print(f"\nSaved summary to {MODELS_DIR / 'classical_baselines_summary.csv'}")
