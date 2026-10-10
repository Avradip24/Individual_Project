"""
SmartSpend — Classical v2, step 1: feature ablation.

Question: which part of the expose's feature set actually improves
generalization to unseen merchants? Trains ONE fixed model (Linear SVM, the
best v1 classical model, fast to train) on four feature sets and compares
them on validation, in-distribution test, and OOD test:

    word              = v1 features (reference point)
    char              = character n-grams only
    word+char
    word+char+hand    = full v2 feature set

Hyperparameters are identical across rows (C=1.0, balanced class weights),
so any difference comes from the features alone.

Run from the project root:
    uv run python src/models/classical_v2_ablation.py
"""

import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.models.classical_features import build_feature_union  # noqa: E402  # pylint: disable=import-error,wrong-import-position

DATA_DIR = Path("data/processed")
REPORT_DIR = Path("reports/classical_v2")
REPORT_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_SEED = 42

FEATURE_SETS = {
    "word (v1)": ("word",),
    "char": ("char",),
    "word+char": ("word", "char"),
    "word+char+hand (v2)": ("word", "char", "hand"),
    "word+hand": ("word", "hand"),
    "hand only": ("hand",),
}


def load_split(name: str):
    df = pd.read_csv(DATA_DIR / f"{name}.csv")
    return df["transaction_description"], df["final_category"]


X_train, y_train = load_split("train")
X_val, y_val = load_split("val")
X_test, y_test = load_split("test_in_distribution")
X_ood, y_ood = load_split("test_ood")
print(
    f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)} | OOD: {len(X_ood)}")

rows = []
for label, parts in FEATURE_SETS.items():
    print(f"\n{'=' * 60}\n{label}\n{'=' * 60}")
    start = time.time()
    union = build_feature_union(parts)
    Xtr = union.fit_transform(X_train)
    Xva, Xte, Xoo = union.transform(X_val), union.transform(
        X_test), union.transform(X_ood)
    feat_time = time.time() - start

    start = time.time()
    clf = LinearSVC(C=1.0, class_weight="balanced",
                    random_state=RANDOM_SEED, max_iter=5000)
    clf.fit(Xtr, y_train)
    fit_time = time.time() - start

    row = {"feature_set": label, "n_features": Xtr.shape[1],
           "feature_time_s": round(feat_time, 1), "fit_time_s": round(fit_time, 1)}
    for split, X, y in [("val", Xva, y_val), ("test", Xte, y_test), ("ood", Xoo, y_ood)]:
        preds = clf.predict(X)
        row[f"{split}_macro_f1"] = round(
            f1_score(y, preds, average="macro"), 4)
        row[f"{split}_accuracy"] = round(accuracy_score(y, preds), 4)
    rows.append(row)
    print({k: v for k, v in row.items() if k != "feature_set"})

results = pd.DataFrame(rows)
baseline_ood = results.loc[0, "ood_macro_f1"]
results["ood_f1_change_vs_v1"] = (
    results["ood_macro_f1"] - baseline_ood).round(4)

print("\n\n=== FEATURE ABLATION (Linear SVM, fixed hyperparameters) ===")
print(results[["feature_set", "n_features", "val_macro_f1", "test_macro_f1",
               "ood_macro_f1", "ood_f1_change_vs_v1"]].to_string(index=False))

out = REPORT_DIR / "feature_ablation.csv"
results.to_csv(out, index=False)
print(f"\nSaved {out}")
