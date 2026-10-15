"""
SmartSpend — Classical v2, extra check: did tuning or the new features
drive Logistic Regression's OOD gain?

In the tuned v2 run, Logistic Regression went from 0.7982 (v1) to 0.8811 OOD
macro F1. Two things changed at once: the features (word -> word+char+hand)
and the regularization strength (C=1 -> C=10). This script fits the same
model on every combination of the two, with everything else fixed:

                    C = 1        C = 10
    v1 features     (v1 result)  tuning effect alone
    v2 features     feature effect alone   (v2 result)

Run from the project root (about 3-5 minutes):
    uv run python src/models/classical_v2_lr_check.py
"""

import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.models.classical_features import build_feature_union  # noqa: E402

DATA_DIR = Path("data/processed")
REPORT_DIR = Path("reports/classical_v2")
REPORT_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_SEED = 42

FEATURE_SETS = {"v1 features (word)": ("word",),
                "v2 features (word+char+hand)": ("word", "char", "hand")}
C_VALUES = [1, 10]


def load_split(name: str):
    df = pd.read_csv(DATA_DIR / f"{name}.csv")
    return df["transaction_description"], df["final_category"]


X_train, y_train = load_split("train")
X_val, y_val = load_split("val")
X_test, y_test = load_split("test_in_distribution")
X_ood, y_ood = load_split("test_ood")

rows = []
for feat_label, parts in FEATURE_SETS.items():
    union = build_feature_union(parts)
    Xtr = union.fit_transform(X_train)
    Xva, Xte, Xoo = union.transform(X_val), union.transform(
        X_test), union.transform(X_ood)

    for c in C_VALUES:
        print(f"\nLogistic Regression | {feat_label} | C={c}", flush=True)
        start = time.time()
        clf = LogisticRegression(C=c, max_iter=1000, class_weight="balanced",
                                 random_state=RANDOM_SEED)
        clf.fit(Xtr, y_train)
        row = {"features": feat_label, "C": c, "n_features": Xtr.shape[1],
               "fit_time_s": round(time.time() - start, 1)}
        for split, X, y in [("val", Xva, y_val), ("test", Xte, y_test), ("ood", Xoo, y_ood)]:
            preds = clf.predict(X)
            row[f"{split}_macro_f1"] = round(
                f1_score(y, preds, average="macro"), 4)
            row[f"{split}_accuracy"] = round(accuracy_score(y, preds), 4)
        rows.append(row)
        print({k: v for k, v in row.items()
              if k not in ("features", "C")}, flush=True)

results = pd.DataFrame(rows)
out = REPORT_DIR / "lr_tuning_vs_features.csv"
results.to_csv(out, index=False)

grid = results.pivot(index="features", columns="C", values="ood_macro_f1")
v1, v2 = list(FEATURE_SETS)
print("\n\n=== Logistic Regression, OOD macro F1 ===")
print(grid.to_string())
print(
    f"\nTuning effect (C 1 -> 10), v1 features:  {grid.loc[v1, 10] - grid.loc[v1, 1]:+.4f}")
print(
    f"Tuning effect (C 1 -> 10), v2 features:  {grid.loc[v2, 10] - grid.loc[v2, 1]:+.4f}")
print(
    f"Feature effect (v1 -> v2), C=1:          {grid.loc[v2, 1] - grid.loc[v1, 1]:+.4f}")
print(
    f"Feature effect (v1 -> v2), C=10:         {grid.loc[v2, 10] - grid.loc[v1, 10]:+.4f}")
print(f"\nSaved {out}")
