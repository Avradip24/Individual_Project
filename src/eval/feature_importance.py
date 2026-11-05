"""
SmartSpend -- Error analysis, part B: what do the classical models actually use?

1. Top features per class for Logistic Regression and Linear SVM (largest positive
   coefficients), top features by gain for XGBoost.
2. Why do Linear SVM / Logistic Regression answer "Other" on the unseen wrapper
   `PREAUTHORIZED WITHDRAWAL TO <merchant>`?  For each row in that cluster where the
   model predicts Other but the truth is not Other, the score gap is split into
        gap = (intercept_Other - intercept_true) + x . (w_Other - w_true)
   i.e. an INTERCEPT part (a default towards Other) and a FEATURE part (evidence from
   the words). If the intercept part is small and the feature part is large, the
   default-to-Other idea is wrong and the Other-evidence comes from the features.
   Also prints the number of active features per row (cluster vs all OOD rows).

Run from the project root:
    uv run python src/eval/feature_importance.py
Options: --term "preauthorized withdrawal"  --top 12
Writes reports/feature_importance/*.csv
"""
# pylint: disable=wrong-import-position,import-error
# ruff: noqa: E402
from scipy import sparse
import pandas as pd
import numpy as np
import joblib
import argparse
import json
import sys
from pathlib import Path

# unpickle classes from src/
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


V2 = Path("models/classical_v2")
FEAT = Path("data/features_v2")
OUT = Path("reports/feature_importance")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--term", default="preauthorized withdrawal")
    ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    classes = json.loads((V2 / "labels.json").read_text())
    idx = {c: i for i, c in enumerate(classes)}
    other = idx["Other"]
    fu = joblib.load(V2 / "feature_union.joblib")
    names = np.asarray(fu.get_feature_names_out())
    ood = pd.read_csv("data/processed/test_ood.csv")
    X = sparse.load_npz(FEAT / "test_ood.npz").tocsr()
    assert X.shape[1] == len(names), (X.shape, len(names))
    y = ood["final_category"].map(idx).to_numpy()
    mask = ood["transaction_description"].astype(str).str.lower().str.contains(
        a.term.lower(), regex=False).to_numpy()
    print(
        f"{X.shape[1]} features; {int(mask.sum())} OOD rows contain '{a.term}'")

    nnz = np.diff(X.indptr)
    print(f"Active features per row: cluster {nnz[mask].mean():.1f} vs all other OOD rows "
          f"{nnz[~mask].mean():.1f}\n")

    # ---- 1. top features -------------------------------------------------------------
    models = {n: joblib.load(V2 / f"{n}.joblib")
              for n in ("logistic_regression", "linear_svm")}
    for n, m in models.items():
        rows = []
        for c, ci in idx.items():
            for r, j in enumerate(np.argsort(m.coef_[ci])[::-1][:a.top], 1):
                rows.append(dict(model=n, cls=c, rank=r,
                            feature=names[j], weight=float(m.coef_[ci][j])))
        pd.DataFrame(rows).to_csv(OUT / f"top_features_{n}.csv", index=False)
        print(f"== {n}: top features for selected classes (full table in CSV)")
        for c in ("Other", "Travel", "Education", "Income"):
            print(f"  {c}: " + ", ".join(r["feature"]
                  for r in rows if r["cls"] == c)[:300])
        print(f"  intercept Other = {m.intercept_[other]:.3f}; all intercepts: "
              + ", ".join(f"{c} {m.intercept_[i]:.2f}" for c, i in idx.items()) + "\n")

    from xgboost import XGBClassifier
    xgb = XGBClassifier()
    xgb.load_model(str(V2 / "gradient_boosting.ubj"))
    imp = np.asarray(xgb.feature_importances_)
    top = np.argsort(imp)[::-1][:30]
    pd.DataFrame(dict(feature=names[top], importance=imp[top])).to_csv(
        OUT / "top_features_xgboost.csv", index=False)
    print("== XGBoost: 15 most important features: " +
          ", ".join(names[top][:15]) + "\n")

    # ---- 2. decomposition of the "Other" answers inside the wrapper cluster ----------
    res = []
    for n, m in models.items():
        S = np.asarray(X @ m.coef_.T) + \
            m.intercept_            # rows x classes
        pred = S.argmax(1)
        acc = (pred[mask] == y[mask]).mean()
        sel = mask & (pred == other) & (y != other)
        print(f"== {n}: accuracy in cluster {acc:.3f}; rows predicted Other with a "
              f"non-Other truth: {int(sel.sum())}")
        if sel.sum() == 0:
            continue
        t = y[sel]
        inter = m.intercept_[other] - m.intercept_[t]
        feat = np.asarray(
            (X[sel].multiply(m.coef_[other] - m.coef_[t])).sum(1)).ravel()
        print(f"   mean score gap Other - true: {np.mean(inter + feat):.3f} = "
              f"intercept part {inter.mean():.3f} + feature part {feat.mean():.3f}")
        print(
            f"   rows where the intercept part alone favours Other: {(inter > 0).mean():.1%}")
        print(
            f"   rows where the feature part alone favours Other:   {(feat > 0).mean():.1%}")
        res.append(dict(model=n, rows=int(sel.sum()), gap=float(np.mean(inter + feat)),
                        intercept_part=float(inter.mean()), feature_part=float(feat.mean()),
                        share_intercept_favours_other=float(
                            (inter > 0).mean()),
                        share_features_favour_other=float((feat > 0).mean())))
    pd.DataFrame(res).to_csv(OUT / "wrapper_decomposition.csv", index=False)
    print(f"\nSaved CSVs to {OUT}")


if __name__ == "__main__":
    main()
