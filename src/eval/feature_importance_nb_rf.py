"""
SmartSpend -- feature importance for the two models feature_importance.py left out
(Naive Bayes, Random Forest), plus the share of importance by feature group
(word / char / hand) for Random Forest and XGBoost.

  Naive Bayes:   for each class, the features whose log-probability in that class is
                 highest relative to the average of the other classes
                 (log P(f|c) - mean over other classes of log P(f|c')).
  Random Forest: impurity-based feature_importances_ (global, not per class).
  XGBoost:       built-in importances, only used here for the group shares.

Run from the project root:
    uv run python src/eval/feature_importance_nb_rf.py
Options: --top 12
Writes reports/feature_importance/top_features_multinomial_nb.csv,
top_features_random_forest.csv and feature_group_share.csv
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# unpickle classes from src/
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


V2 = Path("models/classical_v2")
OUT = Path("reports/feature_importance")


def group_share(names, imp):
    groups = pd.Series([n.split("__")[0] for n in names])
    s = pd.Series(np.asarray(imp)).groupby(groups.values).sum()
    return (s / s.sum()).round(4).to_dict()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    classes = json.loads((V2 / "labels.json").read_text())
    names = np.asarray(joblib.load(
        V2 / "feature_union.joblib").get_feature_names_out())

    # ---- Naive Bayes ------------------------------------------------------------------
    nb = joblib.load(V2 / "multinomial_nb.joblib")
    flp = nb.feature_log_prob_
    assert flp.shape[1] == len(names), (flp.shape, len(names))
    k = flp.shape[0]
    rows = []
    for ci, c in enumerate(classes):
        others = (flp.sum(0) - flp[ci]) / (k - 1)
        score = flp[ci] - others
        for r, j in enumerate(np.argsort(score)[::-1][:a.top], 1):
            rows.append(
                dict(cls=c, rank=r, feature=names[j], log_ratio=float(score[j])))
    pd.DataFrame(rows).to_csv(
        OUT / "top_features_multinomial_nb.csv", index=False)
    print("== Naive Bayes: features with the highest log-probability ratio, selected classes")
    for c in ("Other", "Travel", "Education", "Income"):
        print(f"  {c}: " + ", ".join(r["feature"]
              for r in rows if r["cls"] == c))
    print("  class log-priors: " + ", ".join(
        f"{c} {p:.2f}" for c, p in zip(classes, nb.class_log_prior_)) + "\n")

    # ---- Random Forest ----------------------------------------------------------------
    rf = joblib.load(V2 / "random_forest.joblib")
    rimp = np.asarray(rf.feature_importances_)
    top = np.argsort(rimp)[::-1][:30]
    pd.DataFrame(dict(feature=names[top], importance=rimp[top])).to_csv(
        OUT / "top_features_random_forest.csv", index=False)
    print("== Random Forest: 15 most important features: "
          + ", ".join(names[top][:15]) + "\n")

    # ---- group shares -----------------------------------------------------------------
    from xgboost import XGBClassifier
    xgb = XGBClassifier()
    xgb.load_model(str(V2 / "gradient_boosting.ubj"))
    shares = {"random_forest": group_share(names, rimp),
              "xgboost": group_share(names, xgb.feature_importances_)}
    pd.DataFrame(shares).T.fillna(0).to_csv(OUT / "feature_group_share.csv")
    print("== Share of importance by feature group")
    for m, s in shares.items():
        print(f"  {m}: {s}")
    print(f"\nSaved CSVs to {OUT}")


if __name__ == "__main__":
    main()
