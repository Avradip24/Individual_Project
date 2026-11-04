"""
SmartSpend -- Error analysis, part C: counterfactual test of the wrapper phrase.

Takes the OOD rows that contain `PREAUTHORIZED WITHDRAWAL` and re-scores the classical
models on three versions of each string:
  original      [debit] PREAUTHORIZED WITHDRAWAL TO JETBLUE
  no_phrase     [debit] TO JETBLUE            (phrase removed, "TO" kept)
  merchant_only [debit] JETBLUE               (phrase and "TO" removed)
If accuracy jumps once the phrase is removed, the unseen wrapper is what breaks the
model. If it only jumps when "TO" is also removed, the word "to" is the culprit.
Scores are the raw model predictions (Linear SVM without Platt calibration).

Run from the project root:
    uv run python src/eval/wrapper_ablation.py
Options: --rf (also score Random Forest; loads a 500 MB file)
Writes reports/feature_importance/wrapper_ablation.csv
"""
import pandas as pd
import numpy as np
import joblib
import argparse
import json
import re
import sys
from pathlib import Path

# unpickle classes from src/
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


V2 = Path("models/classical_v2")
OUT = Path("reports/feature_importance")
PHRASE = re.compile(r"preauthorized\s+withdrawal", re.I)
PHRASE_TO = re.compile(r"preauthorized\s+withdrawal\s+to\b\s*", re.I)


def variants(s: str) -> dict:
    return {
        "original": s,
        "no_phrase": re.sub(r"\s+", " ", PHRASE.sub("", s)).strip(),
        "merchant_only": re.sub(r"\s+", " ", PHRASE_TO.sub("", s)).strip(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rf", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    classes = json.loads((V2 / "labels.json").read_text())
    idx = {c: i for i, c in enumerate(classes)}
    other = idx["Other"]
    ood = pd.read_csv("data/processed/test_ood.csv")
    m = ood["transaction_description"].astype(
        str).str.contains(PHRASE).to_numpy()
    sub = ood[m]
    y = sub["final_category"].map(idx).to_numpy()
    texts = {k: [variants(s)[k] for s in sub["transaction_description"].astype(str)]
             for k in ("original", "no_phrase", "merchant_only")}
    print(f"{len(sub)} rows. Example rewrites:")
    for s in sub["transaction_description"].astype(str).head(3):
        print("  ", variants(s))

    union = joblib.load(V2 / "feature_union.joblib")
    names = ["logistic_regression", "linear_svm",
             "multinomial_nb", "gradient_boosting"]
    if a.rf:
        names.append("random_forest")
    rows = []
    for n in names:
        if n == "gradient_boosting":
            from xgboost import XGBClassifier
            model = XGBClassifier()
            model.load_model(str(V2 / "gradient_boosting.ubj"))
        else:
            model = joblib.load(V2 / f"{n}.joblib")
        for k, tx in texts.items():
            pred = np.asarray(model.predict(union.transform(tx)))
            rows.append(dict(model=n, version=k, accuracy=float((pred == y).mean()),
                             share_predicted_other=float((pred == other).mean())))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "wrapper_ablation.csv", index=False)
    print("\nAccuracy inside the cluster (share predicted Other in brackets):")
    for n in names:
        d = df[df.model == n].set_index("version")
        print(f"  {n:20s} " + "  ".join(
            f"{k}: {d.at[k, 'accuracy']:.3f} ({d.at[k, 'share_predicted_other']:.0%})"
            for k in ("original", "no_phrase", "merchant_only")))
    print(f"\nSaved {OUT / 'wrapper_ablation.csv'}")


if __name__ == "__main__":
    main()
