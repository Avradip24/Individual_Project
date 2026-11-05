"""
SmartSpend -- probe: which Income rows do Logistic Regression, Linear SVM and Naive Bayes
send to Utilities (201, 146 and 137 rows)?

For each model it prints the most common text templates (first three tokens, digits -> 0)
among these errors, whether DistilBERT gets the same rows right, and how the same
template is labelled in the TRAINING set and in the OOD set.

Run from the project root:
    uv run python src/eval/income_to_utilities.py
Options: --top 10 --samples 12
Writes reports/error_analysis/income_to_utilities_<model>.csv
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

PROBS = Path("reports/calibration/probs")
OUT = Path("reports/error_analysis")
MODELS = {"LR": "logistic_regression",
          "SVM": "linear_svm", "NB": "multinomial_nb"}


def norm(s) -> str:
    return re.sub(r"\s+", " ", re.sub(r"\d+", "0", str(s).lower())).strip()


def key(s) -> str:
    return " ".join(norm(s).split()[:3])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--samples", type=int, default=12)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    classes = json.loads(Path("models/classical_v2/labels.json").read_text())
    idx = {c: i for i, c in enumerate(classes)}
    ood = pd.read_csv("data/processed/test_ood.csv")
    train = pd.read_csv("data/processed/train.csv")
    y = ood["final_category"].map(idx).to_numpy()
    ood_key = ood["transaction_description"].map(key)
    train_key = train["transaction_description"].map(key)
    dist_pred = np.load(PROBS / "distilbert_finetuned_ood.npy").argmax(1)
    inc, util = idx["Income"], idx["Utilities"]

    for short, m in MODELS.items():
        pred = np.load(PROBS / f"{m}_ood.npy").argmax(1)
        sel = (y == inc) & (pred == util)
        print("=" * 78)
        print(f"{short}: {int(sel.sum())} Income rows predicted Utilities; "
              f"DistilBERT gets {(dist_pred[sel] == inc).mean():.0%} of them right")
        print("=" * 78)
        keys = Counter(ood_key[sel])
        print(f"{len(keys)} distinct templates. Most common (count | OOD rows with this template "
              f"and their true labels | TRAINING rows with it and their labels):")
        for k, c in keys.most_common(a.top):
            o = ood.loc[ood_key == k, "final_category"].value_counts().head(
                3).to_dict()
            t = train.loc[train_key == k,
                          "final_category"].value_counts().head(3).to_dict()
            print(f"  {c:4d}  '{k}'  | OOD {o} | train {t if t else 'none'}")
        print(f"\nSample of {a.samples} rows:")
        sub = ood[sel].sample(min(a.samples, int(sel.sum())), random_state=42)
        for i in sub.index:
            print(
                f"  {ood.at[i, 'transaction_description']}   (DistilBERT: {classes[dist_pred[i]]})")
        ood[sel][["transaction_description", "final_category"]].assign(
            distilbert=[classes[i] for i in dist_pred[sel]]).to_csv(
            OUT / f"income_to_utilities_{m}.csv", index=False)
        print()
    print(f"Saved CSVs to {OUT}")


if __name__ == "__main__":
    main()
