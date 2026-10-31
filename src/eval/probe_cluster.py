"""
Print sample OOD rows containing a substring, with true label and the predictions of
four models, to see what text decides the label inside a generic-phrase cluster.

Run from the project root:
    uv run python src/eval/probe_cluster.py "preauthorized withdrawal" --n 40
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("term")
ap.add_argument("--n", type=int, default=40)
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()

classes = json.loads(Path("models/classical_v2/labels.json").read_text())
ood = pd.read_csv("data/processed/test_ood.csv")
m = ood["transaction_description"].astype(str).str.lower(
).str.contains(a.term.lower(), regex=False).to_numpy()
models = {"LR": "logistic_regression", "SVM": "linear_svm", "XGB": "gradient_boosting",
          "DistilBERT": "distilbert_finetuned", "MiniLM": "minilm_finetuned"}
pred = {k: np.load(f"reports/calibration/probs/{v}_ood.npy").argmax(1)
        for k, v in models.items()}

sub = ood[m]
print(f"{len(sub)} rows. Sample of {a.n} (true | " + " | ".join(models) + ")\n")
for i in sub.sample(min(a.n, len(sub)), random_state=a.seed).index:
    print(f"{ood.at[i, 'transaction_description']}\n   true={ood.at[i, 'final_category']} | "
          + " | ".join(f"{k}={classes[pred[k][i]]}" for k in models))

print("\nAccuracy inside the cluster:")
y = ood.loc[m, "final_category"].map(
    {c: j for j, c in enumerate(classes)}).to_numpy()
for k in models:
    print(f"  {k}: {(pred[k][m] == y).mean():.3f}")
