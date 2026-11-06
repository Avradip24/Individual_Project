"""
SmartSpend -- the consolidated results table: accuracy, macro precision / recall / F1,
weighted F1 and per-category precision / recall / F1 for ALL EIGHT models on BOTH test
sets (in-distribution and OOD), computed from the cached predictions.

Reads reports/calibration/probs/<model>_{test,ood}.npy (written by calibration_all_models.py)
and the two test CSVs. No models are loaded; it runs in seconds. The Linear SVM row uses its
Platt-calibrated predictions (the same ones as the calibration and error analysis).

Run from the project root:
    uv run python src/eval/results_table.py

Writes to reports/results/:
    results_summary.csv          one row per model x split
    per_category_metrics.csv     long format: model, split, category, precision, recall, f1, support
    per_category_f1_test.csv / per_category_f1_ood.csv     wide: category x model
    results_tables.md            the same tables as Markdown, ready to paste into the log/report
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

DATA = Path("data/processed")
PROBS = Path("reports/calibration/probs")
LABELS = Path("models/classical_v2/labels.json")
OUT = Path("reports/results")
SPLITS = {"test": "test_in_distribution.csv", "ood": "test_ood.csv"}
MODELS = {
    "distilbert_finetuned": "Fine-tuned DistilBERT",
    "logistic_regression": "Logistic Regression v2",
    "minilm_finetuned": "Fine-tuned MiniLM",
    "linear_svm": "Linear SVM v2",
    "multinomial_nb": "Naive Bayes v2",
    "gradient_boosting": "XGBoost v2",
    "random_forest": "Random Forest v2",
    "frozen_embedding_logreg": "Frozen MiniLM + LogReg",
}
# OOD macro F1 reported earlier in the methodology log (section 10.5); used as a consistency check.
EXPECTED_OOD_MACRO_F1 = {
    "distilbert_finetuned": 0.9329, "logistic_regression": 0.8811, "minilm_finetuned": 0.8767,
    "linear_svm": 0.8543, "multinomial_nb": 0.8365, "gradient_boosting": 0.8191,
    "random_forest": 0.7450, "frozen_embedding_logreg": 0.6623,
}


def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    head = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(floatfmt.format(v) if isinstance(v, float) else str(v)
                              for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    classes = json.loads(LABELS.read_text())
    idx = {c: i for i, c in enumerate(classes)}
    y = {s: pd.read_csv(DATA / f).final_category.map(idx).to_numpy()
         for s, f in SPLITS.items()}
    assert all(not np.isnan(v.astype(float)).any()
               for v in y.values()), "unknown label in a CSV"

    summary, per_cat = [], []
    for key, title in MODELS.items():
        for split in SPLITS:
            pred = np.load(PROBS / f"{key}_{split}.npy").argmax(1)
            assert len(pred) == len(y[split]), (key,
                                                split, len(pred), len(y[split]))
            p, r, f, sup = precision_recall_fscore_support(
                y[split], pred, labels=range(len(classes)), zero_division=0)
            w = precision_recall_fscore_support(
                y[split], pred, labels=range(len(classes)), average="weighted", zero_division=0)
            summary.append(dict(model=key, title=title, split=split,
                                accuracy=accuracy_score(y[split], pred),
                                macro_precision=p.mean(), macro_recall=r.mean(), macro_f1=f.mean(),
                                weighted_f1=w[2]))
            for i, c in enumerate(classes):
                per_cat.append(dict(model=key, split=split, category=c, precision=p[i],
                                    recall=r[i], f1=f[i], support=int(sup[i])))
    s = pd.DataFrame(summary)
    pc = pd.DataFrame(per_cat)
    s.round(4).to_csv(OUT / "results_summary.csv", index=False)
    pc.round(4).to_csv(OUT / "per_category_metrics.csv", index=False)

    # ---- consistency check against the numbers already in the log ----------------------
    print("Consistency check: OOD macro F1 against the values in the methodology log")
    ood = s[s.split == "ood"].set_index("model")
    for key, exp in EXPECTED_OOD_MACRO_F1.items():
        got = ood.at[key, "macro_f1"]
        flag = "ok" if abs(got - exp) < 5e-4 else "MISMATCH"
        print(f"  {MODELS[key]:24s} {got:.4f} (log {exp:.4f})  {flag}")

    # ---- wide tables ------------------------------------------------------------------
    order = ood.sort_values("macro_f1", ascending=False).index.tolist()
    wide = s.pivot(index="model", columns="split")
    main_tbl = pd.DataFrame({
        "Model": [MODELS[m] for m in order],
        "ID acc": [wide.loc[m, ("accuracy", "test")] for m in order],
        "ID macro F1": [wide.loc[m, ("macro_f1", "test")] for m in order],
        "OOD acc": [wide.loc[m, ("accuracy", "ood")] for m in order],
        "OOD macro P": [wide.loc[m, ("macro_precision", "ood")] for m in order],
        "OOD macro R": [wide.loc[m, ("macro_recall", "ood")] for m in order],
        "OOD macro F1": [wide.loc[m, ("macro_f1", "ood")] for m in order],
        "OOD weighted F1": [wide.loc[m, ("weighted_f1", "ood")] for m in order],
        "Macro F1 drop ID to OOD": [wide.loc[m, ("macro_f1", "test")] - wide.loc[m, ("macro_f1", "ood")]
                                    for m in order],
    }).astype({c: float for c in ["ID acc", "ID macro F1", "OOD acc", "OOD macro P", "OOD macro R",
                                  "OOD macro F1", "OOD weighted F1", "Macro F1 drop ID to OOD"]})

    md = ["## Summary: all eight models, both test sets",
          "", md_table(main_tbl), ""]
    for split, label in (("ood", "OOD test (unseen merchants)"), ("test", "in-distribution test")):
        t = pc[pc.split == split].pivot(
            index="category", columns="model", values="f1")[order]
        t.round(4).to_csv(OUT / f"per_category_f1_{split}.csv")
        sup = pc[(pc.split == split) & (pc.model == order[0])
                 ].set_index("category")["support"]
        t = t.rename(columns=MODELS)
        t.insert(0, "n", sup)
        t = t.reset_index().rename(columns={"category": "Category"})
        t["n"] = t["n"].astype(int)
        md += [f"## Per-category F1, {label}", "", md_table(t), ""]
    (OUT / "results_tables.md").write_text("\n".join(md), encoding="utf-8")

    print("\n" + md_table(main_tbl))
    print(f"\nSaved results_summary.csv, per_category_metrics.csv, per_category_f1_*.csv and "
          f"results_tables.md to {OUT}")


if __name__ == "__main__":
    main()
