"""
SmartSpend -- informal German/English probe on several models (illustrative, NOT a metric).

The 10 strings are the ones from the original DistilBERT probe (predict_distilbert.py,
methodology log section 4.1), run here through the same models for a direct comparison:
fine-tuned DistilBERT (English-only pretraining), fine-tuned multilingual MiniLM, frozen
multilingual MiniLM + LogReg, and Logistic Regression v2 as a classical reference.
Scoring is the same as in the original probe (label must match the expected category).

A second, tiny control set of meaningless strings checks the working hypothesis in log
section 4.1 that Transportation acts as a fallback class for unfamiliar input.

Run from the project root (about a minute on CPU):
    uv run python src/eval/german_probe.py
Writes reports/german_probe/german_probe.csv and german_probe_gibberish.csv
"""
# pylint: disable=wrong-import-position,import-error
# ruff: noqa: E402
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root

import joblib
import pandas as pd

from src.eval.wrapper_ablation_transformers import load_predictor

V2 = Path("models/classical_v2")
OUT = Path("reports/german_probe")

# (text, expected category, language of the string)
PROBE = [
    ("STARBUCKS STORE 1234", "Restaurants", "EN"),
    ("[debit] WITHDRAWAL FROM MIDAS", "Transportation", "EN"),
    ("REWE-SAG-7342", "Groceries", "DE"),
    ("EDEKA 4471 FFM", "Groceries", "DE"),
    ("ALDI SUED 0091", "Groceries", "DE"),
    ("DM DROGERIE MARKT", "Healthcare", "DE"),
    ("TELEKOM DEUTSCHLAND", "Utilities", "DE"),
    ("DB FERNVERKEHR AG", "Travel", "DE"),
    ("MAINOVA AG FFM", "Utilities", "DE"),
    ("APOTHEKE AM DOM", "Healthcare", "DE"),
]
GIBBERISH = ["XQZ KLM 9921", "ZZZZ WWWW", "ASDF QWER 1234", "LOREM IPSUM DOLOR",
             "#### ****", "QPXV LRTN WDBK", "ABCD EFGH IJKL", "0000 1111 2222"]

MODELS = {
    "distilbert_finetuned": "DistilBERT (fine-tuned)",
    "minilm_finetuned": "MiniLM (fine-tuned)",
    "frozen_embedding_logreg": "Frozen MiniLM + LogReg",
    "logistic_regression": "Logistic Regression v2",
}


def load_model(name):
    """Returns fn(list[str]) -> probability matrix for any of the four models."""
    if name == "logistic_regression":
        union = joblib.load(V2 / "feature_union.joblib")
        clf = joblib.load(V2 / "logistic_regression.joblib")
        return lambda texts: clf.predict_proba(union.transform(list(texts)))
    return load_predictor(name)


def main():
    classes = json.loads((V2 / "labels.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    texts = [t for t, _, _ in PROBE]
    rows, gib_rows = [], []
    for key, title in MODELS.items():
        print(f"Loading {title} ...", flush=True)
        predict = load_model(key)
        for (text, expected, lang), p in zip(PROBE, predict(texts)):
            i = int(p.argmax())
            rows.append(dict(model=title, text=text, lang=lang, expected=expected,
                             predicted=classes[i], confidence=round(float(p[i]), 3),
                             correct=classes[i] == expected))
        for text, p in zip(GIBBERISH, predict(GIBBERISH)):
            i = int(p.argmax())
            gib_rows.append(dict(model=title, text=text, predicted=classes[i],
                                 confidence=round(float(p[i]), 3)))
    df = pd.DataFrame(rows)
    gib = pd.DataFrame(gib_rows)
    df.to_csv(OUT / "german_probe.csv", index=False)
    gib.to_csv(OUT / "german_probe_gibberish.csv", index=False)

    print("\n=== PROBE: predictions (confidence) per string ===")
    wide = df.assign(cell=df.predicted + " (" + (df.confidence * 100).round(0)
                     .astype(int).astype(str) + "%)")
    print(wide.pivot(index="text", columns="model", values="cell")
          .reindex(texts).to_string())

    print("\n=== PROBE: correct out of 10 (and out of the 8 German strings) ===")
    for title in MODELS.values():
        sub = df[df.model == title]
        wrong = sub[~sub.correct]
        print(f"{title:26s} {int(sub.correct.sum()):2d}/10 | "
              f"German {int(sub[sub.lang == 'DE'].correct.sum())}/8 | "
              f"mean confidence when wrong "
              f"{wrong.confidence.mean() if len(wrong) else float('nan'):.2f} | "
              f"wrong predicted Transportation: "
              f"{int((wrong.predicted == 'Transportation').sum())}/{len(wrong)}")

    print("\n=== GIBBERISH control: how often is each class predicted? ===")
    for title in MODELS.values():
        sub = gib[gib.model == title]
        counts = {k: int(v) for k, v in sub.predicted.value_counts().items()}
        print(f"{title:26s} {counts} "
              f"(mean confidence {sub.confidence.mean():.2f})")
    print(f"\nSaved to {OUT}/")


if __name__ == "__main__":
    main()