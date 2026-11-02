"""
SmartSpend -- Error analysis, part A: where do the models fail, and are the
failures the model's fault or the labels'?

Uses the cached OOD probabilities of all eight models (reports/calibration/probs/,
written by calibration_all_models.py), so it needs no model loading and runs
in seconds.

What it produces (reports/error_analysis/):
  top_confused_pairs_all_models.csv   top-10 confused (true -> predicted) pairs for
                                      all EIGHT models (v2 classical included)
  examples_<True>_to_<Pred>.csv       30 misclassified OOD transactions per focus
                                      pair, to READ and label by hand (columns
                                      `verdict` and `note` are left blank)

What it prints, for each focus pair (Income -> Other, Shopping -> Transportation):
  * how many of the eight models make the same error on the same transaction
    (if nearly all of them do, the transaction is probably ambiguous or
    mislabelled, not a model weakness)
  * how often the SAME normalised text appears in the training set, and with
    which labels (a direct label-consistency check)
  * the most common leading words among the errors versus among the correctly
    classified rows of the same true category

Run from the project root:
    uv run python src/eval/error_analysis_examples.py
Options: --n 30 (examples per pair), --seed 42
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data/processed")
PROB_DIR = Path("reports/calibration/probs")
V2_DIR = Path("models/classical_v2")
OUT_DIR = Path("reports/error_analysis")

MODELS = ["logistic_regression", "multinomial_nb", "linear_svm", "random_forest",
          "gradient_boosting", "frozen_embedding_logreg", "distilbert_finetuned",
          "minilm_finetuned"]
SHORT = {"logistic_regression": "LR", "multinomial_nb": "NB", "linear_svm": "SVM",
         "random_forest": "RF", "gradient_boosting": "XGB",
         "frozen_embedding_logreg": "Frozen", "distilbert_finetuned": "DistilBERT",
         "minilm_finetuned": "MiniLM"}
PAIRS = [("Income", "Other"), ("Shopping", "Transportation")]
LEAD = "distilbert_finetuned"  # examples are sampled from this model's errors


def normalise(text) -> str:
    """Lower-case, digits -> 0, collapsed whitespace: matches 'same string up to numbers'."""
    return re.sub(r"\s+", " ", re.sub(r"\d+", "0", str(text).lower())).strip()


def first_word(text) -> str:
    parts = normalise(text).split(" ")
    return parts[0] if parts and parts[0] else "(empty)"


def top_pairs(y, pred, classes, k=10):
    n = len(classes)
    cm = np.bincount(y * n + pred, minlength=n * n).reshape(n, n)
    np.fill_diagonal(cm, 0)
    total_err = cm.sum()
    order = np.argsort(cm, axis=None)[::-1][:k]
    return [(classes[i // n], classes[i % n], int(cm.flat[i]),
             round(cm.flat[i] / total_err, 4)) for i in order if cm.flat[i] > 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--probe", nargs="*",
                    default=["zelle payment from", "zelle", "family dollar", "von maur",
                             "ups store", "bloomingdale"],
                    help="substrings to look up in the training and OOD sets")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    classes = json.loads((V2_DIR / "labels.json").read_text())
    idx = {c: i for i, c in enumerate(classes)}
    ood = pd.read_csv(DATA_DIR / "test_ood.csv")
    train = pd.read_csv(DATA_DIR / "train.csv")
    y = ood["final_category"].map(idx).to_numpy()
    texts = ood["transaction_description"].astype(str)

    probs = {m: np.load(PROB_DIR / f"{m}_ood.npy") for m in MODELS}
    pred = {m: p.argmax(1) for m, p in probs.items()}
    conf = {m: p.max(1) for m, p in probs.items()}
    for m in MODELS:
        assert len(pred[m]) == len(
            y), f"{m}: {len(pred[m])} predictions for {len(y)} rows"

    # ---- 1. top confused pairs for all eight models --------------------------------
    rows = []
    print("=" * 78)
    print("TOP CONFUSED PAIRS (OOD, all eight models; count and share of that model's errors)")
    print("=" * 78)
    for m in MODELS:
        pairs = top_pairs(y, pred[m], classes)
        n_err = int((pred[m] != y).sum())
        print(
            f"\n{SHORT[m]}  ({n_err} errors, accuracy {1 - n_err / len(y):.4f})")
        for rank, (t, p, c, share) in enumerate(pairs, 1):
            rows.append(dict(model=m, rank=rank, true=t, predicted=p, count=c,
                             share_of_model_errors=share))
            if rank <= 5:
                print(f"  {rank}. {t} -> {p}: {c}  ({share:.1%})")
    pd.DataFrame(rows).to_csv(
        OUT_DIR / "top_confused_pairs_all_models.csv", index=False)

    # ---- 2.-5. focus pairs ---------------------------------------------------------
    train_norm = train["transaction_description"].map(normalise)
    train_labels = {}
    for t, lab in zip(train_norm, train["final_category"]):
        train_labels.setdefault(t, Counter())[lab] += 1
    norm_ood = texts.map(normalise)

    for true_c, pred_c in PAIRS:
        ti, pi = idx[true_c], idx[pred_c]
        is_true = y == ti
        err = {m: is_true & (pred[m] == pi) for m in MODELS}
        # models making this error, per row
        n_same = np.sum([err[m] for m in MODELS], axis=0)
        lead_err = err[LEAD]

        print("\n" + "=" * 78)
        print(
            f"{true_c.upper()} -> {pred_c.upper()}   ({int(is_true.sum())} {true_c} rows in OOD)")
        print("=" * 78)
        print("Errors per model:", {
              SHORT[m]: int(err[m].sum()) for m in MODELS})
        print(f"\nOf the {true_c} rows, how many models predict {pred_c}?")
        for k in (8, 6, 4, 1):
            c = int((is_true & (n_same >= k)).sum())
            print(
                f"  at least {k} of 8 models: {c} rows ({c / is_true.sum():.1%})")
        shared = int((lead_err & (n_same >= 6)).sum())
        print(f"{LEAD}'s errors on this pair that >=6 models also make: "
              f"{shared} of {int(lead_err.sum())} ({shared / max(lead_err.sum(), 1):.1%})")

        # label-consistency check against the training set
        match = norm_ood[lead_err].map(lambda s: train_labels.get(s))
        found = match.dropna()
        print(f"\nTraining-set check: {len(found)} of {int(lead_err.sum())} of these errors "
              f"have an identical normalised string in the training set.")
        if len(found):
            agg = Counter()
            for c in found:
                agg.update(c)
            tot = sum(agg.values())
            print("  labels of those training matches:",
                  {k: f"{v} ({v / tot:.0%})" for k, v in agg.most_common(5)})

        # leading words
        ok = is_true & (pred[LEAD] == ti)
        fe = Counter(texts[lead_err].map(first_word))
        fo = Counter(texts[ok].map(first_word))
        print(
            f"\nMost common first words, {LEAD} errors ({true_c}->{pred_c}):")
        print("  ", ", ".join(f"{w} ({c})" for w, c in fe.most_common(12)))
        print(f"Most common first words, {true_c} rows classified correctly:")
        print("  ", ", ".join(f"{w} ({c})" for w, c in fo.most_common(12)))

        # examples to read: half with strong consensus (>=6 models agree), half the rest
        cand = np.where(lead_err)[0]
        rng = np.random.default_rng(args.seed)
        strong = [i for i in cand if n_same[i] >= 6]
        weak = [i for i in cand if n_same[i] < 6]
        half = args.n // 2
        take_s = list(rng.choice(strong, size=min(
            half, len(strong)), replace=False)) if strong else []
        take_w = list(rng.choice(weak, size=min(
            args.n - len(take_s), len(weak)), replace=False)) if weak else []
        chosen = sorted(take_s + take_w, key=lambda i: -n_same[i])

        ex = pd.DataFrame({
            "text": texts.iloc[chosen].values,
            "true": true_c,
            "pred_distilbert": pred_c,
            "conf_distilbert": np.round(conf[LEAD][chosen].astype(float), 3),
            "pred_lr": [classes[pred["logistic_regression"][i]] for i in chosen],
            "pred_minilm": [classes[pred["minilm_finetuned"][i]] for i in chosen],
            "models_making_this_error": n_same[chosen],
            "train_labels_same_string": [
                "|".join(f"{k}:{v}" for k, v in
                         train_labels.get(norm_ood.iloc[i], Counter()).most_common(3))
                for i in chosen],
            "verdict": "",   # fill in by hand: label_wrong / ambiguous / model_wrong
            "note": "",
        })
        out = OUT_DIR / f"examples_{true_c}_to_{pred_c}.csv"
        ex.to_csv(out, index=False)
        print(f"\nSaved {len(ex)} examples to {out}")
        print(
            f"\n--- {len(ex)} examples to read (true={true_c}, DistilBERT predicted {pred_c}) ---")
        for _, r in ex.iterrows():
            print(f"[{r.models_making_this_error}/8] {r.text}   "
                  f"(conf {r.conf_distilbert}, LR:{r.pred_lr}, MiniLM:{r.pred_minilm}"
                  f"{', train:' + r.train_labels_same_string if r.train_labels_same_string else ''})")

    # ---- 6. probes: how is a given merchant / template labelled in train vs OOD? -------
    print("\n" + "=" * 78)
    print("PROBES: label counts in the TRAINING set vs the OOD set, and what models predict")
    print("=" * 78)
    low_all = texts.str.lower()
    train_low = train["transaction_description"].astype(str).str.lower()
    for term in args.probe:
        t = term.lower()
        tr = train[train_low.str.contains(t, regex=False)]
        mask = low_all.str.contains(t, regex=False).to_numpy()
        print(f"\n'{term}': {len(tr)} training rows, {int(mask.sum())} OOD rows")
        if len(tr):
            print("  training labels:", {k: int(v) for k, v in
                                         tr["final_category"].value_counts().head(5).items()})
        if mask.any():
            print("  OOD true labels:", {k: int(v) for k, v in
                                         ood.loc[mask, "final_category"].value_counts().head(5).items()})
            for m in (LEAD, "minilm_finetuned", "logistic_regression", "gradient_boosting"):
                got = pd.Series([classes[i]
                                for i in pred[m][mask]]).value_counts().head(4)
                print(f"  {SHORT[m]:>10} predicts:", {k: int(v)
                      for k, v in got.items()})

    print(
        f"\nSaved {OUT_DIR / 'top_confused_pairs_all_models.csv'} and the example files.")


if __name__ == "__main__":
    main()
