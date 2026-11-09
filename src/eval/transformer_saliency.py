"""
SmartSpend -- word-level saliency for a fine-tuned transformer, by OCCLUSION.

Method: remove one word at a time from the transaction text and measure how much the
model's probability for a class drops. For a word w:
    support_pred(w) = P(predicted class | full text) - P(predicted class | text without w)
    support_true(w) = P(true class      | full text) - P(true class      | text without w)
Positive = the word supports that class; negative = the word pulls against it.
(Occlusion is model-agnostic and, unlike attention weights, measures what the output
actually depends on. It looks at one word at a time, so interactions between words are
not captured.)

Part A  Selected examples (Zelle, Family Dollar, Von Maur, UPS Store, Paternity Leave),
        shown word by word.
Part B  The PREAUTHORIZED WITHDRAWAL cluster: whole parts of the string are removed
        (tag, the phrase "PREAUTHORIZED WITHDRAWAL [TO]", the merchant) and the mean
        change in P(true class) and P(Other) is reported separately for rows the model
        gets right and wrong.

Run from the project root (a few minutes on CPU):
    uv run python src/eval/transformer_saliency.py                       # DistilBERT
    uv run python src/eval/transformer_saliency.py --model minilm_finetuned
Options: --per-group 2 --text "any transaction text" (repeatable)
Writes reports/error_analysis/saliency_<model>_examples.csv and _wrapper_roles.csv
"""
# pylint: disable=wrong-import-position,import-error
# ruff: noqa: E402
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from wrapper_ablation_transformers import PHRASE, load_predictor

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root
sys.path.insert(0, str(Path(__file__).resolve().parent))      # sibling module


V2 = Path("models/classical_v2")
OUT = Path("reports/error_analysis")
GROUPS = ["zelle payment from", "family dollar",
          "von maur", "ups store", "paternity leave"]


def word_saliency(predict, text, true_idx):
    words = text.split()
    texts = [text] + [" ".join(words[:i] + words[i + 1:])
                      for i in range(len(words))]
    p = predict(texts)
    full, drops = p[0], p[1:]
    pred = int(full.argmax())
    return words, pred, full, [(full[pred] - d[pred], full[true_idx] - d[true_idx]) for d in drops]


def roles(tokens):
    """Split a wrapper string into index sets: tag, phrase (incl. the following 'to'), merchant."""
    low = [t.lower() for t in tokens]
    tag = {0} if tokens and tokens[0].startswith(
        "[") and tokens[0].endswith("]") else set()
    phrase = {i for i, t in enumerate(low) if t in (
        "preauthorized", "withdrawal")}
    for i, t in enumerate(low):
        if t == "withdrawal" and i + 1 < len(low) and low[i + 1] == "to":
            phrase.add(i + 1)
    merchant = set(range(len(tokens))) - tag - phrase
    return {"tag": tag, "phrase": phrase, "merchant": merchant}


def wrapper_roles(predict, texts, y, other):
    p_full = predict(texts)
    pred = p_full.argmax(1)
    correct = pred == y
    res = {}
    for role in ("tag", "phrase", "merchant"):
        keep, new = [], []
        for i, s in enumerate(texts):
            toks = s.split()
            drop = roles(toks)[role]
            if drop:
                keep.append(i)
                new.append(
                    " ".join(t for j, t in enumerate(toks) if j not in drop))
        p_new = predict(new)
        keep = np.array(keep)
        d_true = p_full[keep, y[keep]] - p_new[np.arange(len(keep)), y[keep]]
        d_other = p_full[keep, other] - p_new[:, other]
        for label, mask in (("right", correct[keep]), ("wrong", ~correct[keep])):
            if mask.any():
                res[(role, label)] = dict(n=int(mask.sum()),
                                          mean_drop_p_true=float(
                                              d_true[mask].mean()),
                                          mean_drop_p_other=float(d_other[mask].mean()))
    return res, correct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilbert_finetuned")
    ap.add_argument("--per-group", type=int, default=2)
    ap.add_argument("--text", action="append", default=[])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    classes = json.loads((V2 / "labels.json").read_text())
    idx = {c: i for i, c in enumerate(classes)}
    ood = pd.read_csv("data/processed/test_ood.csv")
    ood["transaction_description"] = ood["transaction_description"].astype(str)
    predict = load_predictor(a.model)

    # ---- Part A ------------------------------------------------------------------------
    cases = []  # (text, true_label)
    for g in GROUPS:
        hit = ood[ood["transaction_description"].str.lower(
        ).str.contains(g, regex=False)]
        for _, r in hit.sample(min(a.per_group, len(hit)), random_state=42).iterrows():
            cases.append((r["transaction_description"], r["final_category"]))
    cases += [(t, "?") for t in a.text]

    rows = []
    print(f"PART A -- word-level occlusion, {a.model}\n")
    for text, true in cases:
        ti = idx.get(true, 0)
        words, pred, full, sal = word_saliency(predict, text, ti)
        print(f"{text}\n   true={true}  predicted={classes[pred]} (P={full[pred]:.3f}; "
              f"P(true)={full[ti]:.3f})")
        for w, (sp, st) in zip(words, sal):
            print(
                f"     {w:28s} support_pred {sp:+.3f}   support_true {st:+.3f}")
            rows.append(dict(text=text, true=true, predicted=classes[pred], word=w,
                             support_pred=sp, support_true=st))
        print()
    pd.DataFrame(rows).to_csv(
        OUT / f"saliency_{a.model}_examples.csv", index=False)

    # ---- Part B ------------------------------------------------------------------------
    sub = ood[ood["transaction_description"].str.contains(PHRASE)]
    y = sub["final_category"].map(idx).to_numpy()
    res, correct = wrapper_roles(
        predict, sub["transaction_description"].tolist(), y, idx["Other"])
    print("=" * 78)
    print(f"PART B -- PREAUTHORIZED WITHDRAWAL cluster ({len(sub)} rows), {a.model}: "
          f"accuracy {correct.mean():.3f}")
    print("Mean DROP in probability when that part of the string is removed "
          "(positive = the part supported it):")
    print(f"  {'part':10s} {'rows':6s} {'n':>5s}  {'drop in P(true)':>16s}  {'drop in P(Other)':>17s}")
    out = []
    for (role, label), r in res.items():
        print(f"  {role:10s} {label:6s} {r['n']:5d}  {r['mean_drop_p_true']:+16.3f}  "
              f"{r['mean_drop_p_other']:+17.3f}")
        out.append(dict(model=a.model, part=role, rows=label, **r))
    pd.DataFrame(out).to_csv(
        OUT / f"saliency_{a.model}_wrapper_roles.csv", index=False)
    print(f"\nSaved CSVs to {OUT}")


if __name__ == "__main__":
    main()
