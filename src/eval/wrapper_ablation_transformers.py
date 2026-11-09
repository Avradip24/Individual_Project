"""
SmartSpend -- wrapper-removal counterfactual for the TRANSFORMER-type models
(fine-tuned DistilBERT, fine-tuned MiniLM, frozen MiniLM + LogReg). Companion to
wrapper_ablation.py, which did the five classical models.

Takes the 951 OOD rows containing `PREAUTHORIZED WITHDRAWAL` and re-scores them in three
versions: original / phrase removed ("TO" kept) / merchant only. Also checks that the
predictions on the ORIGINAL text reproduce the cached OOD predictions, so you know the
model was loaded correctly (max_length 32, same as training).

Run from the project root (a few minutes on CPU):
    uv run python src/eval/wrapper_ablation_transformers.py
Options: --models distilbert_finetuned minilm_finetuned frozen_embedding_logreg
Writes reports/feature_importance/wrapper_ablation_transformers.csv
"""
# pylint: disable=wrong-import-position,import-error
# ruff: noqa: E402
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root


V2 = Path("models/classical_v2")
PROBS = Path("reports/calibration/probs")
OUT = Path("reports/feature_importance")
MAX_LENGTH = 32
BATCH = 64
ENCODER_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
PHRASE = re.compile(r"preauthorized\s+withdrawal", re.I)
PHRASE_TO = re.compile(r"preauthorized\s+withdrawal\s+to\b\s*", re.I)
TITLES = {"distilbert_finetuned": "Fine-tuned DistilBERT",
          "minilm_finetuned": "Fine-tuned MiniLM",
          "frozen_embedding_logreg": "Frozen MiniLM + LogReg"}


def variants(s: str) -> dict:
    return {
        "original": s,
        "no_phrase": re.sub(r"\s+", " ", PHRASE.sub("", s)).strip(),
        "merchant_only": re.sub(r"\s+", " ", PHRASE_TO.sub("", s)).strip(),
    }


def load_predictor(name: str):
    """Returns fn(list[str]) -> class-probability array (n, 13), classes in alphabetical order."""
    if name == "frozen_embedding_logreg":
        import joblib
        from sentence_transformers import SentenceTransformer
        clf = joblib.load(Path("models/embedding_logreg_classifier.joblib"))
        enc = SentenceTransformer(ENCODER_NAME, device="cpu")

        def predict_frozen(texts):
            emb = enc.encode(list(texts), batch_size=BATCH, show_progress_bar=False,
                             convert_to_numpy=True)
            return clf.predict_proba(emb)

        return predict_frozen

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    path = Path("models") / name / "final"
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).eval()

    def predict_transformer(texts):
        texts = list(texts)
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), BATCH):
                enc = tok(texts[i:i + BATCH], return_tensors="pt", truncation=True,
                          max_length=MAX_LENGTH, padding=True)
                out.append(torch.softmax(model(**enc).logits, -1).numpy())
        return np.concatenate(out)

    return predict_transformer


def evaluate(predict, texts_by_variant, y, other):
    res = {}
    for k, tx in texts_by_variant.items():
        p = predict(tx)
        pred = p.argmax(1)
        res[k] = dict(accuracy=float((pred == y).mean()),
                      share_predicted_other=float((pred == other).mean()), pred=pred)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(TITLES))
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    classes = json.loads((V2 / "labels.json").read_text())
    idx = {c: i for i, c in enumerate(classes)}
    ood = pd.read_csv("data/processed/test_ood.csv")
    m = ood["transaction_description"].astype(
        str).str.contains(PHRASE).to_numpy()
    sub = ood[m]
    y = sub["final_category"].map(idx).to_numpy()
    raw = sub["transaction_description"].astype(str).tolist()
    texts = {k: [variants(s)[k] for s in raw]
             for k in ("original", "no_phrase", "merchant_only")}
    print(f"{len(sub)} rows in the PREAUTHORIZED WITHDRAWAL cluster")

    rows = []
    for name in a.models:
        print(f"\nLoading {name} ...")
        predict = load_predictor(name)
        res = evaluate(predict, texts, y, idx["Other"])
        cached = np.load(PROBS / f"{name}_ood.npy").argmax(1)[m]
        agree = float((res["original"]["pred"] == cached).mean())
        print(f"  sanity: predictions on the original text agree with the cached predictions "
              f"on {agree:.1%} of rows (expect ~100%)")
        for k, r in res.items():
            rows.append(dict(model=name, version=k, accuracy=r["accuracy"],
                             share_predicted_other=r["share_predicted_other"],
                             agreement_with_cached=agree if k == "original" else np.nan))
        print("  " + "  ".join(f"{k}: {res[k]['accuracy']:.3f} ({res[k]['share_predicted_other']:.0%} Other)"
                               for k in res))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "wrapper_ablation_transformers.csv", index=False)
    print("\nAccuracy inside the cluster (share predicted Other in brackets):")
    for name in a.models:
        d = df[df.model == name].set_index("version")
        print(f"  {TITLES[name]:24s} " + "  ".join(
            f"{k}: {d.at[k, 'accuracy']:.3f} ({d.at[k, 'share_predicted_other']:.0%})"
            for k in ("original", "no_phrase", "merchant_only")))
    print(f"\nSaved {OUT / 'wrapper_ablation_transformers.csv'}")


if __name__ == "__main__":
    main()
