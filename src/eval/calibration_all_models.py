"""
SmartSpend -- Calibration (ECE + reliability diagrams) for all eight models.

A model is well calibrated if "I am 80% confident" is right about 80% of the
time. ECE is the sample-weighted average gap between confidence and accuracy
over confidence bins. Evaluated on BOTH the in-distribution test set and the
merchant-held-out OOD test set, because the interesting question is whether
confidence stays honest when the merchants are unseen.

Models (all 13-class, alphabetical label order):
  classical v2 (tuned, from models/classical_v2/): logistic_regression,
      multinomial_nb, linear_svm, random_forest, gradient_boosting
  frozen MiniLM embeddings + LogReg  (models/embedding_logreg_classifier.joblib)
  fine-tuned DistilBERT              (models/distilbert_finetuned/final)
  fine-tuned MiniLM                  (models/minilm_finetuned/final)

Linear SVM has no probabilities. It is wrapped in a sigmoid (Platt) calibrator
fitted on the VALIDATION split with the SVM itself frozen -- no retraining,
test/OOD untouched. This is the one model that receives an explicit
calibration step, so its ECE is not like-for-like with the others; the output
marks it "linear_svm (Platt on val)". Also reported: raw SVM scores pushed
through a softmax are NOT probabilities and are not used.

Outputs (reports/calibration/):
  ece_summary.csv        one row per model x split
  ece_bins.csv           per-bin table behind every diagram
  reliability_ood.png    8-panel reliability diagram, OOD test
  reliability_test.png   same, in-distribution test
  probs/<model>_<split>.npy   cached probabilities (reuse for error analysis)

Run from the project root (transformer inference on CPU: ~10-20 min the first
time, cached afterwards; add --no-transformers to skip them):
    uv run python src/eval/calibration_all_models.py
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import sparse  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV  # noqa: E402
from sklearn.preprocessing import LabelEncoder  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

DATA_DIR = Path("data/processed")
V2_DIR = Path("models/classical_v2")
FEAT_DIR = Path("data/features_v2")
EMB_DIR = Path("data/embeddings")
OUT_DIR = Path("reports/calibration")
PROB_DIR = OUT_DIR / "probs"
for d in (OUT_DIR, PROB_DIR):
    d.mkdir(parents=True, exist_ok=True)

N_BINS = 10
MAX_LENGTH = 32
SPLITS = {"test": "test_in_distribution", "ood": "test_ood"}

CLASSICAL = ["logistic_regression", "multinomial_nb", "linear_svm",
             "random_forest", "gradient_boosting"]
DISPLAY = {
    "logistic_regression": "Logistic Regression v2",
    "multinomial_nb": "Naive Bayes v2",
    "linear_svm": "Linear SVM v2 (Platt on val)",
    "random_forest": "Random Forest v2",
    "gradient_boosting": "XGBoost v2",
    "frozen_embedding_logreg": "Frozen MiniLM + LogReg",
    "distilbert_finetuned": "DistilBERT fine-tuned",
    "minilm_finetuned": "MiniLM fine-tuned",
}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def calibration_bins(conf: np.ndarray, correct: np.ndarray, n_bins: int = N_BINS):
    """Returns (ece, mce, bin DataFrame). Bins are equal-width on [0, 1]."""
    edges = np.linspace(0, 1, n_bins + 1)
    ids = np.digitize(conf, edges[1:-1])
    rows, ece, mce = [], 0.0, 0.0
    for b in range(n_bins):
        m = ids == b
        n = int(m.sum())
        if n == 0:
            continue
        c, a = conf[m].mean(), correct[m].mean()
        ece += n / len(conf) * abs(a - c)
        mce = max(mce, abs(a - c))
        rows.append({"bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}", "count": n,
                     "avg_confidence": c, "accuracy": a, "gap": a - c})
    return ece, mce, pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_labels():
    frames = {s: pd.read_csv(DATA_DIR / f"{f}.csv") for s, f in
              {**SPLITS, "val": "val"}.items()}
    classes = json.loads((V2_DIR / "labels.json").read_text())
    enc = LabelEncoder().fit(classes)
    y = {s: enc.transform(df["final_category"]) for s, df in frames.items()}
    texts = {s: df["transaction_description"].astype(
        str).tolist() for s, df in frames.items()}
    return classes, y, texts


def cache_path(model: str, split: str) -> Path:
    return PROB_DIR / f"{model}_{split}.npy"


def cached(model: str, splits):
    if all(cache_path(model, s).exists() for s in splits):
        return {s: np.load(cache_path(model, s)) for s in splits}
    return None


def save(model: str, probs: dict):
    for s, p in probs.items():
        np.save(cache_path(model, s), p)


# ---------------------------------------------------------------------------
# Probability producers
# ---------------------------------------------------------------------------
def classical_probs(name: str, y_val):
    got = cached(name, list(SPLITS))
    if got:
        return got
    X = {s: sparse.load_npz(FEAT_DIR / f"{f}.npz")
         for s, f in {**SPLITS, "val": "val"}.items()}
    ubj = V2_DIR / f"{name}.ubj"
    if name == "gradient_boosting" and ubj.exists():
        # Native XGBoost format: portable between Colab (GPU) and this machine.
        from xgboost import XGBClassifier
        model = XGBClassifier()
        model.load_model(str(ubj))
    else:
        model = joblib.load(V2_DIR / f"{name}.joblib")
    if name == "linear_svm":
        try:  # scikit-learn >= 1.6
            from sklearn.frozen import FrozenEstimator
            cal = CalibratedClassifierCV(
                FrozenEstimator(model), method="sigmoid")
        except ImportError:
            cal = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
        cal.fit(X["val"], y_val)
        model = cal
    probs = {s: np.asarray(model.predict_proba(X[s])) for s in SPLITS}
    save(name, probs)
    return probs


def frozen_embedding_probs(classes):
    got = cached("frozen_embedding_logreg", list(SPLITS))
    if got:
        return got
    clf = joblib.load(Path("models/embedding_logreg_classifier.joblib"))
    assert list(clf.classes_) == classes, "label order differs from classical_v2"
    probs = {}
    for s, f in SPLITS.items():
        emb = EMB_DIR / f"{f}_embeddings.npy"
        probs[s] = clf.predict_proba(np.load(emb))
    save("frozen_embedding_logreg", probs)
    return probs


def transformer_probs(name: str, path: Path, texts):
    got = cached(name, list(SPLITS))
    if got:
        return got
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).eval()
    probs = {}
    for s in SPLITS:
        out, t = [], texts[s]
        print(f"  {name}/{s}: {len(t)} rows (CPU) ...", flush=True)
        with torch.no_grad():
            for i in range(0, len(t), 128):
                enc = tok(t[i:i + 128], return_tensors="pt", truncation=True,
                          max_length=MAX_LENGTH, padding=True)
                out.append(torch.softmax(model(**enc).logits, dim=-1).numpy())
        probs[s] = np.concatenate(out)
    save(name, probs)
    return probs


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def reliability_figure(split: str, all_bins: dict, summary: pd.DataFrame, out: Path):
    names = list(all_bins)
    cols = 4
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(
        4.2 * cols, 4.2 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.set_visible(False)
    for ax, name in zip(axes.ravel(), names):
        ax.set_visible(True)
        b = all_bins[name][split]
        ece = summary[(summary.model == name) & (
            summary.split == split)].ece.iloc[0]
        ax.plot([0, 1], [0, 1], "--", color="0.6", lw=1, label="perfect")
        ax.scatter(b.avg_confidence, b.accuracy, s=20 + 380 * b["count"] / b["count"].sum(),
                   color="#2a6fbb", alpha=0.8, edgecolor="white")
        ax.plot(b.avg_confidence, b.accuracy, color="#2a6fbb", lw=1)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"{DISPLAY[name]}\nECE = {ece:.3f}", fontsize=10)
        ax.set_xlabel("mean confidence")
        ax.set_ylabel("accuracy")
        ax.grid(alpha=0.25)
    label = "OOD (unseen merchants)" if split == "ood" else "in-distribution"
    fig.suptitle(f"Reliability diagrams, {label} test set "
                 "(dot size = share of predictions; below diagonal = overconfident)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-transformers", action="store_true",
                    help="skip DistilBERT / MiniLM inference")
    args = ap.parse_args()

    classes, y, texts = load_labels()
    producers = {n: (lambda n=n: classical_probs(
        n, y["val"])) for n in CLASSICAL}
    producers["frozen_embedding_logreg"] = lambda: frozen_embedding_probs(
        classes)
    if not args.no_transformers:
        for n in ("distilbert_finetuned", "minilm_finetuned"):
            producers[n] = (lambda n=n: transformer_probs(
                n, Path("models") / n / "final", texts))

    summary_rows, bin_rows, all_bins = [], [], {}
    for name, produce in producers.items():
        print(f"\n== {DISPLAY[name]}", flush=True)
        probs = produce()
        all_bins[name] = {}
        for split in SPLITS:
            p = probs[split]
            assert p.shape == (len(y[split]), len(
                classes)), (name, split, p.shape)
            conf = p.max(axis=1)
            correct = (p.argmax(axis=1) == y[split]).astype(float)
            ece, mce, bins = calibration_bins(conf, correct)
            wrong = correct == 0
            summary_rows.append({
                "model": name, "split": split, "accuracy": round(correct.mean(), 4),
                "mean_confidence": round(conf.mean(), 4), "ece": round(ece, 4),
                "mce": round(mce, 4),
                "conf_when_wrong": round(conf[wrong].mean(), 4) if wrong.any() else np.nan,
                "share_wrong_above_0.9": round(float((conf[wrong] > 0.9).mean()), 4)
                if wrong.any() else np.nan,
            })
            all_bins[name][split] = bins
            bin_rows.append(bins.assign(model=name, split=split))
            print(
                f"  {split}: acc {correct.mean():.4f} | conf {conf.mean():.4f} | ECE {ece:.4f}")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_DIR / "ece_summary.csv", index=False)
    pd.concat(bin_rows).to_csv(OUT_DIR / "ece_bins.csv", index=False)
    for split in SPLITS:
        reliability_figure(split, all_bins, summary,
                           OUT_DIR / f"reliability_{split}.png")

    wide = summary.pivot(index="model", columns="split", values="ece")
    wide["ood_minus_test"] = wide["ood"] - wide["test"]
    print("\n\n=== ECE (lower = better calibrated) ===")
    print(wide.sort_values("ood").round(4).to_string())
    print(f"\nSaved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
