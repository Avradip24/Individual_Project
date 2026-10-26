"""
SmartSpend -- Learning curves (laptop part): classical v2 models + frozen embedding.

Question: how much labelled data does each model need, and do the SMALL
categories suffer first? Every model is retrained on stratified fractions of
the training set and evaluated on the full in-distribution test set and the
merchant-held-out OOD set. Hyperparameters are the TUNED v2 settings (copied
from the saved models), identical across fractions; only the amount of
training data changes.

Category-size groups: the 13 categories are ranked by their size in the full
training set; the 4 smallest are "small", the 4 largest "large", the rest
"medium". Group scores are the mean per-category F1 inside the group, so the
curves show whether low-resource categories degrade faster than big ones.

Outputs (reports/learning_curves/):
  learning_curves_classical.csv   one row per model x seed x fraction x split
  per_category_classical.csv      per-category F1 behind every point
  learning_curves.png             cost-of-data figure (all models found in
                                  learning_curves_*.csv, incl. the Colab DistilBERT run)

Run from the project root. Defaults finish in roughly 15-30 min on a laptop:
    uv run python src/eval/learning_curves.py
    uv run python src/eval/learning_curves.py --models gradient_boosting    # slow on CPU (~1 h)
    uv run python src/eval/learning_curves.py --plot-only                   # redraw / merge Colab results

Method note (state in the report): the v2 feature vocabulary / IDF was fitted
once on the FULL training text (cached matrices) and rows are subsampled; only
the LABELLED data shrinks. This slightly flatters small fractions.
"""

import argparse
import glob
import json
import sys
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.models.train_classical_v2 import fit_model  # noqa: E402

DATA_DIR = Path("data/processed")
FEAT_DIR = Path("data/features_v2")
EMB_DIR = Path("data/embeddings")
V2_DIR = Path("models/classical_v2")
EMB_CLF = Path("models/embedding_logreg_classifier.joblib")
PROB_DIR = Path("reports/calibration/probs")
OUT_DIR = Path("reports/learning_curves")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FRACTIONS = [0.05, 0.1, 0.25, 0.5, 1.0]
DEFAULT_MODELS = ["logistic_regression", "multinomial_nb", "linear_svm",
                  "random_forest", "frozen_embedding_logreg"]
ALL_MODELS = DEFAULT_MODELS + ["gradient_boosting"]
EVAL_SPLITS = {"test": "test_in_distribution", "ood": "test_ood"}
DISPLAY = {
    "logistic_regression": "Logistic Regression v2", "multinomial_nb": "Naive Bayes v2",
    "linear_svm": "Linear SVM v2", "random_forest": "Random Forest v2",
    "gradient_boosting": "XGBoost v2", "frozen_embedding_logreg": "Frozen MiniLM + LogReg",
    "distilbert_finetuned": "DistilBERT fine-tuned", "minilm_finetuned": "MiniLM fine-tuned",
}


# ---------------------------------------------------------------------------
# Shared helpers (the Colab script contains identical copies)
# ---------------------------------------------------------------------------
def size_groups(train_labels, classes):
    """Maps category -> 'small'/'medium'/'large' by full-train size; also returns counts."""
    counts = Counter(train_labels)
    order = sorted(classes, key=lambda c: (counts[c], c))
    groups = {c: "medium" for c in classes}
    for c in order[:4]:
        groups[c] = "small"
    for c in order[-4:]:
        groups[c] = "large"
    return groups, counts


def score(y_true, pred, classes, groups, counts, meta):
    """Returns (summary_row, per_category_rows) for one prediction vector."""
    k = len(classes)
    f1s = f1_score(y_true, pred, labels=range(k),
                   average=None, zero_division=0)
    support = np.bincount(y_true, minlength=k)
    row = dict(meta, macro_f1=round(float(f1s.mean()), 4),
               accuracy=round(float(accuracy_score(y_true, pred)), 4))
    for g in ("small", "medium", "large"):
        sel = [i for i, c in enumerate(classes) if groups[c] == g]
        row[f"f1_{g}"] = round(float(f1s[sel].mean()), 4)
    per_cat = [dict(meta, category=c, size_group=groups[c], n_train_category=counts[c],
                    f1=round(float(f1s[i]), 4), support=int(support[i]))
               for i, c in enumerate(classes)]
    return row, per_cat


def upsert(path: Path, new: pd.DataFrame):
    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat(
            [old[~old["model"].isin(new["model"].unique())], new], ignore_index=True)
    new.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Model templates (tuned hyperparameters taken from the saved v2 models)
# ---------------------------------------------------------------------------
def xgb_template(device: str):
    from xgboost import XGBClassifier
    try:
        return clone(joblib.load(V2_DIR / "gradient_boosting.joblib")).set_params(device=device)
    except Exception:
        # The GPU-pickled model cannot be loaded here, and a .ubj file does NOT keep its
        # training settings (it reports XGBoost defaults), so read the tuned settings that
        # train_classical_v2.py recorded in classical_v2_models.csv instead.
        table = pd.read_csv("reports/classical_v2/classical_v2_models.csv")
        row = table[table.iloc[:, 0] == "gradient_boosting"].iloc[0]
        params = next(json.loads(v)
                      for v in row if isinstance(v, str) and v.startswith("{"))
        print(
            f"  XGBoost tuned settings from classical_v2_models.csv: {params}")
        return XGBClassifier(tree_method="hist", device=device, random_state=42, n_jobs=-1,
                             eval_metric="mlogloss", **params)


def template(name: str, device: str):
    if name == "frozen_embedding_logreg":
        return clone(joblib.load(EMB_CLF))
    if name == "gradient_boosting":
        return xgb_template(device)
    return clone(joblib.load(V2_DIR / f"{name}.joblib"))


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
def run(models, seeds, fractions, device):
    classes = json.loads((V2_DIR / "labels.json").read_text())
    enc = LabelEncoder().fit(classes)
    frames = {s: pd.read_csv(DATA_DIR / f"{s}.csv")
              for s in ["train", *EVAL_SPLITS.values()]}
    y = {s: enc.transform(df["final_category"]) for s, df in frames.items()}
    groups, counts = size_groups(
        frames["train"]["final_category"].tolist(), classes)
    print("Size groups:", {g: [c for c in classes if groups[c] == g]
          for g in ("small", "medium", "large")})
    print("Full-train category sizes:",
          dict(sorted(counts.items(), key=lambda kv: kv[1])))

    def features(name):
        if name == "frozen_embedding_logreg":
            return {s: np.load(EMB_DIR / f"{s}_embeddings.npy") for s in y}
        return {s: sparse.load_npz(FEAT_DIR / f"{s}.npz") for s in y}

    for name in models:
        print(f"\n===== {DISPLAY[name]} =====", flush=True)
        X, tmpl = features(name), template(name, device)
        rows, cats = [], []
        for seed in seeds:
            for frac in fractions:
                if frac < 1.0:
                    idx, _ = train_test_split(np.arange(len(y["train"])), train_size=frac,
                                              stratify=y["train"], random_state=seed)
                else:
                    idx = np.arange(len(y["train"]))
                # tuned settings and pinned random_state; only the data varies
                model = clone(tmpl)
                t0 = time.time()
                if name == "frozen_embedding_logreg":
                    model.fit(X["train"][idx], y["train"][idx])
                else:
                    fit_model(name, model, X["train"][idx], y["train"][idx])
                fit_s = round(time.time() - t0, 1)
                for split, fname in EVAL_SPLITS.items():
                    meta = dict(model=name, seed=seed, fraction=frac, n_train=int(len(idx)),
                                split=split, fit_time_s=fit_s)
                    r, c = score(y[fname], model.predict(
                        X[fname]), classes, groups, counts, meta)
                    rows.append(r)
                    cats.extend(c)
                ood = rows[-1]
                print(f"  seed {seed} | {frac:>5.0%} ({len(idx):>6} rows) | fit {fit_s:>6}s | "
                      f"OOD F1 {ood['macro_f1']:.4f} (small {ood['f1_small']:.4f}, "
                      f"large {ood['f1_large']:.4f})", flush=True)
        upsert(OUT_DIR / "learning_curves_classical.csv", pd.DataFrame(rows))
        upsert(OUT_DIR / "per_category_classical.csv", pd.DataFrame(cats))


# ---------------------------------------------------------------------------
# Existing full-data DistilBERT point (from cached probabilities) + plot
# ---------------------------------------------------------------------------
def add_existing_transformers():
    """100% points of the fine-tuned transformers, from the cached calibration probabilities."""
    classes = json.loads((V2_DIR / "labels.json").read_text())
    enc = LabelEncoder().fit(classes)
    train = pd.read_csv(DATA_DIR / "train.csv")
    groups, counts = size_groups(train["final_category"].tolist(), classes)
    for name in ("distilbert_finetuned", "minilm_finetuned"):
        out = OUT_DIR / f"learning_curves_{name}_full.csv"
        if out.exists() or not all((PROB_DIR / f"{name}_{s}.npy").exists() for s in EVAL_SPLITS):
            continue
        rows, cats = [], []
        for split, fname in EVAL_SPLITS.items():
            yy = enc.transform(pd.read_csv(
                DATA_DIR / f"{fname}.csv")["final_category"])
            meta = dict(model=name, seed=42, fraction=1.0, n_train=len(train),
                        split=split, fit_time_s=np.nan)
            r, c = score(yy, np.load(PROB_DIR / f"{name}_{split}.npy").argmax(1),
                         classes, groups, counts, meta)
            rows.append(r)
            cats.extend(c)
        pd.DataFrame(rows).to_csv(out, index=False)
        pd.DataFrame(cats).to_csv(
            OUT_DIR / f"per_category_{name}_full.csv", index=False)
        print(
            f"Added the existing full-data {name} point from cached probabilities.")


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    add_existing_transformers()
    # learning_curves_summary.csv is this function's own OUTPUT: reading it back in would
    # average stale numbers into the table, so it is excluded.
    files = [f for f in glob.glob(str(OUT_DIR / "learning_curves_*.csv"))
             if not f.endswith("learning_curves_summary.csv")]
    if not files:
        sys.exit("No learning_curves_*.csv found.")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    agg = (df.groupby(["model", "split", "fraction", "n_train"], as_index=False)
           [["macro_f1", "f1_small", "f1_medium", "f1_large"]].mean())
    agg.to_csv(OUT_DIR / "learning_curves_summary.csv", index=False)

    models = [m for m in DISPLAY if m in agg["model"].unique()]
    cmap = plt.get_cmap("tab10")
    markers = "osD^vPX*"
    panels = [("ood", "macro_f1", "OOD macro F1 (all categories)"),
              ("ood", "f1_small", "OOD F1, 4 smallest categories"),
              ("ood", "f1_large", "OOD F1, 4 largest categories")]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), sharey=True)
    for ax, (split, col, title) in zip(axes, panels):
        for i, m in enumerate(models):
            d = agg[(agg.model == m) & (agg.split == split)
                    ].sort_values("n_train")
            ax.plot(d["n_train"], d[col], marker=markers[i % len(markers)], color=cmap(i),
                    lw=1.6, ms=5, label=DISPLAY[m])
        ax.set_xscale("log")
        ticks = sorted(agg["n_train"].unique())
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t / 1000:.1f}k" for t in ticks])
        ax.minorticks_off()
        ax.set_xlabel(
            "labelled training rows (log scale; 5%, 10%, 25%, 50%, 100%)")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("F1 on unseen merchants")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4,
               fontsize=8, frameon=False)
    fig.suptitle(
        "Learning curves: how much labelled data does each model need?", fontsize=11)
    fig.tight_layout(rect=(0, 0.1, 1, 0.96))
    fig.savefig(OUT_DIR / "learning_curves.png", dpi=150)
    plt.close(fig)
    print(
        f"Saved {OUT_DIR / 'learning_curves.png'} and learning_curves_summary.csv")

    pivot = agg[agg.split == "ood"].pivot(
        index="model", columns="n_train", values="macro_f1")
    print("\n=== OOD macro F1 by training-set size ===")
    print(pivot.round(4).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--seeds", default="42",
                    help="comma-separated, e.g. 42,43,44")
    ap.add_argument("--fractions", default=",".join(str(f) for f in FRACTIONS))
    ap.add_argument("--device", default="cpu",
                    choices=["cpu", "cuda"], help="XGBoost only")
    ap.add_argument("--plot-only", action="store_true")
    a = ap.parse_args()

    if not a.plot_only:
        models = [m.strip() for m in a.models.split(",") if m.strip()]
        for m in models:
            if m not in ALL_MODELS:
                sys.exit(f"Unknown model {m}. Choose from {ALL_MODELS}")
        run(models, [int(s) for s in a.seeds.split(",")],
            [float(f) for f in a.fractions.split(",")], a.device)
    plot()


if __name__ == "__main__":
    main()
