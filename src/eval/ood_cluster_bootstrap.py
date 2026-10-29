"""
SmartSpend -- Merchant-clustered bootstrap confidence intervals for the OOD scores.

Why: the OOD test set (16,031 rows) was built by holding out whole merchant
clusters, and the error analysis showed that errors come in whole-merchant
blocks (e.g. every "Zelle payment" row is predicted Other). Rows from one
merchant are therefore not independent observations, and an ordinary
row-level confidence interval is far too narrow. This script resamples
MERCHANT CLUSTERS with replacement (the correct unit), recomputes every
model's OOD macro F1 on each resample, and reports:

  * a 95% interval per model, next to the naive row-level interval, to show
    how much wider the honest interval is;
  * paired differences between models on the SAME resamples (e.g. is
    DistilBERT really better than Logistic Regression?), with the share of
    resamples in which model A beats model B;
  * how concentrated each model's errors are in a few clusters.

Merchant clusters are rebuilt with the same normalisation and fuzzy
threshold as src/data/split_dataset.py (copied below; keep them in sync).
APPROXIMATION: the original clustering ran over ALL merchant keys, this one
over the OOD keys only, so a few cluster boundaries may differ slightly.

Uses the cached OOD probabilities (reports/calibration/probs/); no models are
loaded. Needs rapidfuzz (already a project dependency).

Run from the project root (about a minute):
    uv run python src/eval/ood_cluster_bootstrap.py
Options: --n-boot 2000  --seed 42
"""

import argparse
import itertools
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data/processed")
PROB_DIR = Path("reports/calibration/probs")
V2_DIR = Path("models/classical_v2")
OUT_DIR = Path("reports/error_analysis")
FUZZY_THRESHOLD = 85  # must match split_dataset.py

MODELS = ["logistic_regression", "multinomial_nb", "linear_svm", "random_forest",
          "gradient_boosting", "frozen_embedding_logreg", "distilbert_finetuned",
          "minilm_finetuned"]
SHORT = {"logistic_regression": "Logistic Regression v2", "multinomial_nb": "Naive Bayes v2",
         "linear_svm": "Linear SVM v2 (Platt-calibrated)", "random_forest": "Random Forest v2",
         "gradient_boosting": "XGBoost v2", "frozen_embedding_logreg": "Frozen MiniLM + LogReg",
         "distilbert_finetuned": "DistilBERT fine-tuned", "minilm_finetuned": "MiniLM fine-tuned"}

# ---- copied from src/data/split_dataset.py (keep in sync) ----------------------------
PROCESSOR_PREFIXES = [
    "PAYPAL INST XFER", "PYPL INST XFER", "PYPL", "VENMO PAYMENT TO", "VENMO",
    "CASH APP PAYMENT TO", "CASH APP", "SQ ", "APPLE PAY", "GOOGLE PAY",
    "PREAPPROVED PAYMENT BILL USER PAYMENT:", "SUBSCRIPTION PAYMENT:",
    "EXPRESS CHECKOUT PAYMENT:", "WITHDRAWAL FROM", "PAYMENT TO:",
]


def normalize_merchant_key(desc: str) -> str:
    s = str(desc).upper()
    s = re.sub(r"[\r\n]+", " ", s)
    s = re.sub(r"^\[(DEBIT|CREDIT)\]\s*", "", s)
    orig_co_match = re.search(r"ORIG CO NAME:(.*?)CO ENTRY DESCR", s)
    if orig_co_match:
        s = orig_co_match.group(1).strip()
    else:
        for prefix in PROCESSOR_PREFIXES:
            if s.startswith(prefix):
                s = s[len(prefix):].strip()
                break
    s = re.sub(r"WEB ID.*$", "", s)
    s = re.sub(r"PPD ID.*$", "", s)
    s = re.sub(r"\d+", "", s)
    s = re.sub(r"[^A-Z\s&]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return " ".join(s.split(" ")[:2])
# ---------------------------------------------------------------------------------------


def cluster_keys(keys):
    """Greedy fuzzy clustering, same procedure as split_dataset.py."""
    from rapidfuzz import fuzz  # only needed when test_ood.csv has no merchant_cluster column
    cid, nxt = {}, 0
    for key in keys:
        if key in cid:
            continue
        cid[key] = nxt
        for other in keys:
            if other not in cid and fuzz.ratio(key, other) >= FUZZY_THRESHOLD:
                cid[other] = nxt
        nxt += 1
    return cid


def macro_f1_from_counts(conf):
    """conf: (..., K, K) with rows = true, cols = predicted. F1 = 2TP / (row + col sums)."""
    tp = np.diagonal(conf, axis1=-2, axis2=-1).astype(float)
    denom = conf.sum(-1) + conf.sum(-2)
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    return f1.mean(-1)


def accuracy_from_counts(conf):
    return np.trace(conf, axis1=-2, axis2=-1) / conf.sum((-2, -1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    classes = json.loads((V2_DIR / "labels.json").read_text())
    idx = {c: i for i, c in enumerate(classes)}
    K = len(classes)
    ood = pd.read_csv(DATA_DIR / "test_ood.csv")
    y = ood["final_category"].map(idx).to_numpy()
    n = len(y)

    keys = ood["transaction_description"].map(normalize_merchant_key)
    if "merchant_cluster" in ood.columns:
        # exact clusters, written by split_dataset.py when it built the OOD file
        cluster, _ = pd.factorize(ood["merchant_cluster"])
        ordered_keys = list(dict.fromkeys(keys))
        label_of = keys.groupby(cluster).agg(
            lambda s: s.value_counts().index[0]).to_dict()
        print("Using the merchant_cluster column from test_ood.csv (exact clusters).")
    else:
        print(
            "No merchant_cluster column: rebuilding clusters (approximation, see header).")
        ordered_keys = list(dict.fromkeys(keys))
        cid_of = cluster_keys(ordered_keys)
        cluster = keys.map(cid_of).to_numpy()
        label_of = {c: k for k, c in cid_of.items()}
    n_clusters = int(cluster.max()) + 1
    sizes = np.bincount(cluster, minlength=n_clusters)
    print(
        f"OOD rows: {n} | unique merchant keys: {len(ordered_keys)} | clusters: {n_clusters}")
    top = np.argsort(sizes)[::-1][:8]
    print("Largest clusters:", [
          (label_of.get(int(c), "?"), int(sizes[c])) for c in top])
    print(f"Median rows per cluster: {int(np.median(sizes))} | "
          f"largest cluster holds {sizes.max() / n:.1%} of the rows")

    preds = {m: np.load(PROB_DIR / f"{m}_ood.npy").argmax(1) for m in MODELS}

    # per-cluster confusion counts: (n_clusters, K*K)
    per_cluster = {}
    for m in MODELS:
        flat = y * K + preds[m]
        mat = np.zeros((n_clusters, K * K), dtype=np.int32)
        np.add.at(mat, (cluster, flat), 1)
        per_cluster[m] = mat

    # bootstrap weights: how many times each cluster is drawn (same draws for every model)
    W = rng.multinomial(n_clusters, np.full(
        n_clusters, 1 / n_clusters), size=args.n_boot)
    f1_cl, acc_cl, f1_row = {}, {}, {}
    row_w = None
    for m in MODELS:
        conf = (W @ per_cluster[m]).reshape(args.n_boot, K, K)
        f1_cl[m] = macro_f1_from_counts(conf)
        acc_cl[m] = accuracy_from_counts(conf)
    # naive row-level bootstrap for comparison (same number of replicates)
    row_idx = rng.integers(0, n, size=(args.n_boot, n))
    for m in MODELS:
        flat = y * K + preds[m]
        conf = np.stack([np.bincount(flat[r], minlength=K * K)
                        for r in row_idx]).reshape(args.n_boot, K, K)
        f1_row[m] = macro_f1_from_counts(conf)

    # ---- point estimates + intervals ----
    rows = []
    for m in MODELS:
        full = np.bincount(y * K + preds[m], minlength=K * K).reshape(K, K)
        point = float(macro_f1_from_counts(full))
        lo, hi = np.percentile(f1_cl[m], [2.5, 97.5])
        rlo, rhi = np.percentile(f1_row[m], [2.5, 97.5])
        # share of errors in the 5 clusters with the most errors
        err_per_cluster = np.array([per_cluster[m][c].reshape(K, K).sum() -
                                    np.trace(per_cluster[m][c].reshape(K, K))
                                    for c in range(n_clusters)])
        top5 = np.sort(err_per_cluster)[
            ::-1][:5].sum() / max(err_per_cluster.sum(), 1)
        rows.append(dict(model=m, ood_macro_f1=round(point, 4),
                         ci_cluster_low=round(lo, 4), ci_cluster_high=round(hi, 4),
                         ci_cluster_width=round(hi - lo, 4),
                         ci_rowlevel_low=round(rlo, 4), ci_rowlevel_high=round(rhi, 4),
                         ci_rowlevel_width=round(rhi - rlo, 4),
                         width_ratio=round((hi - lo) / (rhi - rlo), 1),
                         ood_accuracy=round(
                             float(accuracy_from_counts(full)), 4),
                         errors_in_top5_clusters=round(float(top5), 3)))
    res = pd.DataFrame(rows).sort_values("ood_macro_f1", ascending=False)
    res.to_csv(OUT_DIR / "ood_cluster_bootstrap.csv", index=False)

    print("\n" + "=" * 100)
    print(
        f"OOD MACRO F1 WITH 95% INTERVALS ({args.n_boot} resamples; cluster = honest, row-level = naive)")
    print("=" * 100)
    print(f"{'Model':<36}{'F1':>8}  {'cluster 95% CI':<20}{'row-level 95% CI':<20}{'wider by':>9}{'top-5 clusters':>16}")
    for _, r in res.iterrows():
        print(f"{SHORT[r.model]:<36}{r.ood_macro_f1:>8.4f}  "
              f"[{r.ci_cluster_low:.3f}, {r.ci_cluster_high:.3f}]   "
              f"[{r.ci_rowlevel_low:.3f}, {r.ci_rowlevel_high:.3f}]   {r.width_ratio:>6.1f}x"
              f"{r.errors_in_top5_clusters:>14.0%}")

    # ---- paired differences ----
    pair_rows = []
    for a, b in itertools.permutations(MODELS, 2):
        d = f1_cl[a] - f1_cl[b]
        full_a = np.bincount(y * K + preds[a], minlength=K * K).reshape(K, K)
        full_b = np.bincount(y * K + preds[b], minlength=K * K).reshape(K, K)
        lo, hi = np.percentile(d, [2.5, 97.5])
        pair_rows.append(dict(model_a=a, model_b=b,
                              diff=round(float(macro_f1_from_counts(
                                  full_a) - macro_f1_from_counts(full_b)), 4),
                              ci_low=round(lo, 4), ci_high=round(hi, 4),
                              share_a_better=round(float((d > 0).mean()), 3)))
    pairs = pd.DataFrame(pair_rows)
    pairs.to_csv(OUT_DIR / "ood_cluster_bootstrap_pairs.csv", index=False)

    print("\n" + "=" * 100)
    print("PAIRED DIFFERENCES IN OOD MACRO F1 (A minus B, same resamples)")
    print("=" * 100)
    focus = [("distilbert_finetuned", "logistic_regression"), ("distilbert_finetuned", "minilm_finetuned"),
             ("minilm_finetuned",
              "logistic_regression"), ("logistic_regression", "linear_svm"),
             ("logistic_regression",
              "gradient_boosting"), ("distilbert_finetuned", "gradient_boosting"),
             ("logistic_regression", "multinomial_nb")]
    print(f"{'A  vs  B':<46}{'diff':>8}  {'95% CI':<20}{'A better in':>12}")
    for a, b in focus:
        r = pairs[(pairs.model_a == a) & (pairs.model_b == b)].iloc[0]
        sig = "" if r.ci_low <= 0 <= r.ci_high else "  *"
        print(f"{SHORT[a].split(' (')[0] + '  vs  ' + SHORT[b].split(' (')[0]:<46}{r['diff']:>+8.4f}  "
              f"[{r.ci_low:+.3f}, {r.ci_high:+.3f}]   {r.share_a_better:>10.0%}{sig}")
    print("\n* = the 95% interval of the difference excludes zero")

    # ---- sensitivity: remove the largest clusters (point estimates) ----
    order = np.argsort(sizes)[::-1]
    drop_rows = []
    print("\n" + "=" * 100)
    print("OOD MACRO F1 AFTER REMOVING THE LARGEST MERCHANT CLUSTERS (point estimates)")
    print("=" * 100)
    for n_drop in (0, 1, 3, 5):
        keep = ~np.isin(cluster, order[:n_drop])
        vals = {m: float(macro_f1_from_counts(
            np.bincount(y[keep] * K + preds[m][keep], minlength=K * K).reshape(K, K)))
            for m in MODELS}
        drop_rows.append(dict(dropped_largest_clusters=n_drop, rows_left=int(keep.sum()),
                              **{m: round(v, 4) for m, v in vals.items()}))
        print(f"  drop {n_drop} ({int(keep.sum())} rows left): " + ", ".join(
            f"{SHORT[m].split(' ')[0]} {v:.4f}" for m, v in vals.items()))
    pd.DataFrame(drop_rows).to_csv(
        OUT_DIR / "ood_drop_top_clusters.csv", index=False)
    print("  (largest clusters dropped, in order: "
          + ", ".join(f"{label_of.get(int(c), '?')} ({int(sizes[c])})" for c in order[:5]) + ")")

    print(f"\nSaved {OUT_DIR / 'ood_cluster_bootstrap.csv'}, {OUT_DIR / 'ood_cluster_bootstrap_pairs.csv'} "
          f"and {OUT_DIR / 'ood_drop_top_clusters.csv'}")


if __name__ == "__main__":
    main()
