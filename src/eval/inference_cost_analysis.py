"""
SmartSpend -- Inference-cost analysis for all eight models (CPU).

For every model the FULL pipeline is timed, from raw transaction text to a
predicted label (featurisation / tokenisation / embedding included), because
that is what a deployed service pays per request:

  single-transaction latency   batch size 1: median, p95 (ms)
  batch throughput             batch size 256 (transactions / second)
  parameters                   transformers: all weights. Linear models:
                               coefficients. NB: log-prob table. RF / XGBoost:
                               total tree nodes (not comparable to weights --
                               see 'param_kind')
  size on disk (MB)            model files (+ shared featuriser for classical)
  memory (MB)                  RSS increase from loading the model, and the
                               process peak after inference

Each model runs in its OWN subprocess so memory numbers are not polluted by
the models measured before it.

Cost is joined to quality (OOD macro F1 / accuracy) from the cached
probabilities in reports/calibration/probs/ when they exist, and a
latency-vs-quality plot is written.

Run from the project root (about 5-10 minutes, CPU):
    uv run python src/eval/inference_cost_analysis.py
Options:
    --threads N   cap CPU threads (torch / numpy / xgboost); default: library defaults
    --n-single N  transactions for the latency test (default 300)
    --models a,b  subset
Needs psutil for memory figures:  uv add psutil
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path("data/processed")
V2_DIR = Path("models/classical_v2")
OUT_DIR = Path("reports/inference_cost")
PROB_DIR = Path("reports/calibration/probs")

MAX_LENGTH = 32
BATCH = 256
N_BATCH_TEXTS = 2048
WARMUP = 20
SEED = 42
ENCODER_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

CLASSICAL = ["logistic_regression", "multinomial_nb", "linear_svm",
             "random_forest", "gradient_boosting"]
ALL_MODELS = CLASSICAL + ["frozen_embedding_logreg",
                          "distilbert_finetuned", "minilm_finetuned"]
DISPLAY = {
    "logistic_regression": "Logistic Regression v2",
    "multinomial_nb": "Naive Bayes v2",
    "linear_svm": "Linear SVM v2",
    "random_forest": "Random Forest v2",
    "gradient_boosting": "XGBoost v2",
    "frozen_embedding_logreg": "Frozen MiniLM + LogReg",
    "distilbert_finetuned": "DistilBERT fine-tuned",
    "minilm_finetuned": "MiniLM fine-tuned",
}


# ---------------------------------------------------------------------------
# Worker: measures ONE model, prints a single RESULT=<json> line
# ---------------------------------------------------------------------------
def rss_mb():
    try:
        import psutil
    except ImportError:
        return None
    mi = psutil.Process().memory_info()
    return mi.rss / 1e6


def peak_mb():
    try:
        import psutil
    except ImportError:
        return None
    mi = psutil.Process().memory_info()
    peak = getattr(mi, "peak_wset", None)  # Windows keeps a true peak
    return max(mi.rss, peak or 0) / 1e6


def file_mb(paths):
    return sum(Path(p).stat().st_size for p in paths if Path(p).exists()) / 1e6


def load_pipeline(name: str):
    """Returns (predict_fn(list[str]) -> np.ndarray, params, param_kind, size_mb, shared_mb)."""
    if name in CLASSICAL:
        import joblib
        union_path = V2_DIR / "feature_union.joblib"
        union = joblib.load(union_path)
        if name == "gradient_boosting" and (V2_DIR / "gradient_boosting.ubj").exists():
            from xgboost import XGBClassifier
            model = XGBClassifier()
            model.load_model(str(V2_DIR / "gradient_boosting.ubj"))
            model_file = V2_DIR / "gradient_boosting.ubj"
        else:
            model = joblib.load(V2_DIR / f"{name}.joblib")
            model_file = V2_DIR / f"{name}.joblib"

        if name in ("logistic_regression", "linear_svm"):
            params, kind = int(model.coef_.size +
                               model.intercept_.size), "coefficients"
        elif name == "multinomial_nb":
            params, kind = int(model.feature_log_prob_.size +
                               model.class_log_prior_.size), "log-probs"
        elif name == "random_forest":
            params = int(sum(t.tree_.node_count for t in model.estimators_))
            kind = "tree nodes"
        else:
            params = int(len(model.get_booster().trees_to_dataframe()))
            kind = "tree nodes"

        def predict_classical(texts):
            return model.predict(union.transform(texts))

        return predict_classical, params, kind, file_mb([model_file]), file_mb([union_path])

    if name == "frozen_embedding_logreg":
        import joblib
        from sentence_transformers import SentenceTransformer
        clf = joblib.load(Path("models/embedding_logreg_classifier.joblib"))
        enc = SentenceTransformer(ENCODER_NAME, device="cpu")
        n_enc = sum(p.numel() for p in enc.parameters())
        params = int(n_enc + clf.coef_.size + clf.intercept_.size)
        size = n_enc * 4 / 1e6 + \
            file_mb([Path("models/embedding_logreg_classifier.joblib")])

        def predict_embedding(texts):
            emb = enc.encode(texts, batch_size=BATCH, show_progress_bar=False,
                             convert_to_numpy=True)
            return clf.predict(emb)

        return predict_embedding, params, "weights", size, 0.0

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    path = Path("models") / name / "final"
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).eval()
    params = int(sum(p.numel() for p in model.parameters()))
    weights = [f for f in path.iterdir() if f.suffix in (
        ".safetensors", ".bin")]

    def predict_transformer(texts):
        with torch.no_grad():
            enc = tok(texts, return_tensors="pt", truncation=True,
                      max_length=MAX_LENGTH, padding=True)
            return model(**enc).logits.argmax(-1).numpy()

    return predict_transformer, params, "weights", file_mb(weights), 0.0


def worker(name: str, n_single: int, threads: int):
    if threads > 0:
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[var] = str(threads)
    sys.path.insert(0, str(ROOT))
    if name in ("distilbert_finetuned", "minilm_finetuned", "frozen_embedding_logreg"):
        import torch
        if threads > 0:
            torch.set_num_threads(threads)

    texts = (pd.read_csv(DATA_DIR / "test_ood.csv")["transaction_description"]
             .astype(str).sample(N_BATCH_TEXTS, random_state=SEED).tolist())

    base = rss_mb()
    t0 = time.perf_counter()
    predict, params, kind, size_mb, shared_mb = load_pipeline(name)
    load_s = time.perf_counter() - t0
    after_load = rss_mb()

    for t in texts[:WARMUP]:  # warm-up, not timed
        predict([t])

    single = []
    for t in texts[WARMUP:WARMUP + n_single]:
        s = time.perf_counter()
        predict([t])
        single.append((time.perf_counter() - s) * 1000)
    single = np.array(single)

    predict(texts[:BATCH])  # warm the batch path
    s = time.perf_counter()
    for i in range(0, len(texts), BATCH):
        predict(texts[i:i + BATCH])
    batch_s = time.perf_counter() - s

    result = {
        "model": name, "params": params, "param_kind": kind,
        "size_on_disk_mb": round(size_mb, 2), "shared_featurizer_mb": round(shared_mb, 2),
        "load_time_s": round(load_s, 2),
        "ram_after_load_mb": round(after_load - base, 1) if base is not None else None,
        "peak_ram_mb": round(peak_mb(), 1) if peak_mb() is not None else None,
        "latency_p50_ms": round(float(np.median(single)), 2),
        "latency_p95_ms": round(float(np.percentile(single, 95)), 2),
        "throughput_per_s": round(len(texts) / batch_s, 1),
        "n_single": int(len(single)),
    }
    print("RESULT=" + json.dumps(result))


# ---------------------------------------------------------------------------
# Quality join + plot
# ---------------------------------------------------------------------------
def quality_table():
    from sklearn.metrics import accuracy_score, f1_score
    labels = json.loads((V2_DIR / "labels.json").read_text())
    idx = {c: i for i, c in enumerate(labels)}
    y = pd.read_csv(
        DATA_DIR / "test_ood.csv")["final_category"].map(idx).values
    rows = {}
    for name in ALL_MODELS:
        p = PROB_DIR / f"{name}_ood.npy"
        if p.exists():
            pred = np.load(p).argmax(1)
            rows[name] = {"ood_macro_f1": round(f1_score(y, pred, average="macro"), 4),
                          "ood_accuracy": round(accuracy_score(y, pred), 4)}
    return pd.DataFrame.from_dict(rows, orient="index")


def make_plot(df: pd.DataFrame, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = df.dropna(subset=["ood_macro_f1"])
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.scatter(d["latency_p50_ms"], d["ood_macro_f1"], s=60, color="#2a6fbb")
    for _, r in d.iterrows():
        ax.annotate(DISPLAY[r["model"]], (r["latency_p50_ms"], r["ood_macro_f1"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("median single-transaction latency (ms, CPU, log scale)")
    ax.set_ylabel("OOD macro F1")
    ax.set_title("Cost vs. quality on unseen merchants")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", help=argparse.SUPPRESS)
    ap.add_argument("--models", default=",".join(ALL_MODELS))
    ap.add_argument("--n-single", type=int, default=300)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()

    if args.worker:
        worker(args.worker, args.n_single, args.threads)
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        import psutil  # noqa: F401
    except ImportError:
        print("NOTE: psutil not installed -> memory columns will be empty. `uv add psutil`")

    rows = []
    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        print(f"\n== {DISPLAY[name]}", flush=True)
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker", name,
             "--n-single", str(args.n_single), "--threads", str(args.threads)],
            capture_output=True, text=True, cwd=os.getcwd())
        line = next((ln for ln in proc.stdout.splitlines()
                    if ln.startswith("RESULT=")), None)
        if proc.returncode != 0 or line is None:
            print("  FAILED:\n" + (proc.stderr or proc.stdout)[-1500:])
            continue
        r = json.loads(line[len("RESULT="):])
        rows.append(r)
        print(f"  p50 {r['latency_p50_ms']} ms | p95 {r['latency_p95_ms']} ms | "
              f"{r['throughput_per_s']}/s | {r['params']:,} {r['param_kind']} | "
              f"{r['size_on_disk_mb']} MB")

    df = pd.DataFrame(rows)
    df = df.merge(quality_table(), left_on="model",
                  right_index=True, how="left")
    df.insert(1, "display", df["model"].map(DISPLAY))
    df["cpu"] = platform.processor() or platform.machine()
    df["cpu_count"] = os.cpu_count()
    df["threads_setting"] = args.threads or "default"
    out_csv = OUT_DIR / "inference_cost.csv"
    if out_csv.exists():  # merge: a partial rerun must not drop earlier models
        old = pd.read_csv(out_csv)
        df = pd.concat(
            [old[~old["model"].isin(df["model"])], df], ignore_index=True)
    df["_order"] = df["model"].map({m: i for i, m in enumerate(ALL_MODELS)})
    df = df.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    df.to_csv(out_csv, index=False)
    make_plot(df, OUT_DIR / "cost_vs_quality.png")

    show = ["display", "latency_p50_ms", "latency_p95_ms", "throughput_per_s", "params",
            "size_on_disk_mb", "shared_featurizer_mb", "ram_after_load_mb", "peak_ram_mb",
            "ood_macro_f1"]
    print("\n\n=== INFERENCE COST (CPU) ===")
    print(df[[c for c in show if c in df.columns]].to_string(index=False))
    print(
        f"\nSaved to {OUT_DIR}/  (cpu_count={os.cpu_count()}, threads={args.threads or 'default'})")


if __name__ == "__main__":
    main()
