"""
SmartSpend — Classical v2, step 2: tuned classical baselines.

Differences from v1 (train_classical_baselines.py), as required by the expose:
  1. Features: word + character n-gram TF-IDF + hand-engineered features
     (src/models/classical_features.py), fit on the training split only.
  2. ONE hyperparameter search protocol for all five models:
       - same random seed and same budget (N_ITER random candidates, or the
         whole grid if it is smaller),
       - each candidate is fit on a stratified SEARCH_FRACTION of the
         training data and scored by macro F1 on the full validation set,
       - the best candidate is refit on the full training set.
     Test and OOD data are never used for selection.
  3. XGBoost now uses balanced sample weights, like the other models'
     class_weight="balanced" (v1 XGBoost was unweighted). Naive Bayes has no
     class weighting option and is unchanged.

Caveat to report: validation is in-distribution and already saturated
(~0.99 macro F1), so differences between candidates are small. Tuning is
expected to matter less than the feature change.

Run from the project root. Fast models first, slow ones separately:
    uv run python src/models/train_classical_v2.py --models linear_svm,logistic_regression,multinomial_nb
    uv run python src/models/train_classical_v2.py --models random_forest
XGBoost is slow on CPU (about 30 minutes for ONE full-data fit with this feature
set), so run it on a Colab GPU with the same protocol:
    python src/models/train_classical_v2.py --models gradient_boosting --device cuda
Smoke test (minutes):
    uv run python src/models/train_classical_v2.py --n-iter 1 --search-fraction 0.05

Results are merged into reports/classical_v2/*.csv, so separate runs add up.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import ParameterGrid, ParameterSampler, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.models.classical_features import build_feature_union  # noqa: E402

DATA_DIR = Path("data/processed")
FEATURE_CACHE = Path("data/features_v2")
MODELS_DIR = Path("models/classical_v2")
REPORT_DIR = Path("reports/classical_v2")
for d in (FEATURE_CACHE, MODELS_DIR, REPORT_DIR):
    d.mkdir(parents=True, exist_ok=True)

SEED = 42
ALL_MODELS = ["logistic_regression", "multinomial_nb", "linear_svm",
              "random_forest", "gradient_boosting"]

GRIDS = {
    "logistic_regression": {"C": [0.1, 0.3, 1, 3, 10, 30]},
    "multinomial_nb": {"alpha": [0.01, 0.03, 0.1, 0.3, 1.0], "fit_prior": [True, False]},
    "linear_svm": {"C": [0.03, 0.1, 0.3, 1, 3, 10]},
    "random_forest": {
        "n_estimators": [200, 400],
        "max_depth": [None, 40, 80],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", 0.01],
    },
    "gradient_boosting": {
        "n_estimators": [150, 300],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.05, 0.1, 0.2],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.3, 0.6],
    },
}

# v1 results from the methodology log (sections 5 and 8), for comparison only.
V1_TEST_OOD = {
    "linear_svm": (0.9936, 0.8364),
    "multinomial_nb": (0.9747, 0.8240),
    "gradient_boosting": (0.9916, 0.7985),
    "logistic_regression": (0.9906, 0.7982),
    "random_forest": (0.9761, 0.7374),
}


def make_model(name: str, params: dict, device: str = "cpu"):
    if name == "logistic_regression":
        return LogisticRegression(max_iter=1000, class_weight="balanced",
                                  random_state=SEED, **params)
    if name == "multinomial_nb":
        return MultinomialNB(**params)
    if name == "linear_svm":
        return LinearSVC(class_weight="balanced", random_state=SEED, max_iter=5000, **params)
    if name == "random_forest":
        return RandomForestClassifier(class_weight="balanced_subsample", random_state=SEED,
                                      n_jobs=-1, **params)
    if name == "gradient_boosting":
        from xgboost import XGBClassifier  # imported lazily: only needed for this model
        return XGBClassifier(tree_method="hist", device=device, random_state=SEED,
                             n_jobs=-1, eval_metric="mlogloss", **params)
    raise ValueError(f"Unknown model: {name}")


def fit_model(name: str, model, X, y_enc):
    if name == "gradient_boosting":
        model.fit(X, y_enc, sample_weight=compute_sample_weight(
            "balanced", y_enc))
    else:
        model.fit(X, y_enc)


def upsert_csv(path: Path, new_df: pd.DataFrame):
    """Merge rows into a CSV, replacing any existing rows of the same model."""
    if path.exists():
        old = pd.read_csv(path)
        old = old[~old["model"].isin(new_df["model"].unique())]
        new_df = pd.concat([old, new_df], ignore_index=True)
    new_df.to_csv(path, index=False)


def load_features(refit: bool):
    """Fit the v2 feature union on train only (or load the cached matrices)."""
    splits = ["train", "val", "test_in_distribution", "test_ood"]
    texts, labels = {}, {}
    for s in splits:
        df = pd.read_csv(DATA_DIR / f"{s}.csv")
        texts[s], labels[s] = df["transaction_description"], df["final_category"]

    cache_ok = (not refit
                and (MODELS_DIR / "feature_union.joblib").exists()
                and all((FEATURE_CACHE / f"{s}.npz").exists() for s in splits))
    if cache_ok:
        print("Loading cached v2 features (use --refit-features to rebuild)")
        matrices = {s: sparse.load_npz(
            FEATURE_CACHE / f"{s}.npz") for s in splits}
    else:
        print("Fitting v2 feature union on the training split ...")
        start = time.time()
        union = build_feature_union(("word", "char", "hand"))
        matrices = {"train": union.fit_transform(texts["train"]).tocsr()}
        for s in splits[1:]:
            matrices[s] = union.transform(texts[s]).tocsr()
        joblib.dump(union, MODELS_DIR / "feature_union.joblib")
        for s in splits:
            sparse.save_npz(FEATURE_CACHE / f"{s}.npz", matrices[s])
        print(f"  features built in {time.time() - start:.0f}s")
    return matrices, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=",".join(ALL_MODELS),
                        help="comma-separated subset of: " + ", ".join(ALL_MODELS))
    parser.add_argument("--n-iter", type=int, default=8,
                        help="candidates per model")
    parser.add_argument("--search-fraction", type=float, default=0.4,
                        help="share of the training set used while searching")
    parser.add_argument("--refit-features", action="store_true")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                        help="XGBoost device. Use cuda on a Colab GPU; other models ignore it.")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    for m in models:
        if m not in ALL_MODELS:
            sys.exit(f"Unknown model '{m}'. Choose from: {ALL_MODELS}")

    matrices, labels = load_features(args.refit_features)
    encoder = LabelEncoder().fit(pd.concat(labels.values()))
    (MODELS_DIR / "labels.json").write_text(json.dumps(list(encoder.classes_), indent=2))
    y = {s: encoder.transform(v) for s, v in labels.items()}
    X = matrices
    n_features = X["train"].shape[1]
    print(f"Train {X['train'].shape} | Val {X['val'].shape[0]} | "
          f"Test {X['test_in_distribution'].shape[0]} | OOD {X['test_ood'].shape[0]} | "
          f"{n_features} features | {len(encoder.classes_)} classes")

    # Stratified search subset, identical for every model.
    if args.search_fraction < 1.0:
        idx, _ = train_test_split(np.arange(X["train"].shape[0]),
                                  train_size=args.search_fraction,
                                  stratify=y["train"], random_state=SEED)
    else:
        idx = np.arange(X["train"].shape[0])
    X_search, y_search = X["train"][idx], y["train"][idx]
    print(
        f"Search subset: {len(idx)} rows ({args.search_fraction:.0%} of train)")

    for name in models:
        print(f"\n{'=' * 64}\n{name}\n{'=' * 64}")
        grid = GRIDS[name]
        size = len(ParameterGrid(grid))
        n = min(args.n_iter, size)
        candidates = (list(ParameterGrid(grid)) if n == size
                      else list(ParameterSampler(grid, n_iter=n, random_state=SEED)))
        print(f"Searching {len(candidates)} of {size} candidates")

        log_rows, search_start = [], time.time()
        for i, params in enumerate(candidates, 1):
            start = time.time()
            model = make_model(name, params, args.device)
            fit_model(name, model, X_search, y_search)
            fit_s = time.time() - start
            val_f1 = f1_score(y["val"], model.predict(
                X["val"]), average="macro")
            log_rows.append({"model": name, "candidate": i, "params": json.dumps(params),
                             "val_macro_f1": round(val_f1, 4), "fit_time_s": round(fit_s, 1)})
            print(f"  [{i}/{len(candidates)}] val macro F1 {val_f1:.4f} ({fit_s:.0f}s)  {params}",
                  flush=True)
        search_s = time.time() - search_start

        log = pd.DataFrame(log_rows)
        best = log.sort_values(["val_macro_f1", "fit_time_s"], ascending=[
                               False, True]).iloc[0]
        best_params = json.loads(best["params"])
        print(f"Best: {best_params} (val macro F1 {best['val_macro_f1']:.4f})")

        print("Refitting best candidate on the full training set ...")
        start = time.time()
        model = make_model(name, best_params, args.device)
        fit_model(name, model, X["train"], y["train"])
        final_fit_s = time.time() - start
        joblib.dump(model, MODELS_DIR / f"{name}.joblib")

        result_rows, per_cat_rows = [], []
        for split in ["val", "test_in_distribution", "test_ood"]:
            preds = model.predict(X[split])
            acc = accuracy_score(y[split], preds)
            f1 = f1_score(y[split], preds, average="macro")
            result_rows.append({"model": name, "split": split,
                                "accuracy": round(acc, 4), "macro_f1": round(f1, 4)})
            print(f"  {split:22s} accuracy={acc:.4f}  macro_f1={f1:.4f}")
            if split == "test_ood":
                p, r, f, sup = precision_recall_fscore_support(
                    y[split], preds, labels=range(len(encoder.classes_)), zero_division=0)
                for k, cat in enumerate(encoder.classes_):
                    per_cat_rows.append({"model": name, "category": cat,
                                         "precision": round(p[k], 4), "recall": round(r[k], 4),
                                         "f1": round(f[k], 4), "support": int(sup[k])})

        summary = pd.DataFrame([{
            "model": name, "best_params": json.dumps(best_params),
            "n_candidates": len(candidates), "search_time_s": round(search_s, 1),
            "final_fit_time_s": round(final_fit_s, 1), "n_features": n_features,
        }])
        upsert_csv(REPORT_DIR / "tuning_log.csv", log)
        upsert_csv(REPORT_DIR / "classical_v2_results.csv",
                   pd.DataFrame(result_rows))
        upsert_csv(REPORT_DIR / "per_category_ood.csv",
                   pd.DataFrame(per_cat_rows))
        upsert_csv(REPORT_DIR / "classical_v2_models.csv", summary)

    # ----- v1 vs v2 comparison over everything trained so far -----
    res = pd.read_csv(REPORT_DIR / "classical_v2_results.csv")
    rows = []
    for name, (v1_test, v1_ood) in V1_TEST_OOD.items():
        sub = res[res["model"] == name].set_index("split")["macro_f1"]
        if sub.empty:
            continue
        rows.append({"model": name,
                     "v1_test": v1_test, "v2_test": sub["test_in_distribution"],
                     "v1_ood": v1_ood, "v2_ood": sub["test_ood"],
                     "ood_change": round(sub["test_ood"] - v1_ood, 4)})
    comparison = pd.DataFrame(rows)
    comparison.to_csv(REPORT_DIR / "v1_vs_v2_comparison.csv", index=False)
    print("\n\n=== v1 vs v2 macro F1 (all models trained so far) ===")
    print(comparison.to_string(index=False))
    print(f"\nSaved results to {REPORT_DIR}/ and models to {MODELS_DIR}/")


if __name__ == "__main__":
    main()
