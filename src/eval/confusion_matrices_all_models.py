"""
SmartSpend -- Row-normalised confusion-matrix heatmaps for ALL EIGHT models (OOD test set).

Reads the cached OOD probabilities (reports/calibration/probs/, written by
calibration_all_models.py), so it needs no model loading and runs in seconds.
(The Linear SVM probabilities are its Platt-calibrated outputs; argmax matches the
calibrated predictions, which score 0.8543 against 0.8514 for the raw model.)

Writes to reports/confusion_matrices/all_models/ (the older v1 heatmaps are untouched):
  cm_<model>.png                 one annotated heatmap per model (row = true class,
                                 cell = share of that true class predicted as the column)
  cm_grid_all_models.png         all eight side by side (no annotations)
  cm_counts_<model>.csv          raw count matrices
  per_class_recall_ood.csv       diagonal of each matrix: recall per class and model

Run from the project root:
    uv run python src/eval/confusion_matrices_all_models.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

DATA = Path("data/processed/test_ood.csv")
PROBS = Path("reports/calibration/probs")
LABELS = Path("models/classical_v2/labels.json")
OUT = Path("reports/confusion_matrices/all_models")

MODELS = [("distilbert_finetuned", "Fine-tuned DistilBERT"),
          ("logistic_regression", "Logistic Regression v2"),
          ("minilm_finetuned", "Fine-tuned MiniLM"),
          ("linear_svm", "Linear SVM v2"),
          ("multinomial_nb", "Naive Bayes v2"),
          ("gradient_boosting", "XGBoost v2"),
          ("random_forest", "Random Forest v2"),
          ("frozen_embedding_logreg", "Frozen MiniLM + LogReg")]


def draw(ax, cmn, classes, title, annotate):
    n = len(classes)
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(classes, rotation=60, ha="right",
                       fontsize=8 if annotate else 6)
    ax.set_yticklabels(classes, fontsize=8 if annotate else 6)
    ax.set_xlabel("predicted", fontsize=8)
    ax.set_ylabel("true", fontsize=8)
    ax.set_title(title, fontsize=10 if annotate else 9)
    if annotate:
        for i in range(n):
            for j in range(n):
                v = cmn[i, j]
                if v >= 0.01:
                    ax.text(j, i, f"{v:.2f}".lstrip("0") if v < 1 else "1", ha="center",
                            va="center", fontsize=6.5, color="white" if v > 0.5 else "black")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    classes = json.loads(LABELS.read_text())
    idx = {c: i for i, c in enumerate(classes)}
    n = len(classes)
    y = pd.read_csv(DATA)["final_category"].map(idx).to_numpy()

    recalls, mats = {}, {}
    for key, title in MODELS:
        pred = np.load(PROBS / f"{key}_ood.npy").argmax(1)
        assert len(pred) == len(y), (key, len(pred), len(y))
        cm = np.bincount(y * n + pred, minlength=n * n).reshape(n, n)
        cmn = cm / cm.sum(1, keepdims=True)
        mats[key] = (title, cmn)
        recalls[title] = np.diag(cmn)
        pd.DataFrame(cm, index=classes, columns=classes).to_csv(
            OUT / f"cm_counts_{key}.csv")
        fig, ax = plt.subplots(figsize=(7.5, 6.5))
        draw(ax, cmn, classes,
             f"{title}: OOD test, row-normalised (accuracy {(pred == y).mean():.3f})", True)
        fig.tight_layout()
        fig.savefig(OUT / f"cm_{key}.png", dpi=150)
        plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10.5))
    for ax, (key, (title, cmn)) in zip(axes.ravel(), mats.items()):
        draw(ax, cmn, classes, title, False)
    fig.suptitle(
        "OOD confusion matrices, row-normalised (share of each true class)", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "cm_grid_all_models.png", dpi=130)
    plt.close(fig)

    rec = pd.DataFrame(recalls, index=classes).round(4)
    rec.to_csv(OUT / "per_class_recall_ood.csv")
    print("Per-class recall on OOD (diagonal of each matrix):\n")
    print(rec.to_string())
    print("\nLowest-recall class per model:")
    for title in rec.columns:
        c = rec[title].idxmin()
        print(f"  {title:26s} {c} ({rec.at[c, title]:.2f})")
    print(f"\nSaved heatmaps and CSVs to {OUT}")


if __name__ == "__main__":
    main()
