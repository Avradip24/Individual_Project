"""
SmartSpend — Confusion matrices + top confused category pairs.

Evaluated on the OOD test set (where the interesting failures are) for:
  - Linear SVM (your best classical model)
  - Frozen multilingual embedding + LogReg
  - Fine-tuned DistilBERT
  - Fine-tuned MiniLM

For each model: saves a normalized confusion-matrix heatmap (PNG) and a
table of the top confused category pairs (highest off-diagonal values) —
this table is your starting point for the error-analysis chapter's
"confusable category pairs" discussion.

Requires: uv add matplotlib seaborn
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DATA_DIR = Path("data/processed")
MODELS_DIR = Path("models")
EMBEDDINGS_DIR = Path("data/embeddings")
OUT_DIR = Path("reports/confusion_matrices")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABELS = [
    "Education", "Entertainment", "Groceries", "Healthcare", "Housing",
    "Income", "Insurance", "Other", "Restaurants", "Shopping",
    "Transportation", "Travel", "Utilities",
]

df_ood = pd.read_csv(DATA_DIR / "test_ood.csv")
X_ood_text = df_ood["transaction_description"]
y_ood = df_ood["final_category"].values


def plot_and_save(cm_normalized: np.ndarray, model_name: str):
    plt.figure(figsize=(11, 9))
    sns.heatmap(
        cm_normalized, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=LABELS, yticklabels=LABELS, vmin=0, vmax=1,
    )
    plt.title(
        f"Confusion Matrix (row-normalized) — {model_name} — OOD test set")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    out_path = OUT_DIR / f"confusion_matrix_{model_name}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


def top_confused_pairs(cm: np.ndarray, model_name: str, top_n: int = 10):
    rows = []
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            if i != j and cm[i, j] > 0:
                rows.append({
                    "true_category": LABELS[i],
                    "predicted_as": LABELS[j],
                    "count": cm[i, j],
                })
    df = pd.DataFrame(rows).sort_values("count", ascending=False).head(top_n)
    df["model"] = model_name
    print(f"\nTop {top_n} confused pairs — {model_name}:")
    print(df.to_string(index=False))
    return df


all_top_confusions = []

# ---------------------------------------------------------------------------
# 1. Linear SVM (best classical model)
# ---------------------------------------------------------------------------
print("=" * 60)
print("Linear SVM")
print("=" * 60)
vectorizer = joblib.load(MODELS_DIR / "tfidf_vectorizer_shared.joblib")
svm = joblib.load(MODELS_DIR / "linear_svm.joblib")
X_ood_vec = vectorizer.transform(X_ood_text)
preds = svm.predict(X_ood_vec)

cm = confusion_matrix(y_ood, preds, labels=LABELS)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
plot_and_save(cm_norm, "linear_svm")
all_top_confusions.append(top_confused_pairs(cm, "linear_svm"))

# ---------------------------------------------------------------------------
# 2. Frozen multilingual embedding + LogReg
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Frozen multilingual embedding + LogReg")
print("=" * 60)
embedding_clf = joblib.load(MODELS_DIR / "embedding_logreg_classifier.joblib")
X_ood_emb = np.load(EMBEDDINGS_DIR / "test_ood_embeddings.npy")
preds = embedding_clf.predict(X_ood_emb)

cm = confusion_matrix(y_ood, preds, labels=LABELS)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
plot_and_save(cm_norm, "frozen_embedding")
all_top_confusions.append(top_confused_pairs(cm, "frozen_embedding"))


# ---------------------------------------------------------------------------
# Helper: run a fine-tuned HF model over the OOD set
# ---------------------------------------------------------------------------
def run_hf_model(model_path: str, model_name: str):
    print(f"\n{'=' * 60}\n{model_name}\n{'=' * 60}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    texts = X_ood_text.tolist()
    batch_size = 64
    preds_list = []
    print(
        f"Running inference on {len(texts)} examples (CPU — a few minutes)...")
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = tokenizer(batch, return_tensors="pt",
                               truncation=True, max_length=32, padding=True)
            outputs = model(**inputs)
            batch_preds = torch.argmax(outputs.logits, dim=-1)
            preds_list.extend([LABELS[idx] for idx in batch_preds.tolist()])
            if i % (batch_size * 20) == 0:
                print(f"  {i}/{len(texts)}...")

    preds = np.array(preds_list)
    cm = confusion_matrix(y_ood, preds, labels=LABELS)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    plot_and_save(cm_norm, model_name)
    all_top_confusions.append(top_confused_pairs(cm, model_name))


# ---------------------------------------------------------------------------
# 3. Fine-tuned DistilBERT
# ---------------------------------------------------------------------------
run_hf_model("models/distilbert_finetuned/final", "finetuned_distilbert")

# ---------------------------------------------------------------------------
# 4. Fine-tuned MiniLM
# ---------------------------------------------------------------------------
run_hf_model("models/minilm_finetuned/final", "finetuned_minilm")

# ---------------------------------------------------------------------------
# Combined summary
# ---------------------------------------------------------------------------
combined = pd.concat(all_top_confusions, ignore_index=True)
combined.to_csv(OUT_DIR / "top_confused_pairs_all_models.csv", index=False)
print(
    f"\nSaved combined confusion summary to {OUT_DIR / 'top_confused_pairs_all_models.csv'}")
print(f"Confusion matrix heatmaps saved to {OUT_DIR}/")
