"""
SmartSpend — Transformer approach 2: fine-tuned DistilBERT.

Unlike the frozen multilingual embedding approach (train_embedding_baseline.py),
this actually updates the transformer's own internal weights via
backpropagation on the training data ("fine-tuning"), rather than treating
it as a fixed feature extractor. This directly tests whether adapting the
model to this task's specific noisy short-text format succeeds where relying
on unchanged generic pretrained embeddings did not.

This version is configured for the FULL run on Colab GPU, reading/writing
from Google Drive (see DATA_DIR / MODEL_OUT_DIR below) so the trained model
survives even if the Colab session disconnects.

Requires (in Colab): !pip install transformers accelerate -q
"""

import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

DATA_DIR = Path("data/processed")
MODEL_OUT_DIR = Path("models/distilbert_finetuned")
MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 32  # transaction descriptions are short — no need for BERT's usual 128-512
# ceiling, not a target — EarlyStoppingCallback (patience=2) below will
NUM_EPOCHS = 6
# stop training automatically once validation macro F1 stops improving
# for 2 consecutive epochs, so this empirically tests whether more
# epochs help rather than assuming 3 was already optimal
BATCH_SIZE = 16

# --- SMOKE TEST SWITCH (kept for future reuse — leave as None for the real run) ---
SUBSET_SIZE = None


def load_split(name: str):
    df = pd.read_csv(DATA_DIR / f"{name}.csv")
    return df["transaction_description"].tolist(), df["final_category"].tolist()


X_train, y_train = load_split("train")
X_val, y_val = load_split("val")
X_test, y_test = load_split("test_in_distribution")
X_ood, y_ood = load_split("test_ood")

if SUBSET_SIZE is not None:
    print(f"*** SMOKE TEST MODE — using only {SUBSET_SIZE} training rows ***")
    X_train, y_train = X_train[:SUBSET_SIZE], y_train[:SUBSET_SIZE]

# The Trainer re-evaluates on eval_dataset after EVERY epoch, which is
# expensive if that's the full 12k-row validation set (this is what ate
# most of the smoke test's 10 minutes: ~4.5 min per eval pass). Use a small
# random subset for fast per-epoch monitoring during training; the full
# val/test/ood sets are still used for the real, final reported numbers
# after training completes (see the `evaluate()` calls at the bottom).
random.seed(RANDOM_SEED)
MONITOR_EVAL_SIZE = 1500
monitor_idx = random.sample(
    range(len(X_val)), min(MONITOR_EVAL_SIZE, len(X_val)))
X_val_monitor = [X_val[i] for i in monitor_idx]
y_val_monitor = [y_val[i] for i in monitor_idx]

print(f"Train: {len(X_train)} | Val (monitor): {len(X_val_monitor)} | "
      f"Val (full, final eval): {len(X_val)} | Test: {len(X_test)} | OOD: {len(X_ood)}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cpu":
    print("No GPU detected — this will be slow. Fine for the smoke test; "
          "consider Colab/Kaggle for the full run if it's too slow locally.")

# ---------------------------------------------------------------------------
# Labels -> integers (required by the model's classification head)
# ---------------------------------------------------------------------------
label_encoder = LabelEncoder()
# ensure every category is covered
label_encoder.fit(y_train + y_val + y_test + y_ood)
num_labels = len(label_encoder.classes_)
print(f"Num categories: {num_labels}")

y_train_enc = label_encoder.transform(y_train)
y_val_enc = label_encoder.transform(y_val)
y_test_enc = label_encoder.transform(y_test)
y_ood_enc = label_encoder.transform(y_ood)

# ---------------------------------------------------------------------------
# Class weights for the loss — same rationale as class_weight="balanced" in
# the classical models: categories range from 4,000 to 12,000 rows.
# ---------------------------------------------------------------------------
class_weights = compute_class_weight(
    class_weight="balanced", classes=np.arange(num_labels), y=y_train_enc
)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

# ---------------------------------------------------------------------------
# Tokenizer and Dataset
# ---------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


class TransactionDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(
            texts, truncation=True, padding="max_length", max_length=MAX_LENGTH
        )
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


train_dataset = TransactionDataset(X_train, y_train_enc)
val_monitor_dataset = TransactionDataset(
    X_val_monitor, label_encoder.transform(y_val_monitor))
val_dataset = TransactionDataset(X_val, y_val_enc)
test_dataset = TransactionDataset(X_test, y_test_enc)
ood_dataset = TransactionDataset(X_ood, y_ood_enc)

# ---------------------------------------------------------------------------
# Model + weighted-loss Trainer
# ---------------------------------------------------------------------------
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=num_labels)


class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(
            weight=class_weights_tensor.to(logits.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


training_args = TrainingArguments(
    output_dir=str(MODEL_OUT_DIR / "checkpoints"),
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE * 2,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    logging_steps=50,
    save_total_limit=2,
    seed=RANDOM_SEED,
    report_to="none",
)

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    # full validation set — cheap on GPU, more reliable than a subset
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

print("\nFine-tuning DistilBERT...")
start = time.time()
trainer.train()
elapsed = time.time() - start
print(f"\nTraining took {elapsed / 60:.1f} minutes")


def evaluate(name: str, dataset, y_true_enc):
    preds_output = trainer.predict(dataset)
    preds = np.argmax(preds_output.predictions, axis=-1)
    acc = accuracy_score(y_true_enc, preds)
    macro_f1 = f1_score(y_true_enc, preds, average="macro")
    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.4f} | Macro F1: {macro_f1:.4f}")
    print(classification_report(
        y_true_enc, preds, target_names=label_encoder.classes_, zero_division=0
    ))


evaluate("Validation (in-distribution)", val_dataset, y_val_enc)
evaluate("Test (in-distribution)", test_dataset, y_test_enc)
evaluate("Test (Out-of-Distribution / unseen merchants)", ood_dataset, y_ood_enc)

trainer.save_model(str(MODEL_OUT_DIR / "final"))
tokenizer.save_pretrained(str(MODEL_OUT_DIR / "final"))
print(f"\nSaved fine-tuned model to {MODEL_OUT_DIR / 'final'}")
print(f"Total training time: {elapsed / 60:.1f} minutes")

# ---------------------------------------------------------------------------
# Automatic download — runs immediately, no manual step to remember.
# Only works when actually running inside Colab; harmless no-op elsewhere.
# ---------------------------------------------------------------------------
try:
    import shutil

    from google.colab import files

    print("\nZipping model for download...")
    shutil.make_archive("distilbert_finetuned", "zip",
                        str(MODEL_OUT_DIR / "final"))
    print("Starting download — check your browser's downloads.")
    files.download("distilbert_finetuned.zip")
except ImportError:
    print("\n(Not running in Colab — skip auto-download; model is saved locally at "
          f"{MODEL_OUT_DIR / 'final'})")
