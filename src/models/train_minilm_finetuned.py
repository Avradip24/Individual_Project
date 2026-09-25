"""
SmartSpend — Transformer approach 4 (bonus): fine-tuned multilingual MiniLM.

This uses the EXACT SAME base model as Approach 1 (the frozen embedding
model, paraphrase-multilingual-MiniLM-L12-v2) but fine-tunes it end-to-end
instead of freezing it. This isolates a single variable against Approach 1:
frozen vs. fine-tuned, same architecture, same pretraining. It directly
answers whether MiniLM's earlier weak performance (worst OOD relative
robustness of every model tested) was because it was frozen, or because
the architecture itself is a poor fit for this task -- a cleaner question
than comparing across different architectures (DistilBERT vs. MiniLM) at
the same time as frozen-vs-fine-tuned.

Same training setup as both DistilBERT scripts (weighted loss, same data
splits, same epoch-ceiling + early stopping) so all results stay directly
comparable across the full set of transformer experiments.

Run on Colab GPU: upload the 4 CSVs to data/processed/,
!pip install transformers accelerate -q, then run this.
"""

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
MODEL_OUT_DIR = Path("models/minilm_finetuned")
MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
# Same base model as the FROZEN embedding approach (Approach 1) -- this is
# the point of this experiment: same architecture, different training.
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MAX_LENGTH = 32
NUM_EPOCHS = 6  # ceiling — early stopping decides the real stopping point
BATCH_SIZE = 16

SUBSET_SIZE = None  # full run


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

print(
    f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)} | OOD: {len(X_ood)}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cpu":
    print("No GPU detected — this will be slow. Use Colab GPU as before.")

label_encoder = LabelEncoder()
label_encoder.fit(y_train + y_val + y_test + y_ood)
num_labels = len(label_encoder.classes_)
print(f"Num categories: {num_labels}")

y_train_enc = label_encoder.transform(y_train)
y_val_enc = label_encoder.transform(y_val)
y_test_enc = label_encoder.transform(y_test)
y_ood_enc = label_encoder.transform(y_ood)

class_weights = compute_class_weight(
    class_weight="balanced", classes=np.arange(num_labels), y=y_train_enc
)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

print(f"\nLoading tokenizer and model: {MODEL_NAME} ...")
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
val_dataset = TransactionDataset(X_val, y_val_enc)
test_dataset = TransactionDataset(X_test, y_test_enc)
ood_dataset = TransactionDataset(X_ood, y_ood_enc)

# num_labels here adds a fresh classification head on top of MiniLM's
# pretrained encoder -- same "MISSING: classifier.weight" pattern you saw
# when loading DistilBERT is expected here too, for the same reason.
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
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

print("\nFine-tuning multilingual MiniLM...")
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

# Automatic download — same pattern as the DistilBERT scripts
try:
    import shutil

    from google.colab import files

    print("\nZipping model for download...")
    shutil.make_archive("minilm_finetuned", "zip",
                        str(MODEL_OUT_DIR / "final"))
    print("Starting download — check your browser's downloads.")
    files.download("minilm_finetuned.zip")
except ImportError:
    print(
        f"\n(Not running in Colab — model saved locally at {MODEL_OUT_DIR / 'final'})")
