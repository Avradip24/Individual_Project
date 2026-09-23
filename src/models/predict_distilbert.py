"""
SmartSpend — load the fine-tuned DistilBERT model and run predictions.

Run this after extracting the downloaded zip into
models/distilbert_finetuned/final/ (update MODEL_PATH below if you put it
somewhere else).
"""

from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_PATH = "models/distilbert_finetuned/final"

# Exact label order used during training — sklearn's LabelEncoder sorts
# labels alphabetically, confirmed against the order printed in every
# classification_report during training. Must match exactly or predictions
# will be silently mislabeled (e.g. "Groceries" reported as "Healthcare").
LABELS = [
    "Education", "Entertainment", "Groceries", "Healthcare", "Housing",
    "Income", "Insurance", "Other", "Restaurants", "Shopping",
    "Transportation", "Travel", "Utilities",
]

print(f"Loading model from {MODEL_PATH} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()  # inference mode — disables dropout etc.


def predict(text: str):
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=32, padding="max_length"
    )
    with torch.no_grad():  # no need to track gradients for inference — faster, less memory
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)[0]
    pred_idx = torch.argmax(probs).item()
    return LABELS[pred_idx], probs[pred_idx].item()


if __name__ == "__main__":
    # Mix of English (should be easy) and German (never seen in training —
    # this model has zero multilingual pretraining, unlike the frozen
    # MiniLM embedding approach) merchant strings, spanning several
    # categories. Expected category noted as a comment for manual checking
    # — not fed to the model.
    test_strings = [
        ("STARBUCKS STORE 1234", "Restaurants"),
        ("[debit] WITHDRAWAL FROM MIDAS", "Transportation (car maintenance)"),
        ("REWE-SAG-7342", "Groceries"),
        ("EDEKA 4471 FFM", "Groceries"),
        ("ALDI SUED 0091", "Groceries"),
        ("DM DROGERIE MARKT", "Healthcare (drugstore)"),
        ("TELEKOM DEUTSCHLAND", "Utilities"),
        ("DB FERNVERKEHR AG", "Travel (long-distance rail)"),
        ("MAINOVA AG FFM", "Utilities"),
        ("APOTHEKE AM DOM", "Healthcare (pharmacy)"),
    ]
    for text, expected in test_strings:
        label, confidence = predict(text)
        match = "OK" if label.lower() in expected.lower() else "MISS"
        print(
            f"[{match:4s}] {text!r:30s} -> {label:15s} ({confidence:.1%})  expected: {expected}")

    # save the label list alongside the model for future reference — this
    # was missing from the original training script's saved artifacts
    (Path(MODEL_PATH) / "labels.json").write_text(
        "[\n  " + ",\n  ".join(f'"{label}"' for label in LABELS) + "\n]"
    )
    print(
        f"\nSaved label order to {MODEL_PATH}/labels.json for future reference")
