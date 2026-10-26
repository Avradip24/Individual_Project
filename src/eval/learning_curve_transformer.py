"""
SmartSpend -- Learning curves (Colab GPU part): fine-tuned transformer at
several training-set sizes.

Mirrors train_distilbert_finetuned.py exactly (class-weighted loss, 6-epoch
ceiling with early stopping on validation macro F1, batch 16, max_length 32,
default learning rate, seed 42); only the number of TRAINING rows changes
(stratified subsample). Validation stays the full validation set, as in the
original run. Results use the same schema as learning_curves.py, so the laptop
script merges them into the final figure.

The full-data point (100%) already exists from the original run and is added
by learning_curves.py from cached predictions, so the default fractions here
are 0.1, 0.25 and 0.5.

Colab (GPU runtime), after uploading data/processed/{train,val,test_in_distribution,test_ood}.csv
and this file:
    !pip install -q transformers accelerate
    !python learning_curve_transformer.py --data-dir data/processed --out-dir results
Resumable: fractions already present in the output CSV are skipped, and the CSVs are
written after every fraction, so a disconnect loses at most one run.

For fine-tuned MiniLM later:
    !python learning_curve_transformer.py --model-name sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 --tag minilm_finetuned --fractions 0.05,0.1,0.25,0.5 --data-dir . --out-dir results
Then, in a separate notebook cell, download the two result CSVs:
    from google.colab import files
    files.download("results/learning_curves_distilbert_finetuned.csv")
    files.download("results/per_category_distilbert_finetuned.csv")
Put both into reports/learning_curves/ on the laptop and run:
    uv run python src/eval/learning_curves.py --plot-only
"""

import argparse
import gc
import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          EarlyStoppingCallback, Trainer, TrainingArguments)

MAX_LENGTH = 32
NUM_EPOCHS = 6
BATCH_SIZE = 16
SEED = 42
EVAL_SPLITS = {"test": "test_in_distribution", "ood": "test_ood"}


# ---- identical copies of the helpers in learning_curves.py -----------------
def size_groups(train_labels, classes):
    counts = Counter(train_labels)
    order = sorted(classes, key=lambda c: (counts[c], c))
    groups = {c: "medium" for c in classes}
    for c in order[:4]:
        groups[c] = "small"
    for c in order[-4:]:
        groups[c] = "large"
    return groups, counts


def score(y_true, pred, classes, groups, counts, meta):
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


class TxDataset(Dataset):
    def __init__(self, tokenizer, texts, labels):
        self.enc = tokenizer(texts, truncation=True,
                             padding="max_length", max_length=MAX_LENGTH)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(int(self.labels[i]))
        return item


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--model-name", default="distilbert-base-uncased")
    ap.add_argument("--tag", default="distilbert_finetuned")
    ap.add_argument("--fractions", default="0.1,0.25,0.5")
    a = ap.parse_args()

    data, out = Path(a.data_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary_csv = out / f"learning_curves_{a.tag}.csv"
    percat_csv = out / f"per_category_{a.tag}.csv"
    print("Device:", "cuda" if torch.cuda.is_available()
          else "CPU (this will be very slow!)")

    frames = {s: pd.read_csv(data / f"{s}.csv")
              for s in ["train", "val", *EVAL_SPLITS.values()]}
    enc_labels = LabelEncoder().fit(
        pd.concat([f["final_category"] for f in frames.values()]))
    classes = list(enc_labels.classes_)
    y = {s: enc_labels.transform(f["final_category"])
         for s, f in frames.items()}
    texts = {s: f["transaction_description"].astype(
        str).tolist() for s, f in frames.items()}
    groups, counts = size_groups(
        frames["train"]["final_category"].tolist(), classes)

    tokenizer = AutoTokenizer.from_pretrained(a.model_name)
    val_ds = TxDataset(tokenizer, texts["val"], y["val"])
    eval_ds = {s: TxDataset(tokenizer, texts[f], y[f])
               for s, f in EVAL_SPLITS.items()}

    done = set()
    rows = pd.read_csv(summary_csv).to_dict(
        "records") if summary_csv.exists() else []
    cats = pd.read_csv(percat_csv).to_dict(
        "records") if percat_csv.exists() else []
    done = {r["fraction"] for r in rows}

    for frac in [float(f) for f in a.fractions.split(",")]:
        if frac in done:
            print(f"Fraction {frac} already done, skipping.")
            continue
        idx, _ = train_test_split(np.arange(len(y["train"])), train_size=frac,
                                  stratify=y["train"], random_state=SEED)
        print(
            f"\n===== {a.tag} | {frac:.0%} of train = {len(idx)} rows =====", flush=True)
        train_ds = TxDataset(tokenizer, [texts["train"][i]
                             for i in idx], y["train"][idx])
        weights = torch.tensor(compute_class_weight("balanced", classes=np.arange(len(classes)),
                                                    y=y["train"][idx]), dtype=torch.float)

        model = AutoModelForSequenceClassification.from_pretrained(a.model_name,
                                                                   num_labels=len(classes))

        class WeightedTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                labels = inputs.pop("labels")
                outputs = model(**inputs)
                loss = torch.nn.CrossEntropyLoss(
                    weight=weights.to(outputs.logits.device))(outputs.logits, labels)
                return (loss, outputs) if return_outputs else loss

        def compute_metrics(p):
            pred = np.argmax(p[0], axis=-1)
            return {"accuracy": accuracy_score(p[1], pred),
                    "macro_f1": f1_score(p[1], pred, average="macro")}

        work = out / f"_tmp_{a.tag}_{int(frac * 100)}"
        args = TrainingArguments(
            output_dir=str(work), num_train_epochs=NUM_EPOCHS,
            per_device_train_batch_size=BATCH_SIZE, per_device_eval_batch_size=BATCH_SIZE * 2,
            eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
            metric_for_best_model="macro_f1", greater_is_better=True, logging_steps=100,
            save_total_limit=1, seed=SEED, report_to="none")
        trainer = WeightedTrainer(model=model, args=args, train_dataset=train_ds,
                                  eval_dataset=val_ds, compute_metrics=compute_metrics,
                                  callbacks=[EarlyStoppingCallback(early_stopping_patience=2)])
        t0 = time.time()
        trainer.train()
        fit_s = round(time.time() - t0, 1)

        for split, ds in eval_ds.items():
            meta = dict(model=a.tag, seed=SEED, fraction=frac, n_train=int(len(idx)),
                        split=split, fit_time_s=fit_s)
            r, c = score(y[EVAL_SPLITS[split]], trainer.predict(ds).predictions.argmax(-1),
                         classes, groups, counts, meta)
            rows.append(r)
            cats.extend(c)
            print(f"  {split}: macro F1 {r['macro_f1']:.4f} | small {r['f1_small']:.4f} | "
                  f"large {r['f1_large']:.4f}", flush=True)

        pd.DataFrame(rows).to_csv(summary_csv, index=False)
        pd.DataFrame(cats).to_csv(percat_csv, index=False)
        shutil.rmtree(work, ignore_errors=True)
        del trainer, model
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\nDone. Results are in {summary_csv} and {percat_csv}.")
    print("This script cannot download them itself. In Colab, run the download cell "
          "(see the header) or use the Files sidebar.")


if __name__ == "__main__":
    main()
