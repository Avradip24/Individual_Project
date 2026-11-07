# SmartSpend: Classical vs. Transformer NLP for Transaction Categorization

Master's individual project, M.Eng. Information Technology, Frankfurt University of Applied Sciences.
Student: Avradip Mazumdar. Supervisor: Prof. Dr. Andreas Pech. Period: 01/07/2026 to 02/12/2026.

**Question.** Do transformer models categorize noisy bank-transaction descriptions better than
classical ML models, especially for merchants they have never seen?

The study compares eight models on a 13-category task, on an in-distribution test set and on an
out-of-distribution (OOD) set made of held-out merchant clusters. It also includes calibration and
error analysis, and a small FastAPI demonstrator.

## Main results

OOD macro F1 on 16,031 transactions from 1,541 merchant clusters never seen in training.

| Model | In-dist. macro F1 | **OOD macro F1** |
|---|---|---|
| Fine-tuned DistilBERT (6 epochs) | 0.9957 | **0.9329** |
| Logistic Regression (v2 features, tuned) | 0.9957 | 0.8811 |
| Fine-tuned multilingual MiniLM | 0.9942 | 0.8767 |
| Linear SVM (v2, tuned)* | 0.9953 | 0.8543 |
| Naive Bayes (v2, tuned) | 0.9825 | 0.8365 |
| XGBoost (v2, tuned) | 0.9936 | 0.8191 |
| Random Forest (v2, tuned) | 0.9547 | 0.7450 |
| Frozen multilingual MiniLM + LogReg | 0.8626 | 0.6623 |

\* Platt-calibrated probabilities (raw SVM: 0.8514 OOD). The in-distribution value is for the raw model.
The authoritative numbers (accuracy, precision, recall, per-category scores) are in
`reports/results/` after running `src/eval/results_table.py`.

What the results support:

- Fine-tuned DistilBERT beats the best classical model by 5.2 points OOD (cluster-bootstrap 95% CI
  +2.2 to +8.6).
- The benefit comes from fine-tuning, not from "being a transformer": the same MiniLM scores 0.6623
  frozen and 0.8767 fine-tuned, which only matches Logistic Regression.
- Calibration was not improved by the transformers. DistilBERT is almost always maximally confident,
  including when wrong.
- Errors on unseen merchants come in whole templates (for example Zelle transfers labelled Other),
  not as scattered noise. See `docs/methodology_log.md` section 10.

The full experiment record, with every decision and limitation, is `docs/methodology_log.md`.

## Repository layout

```
src/data/      build_interim_dataset.py, split_dataset.py
src/models/    classical + transformer training scripts
src/eval/      calibration, confusion matrices, probes, results table
src/demo/      FastAPI service and web UI
tests/         pytest suite
docs/          methodology_log.md, labeling_guideline.md
reports/       generated tables and figures
data/, models/ NOT in the repository (see below)
```

## Data

Raw data is not redistributed. Download both sources into `data/raw/`:

| Source | Licence |
|---|---|
| [DoDataThings/us-bank-transaction-categories-v2](https://huggingface.co/datasets/DoDataThings/us-bank-transaction-categories-v2) (Hugging Face; the author's model card describes the data as synthetic) | MIT (tagged "Synthetic" on the dataset card) |
| [mitulshah/transaction-categorization](https://huggingface.co/datasets/mitulshah/transaction-categorization) | MIT. Cite: Mitul Shah (2025), "Financial Transaction Categorization Dataset" |

Both are remapped to 13 categories (Groceries, Restaurants, Entertainment, Transportation, Shopping,
Utilities, Housing, Insurance, Healthcare, Education, Travel, Income, Other), balanced to 4,000 rows per
source and raw category, and merged to 96,000 rows. Details are in `docs/methodology_log.md` sections 1 to 3.

Splits (seed 42): train 55,978, validation 11,995, in-distribution test 11,996, and OOD 16,031 (15% of
merchant clusters held out entirely).

## Setup

Python 3.12 and [uv](https://docs.astral.sh/uv/).

```
uv sync
```

## Reproducing the results

Run from the repository root. Trained models and processed data are git-ignored, so everything below
regenerates them.

| Step | Command | Where |
|---|---|---|
| 1. Remap and merge | `uv run python src/data/build_interim_dataset.py` | local |
| 2. Split + OOD holdout | `uv run python src/data/split_dataset.py` | local |
| 3. Classical models (v2 features, tuning) | `uv run python src/models/train_classical_v2.py` | local CPU; XGBoost was run on a Colab T4 GPU |
| 4. Frozen embedding + LogReg | `uv run python src/models/train_embedding_baseline.py` | local CPU (about 16 min) |
| 5. Fine-tuned DistilBERT | `src/models/train_distilbert_finetuned.py` | **Colab T4 GPU** (about 23 min) |
| 6. Fine-tuned MiniLM | `src/models/train_minilm_finetuned.py` | **Colab T4 GPU** (about 28 min) |
| 7. Results table | `uv run python src/eval/results_table.py` | local |

The Colab scripts read and write Google Drive paths; adjust `DATA_DIR` and `MODEL_OUT_DIR` at the top.
Copy the resulting model folders into `models/` (for example `models/distilbert_finetuned/final`).
The calibration, confusion-matrix, probe and saliency scripts in `src/eval/` are described in
`docs/methodology_log.md` section 10, each with a usage line in its docstring.

`src/models/train_classical_baselines.py` and `train_logreg_baseline.py` are the earlier untuned v1
baselines, kept as a reference point; the reported results use v2.

## Demonstrator

A FastAPI service with a one-page UI to categorize a transaction with fine-tuned DistilBERT or Logistic
Regression. It shows the top-3 probabilities and a note that confidence is not reliable.

Locally (needs `models/` in place):

```
uv run uvicorn src.demo.api:app --port 8000
```

Docker (about 2 GB image; models are mounted read-only, not baked in):

```
docker compose up --build
```

Then open http://localhost:8000. Endpoints: `GET /health`, `GET /models`, `POST /predict`.

## Tests and linting

```
uv run pytest
uv run ruff check .
```

The API tests use a stub predictor and need no model files; two integration tests run the real models
and are skipped when `models/` is absent.

## Limitations

- Single training seed; no repeated runs.
- The transformers were not hyperparameter-tuned; the classical models were.
- English-only training data. German and multilingual evaluation was descoped (informal probe only).
- Part of the data is synthetic, so absolute scores may not transfer to real bank statements.
- Calibration (ECE) and learning-curve analyses cover only part of the model set (see the log).
- Occlusion saliency is a rough explanation method, not attribution.

## AI-use disclosure

TODO: confirm and edit before submission. Draft: Claude (Anthropic) was used as an assistant for
planning, code review and debugging, drafting analysis scripts and the demonstrator, and drafting
documentation. All experiments were run, and all results checked and interpreted, by the author, who
is responsible for the content.

## Licence

See `LICENSE`.