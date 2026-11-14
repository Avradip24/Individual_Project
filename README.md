# SmartSpend — Classical vs. Transformer NLP for Personal-Finance Transaction Categorization

**Individual Project — M.Eng. Information Technology**
Frankfurt University of Applied Sciences (FRA-UAS)
Student: Avradip Mazumdar · Supervisor: Prof. Dr. Andreas Pech · Project period: 01.07.2026 – 02.12.2026

---

## Table of Contents

- [Project Context](#project-context)
- [Research Question and Hypotheses](#research-question-and-hypotheses)
- [Dataset](#dataset)
- [Experimental Design](#experimental-design)
- [Models](#models)
- [Results — Accuracy](#results--accuracy)
- [Results — Confidence Calibration](#results--confidence-calibration)
- [Results — Data Efficiency](#results--data-efficiency)
- [Results — Inference Cost](#results--inference-cost)
- [Error Analysis](#error-analysis)
- [Hypothesis Assessment](#hypothesis-assessment)
- [Demonstrator](#demonstrator)
- [Codebase Structure](#codebase-structure)
- [How to Run](#how-to-run)
- [Key Engineering Decisions](#key-engineering-decisions)
- [Limitations](#limitations)
- [AI-Use Disclosure](#ai-use-disclosure)
- [References and Licence](#references-and-licence)

---

<a id="project-context"></a>
## Project Context

SmartSpend asks a practical question from personal-finance apps: **can a model read a noisy bank-transaction
description such as `[debit] STARBUCKS #1234` or `PYPL*BRILLIANT.ORG` and put it into the right spending
category** — and does a transformer do this better than classical machine learning, especially for merchants it
has never seen?

The study compares **eight models** on a **13-category** task, on two test sets:

| Test set | Size | What it measures |
|---|---|---|
| In-distribution test | 11,996 rows | Merchants also seen in training |
| **Out-of-distribution (OOD)** | 16,031 rows | **1,541 merchant clusters held out entirely** — the realistic "new merchant" case |

Besides accuracy, it covers confidence calibration, data efficiency, inference cost, a structured error analysis
and a small FastAPI demonstrator. The complete record of every decision, number and limitation is
[`docs/methodology_log.md`](docs/methodology_log.md); the written report follows the same structure.

---

<a id="research-question-and-hypotheses"></a>
## Research Question and Hypotheses

> Do transformer models categorize noisy transaction descriptions better than classical ML models, with the
> largest gains on unseen merchants and low-resource categories?

The exposé's hypothesis has four parts, each tested separately below: **H1** higher accuracy on unseen merchants,
**H2** better calibrated confidence, **H3** better data efficiency, **H4** larger gains on low-resource categories.

---

<a id="dataset"></a>
## Dataset

Two public Hugging Face datasets (both MIT) were remapped to 13 categories (Groceries, Restaurants,
Entertainment, Transportation, Shopping, Utilities, Housing, Insurance, Healthcare, Education, Travel, Income,
Other), subsampled to 4,000 rows per source and raw category, and merged to **96,000 rows**.
Housing and Income were added to the exposé's 11 categories after inspecting the raw labels.

| Source | Rows used | Licence |
|---|---|---|
| [DoDataThings/us-bank-transaction-categories-v2](https://huggingface.co/datasets/DoDataThings/us-bank-transaction-categories-v2) | 68,000 | MIT (tagged *Synthetic*) |
| [mitulshah/transaction-categorization](https://huggingface.co/datasets/mitulshah/transaction-categorization) | 28,000 (of ~4.5 M) | MIT — cite Mitul Shah (2025) |

<p align="center"><img src="docs/figures/fig02_dataset.png" width="560" alt="Dataset composition by category and source"></p>

*Rows per category and source. Housing, Other, Insurance, Groceries, Education and Travel come from one source
only.* Raw data is **not** redistributed; see [How to Run](#how-to-run).

---

<a id="experimental-design"></a>
## Experimental Design

<p align="center"><img src="docs/figures/fig01_pipeline.png" width="460" alt="Pipeline from raw data to demo"></p>

**The OOD split is the core of the design.** Transactions are clustered by normalized merchant key
(payment-processor prefixes such as `PYPL*` or `VENMO PAYMENT TO` are stripped first, then near-duplicates are
merged with `rapidfuzz`). 15% of merchant clusters are removed from training, validation and in-distribution test
and kept as the OOD set. Splits (seed 42): train 55,978 · validation 11,995 · test 11,996 · OOD 16,031.

Both test sets are scored with **accuracy, macro precision / recall / F1 and per-category F1**. Macro F1 on the
OOD set is the headline metric. Uncertainty uses a **merchant-clustered bootstrap**, because rows of one
merchant are not independent.

---

<a id="models"></a>
## Models

| Group | Model | Input |
|---|---|---|
| Classical (v2) | Logistic Regression, Linear SVM, Naive Bayes, Random Forest, XGBoost | TF-IDF word 1–2-grams + char 2–5-grams (digits collapsed) + 19 hand-made features = 38,914 features |
| Transformer 1 | Frozen `paraphrase-multilingual-MiniLM-L12-v2` + Logistic Regression | 384-d sentence embedding |
| Transformer 2 | **Fine-tuned `distilbert-base-uncased`** | raw text, max 32 tokens |
| Transformer 3 | Fine-tuned multilingual MiniLM (same base as Transformer 1) | raw text, max 32 tokens |

All five classical models share one feature pipeline and **one tuning protocol** (8 candidates, 40% training
subset, validation macro F1). The fine-tuned transformers use one fixed recipe (class-weighted cross-entropy,
6 epochs with early stopping, batch 16, default learning rate 5e-5, seed 42; T4 GPU on Google Colab) and were
**not** hyperparameter-tuned, so the comparison is, if anything, conservative towards the transformers.
An untuned v1 classical set (word n-grams only) is kept as a reference.

---

<a id="results--accuracy"></a>
## Results — Accuracy

OOD macro F1 on unseen merchants (higher is better). Open circles show the in-distribution score of the same model.

<p align="center"><img src="docs/figures/fig03_id_vs_ood.png" width="560" alt="In-distribution vs OOD macro F1"></p>

| Model | In-dist. macro F1 | **OOD macro F1** | OOD accuracy | Drop |
|---|---|---|---|---|
| **Fine-tuned DistilBERT** | 0.9957 | **0.9329** | 0.917 | 0.063 |
| Logistic Regression (v2) | 0.9957 | 0.8811 | 0.854 | 0.115 |
| Fine-tuned MiniLM | 0.9942 | 0.8767 | 0.855 | 0.118 |
| Linear SVM (v2)* | 0.9953 | 0.8543 | 0.833 | 0.141 |
| Naive Bayes (v2) | 0.9825 | 0.8365 | 0.815 | 0.146 |
| XGBoost (v2) | 0.9936 | 0.8191 | 0.780 | 0.175 |
| Random Forest (v2) | 0.9547 | 0.7450 | 0.728 | 0.210 |
| Frozen MiniLM + LogReg | 0.8626 | 0.6623 | 0.670 | 0.200 |

\* Platt-calibrated predictions; the raw SVM scores 0.8514 OOD. All numbers regenerate with
`src/eval/results_table.py` (`reports/results/`).

**What the numbers say**

- The in-distribution test set cannot separate the five best models (0.994–0.996); **the comparison is decided on
  unseen merchants**.
- Fine-tuned DistilBERT beats the best tuned classical model by **5.2 points** — against 9.6 points when compared
  with the untuned v1 baselines. Fair baselines matter.
- The advantage comes from **fine-tuning, not from "being a transformer"**: the same MiniLM scores 0.662 frozen and
  0.877 fine-tuned, which only matches Logistic Regression.

**How certain is that?** Resampling merchants (not rows) gives intervals 7–13× wider than row-level ones:

<p align="center"><img src="docs/figures/fig04_bootstrap_ci.png" width="560" alt="Merchant-clustered bootstrap intervals"></p>

DistilBERT − Logistic Regression = **+5.2 points (95% CI +2.2 to +8.6)** — significant. Fine-tuned MiniLM vs
Logistic Regression (−0.4, CI −3.4 to +2.9) is a tie, and the ordering among the classical models is *not*
established.

**Per-category OOD F1** — DistilBERT is best or tied in 9 of 13 categories; its weak spots are Income, Other,
Transportation and Shopping.

<p align="center"><img src="docs/figures/fig05_per_category_f1.png" width="430" alt="Per-category OOD F1 heatmap"></p>

---

<a id="results--confidence-calibration"></a>
## Results — Confidence Calibration

Expected Calibration Error (ECE, 10 bins) on unseen merchants, and how often a *wrong* prediction is made with more
than 90% confidence.

<p align="center"><img src="docs/figures/fig06_calibration.png" width="560" alt="ECE and overconfident errors"></p>

The transformers win on accuracy but **not on calibration**. 98% of DistilBERT's OOD predictions fall in the
0.9–1.0 confidence bin, and **91% of its wrong predictions are made at over 90% confidence** (MiniLM 89%,
Logistic Regression 24%, XGBoost 13%). A "send low-confidence predictions to review" rule would catch almost none
of the transformers' mistakes. Logistic Regression is the best-calibrated accurate model (ECE 0.038). The Linear SVM
is the only model with an explicit calibration step (Platt scaling fitted on validation).

---

<a id="results--data-efficiency"></a>
## Results — Data Efficiency

Every model was retrained on 5, 10, 25, 50 and 100% of the training set.

<p align="center"><img src="docs/figures/fig07_learning_curves.png" width="560" alt="Learning curves"></p>

- DistilBERT with **5% of the data (2,798 rows)** scores 0.848 — Logistic Regression needs about five times as
  much data to match it.
- **DistilBERT with 25% of the data (0.8995) beats every classical model trained on 100%.**
- The dip at 50% looks like run-to-run variance of fine-tuning (single seed), not a real effect.
- Low-resource *categories* are only weakly supported: small categories score about 5 points lower at 5% data for
  7 of 8 models, but the gap reverses with full data. What is held out matters more than category size.

---

<a id="results--inference-cost"></a>
## Results — Inference Cost

Full pipeline (featurization / tokenization included), CPU, batch size 1, 300 OOD transactions.

<p align="center"><img src="docs/figures/fig08_cost_quality.png" width="520" alt="Cost versus quality"></p>

| Model | Median latency | Size on disk | Peak RAM | OOD macro F1 |
|---|---|---|---|---|
| Logistic Regression v2 | **3.05 ms** | 4.1 MB | 153 MB | 0.8811 |
| Fine-tuned DistilBERT | 62.1 ms | 268 MB | 933 MB | **0.9329** |
| Fine-tuned MiniLM | 32.6 ms | 471 MB | 1,061 MB | 0.8767 |
| Random Forest v2 | 92.6 ms | 515 MB | 529 MB | 0.7450 |

Only **Logistic Regression and fine-tuned DistilBERT** lie on the cost–quality frontier: +5.2 points of OOD macro F1
costs about 20× the latency. Parameter counts are not comparable across families (weights vs. coefficients vs. tree
nodes); the full table is in the log (§11). One machine, one run.

---

<a id="error-analysis"></a>
## Error Analysis

**Errors arrive in whole-merchant blocks, not as scattered noise.** Five templates or merchants account for 803 of
DistilBERT's 1,332 OOD errors (60%).

<p align="center"><img src="docs/figures/fig09_error_concentration.png" width="520" alt="Error concentration"></p>

| Finding | Evidence |
|---|---|
| **Income → Other is one unseen template**, not label noise | All 490 errors are `[credit] Zelle payment from <NAME> <ID>`; the training set contains no Zelle row. An initial "inconsistent labels" explanation was tested and **retracted**. |
| **Income → Utilities** (classical models) | Maternity Leave, Paternity Leave and Social Security templates — also absent from training; DistilBERT gets 99–100% of them right |
| **Shopping → Transportation** | Four unseen merchants: UPS Store, Bloomingdale's, Von Maur, Family Dollar |
| **A wrapper phrase breaks classical models** | 951 rows of `PREAUTHORIZED WITHDRAWAL TO <merchant>`; deleting the phrase restores 0.94–0.98 accuracy (figure below) |
| **Guideline gap** | The labelling guideline has no rule for person-to-person payments (Zelle) |

<p align="center"><img src="docs/figures/fig10_wrapper_ablation.png" width="520" alt="Wrapper phrase counterfactual"></p>

Further analyses (details in the log, §10): confusion matrices and per-class recall for all eight models, feature
importance for the five classical models, score decomposition (Linear SVM decides by class prior, Logistic
Regression by words), and occlusion saliency for the transformers (on wrong rows, the wrapper phrase overrides
the merchant). An informal 10-string German probe (4/10 correct for DistilBERT) is reported as a limitation only;
German evaluation was descoped on the supervisor's advice.

---

<a id="hypothesis-assessment"></a>
## Hypothesis Assessment

| Hypothesis | Verdict | Evidence |
|---|---|---|
| H1 — higher accuracy on unseen merchants | **Supported for fine-tuned DistilBERT only** | +5.2 pts over best classical, CI excludes 0; fine-tuned MiniLM ties Logistic Regression; frozen embedding is worst |
| H2 — better calibration | **Not supported** | OOD ECE 0.078 (DistilBERT) vs. 0.038 / 0.032 (LogReg / XGBoost); 91% of wrong predictions at >0.9 confidence |
| H3 — better data efficiency | **Supported** (fine-tuned transformers) | 5% data = 0.848; 25% beats all classical at 100% |
| H4 — larger gains on low-resource categories | **Weakly supported** | ~5-point penalty at 5% data that vanishes with more data |

---

<a id="demonstrator"></a>
## Demonstrator

A FastAPI service with a one-page UI categorizes one transaction at a time with either fine-tuned DistilBERT or
Logistic Regression, shows the top-3 probabilities and a fixed note that confidence is not a reliable error signal.
The example buttons are chosen to show the findings above.

<p align="center"><img src="docs/figures/fig11_demo.png" width="460" alt="Demo screenshots"></p>

```bash
uv run uvicorn src.demo.api:app --port 8000      # needs models/ in place
docker compose up --build                        # ~2 GB image, models mounted read-only
```

Open <http://localhost:8000>. Endpoints: `GET /health`, `GET /models`, `POST /predict`
(`text` 1–200 chars, `model` = `distilbert` | `logreg`, `direction` = `none` | `debit` | `credit`, `top_k` 1–5).
Models load lazily (`PRELOAD_MODELS=1` preloads them, as the Docker image does); `MODELS_DIR` sets the model folder.

---

<a id="codebase-structure"></a>
## Codebase Structure

```
Individual_Project/
├── src/
│   ├── data/        build_interim_dataset.py, split_dataset.py, inspect_merchant_clusters.py
│   ├── models/      classical_features.py, train_classical_v2.py, train_embedding_baseline.py,
│   │                train_distilbert_finetuned.py, train_minilm_finetuned.py, predict_distilbert.py,
│   │                (legacy v1: train_classical_baselines.py, train_logreg_baseline.py)
│   ├── eval/        results_table.py, calibration_all_models.py, confusion_matrices_all_models.py,
│   │                error_analysis_examples.py, ood_cluster_bootstrap.py, probe_cluster.py,
│   │                feature_importance*.py, wrapper_ablation*.py, transformer_saliency.py,
│   │                income_to_utilities.py, inference_cost_analysis.py, learning_curves.py,
│   │                learning_curve_transformer.py (Colab), german_probe.py
│   └── demo/        api.py, static/index.html
├── tests/           pytest suite (features, API)
├── docs/            methodology_log.md, labeling_guideline.md, figures/
├── reports/         generated tables and figures (results, calibration, error analysis, ...)
├── Dockerfile · docker-compose.yml · requirements-demo.txt
├── pyproject.toml · uv.lock
└── data/, models/   NOT in the repository (git-ignored)
```

The figures in `docs/figures/` are drawn from the numbers in `docs/methodology_log.md`; the underlying tables are
regenerated by the scripts above.

---

<a id="how-to-run"></a>
## How to Run

Python 3.12 and [uv](https://docs.astral.sh/uv/). Run from the repository root.

**1. Setup**

```bash
uv sync
```

**2. Data** — raw data is not redistributed. Download both sources into `data/raw/` as
`us_bank_transactions.csv` and `mitulshah_transaction_categorization.parquet` (concatenate into one file if the
download is several parquet files).

**3. Reproduce the results**

| Step | Command | Where |
|---|---|---|
| 1. Remap and merge | `uv run python src/data/build_interim_dataset.py` | local |
| 2. Split + OOD holdout | `uv run python src/data/split_dataset.py` | local |
| 3. Classical models (v2) | `uv run python src/models/train_classical_v2.py` | local CPU; XGBoost on a Colab T4 GPU |
| 4. Frozen embedding + LogReg | `uv run python src/models/train_embedding_baseline.py` | local CPU (~16 min) |
| 5. Fine-tuned DistilBERT | `src/models/train_distilbert_finetuned.py` | **Colab T4** (~23 min) |
| 6. Fine-tuned MiniLM | `src/models/train_minilm_finetuned.py` | **Colab T4** (~28 min) |
| 7. Results table | `uv run python src/eval/results_table.py` | local |

The Colab scripts read and write Google Drive paths; adjust `DATA_DIR` and `MODEL_OUT_DIR` at the top and copy the
model folders into `models/` (e.g. `models/distilbert_finetuned/final`). Each calibration, probe and saliency script
in `src/eval/` carries a usage line in its docstring (log §10).

**4. Demo, tests, linting**

```bash
uv run uvicorn src.demo.api:app --port 8000
uv run pytest          # 32 passed, 2 skipped without models/
uv run ruff check .
```

The API tests use a stub predictor; two integration tests run the real models and are skipped when `models/` is absent.

---

<a id="key-engineering-decisions"></a>
## Key Engineering Decisions

1. **Merchant-cluster OOD split instead of a random split.** A random split scores every model at ≈0.99 and hides
   the difference between them; holding out whole merchants measures the "new merchant" case.
2. **Processor-prefix stripping before clustering.** Without it, PayPal / Venmo / Cash App rows merge unrelated
   merchants into one cluster. The prefix list was built in three rounds of spot-checking and is not guaranteed
   exhaustive.
3. **One feature pipeline and one tuning protocol for all five classical models**, so differences come from the
   model and not from its preparation. Features and tuning together moved Logistic Regression from 0.798 to 0.881.
4. **Fine-tuned *and* frozen versions of the same MiniLM**, to separate "fine-tuning" from "architecture".
5. **Merchant-clustered bootstrap** instead of row-level intervals, because OOD rows come in merchant blocks.
6. **Platt scaling only for the SVM**, which has no probabilities; flagged as not like-for-like.
7. **Honest corrections kept in the log**: the "label noise" reading of Income → Other and the claim that Random
   Forest scatters Zelle rows were both retracted after checking.
8. **The demo shows its own limits**: the UI states that confidence is not an error signal.

---

<a id="limitations"></a>
## Limitations

- **Single training seed** per model; no repeated runs.
- **Classical models were tuned, transformers were not**; the validation set is in-distribution and saturated, so
  tuning cannot select for unseen-merchant generalization (an XGBoost run with default settings scored about
  2 points higher OOD than the tuned one).
- **English-only, partly synthetic data**: absolute scores may not transfer to real bank statements. German was
  descoped; the informal probe shows a clear generalization gap.
- Errors come in merchant blocks, so the OOD scores rest on far fewer independent observations than 16,031 rows.
- Saliency is word-level occlusion on selected examples, not a full attribution study.
- Cost measured on one machine; GPU inference not measured.

---

<a id="ai-use-disclosure"></a>
## AI-Use Disclosure

An AI assistant (Claude, by Anthropic) was used as a coding and writing aid: planning the work step by step,
writing and debugging parts of the code (including evaluation and error-analysis scripts, the demonstrator and the
tests), suggesting analyses, and drafting the documentation (`docs/methodology_log.md`, this README and the
report). The author ran every experiment on their own machine or on Google Colab, supplied all outputs, checked
the results against the data and made the decisions about scope, models and interpretation. Claims that the
assistant first stated wrongly were corrected after checking (for example the Random Forest / Zelle statement,
log §10.5). The author is responsible for the content of the repository and of the report.

---

<a id="references-and-licence"></a>
## References and Licence

- Mitul Shah (2025). *Financial Transaction Categorization Dataset.* Hugging Face, `mitulshah/transaction-categorization`.
- DoDataThings. *us-bank-transaction-categories-v2.* Hugging Face (synthetic, MIT).
- Sanh et al. (2019). *DistilBERT, a distilled version of BERT.* arXiv:1910.01108.
- Reimers & Gurevych (2019, 2020). *Sentence-BERT* and *Making monolingual sentence embeddings multilingual using knowledge distillation.*
- Hendrycks et al. (2020). *Pretrained transformers improve out-of-distribution robustness.* ACL 2020.
- Guo et al. (2017). *On calibration of modern neural networks.* ICML.

The full reference list is in the report. Licence: see `LICENSE`.