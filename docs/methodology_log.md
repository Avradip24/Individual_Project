# SmartSpend — Methodology & Experiment Log

Running record of technical decisions, rationale, and results. Update this
as the project progresses — this is written to be adaptable almost
directly into the report's Methodology and Results chapters.

---

## 0. Progress Summary (as of 7 November 2026 — Phases 4-5)

Status relative to the exposé's phase plan: the experiments and error
analysis (Phase 3) and the demonstrator (Phase 4) are done; what remains
is the repository clean-up, the report and the presentation (Phase 5).
The project ends on 2 December 2026.

**Completed so far:**

- **Dataset curated and documented.** Combined two independent
  English-language transaction datasets (~96,000 rows total after
  remapping, subsampling, and merge — see §1). Extended the category
  taxonomy from the exposé's 11 categories to **13** (added Housing,
  Income) based on inspecting the real label distributions in both
  sources (see §2). Built a merchant-based OOD test partition (unseen
  merchants held out entirely from training) with documented
  limitations (see §3).
- **Five classical ML baselines** — Logistic Regression, Naive Bayes,
  Linear SVM, Random Forest, XGBoost (Gradient Boosting) — satisfying
  the exposé's "at least three baselines" requirement. v1 (§5) used
  untuned defaults on word-level TF-IDF; **v2 (§5.1)** adds the
  exposé's character n-grams and hand-engineered features and tunes all
  five models with one validation-set search protocol.
- **Both required transformer approaches built**: a frozen multilingual
  sentence-embedding model + lightweight classifier (§6), and a
  fine-tuned DistilBERT (§7), including a 3-vs-6 epoch ablation. A
  fine-tuned multilingual MiniLM (§7.1) was added as a controlled
  frozen-vs-fine-tuned comparison on the same base model.
- **Confidence calibration (ECE + reliability diagrams) for all eight
  models** (§9): the five tuned v2 classical models (Linear SVM via a
  sigmoid calibrator fitted on validation), the frozen embedding, and
  both fine-tuned transformers, on the in-distribution and OOD test
  sets. Replaces the earlier 3-model preliminary run.
- **Inference-cost analysis** (§11): per-transaction latency, batch
  throughput, parameter count, model size and memory for all eight
  models on CPU.
- **Learning curves** (§12): all eight models retrained at 5, 10, 25,
  50 and 100% of the training data, with scores split by category-size
  group to test the exposé's low-resource claim.
- **Error analysis** completed for all eight models on the OOD set (§10):
  confusion matrices and per-class recall, read examples and merchant
  probes, merchant-clustered bootstrap intervals, feature importance for
  the five classical models, word-occlusion saliency for the two
  transformers, and counterfactual tests of the wrapper template — with
  a genuinely interesting cross-model finding (see below).
- **Consolidated results table** (§8.1): accuracy, macro precision,
  recall and F1, and per-category F1 for all eight models on both test
  sets.
- **Demonstrator** (§13): a FastAPI service with a one-page web UI, a
  Docker image and a pytest suite.
- **Reproducible pipeline**, versioned in a private Git repo (see §14 for
  what is and is not in the repository).

**Headline finding so far:** the central hypothesis is supported for
one of two fine-tuned transformers, with important nuances (full table
in §8). On unseen merchants, fine-tuned DistilBERT reaches **93.3%** OOD
macro F1 against **88.1%** for the best tuned classical model (Logistic
Regression, v2 features): a gap of about 5 points, roughly half the
9.6-point gap measured against the untuned v1 baselines. A *frozen*
transformer embedding **underperforms every classical baseline**
(66.2%), and the same MiniLM model fine-tuned reaches 87.7% (+21.4
points), so the benefit comes from **fine-tuning**, not from the
transformer architecture per se. Fine-tuned MiniLM only matches the
best classical model (87.7% vs 88.1%, within noise), so a fine-tuned
transformer does not automatically beat well-tuned classical models;
DistilBERT does.

**Second headline finding:** the confusion-matrix analysis (§10) shows
that Income→Other is the single most confused category pair for six of
the eight models (all except Random Forest and XGBoost), with
near-identical error counts (490-509) despite very different
architectures and overall OOD accuracy. Reading the errors (§10.5) shows
this is **one transaction template, not diffuse label noise**: all 490
errors are `[credit] Zelle payment from <NAME> <ID>`, labelled Income
consistently in the OOD set, with **zero Zelle rows in the training set**
(the merchant was held out by the OOD split). The models have no example
to learn from and default to Other. This explains the 0.70 Income recall
flagged in §7 (490 of the 1,661 OOD Income rows are this template) and
means the earlier "inconsistent labelling in the source data" reading was
**not supported**. More generally, OOD errors arrive in whole-merchant
blocks: five templates or merchants account for 60% of fine-tuned
DistilBERT's errors, so the OOD scores rest on far fewer independent
observations than the 16,031 rows suggest. A merchant-clustered bootstrap (§10.5) makes this concrete: intervals are 7 to 13 times wider than row-level ones (DistilBERT 0.933, 95% CI 0.892 to 0.973), DistilBERT's 5-point lead over Logistic Regression stays significant, and differences among the classical models and between fine-tuned MiniLM and Logistic Regression are not.

**Third headline finding (calibration and data efficiency):** the
transformers win on accuracy and data efficiency but **not on
calibration**. On unseen merchants fine-tuned DistilBERT has an ECE of
0.078 against 0.038 for Logistic Regression and 0.032 for XGBoost, and
its confidence carries almost no information about correctness: **91% of
its wrong OOD predictions are made at over 90% confidence** (89% for
fine-tuned MiniLM, 24% for Logistic Regression, 13% for XGBoost), so a
confidence threshold cannot be used to flag its mistakes (§9). On the
other side, DistilBERT trained on 5% of the data (2,798 rows) already
reaches 84.8% OOD macro F1, which Logistic Regression needs about five
times as much labelled data to match, and DistilBERT with 25% of the
data beats every classical model trained on 100% (§12). The exposé's
hypothesis that transformers improve *both* accuracy and calibration is
therefore supported for accuracy (DistilBERT) and **not** for
calibration.

**Open limitation — German/multilingual input:** an informal live check
with real German merchant strings (§4.1) showed a clear generalization
gap; the training data is English-only, so this is expected. Per
supervisor guidance, German-language evaluation is not a focus of the
project: the multilingual model variants and the informal probe are
reported as a limitation and future work, and no personal bank
statement is required (§4). A follow-up probe (§4.2) found that the
fine-tuned multilingual MiniLM gets the same 4/10 as English-only
DistilBERT, so multilingual pretraining did not help on these strings.

**Not yet done (explicitly tracked):**
- Limitations to state in the report (the error analysis itself is
  complete, §10): saliency is word-level occlusion on a handful of
  examples plus one aggregate over the wrapper cluster, not a full
  attribution study; the models are single-seed; the wrapper
  counterfactual covers one template family; the hand-labelling columns
  of the two example CSVs (`verdict`, `note`) were not filled in.
- Optional extensions of the calibration work: risk-coverage
  (selective-prediction) curves and temperature scaling for the
  transformers (§9).
- Repository clean-up (remove draft files, fix script usage lines),
  public repository with README and regeneration instructions (the README
  is drafted; the repository is still private and must be made public at
  the end), AI-use disclosure, report and presentation (Phase 5 of the
  exposé's plan).
- A single training-cost and parameter-count summary for the report
  (§8, §11 and §14 hold the parts).

---

## 1. Data Sources

| Source | Format | Rows used | Categories provided | License note |
|---|---|---|---|---|
| DoDataThings/us-bank-transaction-categories-v2 (Hugging Face) | CSV | 68,000 (4,000/category × 17 raw categories) | Insurance, Mortgage, Entertainment, Utilities, Groceries, Personal Care, Shopping, Subscription, Healthcare, Rent, Fees, Income, Transportation, Education, Restaurants, Transfer, Travel | MIT (dataset card). Tagged "Synthetic" on the card; 68k rows, English, CSV. Author's model card says it is generated by an open-source generator (github.com/wnstnb/foliome) and models real bank-statement patterns |
| mitulshah/transaction-categorization (Hugging Face) | Parquet | 28,000 (subsampled to 4,000/category, from ~4.5M available) | Utilities & Services, Government & Legal, Financial Services, Income, Charity & Donations, Shopping & Retail, Healthcare & Medical, Entertainment & Recreation, Transportation, Food & Dining | MIT (dataset card); the card asks research users to cite: Mitul Shah (2025), "Financial Transaction Categorization Dataset", huggingface.co/datasets/mitulshah/transaction-categorization. Its out-of-scope uses (identifying individuals, financial-privacy-compromising models, apps needing real financial data) do not apply here |

**Why two sources, not one:** single-source training risks the model learning that source's specific formatting conventions rather than generalizable category signal. Combining two independently-constructed sources with different formatting styles (see confirmed cases: DoDataThings uses real bank-statement style like `[debit] MERCHANT...`; Mitulshah spans five countries) is a deliberate robustness choice.

**Why subsample Mitulshah:** it is ~66x larger than US Bank v2 per category. Merging without subsampling would let one source's formatting style dominate training, defeating the purpose of combining two sources. Capped at 4,000/category to match US Bank v2 exactly — a balanced 50/50 blend.

**Dropped categories:** Government & Legal, Financial Services, Charity & Donations (Mitulshah) — ~1.35M rows dropped. These have no home in the project's taxonomy; forcing them into "Other" would have made that category ~30% of the dataset, violating the design principle that "Other" stays a small catch-all.

### 1.1 Exact Category Remapping

**US Bank v2 → final taxonomy** (all 17 raw categories map cleanly, none dropped):

| Raw category | → Final category |
|---|---|
| Insurance, Entertainment, Utilities, Groceries, Shopping, Healthcare, Transportation, Education, Restaurants, Travel | *(unchanged — direct match)* |
| Mortgage, Rent | → **Housing** |
| Income | → **Income** |
| Personal Care | → **Shopping** |
| Subscription | → **Entertainment** |
| Fees, Transfer | → **Other** |

**Mitulshah → final taxonomy:**

| Raw category | → Final category |
|---|---|
| Food & Dining | → Restaurants |
| Healthcare & Medical | → Healthcare |
| Shopping & Retail | → Shopping |
| Utilities & Services | → Utilities |
| Entertainment & Recreation | → Entertainment |
| Transportation | → Transportation |
| Income | → Income |
| Government & Legal, Financial Services, Charity & Donations | → **dropped entirely** (~1.35M rows) |

### 1.2 Final Merged Dataset Composition

96,000 rows total, after remapping + subsampling + merge. Random seed 42 used throughout for reproducibility. Rows shuffled (`.sample(frac=1)`) after merge so source order doesn't bias any downstream inspection or naive slicing.

| Category | Total rows | From US Bank v2 | From Mitulshah |
|---|---|---|---|
| Entertainment | 12,000 | 8,000 | 4,000 |
| Shopping | 12,000 | 8,000 | 4,000 |
| Housing | 8,000 | 8,000 | 0 |
| Utilities | 8,000 | 4,000 | 4,000 |
| Healthcare | 8,000 | 4,000 | 4,000 |
| Other | 8,000 | 8,000 | 0 |
| Income | 8,000 | 4,000 | 4,000 |
| Transportation | 8,000 | 4,000 | 4,000 |
| Restaurants | 8,000 | 4,000 | 4,000 |
| Insurance | 4,000 | 4,000 | 0 |
| Groceries | 4,000 | 4,000 | 0 |
| Education | 4,000 | 4,000 | 0 |
| Travel | 4,000 | 4,000 | 0 |

Four categories (Housing, Other, Insurance, Groceries, Education, Travel) are entirely single-source — this is the concrete data behind the "single-source category" limitation noted above.

**Known limitation — single-source categories:** Education, Insurance, Travel, and Groceries exist only in US Bank v2 (Mitulshah has no equivalent label). These four categories are trained on one source's formatting style only. Flagged explicitly here so it isn't rediscovered as a surprise during error analysis — it should be stated directly in the report's limitations section.

**Known limitation — no German/multilingual data in the training set.** Both sources are English-language, drawn from US/UK/Australia/India/Canada. A German-language test set was planned from the student's own bank export but descoped per supervisor guidance (see §4).

**Known limitation — part of the data is synthetic.** The DoDataThings source is tagged "Synthetic" on its dataset card and described by its author as generated to model real bank-statement patterns, so about half of the rows are generated text. Absolute scores may therefore not transfer to real bank statements, and templated generation may explain why whole unseen merchants and templates fail as blocks (§10.5). Whether the mitulshah rows are real or synthetic is not stated on its card as far as checked, so the share of synthetic data overall is uncertain. State this in the report's limitations.

---

## 2. Category Taxonomy (v0.2)

Started from the expose's 11 categories. Expanded to **13** after inspecting the real label distributions in both sources (see `docs/labeling_guideline.md` for full definitions and confusable-pair tie-break rules):

- Added **Housing** (rent, mortgage — previously would have been forced into "Other" or "Utilities", neither of which fit)
- Added **Income** (salary/wage deposits — a non-spending category, kept separate so it doesn't distort spending-category metrics)

Final 13: Groceries, Restaurants, Entertainment, Transportation, Shopping, Utilities, Housing, Insurance, Healthcare, Education, Travel, Income, Other.

---

## 3. Preprocessing — Merchant Extraction for the OOD Split

**Problem:** many raw transaction strings route through payment intermediaries (PayPal, Venmo, Cash App) or use structured ACH banking fields (`ORIG CO NAME:X CO ENTRY DESCR:...`), which obscure the real merchant behind a generic prefix. A naive "first N tokens" merchant key would cluster completely unrelated merchants together under one prefix.

**Fix applied:** a list of known processor/intermediary prefixes is stripped before extracting the merchant key, plus a dedicated regex for the `ORIG CO NAME:X CO ENTRY DESCR` banking-field pattern. This list was built **iteratively**, by manually spot-checking the largest resulting merchant clusters after each pass and adding whatever new pattern surfaced — this is not a fix applied once, it took three rounds of inspection. **Not guaranteed exhaustive** — a real limitation worth naming in the methodology chapter, since production systems face this exact problem with no clean general solution.

**Full final prefix list** (order matters — more specific patterns are checked before generic ones, e.g. `VENMO PAYMENT TO` before plain `VENMO`, so a real merchant name isn't swallowed by a shorter generic match):

`PAYPAL INST XFER`, `PYPL INST XFER`, `PYPL`, `VENMO PAYMENT TO`, `VENMO`, `CASH APP PAYMENT TO`, `CASH APP`, `SQ `, `APPLE PAY`, `GOOGLE PAY`, `PREAPPROVED PAYMENT BILL USER PAYMENT:`, `SUBSCRIPTION PAYMENT:`, `EXPRESS CHECKOUT PAYMENT:`, `WITHDRAWAL FROM`, `PAYMENT TO:` — plus the separate `ORIG CO NAME:X CO ENTRY DESCR` regex extraction (used for ACH payroll-style deposits, where the real employer name sits *between* two field markers rather than after a simple prefix).

**Validation evidence (manual spot-check, final run):** after all fixes, correctly-merged clusters were confirmed for cases spanning very different formatting styles per merchant — e.g. Brilliant.org (merged across 4 different prefix styles: plain, `PreApproved Payment Bill User Payment:`, `PYPL*`, `Express Checkout Payment:`), five distinct property-management companies each correctly clustering their own `WITHDRAWAL FROM X` / `X RENT PMT ... WEB ID:` / `X MONTHLY RENT` variants *without* merging into each other (Camden Property, Avalon Communities, Invitation Homes, UDR Inc, Irvine Company, Related Cos), plus Amazon, Rite Aid, T-Mobile, and Target all correctly clustering despite heavy format variation (store numbers, case, `*` suffixes, embedded addresses). One acceptable non-merchant cluster remains (`ACH DEPOSIT INTERNET TRANSFER FROM ACCOUNT ENDING IN [masked digits]`) — a real inter-account transfer pattern, not a bug, correctly treated as one coherent group rather than fragmenting by masked account number.

**Clustering approach:** normalized merchant keys (first 2 tokens after stripping digits/punctuation/known prefixes) are fuzzy-matched using `rapidfuzz` (`fuzz.ratio >= 85`) to collapse near-duplicate strings (e.g. different store-number suffixes, `LLC`/`Inc.` suffixes) into one merchant cluster.

**OOD holdout:** 15% of merchant clusters (by cluster ID) are held out entirely — none of their rows appear in train/val/test. The remaining data is split 70/15/15 into train/val/test, stratified by category.

**Final split sizes** (after fixes): Train 55,978 | Val 11,995 | Test (in-distribution) 11,996 | OOD 16,031.

---

## 4. German / Multilingual Evaluation (descoped)

**Status: descoped per supervisor guidance.** The exposé only asks for a multilingual model variant "where feasible" to assess robustness on mixed German/English merchant strings, and the supervisor advised that a personal German bank statement is not required and that the project should not focus heavily on the German language; the correctness of the findings matters more. The multilingual MiniLM variants (§6, §7.1) and the informal probe below are therefore reported as a limitation and as future work, and no hand-labeled German partition will be built. (An optional lightweight alternative, if time allows: a small set of 30-50 well-known German merchant strings with no personal data.)

**Original plan (not pursued):** hand-label the student's own Commerzbank export (~6-12 months), preprocess it with the same normalization pipeline, and save it as `data/ood/test_ood_multilingual.csv`, never touched during training. That would have tested a *stronger* form of generalization than the merchant-based OOD set (different language, bank format and country, with zero vocabulary overlap with the training data).

### 4.1 Preliminary Qualitative Probe (fine-tuned DistilBERT, 10 hand-picked German/English strings)

Before the formal hand-labeled German OOD set exists, a quick informal check was run on the fine-tuned DistilBERT model using 10 well-known merchant strings (mix of English controls and German brands), to sanity-check expectations ahead of the real evaluation. **This is illustrative, not a formal metric** — small sample, hand-picked strings, no statistical claims should be drawn from it directly.

**Result: 4/10 correct (40%)** — compare to ~93% OOD macro F1 on unseen-but-English merchants from the same model. Confirms the expected finding: fine-tuning on English-only data does not transfer to a different language, even though the underlying *concepts* (grocery store, pharmacy, utility company) are identical.

**Calibration red flag, consistent with what formal ECE analysis should quantify:** confidence on the 6 wrong predictions was mostly extremely high (100%, 82.9%, 100%, 99.9%, 81.4%) — the model is not just wrong on foreign-language input, it's confidently wrong. The one prediction with meaningfully lower confidence (52.9%) happened to be the only correct German prediction among the ambiguous cases. This is a concrete, real illustration of exactly the calibration failure mode the expose's ECE analysis is designed to catch formally.

**Notable pattern worth investigating further — "Transportation" as an apparent fallback:** 4 of the 6 wrong predictions (REWE, DM Drogerie Markt, DB Fernverkehr, Apotheke am Dom — true categories Groceries/Healthcare/Travel/Healthcare) were all misclassified specifically as **Transportation**, not scattered across categories. Working hypothesis: Transportation's broad, heterogeneous training definition (fuel, tolls, parking, car maintenance, public transit) may make its decision region behave like a "default" for unfamiliar/out-of-distribution input. Not confirmed — would need a follow-up test with pure gibberish/non-language strings to check whether this is a genuine attractor behavior or coincidence from this small sample.

**Follow-up:** done on 9 November 2026 with the other models and a gibberish control, see §4.2. In the re-run, DistilBERT again scores 4/10, but 5 of its 6 wrong predictions (not 4 of 6) are Transportation: EDEKA is also predicted Transportation at 83%, which matches the 82.9% confidence listed above, so the count above appears to have missed it. The re-run output is the reference.

### 4.2 Follow-up Probe: Four Models and a Gibberish Control (illustrative, not a metric)

`src/eval/german_probe.py` runs the same 10 strings (2 English controls, 8 German) through fine-tuned DistilBERT, fine-tuned multilingual MiniLM, frozen multilingual MiniLM + LogReg and Logistic Regression v2, with the same scoring as §4.1 (predicted label must equal the expected category). Output: `reports/german_probe/german_probe.csv`. **Ten hand-picked strings: no statistical claim can be drawn from this.**

| Model | Correct (of 10) | German (of 8) | Mean confidence when wrong | Wrong predictions that are Transportation |
|---|---|---|---|---|
| Fine-tuned DistilBERT | 4 | 2 | 0.94 | 5 of 6 |
| Fine-tuned MiniLM (multilingual) | 4 | 2 | 0.98 | 0 of 6 (mostly Shopping) |
| Frozen MiniLM + LogReg | 0 | 0 | 0.57 | 2 of 10 |
| Logistic Regression v2 | 3 | 1 | 0.47 | 3 of 7 |

- **Multilingual pretraining did not help.** Fine-tuned MiniLM, whose base model was pretrained on many languages, gets the same 4/10 as English-only DistilBERT, and the same four strings right (both English controls, ALDI SUED, TELEKOM DEUTSCHLAND). The mechanism proposed in §6 (pretrained world knowledge about German brands transferring to unseen strings) is **not supported** by this probe. After fine-tuning on English-only transaction data, the multilingual model did not behave differently from the English one on these strings.
- **The frozen multilingual embedding scored 0/10, including the English controls** (STARBUCKS STORE 1234 is predicted Shopping at 54%), consistent with its weak OOD result (§6). It was much less confident when wrong (mean 0.57) than the fine-tuned transformers.
- **Both fine-tuned transformers are confidently wrong:** mean confidence 0.94 (DistilBERT) and 0.98 (MiniLM) on their wrong answers, against 0.47 for Logistic Regression. This repeats the calibration finding of §9 on a small, different kind of input.
- **The Transportation-fallback hypothesis (§4.1) is only partly supported.** On eight meaningless strings (`XQZ KLM 9921`, `ZZZZ WWWW`, ...), DistilBERT predicts Transportation 4 times (mean confidence 0.92) and Logistic Regression 4 times, but fine-tuned MiniLM mostly predicts Shopping (3) and spreads the rest, and the frozen model spreads its predictions. So the effect appears for the two English-vocabulary models and not for the multilingual one, which suggests it depends on the model and is not a property of the Transportation class. With eight strings this is a hint, not a result.
- **No model predicts Other for any of the 8 meaningless strings** (0 of 32 predictions). In this dataset Other mostly holds fees and transfers, not "unknown", so the models have never been taught an "I don't know" class; a deployed system would need a rejection mechanism, which the poor confidence signal (§9) cannot provide on its own.

**Limitation:** 10 strings chosen by hand, expected categories assigned by the author (DM Drogerie Markt as Healthcare is arguable: Shopping is also defensible), one run, no repeated seeds. The result is reported as a limitation and future work (a proper German test set was descoped, §4).

---

## 5. Classical ML Baselines — Results

Five models, one shared TF-IDF feature space (word unigrams+bigrams, `min_df=2`, `max_features=20,000`), fit once and reused across all models for a fair comparison. `class_weight="balanced"` used where supported, to handle the ~3x spread between smallest (Insurance, 4,000) and largest (Entertainment/Shopping, 12,000) categories.

| Model | Val Macro F1 | Test (in-dist) Macro F1 | **Test OOD Macro F1** | Training time |
|---|---|---|---|---|
| **Linear SVM** | 0.9929 | 0.9936 | **0.8364** (best) | fast |
| Multinomial Naive Bayes | 0.9729 | 0.9747 | 0.8240 | fastest |
| Gradient Boosting (XGBoost) | 0.9919 | 0.9916 | 0.7985 | slow (minutes) |
| Logistic Regression | 0.9901 | 0.9906 | 0.7982 | fast |
| Random Forest | 0.9802 | 0.9761 | 0.7374 (worst) | moderate (1.5-2 min) |

**v1 best classical baseline: Linear SVM, 83.64% OOD macro F1.** These v1 models use untuned defaults on word-level features only, so they are kept as a reference point; the v2 models in §5.1 are the final classical baselines used in the comparison (§8).

**Key finding — complexity vs. generalization inversely related here:** Random Forest is strongest in-distribution among the weaker performers but has the *largest* in-distribution-to-OOD gap (23.9 points) — consistent with it overfitting to memorized merchant-specific tokens. Naive Bayes is the *weakest* in-distribution model but the *second-best* on OOD — its simpler independence assumption appears to generalize better precisely because it can't memorize as precisely. Worth developing further in the discussion chapter: model expressiveness on this task correlates with in-distribution performance but not with OOD robustness, and may even trend inversely.

### 5.1 Classical v2 — Extended Features and One Tuning Protocol

**Why v2:** the exposé specifies TF-IDF over words *and* character n-grams, hand-engineered features (string length, digit ratio, merchant cues), and hyperparameters tuned on the validation set under a consistent protocol. v1 used word 1-2-grams with defaults, so it met neither. v2 closes both gaps. Everything below is on a separate git branch (`classical-v2`), saved to `models/classical_v2/` and `reports/classical_v2/`, so the v1 results stay intact.

**Features** (`src/models/classical_features.py`; fit on the training split only, shared by all five models):
- **word:** TF-IDF word 1-2-grams, v1 settings (`min_df=2`, up to 20,000 features).
- **char:** TF-IDF `char_wb` 2-5-grams, sublinear tf, `min_df=3`, up to 60,000 features. Digit runs are collapsed to "0" first, so store numbers and transaction codes do not create thousands of one-off n-grams.
- **hand:** 19 features — length, token count, digit / uppercase / punctuation ratios, flags for `*`, `#`, a digit run of 4+, a domain suffix, mean token length, and 9 binary merchant-cue groups (legal suffix, payroll, transfer, fee, housing, store, food, travel, subscription). Scaled with `MaxAbsScaler` so they are non-negative (required by MultinomialNB).
- Total: 38,914 features (v1: 14,438). 13 unit tests in `tests/test_classical_features.py`.

**Feature ablation** (Linear SVM, C=1, balanced class weights, everything else fixed; `reports/classical_v2/feature_ablation.csv`):

| Feature set | Features | Val F1 | Test F1 | OOD F1 | OOD change vs v1 |
|---|---|---|---|---|---|
| word (v1) | 14,438 | 0.9929 | 0.9936 | 0.8364 | – |
| char | 24,457 | 0.9938 | 0.9950 | 0.8189 | −1.75 pts |
| word + char | 38,895 | 0.9956 | 0.9955 | 0.8403 | +0.39 pts |
| **word + char + hand (v2)** | 38,914 | 0.9958 | 0.9954 | **0.8519** | **+1.55 pts** |
| word + hand | 14,457 | 0.9941 | 0.9951 | 0.8290 | −0.74 pts |
| hand only | 19 | 0.3250 | 0.3273 | 0.2595 | −57.7 pts |

**Reading:** for Linear SVM no single new feature type helps on its own (char alone and word + hand are below v1); only the full combination beats v1, by 1.55 points. This is one run on OOD rows that come in merchant clusters, so differences under about a point should not be over-interpreted. Hand-engineered features alone carry weak signal (0.26 against ~0.08 for chance), which is expected since they describe the shape of a string and not what the merchant is. Character n-grams raise the in-distribution score but lower the OOD score when used alone; a possible explanation (untested) is that they memorize merchant-specific fragments.

**Tuning protocol** (`src/models/train_classical_v2.py`), identical for all five models: seed 42; 8 random candidates per model (the whole grid when it is smaller: Logistic Regression 6, Linear SVM 6; Naive Bayes 8 of 10, Random Forest 8 of 36, XGBoost 8 of 72); each candidate is fit on a stratified 40% of the training set (22,391 rows) and scored by macro F1 on the **full validation set**; the best candidate is refit on the full training set. Test and OOD data are never used for selection. Grids: Logistic Regression C ∈ {0.1, 0.3, 1, 3, 10, 30}; Naive Bayes alpha ∈ {0.01, 0.03, 0.1, 0.3, 1.0} × fit_prior; Linear SVM C ∈ {0.03 … 10}; Random Forest n_estimators {200, 400}, max_depth {None, 40, 80}, min_samples_leaf {1, 2, 4}, max_features {sqrt, 0.01}; XGBoost n_estimators {150, 300}, max_depth {4, 6, 8}, learning_rate {0.05, 0.1, 0.2}, subsample {0.8, 1.0}, colsample_bytree {0.3, 0.6}. Class imbalance: balanced class weights (Random Forest: balanced_subsample); **XGBoost now uses balanced sample weights, which v1 XGBoost did not**; Naive Bayes has no such option.

**Selected settings:** Logistic Regression C=10; Naive Bayes alpha=0.01, fit_prior=True; Linear SVM C=3; Random Forest 400 trees, max_depth 80, min_samples_leaf 1, max_features 0.01; XGBoost 300 trees, max_depth 4, learning_rate 0.2, colsample_bytree 0.6, subsample 0.8. Full candidate scores are in `reports/classical_v2/tuning_log.csv`.

**Results** (`reports/classical_v2/classical_v2_results.csv`, `v1_vs_v2_comparison.csv`; per-category OOD scores in `per_category_ood.csv`):

| Model | Val F1 | Test F1 | **OOD F1** | v1 OOD F1 | Change |
|---|---|---|---|---|---|
| **Logistic Regression** | 0.9956 | 0.9957 | **0.8811** | 0.7982 | +8.29 pts |
| Linear SVM | 0.9957 | 0.9953 | 0.8514 | 0.8364 | +1.50 pts |
| Naive Bayes | 0.9813 | 0.9825 | 0.8365 | 0.8240 | +1.25 pts |
| XGBoost | 0.9936 | 0.9936 | 0.8191 | 0.7985 | +2.06 pts |
| Random Forest | 0.9570 | 0.9547 | 0.7450 | 0.7374 | +0.76 pts |

**Best classical baseline is now Logistic Regression, 88.11% OOD macro F1** (v1: Linear SVM, 83.64%). All five models improved on OOD; Random Forest remains the weakest.

**Did tuning or features drive the Logistic Regression gain?** Logistic Regression, OOD macro F1, `reports/classical_v2/lr_tuning_vs_features.csv`:

| Features | C = 1 | C = 10 | Effect of tuning (C 1→10) |
|---|---|---|---|
| v1 (word) | 0.7982 | 0.8311 | +3.3 pts |
| v2 (word + char + hand) | 0.8692 | **0.8811** | +1.2 pts |
| **Effect of the new features** | **+7.1 pts** | **+5.0 pts** | |

Both changes mattered and their effects overlap, so they should not be added up: of the +8.3-point total, roughly 5 points come from the features and 3 from tuning. The benefit of the extended feature set is model-dependent: about +1.5 points for Linear SVM (ablation above) against +5 to +7 for Logistic Regression.

**Caveats to state in the report:**
- The validation set is in-distribution and already saturated (~0.99 macro F1), so candidate scores differ little and the selection signal is weak. Candidates were also scored after fitting on 40% of the training data, which may favor stronger regularization than the full-data optimum.
- Several selected settings sit at the edge of their grid (Random Forest: most trees, deepest, smallest leaves, larger feature share; XGBoost: most trees, highest learning rate), so wider grids might score higher. Not pursued, mainly because of cost.
- XGBoost's search and final fit ran on a Colab T4 GPU (`device="cuda"`) because one full-data CPU fit with this feature set took about 32 minutes; GPU results can differ slightly from CPU results. The other four models ran on CPU.
- Single seed and no confidence intervals; OOD rows are clustered by merchant, so differences under about a point are not reliable.
- The transformers were trained with default hyperparameters (no search), so the classical-vs-transformer comparison is, if anything, conservative against the transformers.

**Done since this section was first written:** ECE / reliability diagrams for the v2 models (§9), inference cost (§11) and learning curves (§12). **Still open for the classical models:** feature-importance analysis (exposé, error-analysis phase).

### 5.2 Side observation — XGBoost with default settings scored higher OOD than the tuned model

While building the learning curves (§12), the XGBoost curve was first run with the wrong settings: the script tried to recover the tuned hyperparameters from the saved `.ubj` file, but a model loaded from `.ubj` keeps its trees, not its training settings, so XGBoost's *defaults* were used (300 rounds, max_depth 6, learning_rate 0.3, subsample 1.0, colsample_bytree 1.0, balanced sample weights, CPU). This was a bug, caught because the 100% point (0.8472) disagreed with the reported tuned result (0.8191). The script now reads the tuned settings from `classical_v2_models.csv`. The unplanned run is kept as evidence:

| Training data | 5% | 10% | 25% | 50% | 100% |
|---|---|---|---|---|---|
| XGBoost, default settings (OOD macro F1) | 0.6186 | 0.6934 | 0.7801 | 0.8214 | **0.8472** |
| XGBoost, tuned (CPU re-run) | 0.6244 | 0.7108 | 0.7847 | 0.8172 | 0.8260 |

At full data the untuned model is about **2.1 points above the tuned CPU run (0.8260) and 2.8 above the tuned GPU run (0.8191, the headline)**. The default-settings model is level with or slightly below the tuned one up to 25% (0.5 to 1.7 points lower) and pulls ahead only at 50% (+0.4) and 100% (+2.1). **Interpretation (untested):** the tuning protocol selects on a validation set that is in-distribution and saturated (~0.99), so it cannot tell settings that generalise to unseen merchants from those that do not; the selected max_depth 4 may be too shallow for OOD, while depth 6 was in the grid but learning_rate 0.3 was not. This is one run per setting with a single seed, so it is an observation to report with that caveat, not an established effect. It also suggests the headline classical numbers may be pessimistic or optimistic depending on how each model's selected settings generalise. The same CPU-vs-GPU difference (0.8260 vs 0.8191 for identical tuned settings) shows how much run-to-run and hardware variation to expect.

---

## 6. Transformer Approach 1 — Multilingual Sentence Embeddings + Lightweight Classifier

**Model:** `paraphrase-multilingual-MiniLM-L12-v2` (Hugging Face / sentence-transformers), ~470MB, downloaded once and cached locally thereafter (no repeat internet dependency after first run).

**Why this model specifically:** pretrained on large-scale multilingual web text, so it likely already encodes real-world associations (e.g. "ALDI," "REWE" are supermarket chains) purely from pretraining — not from anything in this project's entirely-English training data. This is the specific mechanism that could let it generalize to unseen and non-English merchants in a way the classical TF-IDF models structurally cannot, since TF-IDF has zero information about any token it never saw in training.

**Setup:** the sentence-transformer itself is kept **frozen** (not fine-tuned) — only a Logistic Regression classifier is trained on top of its output embeddings. This is deliberately the "cheap" transformer approach; the second transformer approach (fine-tuned DistilBERT) updates the transformer's own weights.

**Embeddings are cached to `data/embeddings/*.npy`** after first computation, so re-running with a different classifier on top (e.g. a shallow MLP instead of Logistic Regression, as the proposal allows) doesn't require re-encoding all ~96k transaction strings.

**Result:**

| Split | Accuracy | Macro F1 |
|---|---|---|
| Val | 0.8663 | 0.8637 |
| Test (in-distribution) | 0.8645 | 0.8626 |
| Test (OOD) | 0.6701 | 0.6623 |

Embedding dimension: 384. Encoding cost: ~16 minutes total for ~96k transaction strings on CPU (538s train + 149s val + 114s test + 143s OOD), plus a one-time 470MB model download — a real, worth-recording inference-cost data point given the expose's cost-tracking requirement.

**Honest finding: this approach underperforms every one of the five classical baselines, on both in-distribution and OOD metrics, and does NOT show the hypothesized robustness benefit.**

| Model | In-dist Macro F1 | OOD Macro F1 | Relative drop |
|---|---|---|---|
| Linear SVM (best classical) | 0.9936 | 0.8364 | 15.8% |
| Random Forest (worst classical) | 0.9761 | 0.7374 | 24.4% |
| Multilingual embedding + LogReg | 0.8626 | 0.6623 | 23.2% |

Not only is the embedding approach's absolute OOD score (0.6623) below every classical model's OOD score (worst classical was 0.7374) — its *relative* degradation from in-distribution to OOD (23.2%) is worse than Linear SVM's (15.8%), nearly matching the worst classical model. The central hypothesis's expected mechanism (pretrained multilingual world knowledge → better unseen-merchant generalization) is **not supported by this specific approach** as implemented.

**Working explanation:** `paraphrase-multilingual-MiniLM-L12-v2` was pretrained to embed natural sentences for semantic similarity (paraphrase/translation matching) — not cryptic abbreviated bank codes like `REWE-SAG-7342`. Classical TF-IDF directly exploits exact surface tokens and store-code patterns, which appears to carry most of the real classification signal in this task. A frozen, general-purpose sentence embedding compresses that surface detail away into a semantic space built for different text, and a simple Logistic Regression head has no mechanism to recover it.

**Notable failure mode, different from the classical models:** Groceries and Travel show very poor *precision* here (0.38, 0.40) despite high recall — the model over-triggers these categories on unrelated text. Plausible mechanism: the embedding space may cluster loosely-related concepts (anything place-like → Travel, anything market-like → Groceries) even without literal token evidence — a form of semantic over-generalization the classical token-based models don't exhibit.

**Implication for next step:** this result should NOT be read as "transformers don't help" — it's specifically a *frozen-embedding* result. The remaining transformer approach (fine-tuned DistilBERT) is mechanically different: it updates the model's own weights to this specific noisy short-text task, rather than relying on unchanged generic pretrained embeddings. This negative result sharpens the hypothesis into: does *fine-tuning* (as opposed to frozen embeddings) close the gap.

**Still worth doing regardless of the numbers above:** the planned live sanity check with real German merchant strings (ALDI SÜD, REWE, dm) — the mechanism being tested there (pretrained world-knowledge transfer) is somewhat orthogonal to overall task accuracy, and is worth checking qualitatively even though this model isn't the strongest numeric performer.

---

## 7. Transformer Approach 2 — Fine-Tuned DistilBERT

**Model:** `distilbert-base-uncased`, fine-tuned end-to-end (unlike Approach 1, the transformer's own weights are updated via backpropagation, not frozen). Class-weighted cross-entropy loss (same `balanced` rationale as the classical models). Trained on Google Colab's free T4 GPU after CPU training proved impractically slow (~4-5+ hour estimate for the full dataset on CPU vs. minutes on GPU) — matches the proposal's stated plan to use free-tier cloud GPU for larger experiments.

**Epoch-count ablation** (early-stopping ceiling set to 6, `patience=2`; never triggered early — validation metric kept climbing, with one minor dip at epoch 2):

| | 3 epochs | 6 epochs |
|---|---|---|
| Training time (GPU) | 9.9 min | 23.2 min |
| Val Macro F1 | 0.9932 | 0.9959 |
| Test (in-dist) Macro F1 | 0.9961 | 0.9957 |
| **Test OOD Macro F1** | 0.9283 | **0.9329** |

**Finding:** more training epochs bought a marginal +0.46 point OOD improvement for 2.3x the training cost — a clear diminishing-returns pattern, consistent with standard transformer fine-tuning practice (2-4 epochs is the typical recommended range; see BERT paper). The improvement also was NOT uniform across categories — Groceries and Restaurants improved substantially (Groceries F1: 0.89→0.97, driven by precision jumping 0.79→0.96), while Education recall (1.00→0.91) and Other precision (0.74→0.70) both got slightly worse. This shows additional training reshapes decision boundaries rather than uniformly improving everything — worth stating explicitly rather than implying "more epochs = strictly better."

**Final adopted result (6-epoch model):**

| Split | Accuracy | Macro F1 |
|---|---|---|
| Val | 0.9948 | 0.9959 |
| Test (in-distribution) | 0.9945 | 0.9957 |
| **Test (OOD)** | **0.9169** | **0.9329** |

**Recurring weak spot, consistent across every model tried so far:** Income recall on OOD is 0.70 — the lowest recall of any category, and this exact pattern (Income being hard on unseen data) has now shown up identically in every classical model AND both transformer approaches. This is a genuine, robust finding about the task itself (unseen employer names carry little generalizable signal), not a quirk of any one model — worth a dedicated paragraph in the error analysis chapter.

### 7.1 Transformer Approach 4 (bonus) — Fine-Tuned Multilingual MiniLM

**Model:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` — exactly the base model of Approach 1 (§6), loaded as `BertForSequenceClassification` with a freshly initialized classification head (the "classifier.weight / classifier.bias MISSING" load report is expected) and fine-tuned end to end.

**Purpose:** isolate one variable. Approach 1 used this model frozen and performed worst of all models; comparing it with the same model fine-tuned separates "frozen vs. fine-tuned" from "architecture A vs. architecture B", which a DistilBERT comparison alone cannot do.

**Setup:** same as the DistilBERT runs — class-weighted cross-entropy, same splits, ceiling of 6 epochs with early stopping (patience 2), batch size 16, max length 32, Colab T4 GPU. Early stopping did not trigger; all 6 epochs ran (27.8 minutes). Validation macro F1 by epoch: 0.9885, 0.9915, 0.9941, 0.9938, 0.9948, 0.9957 — still improving at the ceiling.

| Split | Accuracy | Macro F1 |
|---|---|---|
| Val | 0.9948 | 0.9957 |
| Test (in-distribution) | 0.9927 | 0.9942 |
| **Test (OOD)** | **0.8550** | **0.8767** |

**Findings:**
- **Fine-tuning, not the architecture, explains MiniLM's earlier weak result:** the same model goes from 0.6623 OOD macro F1 frozen to 0.8767 fine-tuned (+21.4 points).
- **It does not beat the best tuned classical model:** 0.8767 against 0.8811 for Logistic Regression with v2 features (§5.1), a difference within noise. It is 5.6 points below fine-tuned DistilBERT, but the two differ in vocabulary (30k English uncased vs. 250k multilingual), size and pretraining, so the gap cannot be attributed to one cause. Multilingual pretraining did not help on English unseen merchants.
- **The same weak spots as every other model on OOD:** Income has the lowest recall (0.65); Healthcare (0.74), Entertainment (0.78) and Shopping (0.79) recall are also low; Other (0.68), Shopping (0.69) and Groceries (0.72) have the lowest precision, consistent with the Income→Other and Shopping confusions in §10.
- **Limitation:** validation F1 was still rising at epoch 6, so the epoch ceiling, not early stopping, ended training. Hyperparameters were not searched.

---

## 8. Full Model Comparison (All Approaches, To Date)

Sorted by OOD Macro F1 — the metric that actually tests the central hypothesis:

Classical models are the tuned **v2** versions (§5.1). Relative drop = (in-distribution test F1 − OOD F1) / in-distribution test F1. Training costs are approximate; inference cost is in §11, calibration in §9 and learning curves in §12.

| Model | In-dist Macro F1 | **OOD Macro F1** | Relative drop | Training cost |
|---|---|---|---|---|
| **Fine-tuned DistilBERT (6 epochs)** | 0.9957 | **0.9329** | **6.3%** | 23.2 min (T4 GPU) |
| Fine-tuned DistilBERT (3 epochs) | 0.9961 | 0.9283 | 6.8% | 9.9 min (T4 GPU) |
| Logistic Regression (v2, tuned) | 0.9957 | 0.8811 | 11.5% | ~20 s per fit (CPU) |
| Fine-tuned MiniLM (6 epochs) | 0.9942 | 0.8767 | 11.8% | 27.8 min (T4 GPU) |
| Linear SVM (v2, tuned) | 0.9953 | 0.8514 | 14.5% | seconds to ~1 min (CPU) |
| Multinomial NB (v2, tuned) | 0.9825 | 0.8365 | 14.9% | seconds (CPU) |
| XGBoost (v2, tuned) | 0.9936 | 0.8191 | 17.6% | ~6 min for the full search on a T4 GPU; ~32 min per full-data fit on CPU |
| Random Forest (v2, tuned) | 0.9547 | 0.7450 | 22.0% | seconds to ~1 min per fit (CPU) |
| Frozen multilingual MiniLM + LogReg | 0.8626 | 0.6623 | 23.2% | ~16 min encoding (CPU) |

For reference, the untuned v1 baselines (§5) scored 0.8364 (Linear SVM), 0.8240 (Naive Bayes), 0.7985 (XGBoost), 0.7982 (Logistic Regression) and 0.7374 (Random Forest) OOD macro F1.

**Central hypothesis: partly supported, with several nuances.** The exposé's hypothesis is that transformer models improve accuracy and calibration over classical models, with the largest gains on unseen merchants and low-resource categories. So far:
- **Accuracy on unseen merchants:** fine-tuned DistilBERT beats the best tuned classical model (Logistic Regression, v2) by 5.2 points and shows the smallest in-distribution-to-OOD drop (6.3%, against 11.5-22.0% for the classical models). Fine-tuned MiniLM does **not** beat it (87.7% vs 88.1%, within noise), although its relative drop (11.8%) is similar to the best classical model's. Under a merchant-clustered bootstrap the DistilBERT advantage holds (+5.2 points over Logistic Regression, 95% CI +2.2 to +8.6), while fine-tuned MiniLM and Logistic Regression are statistically indistinguishable (−0.4 points, CI −3.4 to +2.9) and the ordering among the tuned classical models is not established (§10.5).
- **Fine-tuning, not the architecture:** the frozen MiniLM embedding underperformed every classical model (66.2%), while the same model fine-tuned reaches 87.7% (+21.4 points). "Transformers" as a category did not automatically win; adapting a pretrained transformer to this noisy short-text task is what helped.
- **Fair baselines matter:** against the untuned v1 baselines the gap to DistilBERT looked like 9.6 points; with the exposé's character n-grams, hand-engineered features and one tuning protocol it is 5.2 points. The report should present the v2 numbers as the main comparison and v1 as a reference. The transformers were not tuned, so this comparison is conservative against them.
- **Calibration: not supported.** On unseen merchants the transformers are *not* better calibrated (OOD ECE: XGBoost 0.032, Logistic Regression 0.038, DistilBERT 0.078, Linear SVM 0.074 after calibration, fine-tuned MiniLM 0.133). DistilBERT and MiniLM are near-certain about almost everything: 91% and 89% of their wrong OOD predictions come with over 90% confidence, against 24% for Logistic Regression and 13% for XGBoost (§9). For a user-facing tool this means a confidence threshold would work with the classical models and not with the transformers.
- **Data efficiency: supported for the fine-tuned transformers.** With 5% of the training data DistilBERT scores 0.848 OOD macro F1 and fine-tuned MiniLM 0.762, against 0.752 for the best classical model at that size. Logistic Regression needs roughly 14,000 rows (five times as much) to match DistilBERT's 5% score, and DistilBERT with 25% of the data beats every classical model trained on 100% (§12). The frozen embedding is the worst model at every size.
- **Low-resource categories: only weakly supported.** At the smallest data size, small categories score about 5 points lower than large ones for 7 of 8 models, but the gap reverses by full data and category size does not predict difficulty; the medium group, which contains Income and Other, is hardest, largely because of one unseen Income template (Zelle payments, §10.5) (§12).
- **Cost:** DistilBERT is about 20 times slower per transaction than Logistic Regression (62 ms against 3 ms median on CPU) for +5.2 points of OOD macro F1; fine-tuned MiniLM is slower than Logistic Regression and not better (§11).

### 8.1 Full Metrics for All Eight Models (accuracy, macro precision/recall/F1, per-category F1)

**Source:** `src/eval/results_table.py` computes everything from the cached predictions (`reports/calibration/probs/`); outputs in `reports/results/` (`results_summary.csv`, `per_category_metrics.csv`, `per_category_f1_test.csv`, `per_category_f1_ood.csv`, `results_tables.md`). All eight OOD macro-F1 values reproduce the numbers above (consistency check built into the script). **One difference from the table above:** the Linear SVM row here uses its Platt-calibrated predictions (OOD macro F1 0.8543, as in §9, §10 and the bootstrap); the raw model scores 0.8514. Macro averages weight the 13 categories equally; class sizes differ between the two test sets (for example Income 951 in-distribution against 1,661 OOD).

| Model | ID acc | ID macro F1 | OOD acc | OOD macro P | OOD macro R | OOD macro F1 | OOD weighted F1 | Macro F1 drop ID to OOD |
|---|---|---|---|---|---|---|---|---|
| Fine-tuned DistilBERT | 0.994 | 0.996 | 0.917 | 0.942 | 0.934 | 0.933 | 0.918 | 0.063 |
| Logistic Regression v2 | 0.994 | 0.996 | 0.854 | 0.881 | 0.896 | 0.881 | 0.849 | 0.115 |
| Fine-tuned MiniLM | 0.993 | 0.994 | 0.855 | 0.889 | 0.882 | 0.877 | 0.856 | 0.118 |
| Linear SVM v2 | 0.994 | 0.995 | 0.833 | 0.875 | 0.860 | 0.854 | 0.832 | 0.141 |
| Naive Bayes v2 | 0.980 | 0.983 | 0.815 | 0.844 | 0.848 | 0.836 | 0.815 | 0.146 |
| XGBoost v2 | 0.992 | 0.994 | 0.780 | 0.836 | 0.813 | 0.819 | 0.783 | 0.175 |
| Random Forest v2 | 0.949 | 0.955 | 0.728 | 0.775 | 0.740 | 0.745 | 0.726 | 0.210 |
| Frozen MiniLM + LogReg | 0.864 | 0.863 | 0.670 | 0.661 | 0.704 | 0.662 | 0.669 | 0.200 |

**Per-category F1, OOD test (unseen merchants):**

| Category | n | DistilBERT | LR v2 | MiniLM | SVM v2 | NB v2 | XGB v2 | RF v2 | Frozen |
|---|---|---|---|---|---|---|---|---|---|
| Education | 535 | 0.953 | 1.000 | 0.945 | 0.909 | 0.890 | 0.928 | 0.746 | 0.633 |
| Entertainment | 2158 | 0.967 | 0.819 | 0.844 | 0.805 | 0.778 | 0.754 | 0.586 | 0.722 |
| Groceries | 432 | 0.975 | 0.889 | 0.836 | 0.893 | 0.890 | 0.969 | 0.699 | 0.532 |
| Healthcare | 940 | 0.966 | 0.922 | 0.840 | 0.885 | 0.864 | 0.782 | 0.746 | 0.730 |
| Housing | 1192 | 1.000 | 1.000 | 0.995 | 1.000 | 0.986 | 1.000 | 0.984 | 0.835 |
| Income | 1661 | 0.821 | 0.679 | 0.787 | 0.693 | 0.708 | 0.738 | 0.935 | 0.639 |
| Insurance | 535 | 0.994 | 0.987 | 0.958 | 0.944 | 0.931 | 0.973 | 0.932 | 0.665 |
| Other | 1557 | 0.814 | 0.826 | 0.787 | 0.733 | 0.679 | 0.654 | 0.798 | 0.670 |
| Restaurants | 1400 | 0.974 | 0.849 | 0.903 | 0.902 | 0.876 | 0.757 | 0.687 | 0.670 |
| Shopping | 2082 | 0.853 | 0.757 | 0.737 | 0.753 | 0.772 | 0.625 | 0.519 | 0.438 |
| Transportation | 1377 | 0.841 | 0.860 | 0.862 | 0.830 | 0.823 | 0.747 | 0.576 | 0.688 |
| Travel | 597 | 0.983 | 0.975 | 0.975 | 0.847 | 0.807 | 0.794 | 0.672 | 0.531 |
| Utilities | 1565 | 0.987 | 0.891 | 0.928 | 0.910 | 0.871 | 0.926 | 0.806 | 0.855 |

**Per-category F1, in-distribution test:**

| Category | n | DistilBERT | LR v2 | MiniLM | SVM v2 | NB v2 | XGB v2 | RF v2 | Frozen |
|---|---|---|---|---|---|---|---|---|---|
| Education | 520 | 1.000 | 1.000 | 0.999 | 1.000 | 0.991 | 0.998 | 0.983 | 0.833 |
| Entertainment | 1477 | 1.000 | 0.999 | 0.997 | 1.000 | 0.975 | 0.992 | 0.908 | 0.820 |
| Groceries | 535 | 1.000 | 1.000 | 0.995 | 0.998 | 0.985 | 0.996 | 0.914 | 0.746 |
| Healthcare | 1059 | 0.979 | 0.981 | 0.975 | 0.981 | 0.972 | 0.979 | 0.951 | 0.923 |
| Housing | 1021 | 1.000 | 1.000 | 1.000 | 1.000 | 0.999 | 1.000 | 0.998 | 0.972 |
| Income | 951 | 1.000 | 1.000 | 1.000 | 1.000 | 0.994 | 1.000 | 0.998 | 0.958 |
| Insurance | 519 | 1.000 | 1.000 | 0.999 | 1.000 | 0.984 | 0.999 | 0.985 | 0.856 |
| Other | 967 | 0.999 | 1.000 | 1.000 | 1.000 | 0.997 | 1.000 | 0.998 | 0.949 |
| Restaurants | 990 | 0.993 | 0.992 | 0.989 | 0.991 | 0.979 | 0.991 | 0.929 | 0.844 |
| Shopping | 1488 | 0.981 | 0.980 | 0.975 | 0.981 | 0.944 | 0.973 | 0.915 | 0.776 |
| Transportation | 994 | 0.996 | 0.996 | 0.997 | 0.995 | 0.978 | 0.995 | 0.888 | 0.797 |
| Travel | 510 | 1.000 | 1.000 | 1.000 | 1.000 | 0.979 | 0.997 | 0.965 | 0.834 |
| Utilities | 965 | 0.996 | 0.996 | 0.997 | 0.995 | 0.995 | 0.996 | 0.978 | 0.907 |

**Findings:**
- **The in-distribution test set does not discriminate between the best models.** Five models score 0.992 to 0.994 accuracy and 0.994 to 0.996 macro F1 (DistilBERT, Logistic Regression, MiniLM, Linear SVM, XGBoost); differences are in the third decimal and were not tested. Only Naive Bayes (0.983), Random Forest (0.955) and the frozen embedding (0.863) are visibly lower. The comparison is decided on the OOD set.
- **Absolute macro-F1 drop to OOD** (the table's last column, an absolute difference; §8 reports the relative drop): 0.063 for DistilBERT, 0.115 to 0.146 for Logistic Regression, MiniLM, Linear SVM and Naive Bayes, 0.175 for XGBoost, and 0.200 to 0.210 for the frozen embedding and Random Forest.
- **Other is the lowest-precision class** for DistilBERT (0.698), Logistic Regression (0.713), MiniLM (0.679), Linear SVM (0.586) and Naive Bayes (0.545), and by a hair for XGBoost (0.605 against 0.606 for Shopping), while its recall is high (0.90 to 0.98 for the first five). Many errors end up in Other, consistent with the Zelle and wrapper findings in §10.5 (unseen templates default to Other).
- **DistilBERT's weak OOD classes are exactly the confused pairs.** Its OOD F1 is below 0.86 only for Other (0.814), Income (0.821), Transportation (0.841) and Shopping (0.853) (the Income→Other and Shopping→Transportation pairs of §10); every other class is at least 0.953.
- **DistilBERT is best or tied in 9 of 13 OOD categories and not best in 4:** Education (Logistic Regression 1.000), Income (Random Forest 0.935), Other (Logistic Regression 0.826) and Transportation (MiniLM 0.862, Logistic Regression 0.860). Its advantage is therefore broad but not universal.
- **Random Forest** has the best Income F1 of any model (0.935) and the worst or near-worst F1 in most other categories (for example Shopping 0.519, Transportation 0.576), matching §10.6.
- **Hardest in-distribution classes:** for the five best models the two lowest in-distribution F1 values are Healthcare and Shopping (0.973 to 0.981; every other class is at least 0.989). Why was not examined (possibly overlapping category definitions).

---

## 9. Confidence Calibration (Expected Calibration Error)

**Status: complete for all eight models** (`src/eval/calibration_all_models.py`; outputs in `reports/calibration/`: `ece_summary.csv`, `ece_bins.csv`, `reliability_ood.png`, `reliability_test.png`; cached probabilities in `reports/calibration/probs/`, not committed). This replaces the earlier 3-model preliminary run, which used the untuned v1 Logistic Regression (OOD ECE 0.1202, underconfident) and is superseded.

**Method.** ECE = sample-weighted average gap between stated confidence (the highest class probability) and actual accuracy over 10 equal-width confidence bins. Computed on both the in-distribution test set (11,996 rows) and the OOD test set (16,031 rows). Models: the five tuned v2 classical models (native probabilities; Random Forest and XGBoost use their predicted probabilities), the frozen embedding + Logistic Regression, fine-tuned DistilBERT and fine-tuned MiniLM (softmax over logits). **Linear SVM has no probabilities**, so it is wrapped in a sigmoid (Platt) calibrator fitted on the **validation** split with the SVM itself frozen (`CalibratedClassifierCV` with `FrozenEstimator`): no retraining, and test and OOD are untouched. It is the only model that received an explicit calibration step, so its ECE is not like-for-like with the others. XGBoost was loaded from its native `.ubj` file because the GPU-trained pickle could not be read on the CPU machine (same model).

**Results** (`ece_summary.csv`; "wrong >0.9" = share of wrong predictions made with over 90% confidence):

| Model | ECE, in-dist test | **ECE, OOD** | OOD accuracy | OOD mean confidence | OOD mean confidence when wrong | OOD wrong >0.9 |
|---|---|---|---|---|---|---|
| XGBoost v2 | 0.0403 | **0.0318** | 0.7805 | 0.804 | 0.566 | 13.4% |
| Logistic Regression v2 | 0.0134 | **0.0378** | 0.8535 | 0.870 | 0.622 | 24.2% |
| Linear SVM v2 (Platt on val) | 0.0045 | 0.0736 | 0.8331 | 0.907 | 0.711 | 32.0% |
| Fine-tuned DistilBERT | 0.0015 | 0.0778 | 0.9169 | 0.995 | 0.971 | **91.3%** |
| Frozen MiniLM + LogReg | 0.0410 | 0.0986 | 0.6701 | 0.764 | 0.608 | 13.8% |
| Fine-tuned MiniLM | 0.0043 | 0.1333 | 0.8550 | 0.988 | 0.960 | **89.1%** |
| Naive Bayes v2 | 0.0138 | 0.1509 | 0.8150 | 0.966 | 0.875 | 63.7% |
| Random Forest v2 | 0.1550 | 0.1800 | 0.7275 | 0.548 | 0.340 | 0.0% |

**Reading the table:**
- **Every model except XGBoost is better calibrated in-distribution than on unseen merchants**, and the OOD penalty is largest for the models that are most confident: fine-tuned MiniLM (+0.129), Naive Bayes (+0.137), Linear SVM (+0.069), DistilBERT (+0.076). In-distribution ECE for DistilBERT (0.0015) and MiniLM (0.0043) is excellent; the failure appears only on unseen merchants.
- **The transformers are almost always maximally confident.** 15,779 of DistilBERT's 16,031 OOD predictions (98.4%) fall in the 0.9-1.0 bin, which states 99.9% and delivers 92.3% accuracy; for MiniLM 15,486 (96.6%) fall there, stating 99.8% and delivering 86.6%. Their few predictions at lower confidence are worse calibrated still (DistilBERT's 0.8-0.9 bin states 86.3% and delivers 57.8%, on only 45 rows), but they are too rare to move the aggregate ECE. DistilBERT's low aggregate ECE is therefore largely a distributional artifact, not evidence of well-calibrated confidence across its range.
- **For the transformers, confidence says almost nothing about which predictions are wrong.** Their wrong predictions carry a mean confidence of 0.97 and 0.96, and 91% and 89% of them are above 0.9. A "send low-confidence predictions for review" rule would catch almost none of their errors. For Logistic Regression (24%) and XGBoost (13%), errors do sit at lower confidence, so such a rule would work. This matches the informal German-merchant probe (§4.1), where the model was 80-100% confident while wrong.
- **Logistic Regression v2 is the best-calibrated accurate model** (OOD ECE 0.038); it is no longer underconfident as the untuned v1 model was. **XGBoost's OOD ECE (0.032) is lower than its in-distribution ECE (0.040), most likely a coincidence**: it is underconfident in-distribution (confidence 0.95 against accuracy 0.99), and on OOD the accuracy drop happens to land near its confidence. It should not be read as "more reliable on unseen merchants".
- **Random Forest is underconfident everywhere** (mean confidence 0.55 against accuracy 0.73 OOD; ECE 0.155 even in-distribution): its averaged tree votes give probabilities that are systematically too low.
- **Naive Bayes is badly overconfident** (OOD mean confidence 0.966 against accuracy 0.815), the usual behaviour of its independence assumption.

**Caveats to state in the report:**
- The Linear SVM's calibrator was fitted on the in-distribution validation split, so its OOD ECE reflects a calibrator that never saw unseen merchants, and it is the only model with an explicit calibration step.
- With 10 equal-width bins and most predictions in the top bin for the confident models, ECE is dominated by that single bin; the "mean confidence when wrong" and "wrong >0.9" columns describe the overconfidence more directly.
- Single run of each model; the OOD rows are clustered by merchant.
- The calibration of fine-tuned models depends on training settings that were not tuned (default learning rate, 6-epoch ceiling with early stopping).

**Optional extensions, not done:** temperature scaling of the transformers fitted on the validation split (expected to fix in-distribution calibration but, since validation is in-distribution, not necessarily OOD), and risk-coverage curves (OOD accuracy if each model may abstain on its lowest-confidence predictions), which would show the practical consequence of the finding above.

---

## 10. Error Analysis — Confusion Matrices (run complete)

**Status: executed.** `confusion_matrix_analysis.py` ran cleanly against all four models and the full OOD set (16,031 rows), with no errors. Outputs saved: four row-normalized confusion-matrix heatmaps (`reports/confusion_matrices/confusion_matrix_<model>.png`) and the combined top-10-confused-pairs table (`reports/confusion_matrices/top_confused_pairs_all_models.csv`).

**Scope:** evaluated on the OOD test set specifically (16,031 rows) rather than in-distribution test — this is deliberately where the interesting failures are, since every model scores >0.97 in-distribution and the real discriminating behavior shows up only on unseen merchants.

**Models covered:** Linear SVM (best **v1** classical baseline), frozen multilingual embedding + LogReg, fine-tuned DistilBERT, fine-tuned MiniLM. The classical model analyzed here is the untuned v1 Linear SVM; the v2 models (best classical model is now Logistic Regression, §5.1) have not been run through this analysis yet. The top confused pairs for the v2 models are in §10.5; row-normalised heatmaps and per-class recall for all eight models are in §10.6 (`reports/confusion_matrices/all_models/`).

### 10.1 Results — Top Confused Pairs per Model

| Rank | Linear SVM | Frozen Embedding | Fine-tuned DistilBERT | Fine-tuned MiniLM |
|---|---|---|---|---|
| 1 | Income→Other (490) | Income→Other (509) | Income→Other (491) | Income→Other (490) |
| 2 | Shopping→Transportation (269) | Shopping→Restaurants (385) | Shopping→Transportation (313) | Entertainment→Shopping (243) |
| 3 | Entertainment→Shopping (202) | Shopping→Entertainment (256) | Transportation→Shopping (82) | Healthcare→Shopping (206) |
| 4 | Income→Transportation (174) | Entertainment→Travel (173) | Shopping→Other (69) | Shopping→Transportation (157) |
| 5 | Entertainment→Transportation (153) | Shopping→Groceries (151) | Education→Other (48) | Entertainment→Transportation (142) |
| 6 | Travel→Other (134) | Entertainment→Transportation (151) | Healthcare→Shopping (38) | Utilities→Entertainment (141) |
| 7 | Other→Education (127) | Income→Travel (148) | Entertainment→Transportation (36) | Restaurants→Shopping (102) |
| 8 | Entertainment→Other (118) | Shopping→Travel (142) | Entertainment→Restaurants (32) | Shopping→Restaurants (99) |
| 9 | Shopping→Other (106) | Housing→Other (129) | Other→Entertainment (28) | Other→Shopping (83) |
| 10 | Restaurants→Transportation (95) | Shopping→Insurance (125) | Restaurants→Transportation (22) | Income→Shopping (80) |

### 10.2 Key Finding — Income→Other Is One Unseen Template (revised after §10.5)

**Observation:** Income→Other is the most confused pair for Linear SVM, Logistic Regression, Naive Bayes, the frozen embedding, fine-tuned DistilBERT and fine-tuned MiniLM, with nearly identical counts (490, 490, 490, 509, 491, 490). Four architecturally very different models converging on the same number of errors on the same held-out set is what first suggested a data problem.

**Original interpretation (not supported):** that borderline transfers were labelled inconsistently in the source data. Reading the errors (§10.5) shows otherwise: every one of these errors is a string of the form `[credit] Zelle payment from <NAME> <ID>`; all 490 such rows carry the label Income; **no training row contains "Zelle"** (the merchant was held out by the OOD split); and DistilBERT, MiniLM and Logistic Regression each predict Other for all 490 of them. So the labels are consistent and the models fail on a transaction type that is absent from training, where the correct category depends on a convention (a received person-to-person payment counts as Income) that the models do not apply to this string form (they do apply it to Venmo and Cash App credits, see below). The remaining 152 OOD rows that contain "Zelle" are labelled Other and are predicted Other by the same models, so the models treat the whole Zelle family alike.

**Consequences:** (i) Zelle alone caps Income recall at about 0.70 (490 of 1,661 OOD Income rows), which matches DistilBERT's 0.70 flagged in §7; the classical models are lower still (0.53 to 0.56) because they also send Income rows to Utilities (§10.6); (ii) solving this template alone would raise DistilBERT's OOD accuracy by about 3 points (0.917 to roughly 0.947); (iii) the near-identical counts reflect one shared failure on one template, not agreement about a hard boundary.

**Checked afterwards:** (i) `docs/labeling_guideline.md` has **no rule for person-to-person payments** such as Zelle. Its only transfer rules point to Other (transfers between own accounts; bank/account mechanics such as fees, transfers and withdrawals), so the models' choice of Other is consistent with the guideline as written, while the dataset labels incoming Zelle as Income. This is a gap in the labelling guideline to state in the limitations (and fix in the guideline). (ii) The data does follow a direction convention: the 490 `[credit] Zelle payment from ...` rows are labelled Income and the 152 `[debit] Zelle payment to ...` rows are labelled Other, and every model predicts Other for all 642. The `[credit]`/`[debit]` tag (and "from"/"to") is in the string and separates the two labels perfectly within this merchant, but no Zelle row is in the training set to learn it from. (iii) **The convention is learnable and the models do apply it elsewhere.** Training contains 88 Venmo and 90 Cash App rows, all labelled Other and none starting with `[credit] venmo`. Yet in the OOD set DistilBERT, MiniLM, Logistic Regression and XGBoost label all 56 `[credit] Venmo` rows and all 51 `[credit]` Cash App rows Income, and the 22 and 7 debit rows Other: exactly right. They apparently use the `[credit]` tag (which starts 397 correctly classified Income rows) even for merchants whose training rows are all Other, but do not use it for `Zelle payment from <NAME> <ID>`. Why is untested; one possible reason is that the shape "payment ... <name> <ID>" resembles bank-payment strings labelled Other in training.

### 10.3 Key Finding — Shopping Confusions Reveal a Model-Type Split

**Shopping is the second major error source for every model, but *which* categories it's confused with differs by model type:**

- **Classical/fine-tuned models** (Linear SVM, DistilBERT, MiniLM) confuse Shopping predominantly with **Transportation** (269 / 313 / 157) and, for MiniLM and SVM, **Entertainment**.
- **The frozen embedding model** confuses Shopping far more broadly — with Restaurants (385), Entertainment (256), Groceries (151), Travel (142), and even Insurance (125), five different categories all appearing in its top 10.

This matches and reinforces the "semantic over-generalization" failure mode already documented qualitatively in §6: the frozen embedding's errors are spread thin across many semantically-adjacent categories rather than concentrated on one or two, consistent with it relying on coarse semantic similarity rather than the token-level precision the fine-tuned and classical models have.

### 10.4 Key Finding — Fine-tuned DistilBERT Has the Most Concentrated Error Profile

**DistilBERT's errors are markedly more concentrated than the other three models':** its top two pairs (Income→Other: 491, Shopping→Transportation: 313) dominate its error mass, then counts drop sharply into the double digits (82 and below) for the rest of its top 10. By contrast, Linear SVM, the frozen embedding, and MiniLM all have errors spread more evenly across their top 10 pairs (values declining more gradually from ~500 down to ~80-100). A more concentrated error profile is arguably more useful in deployment — it means DistilBERT's mistakes are predictable and limited to two well-understood categories, rather than being spread unpredictably across many category pairs.

### 10.5 Reading the Errors — Merchant Blocks, Not Scattered Mistakes

**Method** (`src/eval/error_analysis_examples.py`; outputs in `reports/error_analysis/`): the cached OOD predictions of all eight models were used to (a) list each model's top confused pairs, including the v2 classical models that §10.1 lacked; (b) count how many models make the same error on the same transaction; (c) read 30 DistilBERT errors for each of the two main pairs (half sampled from errors that at least six models share, half from the rest); and (d) look up chosen merchant strings in the training and OOD sets ("probes").

**Top confused pairs, v2 classical models** (OOD; count):

| Model | 1st | 2nd | 3rd |
|---|---|---|---|
| Logistic Regression v2 | Income→Other (490) | Entertainment→Shopping (251) | Shopping→Transportation (228) |
| Linear SVM v2 | Income→Other (490) | Entertainment→Shopping (283) | Shopping→Transportation (215) |
| Naive Bayes v2 | Income→Other (490) | Entertainment→Shopping (305) | Entertainment→Other (209) |
| Random Forest v2 | Entertainment→Transportation (478) | Shopping→Restaurants (281) | Shopping→Transportation (260) |
| XGBoost v2 | Other→Shopping (415) | Income→Entertainment (252) | Income→Transportation (216) |

XGBoost (6 errors) does not make Income→Other because it scatters the Zelle rows over other categories (probe: Entertainment 185, Income 172, Shopping 74; its 2nd and 3rd errors are Income→Entertainment and Income→Transportation). A person's name and a random ID carry no category signal, so Other is the default for the other models. **Correction:** an earlier version of this note said Random Forest scatters the Zelle rows too. Random Forest makes 93 Income→Other errors and has the *highest* Income recall of all eight models (0.919, §10.6), so it labels most Zelle rows Income; why was not examined.

**Probes** (substring lookups; every merchant below has **zero training rows**, i.e. it was held out; model columns show the most frequent prediction):

| String | OOD rows (true label) | DistilBERT | Fine-tuned MiniLM | Logistic Regression | XGBoost |
|---|---|---|---|---|---|
| "zelle payment from" | 490 (Income) | Other 490 | Other 490 | Other 490 | Entertainment 185, Income 172, Shopping 74 |
| "family dollar" | 76 (Shopping) | Transportation 74 | Shopping 59, Insurance 16 | Transportation 76 | Transportation 60, Shopping 16 |
| "von maur" | 79 (Shopping) | Transportation 77 | Shopping 77 | Transportation 73 | Transportation 50, Utilities 19 |
| "ups store" | 89 (Shopping) | Transportation 83 | Transportation 55, Shopping 34 | Transportation 71 | **Shopping 89** |
| "bloomingdale" | 90 (Shopping) | Transportation 79, Groceries 11 | Restaurants 85 | Restaurants 71 | Restaurants 90 |

**Findings:**
- **Errors come in whole-merchant blocks.** Within one model, nearly every row of an unseen merchant gets the same wrong category. Five templates or merchants (Zelle 491, UPS Store 82, Bloomingdale's 79, Von Maur 77, Family Dollar 74) account for 803 of DistilBERT's 1,332 OOD errors (60%).
- **Which wrong category is model-specific.** Bloomingdale's goes to Transportation for DistilBERT and to Restaurants for the other three; Family Dollar to Transportation for DistilBERT and Logistic Regression but mostly Shopping for MiniLM; XGBoost gets all 89 UPS Store rows right and none of the Bloomingdale's rows. No model is uniformly best at merchant generalisation, and only 52 of DistilBERT's 313 Shopping→Transportation errors are shared by six or more models, in contrast to Income→Other (456 of 491).
- **The 30 Shopping→Transportation examples:** 23 are clear model failures on obvious shops (Family Dollar 14, Von Maur 8, Bloomingdale's 1) and 7 are UPS Store, where the label Shopping is defensible but not obviously better than Other or Transportation. **The 30 Income→Other examples are all the same Zelle template.**
- **Statistical consequence:** the effective sample size of the OOD set is closer to the number of held-out merchant clusters than to 16,031 rows, so differences of a few points between models can come down to a handful of merchants; this is quantified in the next subsection.

**Merchant-clustered uncertainty** (`src/eval/ood_cluster_bootstrap.py`; `reports/error_analysis/ood_cluster_bootstrap.csv`, `ood_cluster_bootstrap_pairs.csv`, `ood_drop_top_clusters.csv`). The OOD set has 16,031 rows in **1,541 merchant clusters** (the clusters written by `split_dataset.py`; median 2 rows per cluster). The largest are PREAUTHORIZED WITHDRAWAL (951 rows, 5.9%), ZELLE PAYMENT (642), MONTHLY INSTALLMENTS (488), NHP FOUNDATION (279), JBG SMITH (259), LYFT TRIP (238), ESSEX PROPERTY (216) and MATERNITY LEAVE (207). Clusters were resampled with replacement (1,000 resamples, same draws for every model), and the row-level bootstrap is shown for comparison:

| Model | OOD macro F1 | Cluster 95% CI | Row-level 95% CI | Cluster CI wider by | Errors in 5 worst clusters |
|---|---|---|---|---|---|
| Fine-tuned DistilBERT | 0.9329 | 0.892 to 0.973 | 0.930 to 0.936 | 12.3x | 68% |
| Logistic Regression v2 | 0.8811 | 0.838 to 0.928 | 0.876 to 0.885 | 10.0x | 43% |
| Fine-tuned MiniLM | 0.8767 | 0.833 to 0.925 | 0.872 to 0.881 | 9.6x | 45% |
| Linear SVM v2 (Platt-calibrated predictions) | 0.8543 | 0.793 to 0.926 | 0.849 to 0.860 | 13.2x | 55% |
| Naive Bayes v2 | 0.8365 | 0.779 to 0.903 | 0.831 to 0.842 | 11.4x | 53% |
| XGBoost v2 | 0.8191 | 0.761 to 0.889 | 0.814 to 0.825 | 11.7x | 50% |
| Random Forest v2 | 0.7450 | 0.683 to 0.803 | 0.738 to 0.752 | 8.8x | 29% |
| Frozen MiniLM + LogReg | 0.6623 | 0.607 to 0.711 | 0.655 to 0.670 | 6.8x | 25% |

Paired differences in OOD macro F1 (same resamples; first minus second):

| Comparison | Difference | 95% CI | First model better in |
|---|---|---|---|
| DistilBERT − Logistic Regression | +0.0519 | +0.022 to +0.086 | 100% (excludes 0) |
| DistilBERT − fine-tuned MiniLM | +0.0563 | +0.030 to +0.083 | 100% (excludes 0) |
| DistilBERT − XGBoost | +0.1138 | +0.052 to +0.168 | 100% (excludes 0) |
| Fine-tuned MiniLM − Logistic Regression | −0.0044 | −0.034 to +0.029 | 42% |
| Logistic Regression − Linear SVM | +0.0268 | −0.015 to +0.074 | 72% |
| Logistic Regression − XGBoost | +0.0620 | −0.003 to +0.120 | 96% |
| Logistic Regression − Naive Bayes | +0.0446 | −0.000 to +0.089 | 97% |

OOD macro F1 after removing the largest merchant clusters in turn (point estimates; the clusters removed are PREAUTHORIZED WITHDRAWAL, then ZELLE PAYMENT and MONTHLY INSTALLMENTS):

| Clusters removed | Rows left | DistilBERT | Logistic Regression | MiniLM | Linear SVM | Naive Bayes | XGBoost | Random Forest | Frozen |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 16,031 | 0.9329 | 0.8811 | 0.8767 | 0.8543 | 0.8365 | 0.8191 | 0.7450 | 0.6623 |
| 1 | 15,080 | 0.9405 | 0.8803 | 0.8835 | 0.8861 | 0.8616 | 0.8511 | 0.7745 | 0.6567 |
| 3 | 13,950 | 0.9633 | 0.9018 | 0.9090 | 0.9085 | 0.8821 | 0.8792 | 0.7765 | 0.6751 |
| 5 | 13,412 | 0.9633 | 0.9017 | 0.9087 | 0.9085 | 0.8812 | 0.8792 | 0.7755 | 0.6810 |

**Findings:**
- **Row-level intervals are far too narrow.** The merchant-clustered intervals are 7 to 13 times wider (DistilBERT: 0.892 to 0.973 against 0.930 to 0.936). Any statement of uncertainty in the report should use the cluster intervals.
- **DistilBERT's advantage is robust.** Its lead over Logistic Regression (+5.2 points, 95% CI +2.2 to +8.6) and over fine-tuned MiniLM (+5.6) excludes zero, it wins in 100% of the resamples, and a gap of about 5.5 points to the best classical model persists after the three largest clusters are removed (0.963 against 0.909 for Linear SVM).
- **Much else is not established.** Fine-tuned MiniLM and Logistic Regression are statistically indistinguishable (−0.4 points, CI −3.4 to +2.9). The ordering among Logistic Regression, Linear SVM, Naive Bayes and XGBoost is not established (the Logistic Regression advantages over Linear SVM, XGBoost and Naive Bayes have intervals that include or nearly touch zero).
- **"Logistic Regression is the best classical model" depends on one cluster.** With the 951-row PREAUTHORIZED WITHDRAWAL cluster removed, Linear SVM scores higher (0.886 against 0.880), and it stays ahead after the three largest clusters are removed (0.909 against 0.902). That cluster costs Linear SVM, Naive Bayes, XGBoost and Random Forest 2.5 to 3.2 points of macro F1 but costs Logistic Regression and the two transformers almost nothing. Reading the rows shows why: it is a wrapper template (below).
- **Concentration:** the five clusters with the most errors hold 68% of DistilBERT's errors and 43 to 55% of those of most other models.

**Caveats:** resampling merchants captures the variability of the OOD sample only, not training randomness (single seed), the tuning, or the choice of model settings. The clusters come from the split script's normalised two-token key plus fuzzy matching, so rows of one real merchant under different keys can count as separate clusters, which leaves the intervals somewhat too narrow. The Linear SVM row uses its Platt-calibrated predictions (0.8543; the raw model scores 0.8514).

**The PREAUTHORIZED WITHDRAWAL cluster** (`src/eval/probe_cluster.py`). All 951 rows have the form `[debit|credit] PREAUTHORIZED WITHDRAWAL TO <merchant>` (case varies), with no training rows containing the phrase. The label is set by the merchant, not the phrase (Pearson, Skillshare: Education; Airbnb, JetBlue, VRBO: Travel; BetterHelp, Costco Pharmacy: Healthcare; Dave & Buster's, Roblox: Entertainment; CTA: Transportation; gyms, salons and YMCA: Shopping). True labels in the cluster are mixed (Shopping 232, Travel 215, Entertainment 198, Education 138, Healthcare 94, others). The monthly-installments cluster (488 rows, all Other, also unseen) is the opposite case: one label, which Logistic Regression and DistilBERT get 488/488 right, MiniLM 423/488 and XGBoost 86/488 (402 predicted Shopping).

Accuracy inside the PREAUTHORIZED WITHDRAWAL cluster: Logistic Regression 0.875, DistilBERT 0.832, MiniLM 0.783, Linear SVM 0.384, XGBoost 0.248. In a 40-row sample, Linear SVM and XGBoost mostly answer Other even where the merchant is clear (Airbnb, JetBlue, Pearson), whereas Logistic Regression and the transformers classify by the merchant after "TO". This is what drives the ordering result above: the classical-model ranking depends on how a model behaves when an unseen generic phrase surrounds a known merchant. **Score decomposition** (`src/eval/feature_importance.py`; `reports/feature_importance/`). For rows where a model predicts Other but the truth is not Other, the score gap Other minus true class was split into an intercept part and a feature part. *Logistic Regression* (111 such rows): intercept part −7.08, feature part +7.52; the intercept favours Other in 0.9% of rows and the words in 99.1%, so its errors come from the words (`to` is among its top-12 Other features and the wrapper contains "TO"; not measured per row). *Linear SVM* (623 rows, raw-score accuracy in the cluster 0.345): mean gap 0.41 = intercept +0.58 + features −0.17; the intercept alone favours Other in 67.6% of rows, because Travel (−3.86), Education (−2.67) and Healthcare (−2.16) have lower intercepts than Other (−1.27), while the words contribute almost nothing. So Linear SVM is deciding by class prior when the wrapper leaves it with little word evidence. Cluster rows also activate more features than other OOD rows (164.6 against 94.6), mostly character n-grams of the unseen wrapper words.

**Counterfactual test** (`src/eval/wrapper_ablation.py`; `reports/feature_importance/wrapper_ablation.csv`). The 951 rows were rewritten in two ways and re-scored with the raw models (accuracy inside the cluster):

| Model | Original | Phrase removed, "TO" kept | Merchant only |
|---|---|---|---|
| Logistic Regression | 0.875 | 0.961 | 1.000 |
| Linear SVM | 0.345 | 0.938 | 1.000 |
| Naive Bayes | 0.342 | 0.973 | 1.000 |
| XGBoost | 0.248 | 0.977 | 0.984 |
| Random Forest | 0.178 | 0.957 | 0.938 |

(Examples: `[debit] PREAUTHORIZED WITHDRAWAL TO MARRIOTT` becomes `[debit] TO MARRIOTT` and `[debit] MARRIOTT`.) Share predicted Other falls from 12 to 75% to 0% in every model.

**Findings:** (i) **the unseen wrapper phrase causes the failure.** Deleting it restores 0.94 to 0.98 accuracy in all five models, and with the merchant alone three models reach 1.000, so the models classify these merchants correctly when the phrase is absent (the merchant strings are probably familiar from training; not checked). This is a surface-form robustness failure, not missing knowledge. (ii) The word "to" costs 4 to 6 points for Logistic Regression and Linear SVM. (iii) **The earlier bigram explanation (the wrapper breaking `debit <merchant>` bigrams) is not supported:** `[debit] TO MARRIOTT` breaks that bigram too, yet Linear SVM scores 0.938. The phrase's own words and character n-grams, which add features and class-prior pull towards Other (intercept result above), are the more likely cause, but that is an inference from the decomposition, not a separate test.

**Other observation:** `hand__length` and `hand__punct_ratio` carry top weights for Travel and Education in Logistic Regression and Linear SVM, i.e. the models partly use string length and punctuation, which may be a shortcut. Some transformer errors in the sample are defensible given label conventions (Walgreens labelled Healthcare, Orangetheory and YMCA labelled Shopping).

**Transformers: the same counterfactual** (`src/eval/wrapper_ablation_transformers.py`; `reports/feature_importance/wrapper_ablation_transformers.csv`; predictions on the original text reproduce the cached predictions on 100% of rows for all three models):

| Model | Original | Phrase removed, "TO" kept | Merchant only |
|---|---|---|---|
| Fine-tuned DistilBERT | 0.832 | 1.000 | 1.000 |
| Fine-tuned MiniLM | 0.783 | 0.989 | 0.991 |
| Frozen MiniLM + LogReg | 0.597 | 0.711 | 0.814 |

Deleting the phrase removes all 160 of DistilBERT's errors and nearly all of MiniLM's 206 in this cluster. The frozen embedding model improves only to 0.71 and 0.81, so for it the wrapper explains part of the failure and its generally weak OOD accuracy the rest. Across all eight models the wrapper phrase is therefore a cause of the cluster errors, and for DistilBERT and MiniLM it is the only cause found.

**Saliency by occlusion** (`src/eval/transformer_saliency.py`; `reports/error_analysis/saliency_<model>_*.csv`). A word or part of the string is removed and the change in probability is measured ("drop" is positive when the part supported the class; a negative drop means probability rose when the part was removed). It examines one removal at a time, so interactions between words appear only indirectly, and a word that is redundant with another can show no effect.
- *Wrapper cluster (aggregate).* For the rows the model gets wrong, removing the phrase `PREAUTHORIZED WITHDRAWAL [TO]` raises P(true class) by 0.95 for DistilBERT (n=160) and 0.88 for MiniLM (n=206) and lowers P(Other) by the same amounts, whereas removing the merchant changes P(true) by only +0.05 and +0.09: the phrase overrides the merchant. For rows the model gets right, removing the merchant lowers P(true) by 0.99 and 0.96 and removing the phrase changes it by less than 0.04. Removing the `[debit]`/`[credit]` tag also moves wrong rows towards the true class (P(true) up by 0.21 and 0.30), so the tag adds some pull towards Other.
- *Zelle (DistilBERT).* The prediction Other rests on the words "payment" (support +1.00 in one example) and "from"; the name and the ID contribute nothing, and "Zelle" itself shows no effect. This fits an absence of Zelle knowledge: the model reads the generic payment wording.
- *Family Dollar and UPS (DistilBERT).* Transportation rests on single words: "Dollar" (support +0.985 and +0.777) and "UPS" (+1.00 and +0.74). Why those words point to Transportation was not tested (a car-rental brand called Dollar and parcel shipping are possible associations, not checked against the training set).
- *Von Maur (DistilBERT).* No single word matters (all effects near zero), so the prediction depends on the combination of tokens or on redundant evidence; word-level occlusion cannot say more.
- *Paternity Leave.* DistilBERT is correct and relies on the word "Leave" (support +0.55 in one example, +0.05 in another), not on "Paternity". Fine-tuned MiniLM predicts Shopping, and the prediction depends on "Paternity" together with the ID token (removing either one removes the Shopping prediction).

**Not yet done in the error analysis:** nothing planned; limitations are listed in §0.

### 10.6 Per-Class Recall and Feature Importance, All Models

**Method:** `src/eval/confusion_matrices_all_models.py` builds row-normalised confusion matrices for all eight models from the cached OOD predictions (heatmaps, count CSVs and `per_class_recall_ood.csv` in `reports/confusion_matrices/all_models/`; the Linear SVM uses its Platt-calibrated predictions). Feature importance: `src/eval/feature_importance.py` (Logistic Regression, Linear SVM, XGBoost) and `src/eval/feature_importance_nb_rf.py` (Naive Bayes, Random Forest); outputs in `reports/feature_importance/`. Transformers have no coefficients; their behaviour is characterised by the error analysis and probes above, not by feature attribution (saliency not done).

**Per-class recall, OOD test (diagonal of the confusion matrix):**

| Class | DistilBERT | LR | MiniLM | SVM | NB | XGB | RF | Frozen |
|---|---|---|---|---|---|---|---|---|
| Education | 0.910 | 1.000 | 0.903 | 0.836 | 0.864 | 0.867 | 0.677 | 0.682 |
| Entertainment | 0.950 | 0.772 | 0.785 | 0.737 | 0.691 | 0.745 | 0.564 | 0.695 |
| Groceries | 0.993 | 0.998 | 0.998 | 0.991 | 0.958 | 0.988 | 0.743 | 0.875 |
| Healthcare | 0.947 | 0.900 | 0.736 | 0.892 | 0.909 | 0.669 | 0.604 | 0.685 |
| Housing | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.998 | 0.837 |
| Income | 0.702 | 0.539 | 0.651 | 0.535 | 0.555 | 0.632 | 0.919 | 0.518 |
| Insurance | 1.000 | 0.998 | 0.994 | 1.000 | 0.989 | 0.998 | 0.944 | 0.738 |
| Other | 0.975 | 0.981 | 0.936 | 0.978 | 0.899 | 0.710 | 0.952 | 0.736 |
| Restaurants | 0.978 | 0.884 | 0.889 | 0.923 | 0.912 | 0.794 | 0.704 | 0.740 |
| Shopping | 0.793 | 0.723 | 0.795 | 0.712 | 0.733 | 0.647 | 0.456 | 0.346 |
| Transportation | 0.925 | 0.935 | 0.926 | 0.915 | 0.847 | 0.870 | 0.722 | 0.647 |
| Travel | 0.967 | 0.998 | 0.968 | 0.744 | 0.812 | 0.764 | 0.538 | 0.802 |
| Utilities | 0.997 | 0.914 | 0.888 | 0.918 | 0.861 | 0.888 | 0.793 | 0.849 |

**Findings:**
- **Income is the lowest-recall class for six of the eight models** (DistilBERT 0.70, MiniLM 0.65, XGBoost 0.63, Naive Bayes 0.56, Logistic Regression 0.54, Linear SVM 0.53). For the three linear/probabilistic classical models the shortfall is larger than the Zelle template alone explains (Zelle caps recall at about 0.70): their top confused pairs also include Income→Utilities (201, 146 and 137 rows). **These are one more set of held-out templates** (`src/eval/income_to_utilities.py`; `reports/error_analysis/income_to_utilities_*.csv`): Maternity Leave, Paternity Leave and Social Security rows, all labelled Income in the OOD set. The ten most common templates cover 176 of Logistic Regression's 201 Income→Utilities errors, 131 of Linear SVM's 146 and 123 of Naive Bayes's 137 (26, 22 and 22 distinct templates in total). DistilBERT classifies 100%, 99% and 99% of these same rows correctly (see the substring probes below). Zelle (490 rows) plus these rows (201) account for 691 of Logistic Regression's roughly 766 Income errors, so its Income recall of 0.54 comes almost entirely from two held-out template families. That the transformer handles them may reflect language knowledge of the words "maternity" and "paternity"; that is an untested explanation. **Confirmed unseen** (substring probes, `error_analysis_examples.py --probe`): the training set has 0 rows containing "maternity leave", "paternity leave" or "social security", against 106, 101 and 82 OOD rows, all labelled Income. Most frequent predictions per family:

| Model | Maternity leave (106) | Paternity leave (101) | Social security (82) |
|---|---|---|---|
| DistilBERT | Income 106 | Income 100 | Income 82 |
| Fine-tuned MiniLM | Income 106 | Shopping 80, Income 13 | Income 81 |
| Logistic Regression | Utilities 100 | Utilities 87 | Entertainment 31, Transportation 19, Utilities 14, Income 9 |
| XGBoost | Transportation 79 | Transportation 69 | Transportation 68 |

So the transformers do not generalise uniformly either: MiniLM classifies Maternity Leave correctly but sends 80 of 101 Paternity Leave rows to Shopping, and Logistic Regression's errors on Social Security are spread over several classes, not concentrated in Utilities.
- **Random Forest and the frozen embedding have a different weak class:** Shopping (0.46 and 0.35). Random Forest has the best Income recall (0.919) but the worst recall on Entertainment, Travel and Healthcare.
- **The PREAUTHORIZED WITHDRAWAL effect is visible in the per-class recall.** Logistic Regression scores 1.000 on Education and 0.998 on Travel, Linear SVM 0.836 and 0.744; those are two of the categories the wrapper cluster holds (§10.5). This is consistent with, not separate evidence for, the cluster explanation.
- **Easy classes:** Housing and Insurance are at or near 1.000 for every model except Random Forest (Insurance 0.944) and the frozen embedding (0.837 and 0.738).
- **DistilBERT's weak classes are Income (0.70) and Shopping (0.79)**; every other class is at least 0.91. Fine-tuned MiniLM's weak classes are Income, Healthcare (0.736) and Entertainment (0.785).
- **XGBoost's weak classes** are Healthcare (0.669), Shopping (0.647) and Other (0.710); the low Other recall is the monthly-installments cluster predicted as Shopping (§10.5).

**Feature importance:**
- **Logistic Regression and Linear SVM** (§10.5): sensible merchant and keyword features per class (`expedia`, `pluralsight`, `payroll`, `fee`, `atm`); `word__to` is a top Other feature in both; `hand__length` and `hand__punct_ratio` carry top coefficients for Travel and Education.
- **Naive Bayes** (highest log-probability ratio per class): Other `fee`, `atm`, `debit atm`, `account`, `transaction`; Travel `cruise`, `booking`, `airlines`, `inn`; Education `tuition`, `pluralsight`, `college`, `tutoring`; Income `payroll`, `interest`, `salary`, `refund`, `hand__cue_payroll`. Many are character n-gram fragments of the same words.
- **Random Forest** (global impurity importance): the top features are `hand__cue_housing`, the character n-gram ` fee `, `word__debit`, ` ent `, `hand__punct_ratio`, `hand__digit_ratio`, `hand__cue_fee`, and fragments of the `[debit]` tag (`[debi`, `t]`, `bi`). It leans on structural and tag features more than on merchant words, which fits its being the weakest classical model on OOD; this is an association, not a tested cause.
- **XGBoost:** top features are character n-grams (` crui`, ` pre`, `arket`, `uitio`) and `word__atm`, `word__learning`.
- **Share of importance by feature group** (Random Forest, XGBoost): character n-grams 87.9% and 83.6%, word n-grams 8.7% and 15.1%, hand-crafted features 3.4% and 1.3%. The hand-crafted features are therefore a small share of the tree models' importance, although for Random Forest three of its six most important features are hand-crafted.

---

## 11. Inference Cost (CPU)

**Status: complete for all eight models** (`src/eval/inference_cost_analysis.py`; `reports/inference_cost/inference_cost.csv`, `cost_vs_quality.png`).

**Method.** The *full* pipeline is timed from raw text to predicted label (featurisation, tokenisation or embedding included), because that is what a deployed service pays per request. Each model runs in its own subprocess so memory figures are not polluted by earlier models. Single-transaction latency: 300 random OOD transactions at batch size 1 after 20 warm-up calls, median and 95th percentile. Batch throughput: 2,048 transactions in batches of 256. Memory: RSS increase from loading the model, and process peak after inference. Hardware: one AMD CPU, 12 logical cores, library-default thread settings, Windows. All numbers are specific to this machine.

| Model | Params (kind) | Size on disk | Median latency | p95 latency | Batch throughput | RAM after load | Peak RAM | OOD macro F1 |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression v2 | 505,895 (coefficients) | 4.1 MB | **3.05 ms** | 4.9 ms | 6,902/s | 46 MB | 153 MB | 0.8811 |
| Linear SVM v2 | 505,895 (coefficients) | 4.1 MB | 3.22 ms | 6.1 ms | 5,381/s | 93 MB | 182 MB | 0.8514 |
| Naive Bayes v2 | 505,895 (log-probs) | 8.1 MB | 6.70 ms | 9.4 ms | 6,559/s | 93 MB | 186 MB | 0.8365 |
| XGBoost v2 | 56,884 (tree nodes) | 4.5 MB | 9.58 ms | 15.3 ms | 1,678/s | 114 MB | 245 MB | 0.8191 |
| Random Forest v2 | 3,063,818 (tree nodes) | 515 MB | 92.6 ms | 232 ms | 595/s | 74 MB | 529 MB | 0.7450 |
| Fine-tuned MiniLM | 117,658,765 (weights) | 471 MB | 32.6 ms | 40.3 ms | 79/s | 497 MB | 1,061 MB | 0.8767 |
| Fine-tuned DistilBERT | 66,963,469 (weights) | 268 MB | 62.1 ms | 104 ms | 48/s | 191 MB | 933 MB | **0.9329** |
| Frozen MiniLM + LogReg | 117,658,765 (weights) | 471 MB | 88.8 ms | 326 ms | 56/s | 337 MB | 889 MB | 0.6623 |

The classical models share a 1.4 MB fitted feature pipeline (not included in their size). OOD macro F1 for Linear SVM is the raw model's 0.8514 (the cost CSV shows 0.8543 because it was computed from the Platt-calibrated predictions).

**Findings:**
- **Only two models are on the cost-quality frontier: Logistic Regression and fine-tuned DistilBERT.** Moving from Logistic Regression to DistilBERT buys +5.2 points of OOD macro F1 at about 20 times the median latency (3 ms to 62 ms), about 140 times the throughput cost (6,900 to 48 transactions/s) and roughly six times the peak memory.
- **Every other model is dominated.** Fine-tuned MiniLM is slower than Logistic Regression (33 ms against 3 ms) and slightly worse; Linear SVM, Naive Bayes, Random Forest and XGBoost are no faster than Logistic Regression and less accurate; the frozen embedding is slower than all of them and the worst.
- **Parameter count is not compute.** MiniLM has more parameters (117.7M) than DistilBERT (67.0M) but is about twice as fast: most of its parameters are the 250k-word multilingual embedding table, which is looked up and not multiplied, while DistilBERT has wider (768) layers. Parameter counts are also not comparable across families (weights, coefficients, tree nodes), which the `kind` column records.
- **Random Forest is the expensive classical model** (93 ms, 515 MB) for the lowest classical accuracy.

**Caveats:** one machine and one run; the frozen MiniLM latency is noisy (p95 is 3.7 times its median, and it shares the architecture of fine-tuned MiniLM, which is much faster) and was not re-measured; Naive Bayes being slower than Logistic Regression (6.7 against 3.05 ms) is unexplained; batch throughput uses library-default threading; GPU inference was not measured. XGBoost was trained on a GPU and exported to its native format for CPU evaluation (the model is unchanged).


### 11.1 Training and Inference Cost: Combined Summary

One table for the report, drawing together the training costs (§5.1, §6, §7, §7.1, §8) and the inference costs above. Hardware: laptop CPU (AMD, 12 logical cores) or a free Colab T4 GPU, as stated; times from different hardware are not comparable.

| Model | Parameters | Trained on | Training time | Size on disk | Median latency | Peak RAM | OOD macro F1 |
|---|---|---|---|---|---|---|---|
| Logistic Regression v2 | 505,895 (coefficients) | laptop CPU | about 20 s per full-data fit | 4.1 MB | 3.05 ms | 153 MB | 0.8811 |
| Linear SVM v2 | 505,895 (coefficients) | laptop CPU | seconds to about 1 min per fit | 4.1 MB | 3.22 ms | 182 MB | 0.8514 |
| Naive Bayes v2 | 505,895 (log-probabilities) | laptop CPU | seconds per fit | 8.1 MB | 6.70 ms | 186 MB | 0.8365 |
| XGBoost v2 | 56,884 (tree nodes) | Colab T4 GPU | about 6 min for the whole search and final fit; about 32 min per full-data fit on CPU | 4.5 MB | 9.58 ms | 245 MB | 0.8191 |
| Random Forest v2 | 3,063,818 (tree nodes) | laptop CPU | seconds to about 1 min per fit | 515 MB | 92.6 ms | 529 MB | 0.7450 |
| Frozen MiniLM + LogReg | 117,658,765 frozen weights, plus a 5,005-parameter classifier trained (13 x 384 weights + 13 biases, calculated) | laptop CPU | about 16 min to encode the 96k strings once; the classifier fit is quick | 471 MB | 88.8 ms | 889 MB | 0.6623 |
| Fine-tuned DistilBERT | 66,963,469 (all fine-tuned) | Colab T4 GPU | 23.2 min (6 epochs); 9.9 min for the 3-epoch variant | 268 MB | 62.1 ms | 933 MB | 0.9329 |
| Fine-tuned MiniLM | 117,658,765 (all fine-tuned) | Colab T4 GPU | 27.8 min (6 epochs) | 471 MB | 32.6 ms | 1,061 MB | 0.8767 |

**Reading.** Fine-tuning DistilBERT costs about 23 minutes on a free GPU, which is not a large training bill; the price of the +5.2 points over Logistic Regression is paid mainly at inference (about 20 times the latency, §11) and in memory. Parameter counts are not comparable across families (coefficients, tree nodes, weights), and the classical models' parameter counts reflect the 38,914-feature input, not model complexity.

**Caveats.** Classical training times are rough single measurements, not a controlled benchmark, and exclude the hyperparameter search except for XGBoost. The transformers were not tuned, so their cost is one training run. The three-epoch DistilBERT run, the learning-curve retrainings (§12) and the Colab sessions' set-up time are not included. GPU time is a free Colab T4 and varies by session. Energy use was not measured.

---

## 12. Learning Curves and Low-Resource Categories

**Status: complete for all eight models** (`src/eval/learning_curves.py` for the laptop runs and the figure; `src/eval/learning_curve_transformer.py` for the Colab GPU runs; `reports/learning_curves/`: `learning_curves_summary.csv`, per-category CSVs, `learning_curves.png`).

**Method.** Each model is retrained on a stratified 5%, 10%, 25%, 50% and 100% of the training set (2,798 to 55,978 rows) and scored on the full in-distribution and OOD test sets. Settings are held at each model's tuned (v2) values; only the amount of training data changes. The fine-tuned transformers use the original training recipe (class-weighted loss, 6-epoch ceiling, early stopping on validation macro F1); their 100% points are the original runs, scored from cached predictions. **Category-size groups:** the 13 categories ranked by training-set size; *small* = Travel, Education, Insurance, Groceries (2,382-2,498 rows); *medium* = Income, Utilities, Other, Restaurants, Transportation (4,437-4,636); *large* = Housing, Healthcare, Entertainment, Shopping (4,766-6,943). A group's score is the mean per-category F1 inside it.

**OOD macro F1 by training-set size:**

| Model | 2,798 (5%) | 5,597 (10%) | 13,994 (25%) | 27,989 (50%) | 55,978 (100%) |
|---|---|---|---|---|---|
| Fine-tuned DistilBERT | **0.8478** | **0.8629** | **0.8995** | 0.8761 | **0.9329** |
| Fine-tuned MiniLM | 0.7616 | 0.8125 | 0.8641 | 0.8668 | 0.8767 |
| Logistic Regression v2 | 0.7268 | 0.7881 | 0.8442 | 0.8696 | 0.8811 |
| Linear SVM v2 | 0.7518 | 0.7887 | 0.8291 | 0.8500 | 0.8514 |
| Naive Bayes v2 | 0.6914 | 0.7348 | 0.7917 | 0.8148 | 0.8365 |
| XGBoost v2 (tuned, CPU re-run) | 0.6244 | 0.7108 | 0.7847 | 0.8172 | 0.8260 |
| Random Forest v2 | 0.6158 | 0.6693 | 0.6982 | 0.7276 | 0.7450 |
| Frozen MiniLM + LogReg | 0.5632 | 0.5984 | 0.6316 | 0.6558 | 0.6623 |

**Findings on data efficiency:**
- **The fine-tuned transformers need far less labelled data.** DistilBERT with 5% of the data beats the best classical model at that size by 9.6 points (0.848 against 0.752, Linear SVM); Logistic Regression needs about 25% (13,994 rows, five times as many) to reach 0.844. **DistilBERT with 25% of the data (0.8995) beats every classical model trained on 100% (best: Logistic Regression, 0.8811).** Fine-tuned MiniLM leads the classical models from 5% to 25% (by 1.0 to 2.4 points) and is level with Logistic Regression from 50% upward.
- **The best classical model depends on data size:** Linear SVM at 5%, a tie at 10%, Logistic Regression from 25%.
- **Diminishing returns for classical models.** Linear SVM, Random Forest and the frozen embedding are nearly flat from 50% to 100%; no classical model approaches DistilBERT at any size, so more labelled data of the same kind will not close the gap, which is about unseen merchants.
- **The frozen embedding is the worst model at every size.**
- **DistilBERT's score dips at 50% (0.8761 below 0.8995 at 25%).** The other curves never move by more than about a point between neighbouring sizes, so this looks like run-to-run variance of fine-tuning (single seed, early stopping) and not a real effect. The dip sits in the medium and large groups (medium 0.877 to 0.839, large 0.911 to 0.887), not the small one (0.916 to 0.912), which points at the Income/Other boundary (§10.2); this was not examined further.

**Findings on low-resource categories** (small-group F1 minus large-group F1, OOD, in points; negative = small categories do worse):

| Model | 5% | 10% | 25% | 100% |
|---|---|---|---|---|
| Fine-tuned DistilBERT | −3.5 | −6.9 | +0.4 | +3.0 |
| Fine-tuned MiniLM | −5.2 | +1.9 | +4.8 | +7.5 |
| Logistic Regression v2 | +0.4 | +2.0 | +2.8 | +8.8 |
| Linear SVM v2 | −4.3 | −0.7 | −1.2 | +4.1 |
| Naive Bayes v2 | −6.7 | −5.9 | −3.1 | +3.0 |
| Random Forest v2 | −4.4 | −5.1 | −0.2 | +5.3 |
| XGBoost v2 (tuned) | −1.6 | +3.7 | +8.4 | +13.0 |
| Frozen MiniLM + LogReg | −15.9 | −13.0 | −12.3 | −9.1 |

- **At the smallest data size (about 120 rows per small category) seven of eight models score lower on small categories than on large ones, by about 5 points on average**; Logistic Regression is the exception. The penalty is modest.
- **It disappears with more data, and at full data seven of eight models score *higher* on the small categories.** The small categories (Education, Groceries, Insurance, Travel) have distinctive vocabulary; the large ones (Shopping, Entertainment, Housing, Healthcare) are heterogeneous.
- **The frozen embedding is the one model with a consistent low-resource penalty** (small-group F1 0.44 at 5% and 0.59 at 100%, against 0.68 for the large group at 100%).
- **The fine-tuned transformers cope better with scarce data in absolute terms:** at 5%, DistilBERT's small-group F1 is 0.840 against 0.756 for the best classical model (+8.4 points), though its own small-versus-large gap at 10% (−6.9 points) is the largest of the transformers.
- **Category size does not predict difficulty; what is held out does.** At 100% the *medium* group is the hardest for five of the eight models (OOD group F1: Logistic Regression 0.821, Linear SVM 0.812, Naive Bayes 0.791, DistilBERT 0.888, fine-tuned MiniLM 0.854, essentially tied with its large group), and it contains Income, 490 of whose 1,661 OOD rows are one unseen template (Zelle payments) that six models send to Other (§10.2, §10.5); the group's low score is therefore partly an artifact of one held-out merchant, not evidence that medium-sized categories are intrinsically harder.

**Caveats to state in the report:**
- Single seed per point; group scores average only four categories, so differences under about 2-3 points are within noise.
- "Small" categories have about 2,400 training rows each (about a third to a half the size of the large ones), so this is not an extreme low-resource setting; the 5% point (about 120 rows per small category) is the closest test of the claim.
- The v2 feature vocabulary and IDF were fitted once on the full training text and rows were subsampled, so only the labelled data shrinks; this slightly flatters small fractions. Tuned settings were chosen on the full training set and may be slightly off at 5%.
- XGBoost: the tuned CPU re-run scores 0.8260 at 100% against 0.8191 from the GPU run used elsewhere (hardware/run variation), and an earlier run with default settings scored higher (§5.2).
- DistilBERT and MiniLM were retrained on Colab; the learning-curve points are single trainings.

---

## 13. Demonstrator (FastAPI, Web UI, Docker, Tests)

**Status: complete** (`src/demo/api.py`, `src/demo/static/index.html`, `tests/test_api.py`, `tests/conftest.py`, `Dockerfile`, `docker-compose.yml`, `requirements-demo.txt`).

**What it is.** A small web service that categorizes one transaction description at a time. It serves a one-page UI and a JSON API. Two models are offered: fine-tuned DistilBERT (best accuracy) and Logistic Regression v2 (the cost-quality alternative, §11), the two models on the cost-quality frontier.

**API.** `GET /health`; `GET /models` (the available models with a description each); `POST /predict` with `text` (1-200 characters, whitespace collapsed), `model` (`distilbert` or `logreg`), `direction` (`none`, `debit` or `credit`) and `top_k` (1-5). The response has the predicted category, its confidence, the top-k categories with probabilities, the exact model input, the latency in milliseconds and a note. Invalid input returns 422; missing model files return 503 with a readable message. Request and response models use pydantic v2.

**Design choices.**
- **Direction tag.** The models were trained on strings that carry a `[debit]` or `[credit]` tag, so the UI offers a direction selector and prepends the tag unless the text already has one. The tag is part of the model input and can change the prediction (the credit-means-Income convention, §10.2).
- **Loading.** Models are loaded lazily on first use and cached. Setting `PRELOAD_MODELS=1` loads both at startup (the Docker image sets it), so the first request does not pay the load time (about 1 s for DistilBERT without preload). Models are found through the `MODELS_DIR` environment variable.
- **Honest output.** The UI shows the top three probabilities and a fixed note that the confidence is not a reliable indicator of correctness, because of the calibration finding in §9 (91% of DistilBERT's wrong OOD predictions are made at over 90% confidence).
- **Examples.** The UI has example buttons chosen to show the findings: an ordinary merchant (Kroger, Starbucks), the Zelle template (predicted Other although labelled Income in the data), unseen merchants (Family Dollar), the PREAUTHORIZED WITHDRAWAL wrapper template and a Paternity Leave string (§10.5, §10.6). A German merchant (REWE) was removed from the examples, because German input is outside the scope (§4).

**Docker.** `python:3.12-slim`, CPU-only PyTorch from the PyTorch CPU index, other packages pinned in `requirements-demo.txt` (scikit-learn must match the version that pickled the classical models), runs as a non-root user, port 8000, with a health check (90 s start period for the preload). **Model files are not baked into the image**: `docker-compose.yml` mounts `./models` read-only at `/models`. Image size 2.03 GB; the first build takes about 5-6 minutes. Offline mode for Hugging Face is set, so the container makes no network calls at run time.

**Latency in the container** (CPU, this laptop): DistilBERT about 69 ms in steady state (range 69-256 ms in ad-hoc use), in line with the 62 ms median measured in §11. Logistic Regression was not timed in the container.

**Tests.** `uv run pytest` runs 34 tests, all passing: 13 feature-pipeline tests (§5.1), 19 API tests with a stub predictor (no model files needed: health, model list, top-k sorting, direction-tag handling, no duplicated tag, whitespace collapsing, eight invalid-input cases, 503 on missing models, preload on / off / failure) and 2 integration tests that run the real models and are skipped when `models/` is absent. Static checks: `ruff` and `pylint`.

**Limitations.** Single-request, CPU-only, not hardened for production (no authentication or rate limiting); the pickled classical models only load with the same scikit-learn version; the model files are not in the repository (§14), so the demo needs them regenerated or supplied.

---

## 14. Reproducibility and Repository Notes

**What is in the repository:** source code (`src/`), tests, the Dockerfile and compose file, documentation (`docs/`), and small generated reports (`reports/`, tables and figures).

**What is not (git-ignored):** raw, interim and processed data (`data/raw/`, `data/interim/`, `data/processed/`, `data/embeddings/`, `data/features_v2/`), all trained models (`/models/`, including the 515 MB Random Forest), cached probabilities (`reports/calibration/probs/`) and `*.bak` files. Checked on 7 November 2026: `git ls-files data models` lists nothing, so no data or model file is tracked. Everything ignored is regenerated by the pipeline (README, "Reproducing the results"). The `data/ood/` folder is not ignored; it holds nothing tracked.

**Where each step ran:**
- **Laptop (CPU):** data build and split, the classical models (v1 and v2, except XGBoost's tuning), the frozen-embedding encoding and classifier (about 16 min), calibration, error analysis, inference cost, the results table and the demo.
- **Google Colab (T4 GPU):** fine-tuning of DistilBERT (3 and 6 epochs) and of multilingual MiniLM, the XGBoost search and final fit, and the transformer learning-curve runs (`learning_curve_transformer.py`, §12). The Colab scripts read and write Google Drive paths set at the top of each file.

**Data provenance:** both raw sources are public Hugging Face datasets under MIT (§1); nothing personal was used.

**Seeds and versions:** seed 42 for the data split and sampling. Python 3.12, dependencies locked in `uv.lock`.

**Known clean-up items (before the repository goes public):** remove draft copies of this log, `main.py` and the empty `mkdir` file; fix script usage lines; mark the v1 scripts as legacy; replace the stale `docs/methodology_log.md` with this file.

---

## Changelog
- Initial version — covers data sourcing through classical baseline results, transformer approach 1 in progress.
- Added transformer approach 1 (frozen multilingual embedding) results — underperformed all classical baselines, a genuine negative result.
- Added transformer approach 2 (fine-tuned DistilBERT) results, including a 3-vs-6 epoch ablation, and the full cross-model comparison. Central hypothesis confirmed via fine-tuning specifically, not transformers generically.
- Added live German-merchant qualitative probe (§4.1) and full ECE/calibration analysis (§9). Key finding: DistilBERT's low aggregate ECE masks a lopsided confidence distribution (98% of predictions maximally confident) with much worse calibration in its rare non-extreme predictions.
- Added §10: confusion-matrix / confusable-pair error-analysis script written and documented (method, scope, rationale). Not yet executed — no results yet, flagged explicitly as pending rather than left silent.
- §10 updated with actual results from the executed run across all four models. Key findings: Income→Other is the top confused pair for every model with near-identical counts (490-509), pointing to genuine label ambiguity rather than a model-specific weakness; Shopping confusions split by model type (frozen embedding spreads broadly, classical/fine-tuned models concentrate on Transportation); fine-tuned DistilBERT has the most concentrated error profile of the four. Added §0 progress summary section covering all work completed to date.
- Added §5.1 (classical v2): exposé-specified character n-gram and hand-engineered features, a Linear SVM feature ablation, one validation-set tuning protocol for all five models, results, and a Logistic Regression tuning-vs-features check. Best classical baseline is now Logistic Regression (88.1% OOD macro F1, up from 79.8% untuned; v1 best was Linear SVM at 83.6%). XGBoost was tuned on a Colab GPU.
- Added §7.1 (fine-tuned multilingual MiniLM, 87.7% OOD macro F1): same base model as the frozen embedding approach, +21.4 points from fine-tuning, roughly level with the best tuned classical model and 5.6 points below DistilBERT.
- Rewrote §8 with the v2 classical models and MiniLM; central hypothesis restated as partly supported (gap between DistilBERT and the best classical model is 5.2 points, not 9.6). Descoped German/multilingual evaluation (§4) per supervisor guidance. Updated §0 progress summary and "not yet done" list; flagged that §9 (ECE) and §10 (confusion matrices) still use v1 classical models.
- Replaced the 3-model preliminary calibration section (§9) with ECE and reliability diagrams for all eight models on both test sets (Linear SVM via a sigmoid calibrator fitted on validation). Key finding: the fine-tuned transformers are not better calibrated than the classical models on unseen merchants (OOD ECE: DistilBERT 0.078, fine-tuned MiniLM 0.133, against 0.032 XGBoost and 0.038 Logistic Regression), and 91% / 89% of their wrong OOD predictions are made at over 90% confidence, so confidence cannot flag their errors.
- Added §11 (inference cost on CPU for all eight models): only Logistic Regression and fine-tuned DistilBERT are on the cost-quality frontier (62 ms against 3 ms median latency for +5.2 points OOD macro F1); parameter count is not compute (MiniLM has more parameters than DistilBERT but is twice as fast).
- Added §12 (learning curves for all eight models at 5-100% of the training data, with small/medium/large category-size groups): DistilBERT with 5% of the data beats the best classical model at 5% by 9.6 points, and with 25% beats every classical model at 100%; the low-resource-category claim is only weakly supported (small categories score about 5 points lower at 5% for 7 of 8 models, reversing by full data; the frozen embedding is the exception) and category ambiguity, not size, predicts difficulty.
- Added §5.2: an unplanned XGBoost run with default settings (caused by a script bug, since fixed) scored 0.8472 OOD macro F1 at full data, above the tuned model (0.8260 CPU, 0.8191 GPU), suggesting the validation-based tuning protocol does not select for unseen-merchant generalisation. Updated §0 (progress summary, third headline finding, not-yet-done list) and §8 (hypothesis assessment now covers calibration, data efficiency, low-resource categories and cost).
- Added §10.5 (error analysis, part A): top confused pairs for all eight models, 30 read examples per main pair, and merchant probes. Key finding: the Income→Other errors are one unseen template (`[credit] Zelle payment from <NAME> <ID>`, 490 rows labelled Income, no Zelle in training), so the earlier "inconsistent labels in the source data" interpretation (§10.2, §0) was **retracted**; Shopping→Transportation errors are four unseen merchants (UPS Store, Bloomingdale's, Von Maur, Family Dollar) that each model mislabels almost wholesale. Five templates or merchants account for 60% of DistilBERT's OOD errors, so OOD scores rest on far fewer independent observations than rows; merchant-clustered confidence intervals are still to do.
- Error analysis follow-up: the labelling guideline has no rule for person-to-person payments (its transfer rules point to Other), and the dataset labels `[credit] Zelle payment from` as Income (490 rows) and `[debit] Zelle payment to` as Other (152 rows); all models predict Other for all 642, so the unseen Zelle template, not label noise, explains the Income→Other errors. Documented as a guideline gap and limitation (§10.2).
- Added merchant-clustered bootstrap intervals for the OOD scores (§10.5; 1,541 clusters): cluster intervals are 7-13 times wider than row-level ones; DistilBERT's lead over Logistic Regression (+5.2 points, CI +2.2 to +8.6) and over fine-tuned MiniLM is significant, whereas MiniLM vs Logistic Regression and the ordering among the classical models are not; removing one 951-row cluster flips Linear SVM above Logistic Regression. Also added to §10.2: the Venmo / Cash App probes show the models do apply the credit-means-Income convention to other merchants (56/56 and 51/51 credits labelled Income), so the Zelle failure is specific to that string form.
- §10.5: PREAUTHORIZED WITHDRAWAL identified as a wrapper template (`... TO <merchant>`, label set by the merchant); in-cluster accuracy LR 0.875, DistilBERT 0.832, MiniLM 0.783, SVM 0.384, XGBoost 0.248. Mechanism untested.
- §10.5: feature importance and score decomposition for the classical models. Linear SVM's wrapper-cluster errors are decided by the intercept (class prior), Logistic Regression's by word evidence; the bigram-breaking explanation is a hypothesis; string length/punctuation features carry top weights.
- §10.5: wrapper-removal counterfactual. Deleting `PREAUTHORIZED WITHDRAWAL` restores 0.94-0.98 accuracy in all five classical models (merchant only: LR, SVM, NB 1.000), so the failure is the unseen surface form, not missing merchant knowledge. The earlier bigram hypothesis is **not supported** (removed).
- Added §10.6 (per-class recall table for all eight models, heatmaps in `reports/confusion_matrices/all_models/`; feature importance for Naive Bayes and Random Forest). **Corrected §10.5:** Random Forest does not scatter the Zelle rows; it has the highest Income recall (0.919). §10.2 now notes that Income recall is 0.53-0.56 for the classical linear models (Zelle plus Income→Utilities), not 0.70.
- §10.6: Income→Utilities errors of LR/SVM/NB are Maternity Leave, Paternity Leave and Social Security templates (all labelled Income, none starting with the same first three tokens in training); DistilBERT gets 99-100% of them right. Together with Zelle they explain about 90% of LR's Income errors.
- §10.6: confirmed by substring probes that Maternity Leave, Paternity Leave and Social Security have zero training rows (OOD: 106, 101, 82 rows, all Income); added per-model predictions. Corrected a wrong cross-reference to §10.5. Note: re-running `error_analysis_examples.py` rewrites the example CSVs and blanks their verdict columns.
- §10.5: transformer wrapper test (DistilBERT 0.832 to 1.000, MiniLM 0.783 to 0.989, frozen 0.597 to 0.711/0.814) and occlusion saliency for DistilBERT and MiniLM (the wrapper phrase overrides the merchant on the wrong rows). §0: error analysis marked complete, with limitations listed.
- Added §8.1 (accuracy, macro precision/recall/F1 and per-category F1 for all eight models on both test sets, from `src/eval/results_table.py`). Linear SVM uses its Platt-calibrated predictions there (0.8543, against 0.8514 raw in §8).
- Added §13 (demonstrator: FastAPI, web UI, Docker, 34 passing tests) and §14 (reproducibility: what is git-ignored, what ran on Colab versus the laptop, clean-up items). Updated §0 to 7 November 2026 (phases 4-5; demo and error analysis done), recorded the dataset licences in §1 (both MIT; DoDataThings tagged Synthetic) and added a synthetic-data limitation. Merged and checked against all earlier draft copies of the log: nothing in them was missing from this version.
- Added §11.1: one combined training-and-inference cost table (parameters, hardware, training time, size, latency, memory, OOD macro F1) for all eight models, for the report.
- Added §4.2: the planned follow-up to the German probe (same 10 strings on DistilBERT, fine-tuned MiniLM, frozen MiniLM + LogReg and Logistic Regression v2, plus a gibberish control, `src/eval/german_probe.py`). Multilingual pretraining did not help (MiniLM 4/10 = DistilBERT 4/10); the Transportation-fallback hypothesis is only partly supported; no model predicts Other for meaningless strings. Corrected the §4.1 count to 5 of 6 wrong predictions being Transportation.