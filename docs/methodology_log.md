# SmartSpend — Methodology & Experiment Log

Running record of technical decisions, rationale, and results. Update this
as the project progresses — this is written to be adaptable almost
directly into the report's Methodology and Results chapters.

---

## 1. Data Sources

| Source | Format | Rows used | Categories provided | License note |
|---|---|---|---|---|
| DoDataThings/us-bank-transaction-categories-v2 (Hugging Face) | CSV | 68,000 (4,000/category × 17 raw categories) | Insurance, Mortgage, Entertainment, Utilities, Groceries, Personal Care, Shopping, Subscription, Healthcare, Rent, Fees, Income, Transportation, Education, Restaurants, Transfer, Travel | Check HF dataset card before citing in public repo |
| mitulshah/transaction-categorization (Hugging Face) | Parquet | 28,000 (subsampled to 4,000/category, from ~4.5M available) | Utilities & Services, Government & Legal, Financial Services, Income, Charity & Donations, Shopping & Retail, Healthcare & Medical, Entertainment & Recreation, Transportation, Food & Dining | Check HF dataset card before citing |

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

**Known limitation — no German/multilingual data in the training set.** Both sources are English-language, drawn from US/UK/Australia/India/Canada. A German-language OOD test set is planned separately from the student's own bank export (see §4).

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

## 4. Planned: German / Multilingual OOD Partition

Separate from the merchant-based OOD split above. The student's own bank export (Commerzbank, ~6-12 months) will be:
1. Preprocessed with the same normalization pipeline.
2. Hand-labeled using `docs/labeling_guideline.md`.
3. Saved as `data/ood/test_ood_multilingual.csv` — never touched during training.

This tests a *stronger* form of generalization than the merchant-based OOD set: different language, different bank formatting conventions, different country — automatically "unseen" without needing any merchant-matching logic, since there is zero vocabulary overlap with the training data.

### 4.1 Preliminary Qualitative Probe (fine-tuned DistilBERT, 10 hand-picked German/English strings)

Before the formal hand-labeled German OOD set exists, a quick informal check was run on the fine-tuned DistilBERT model using 10 well-known merchant strings (mix of English controls and German brands), to sanity-check expectations ahead of the real evaluation. **This is illustrative, not a formal metric** — small sample, hand-picked strings, no statistical claims should be drawn from it directly.

**Result: 4/10 correct (40%)** — compare to ~93% OOD macro F1 on unseen-but-English merchants from the same model. Confirms the expected finding: fine-tuning on English-only data does not transfer to a different language, even though the underlying *concepts* (grocery store, pharmacy, utility company) are identical.

**Calibration red flag, consistent with what formal ECE analysis should quantify:** confidence on the 6 wrong predictions was mostly extremely high (100%, 82.9%, 100%, 99.9%, 81.4%) — the model is not just wrong on foreign-language input, it's confidently wrong. The one prediction with meaningfully lower confidence (52.9%) happened to be the only correct German prediction among the ambiguous cases. This is a concrete, real illustration of exactly the calibration failure mode the expose's ECE analysis is designed to catch formally.

**Notable pattern worth investigating further — "Transportation" as an apparent fallback:** 4 of the 6 wrong predictions (REWE, DM Drogerie Markt, DB Fernverkehr, Apotheke am Dom — true categories Groceries/Healthcare/Travel/Healthcare) were all misclassified specifically as **Transportation**, not scattered across categories. Working hypothesis: Transportation's broad, heterogeneous training definition (fuel, tolls, parking, car maintenance, public transit) may make its decision region behave like a "default" for unfamiliar/out-of-distribution input. Not confirmed — would need a follow-up test with pure gibberish/non-language strings to check whether this is a genuine attractor behavior or coincidence from this small sample.

**Planned follow-up (not yet done):** run the same 10 strings through the frozen multilingual embedding model (Approach 1) for direct comparison — that model has actual multilingual pretraining despite losing on every aggregate metric, so it may handle real German text differently (better or worse) than DistilBERT despite the aggregate ranking. Worth doing before the report is finalized, even though deprioritized for now in favor of the formal calibration analysis.

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

**Best classical baseline: Linear SVM, 83.64% OOD macro F1.** This is the number the transformer models need to beat for the central hypothesis to hold.

**Key finding — complexity vs. generalization inversely related here:** Random Forest is strongest in-distribution among the weaker performers but has the *largest* in-distribution-to-OOD gap (23.9 points) — consistent with it overfitting to memorized merchant-specific tokens. Naive Bayes is the *weakest* in-distribution model but the *second-best* on OOD — its simpler independence assumption appears to generalize better precisely because it can't memorize as precisely. Worth developing further in the discussion chapter: model expressiveness on this task correlates with in-distribution performance but not with OOD robustness, and may even trend inversely.

**Not yet done — explicitly tracked, nothing built yet:**
- Hyperparameter tuning (all results above use reasonable defaults, not a validation-set search)
- **Confidence calibration / Expected Calibration Error (ECE) — zero calibration work exists at this point in the project.** Only accuracy and macro F1 have been measured so far, across all models (classical and the embedding-based transformer). This is required by the expose's evaluation protocol and comes after the transformer results are in.
- Confusion matrices
- Per-category / confusable-pair error analysis
- Learning-curve analysis (performance under reduced training data)
- Inference-cost analysis (latency, memory footprint)

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

---

## 8. Full Model Comparison (All Approaches, To Date)

Sorted by OOD Macro F1 — the metric that actually tests the central hypothesis:

| Model | In-dist Macro F1 | **OOD Macro F1** | Relative drop | Training cost |
|---|---|---|---|---|
| **Fine-tuned DistilBERT (6 epochs)** | 0.9957 | **0.9329** | **6.3%** | 23.2 min (GPU) |
| Fine-tuned DistilBERT (3 epochs) | 0.9961 | 0.9283 | 6.8% | 9.9 min (GPU) |
| Linear SVM | 0.9936 | 0.8364 | 15.8% | seconds (CPU) |
| Multinomial NB | 0.9747 | 0.8240 | 15.5% | seconds (CPU) |
| Gradient Boosting (XGBoost) | 0.9916 | 0.7985 | 19.5% | several min (CPU) |
| Logistic Regression | 0.9906 | 0.7982 | 19.4% | seconds (CPU) |
| Frozen multilingual embedding + LogReg | 0.8626 | 0.6623 | 23.2% | ~16 min encoding (CPU) |
| Random Forest | 0.9761 | 0.7374 | 24.4% | 1.5-2 min (CPU) |

**Central hypothesis: confirmed, with an important nuance.** Fine-tuned DistilBERT beats every classical model on OOD by a wide margin (+9.6 points over the best classical model, Linear SVM) and shows by far the smallest relative degradation from in-distribution to OOD. However, this benefit comes specifically from **fine-tuning** — actually updating the transformer's weights — not from transformer architectures generically. The frozen-embedding transformer approach (Approach 1) underperformed every single classical model. The report's discussion should be explicit about this: "transformers" as a category didn't automatically win; adapting a pretrained transformer to the task's specific noisy format is what won.

---

## Changelog
- Initial version — covers data sourcing through classical baseline results, transformer approach 1 in progress.
- Added transformer approach 1 (frozen multilingual embedding) results — underperformed all classical baselines, a genuine negative result.
- Added transformer approach 2 (fine-tuned DistilBERT) results, including a 3-vs-6 epoch ablation, and the full cross-model comparison. Central hypothesis confirmed via fine-tuning specifically, not transformers generically.