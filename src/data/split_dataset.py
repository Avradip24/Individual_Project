"""
SmartSpend — Step 3b: split the merged dataset into train/val/test plus a
held-out unseen-merchant OOD partition.

Approach:
  1. Normalize each transaction_description into a rough "merchant key"
     (strip debit/credit tags, digits, store codes, punctuation noise).
  2. Fuzzy-cluster the *unique* merchant keys (much smaller than the full
     row count) so near-duplicate strings like "REWE SAGT 7342" and
     "REWE MARKET 123" collapse into the same merchant cluster.
  3. Hold out a random subset of merchant clusters entirely -> OOD test set.
     None of their rows appear anywhere in train/val/test.
  4. Split the remaining (non-OOD) rows into train/val/test, stratified by
     final_category.

Requires: uv add rapidfuzz
"""

import re
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz
from sklearn.model_selection import train_test_split

INTERIM_DIR = Path("data/interim")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
OOD_MERCHANT_FRACTION = 0.15   # ~15% of unique merchant clusters held out
FUZZY_THRESHOLD = 85           # rapidfuzz ratio, 0-100; higher = stricter grouping

# ---------------------------------------------------------------------------
# 1. Load merged, remapped, shuffled dataset
# ---------------------------------------------------------------------------
df = pd.read_csv(INTERIM_DIR / "merged_remapped.csv")
print(f"Loaded {len(df)} rows")

# ---------------------------------------------------------------------------
# 2. Normalize into a rough merchant key
# ---------------------------------------------------------------------------

# Known payment-processor / intermediary prefixes that would otherwise
# swallow the real merchant name into a single meaningless cluster
# (e.g. "PAYPAL INST XFER NATIONAL CAR RENTAL..." was clustering as
# just "PAYPAL INST" regardless of the actual merchant).
PROCESSOR_PREFIXES = [
    "PAYPAL INST XFER",
    "PYPL INST XFER",
    "PYPL",
    "VENMO PAYMENT TO",     # more specific — must come before plain "VENMO"
    "VENMO",
    "CASH APP PAYMENT TO",  # more specific — must come before plain "CASH APP"
    "CASH APP",
    "SQ ",
    "APPLE PAY",
    "GOOGLE PAY",
    "PREAPPROVED PAYMENT BILL USER PAYMENT:",
    "SUBSCRIPTION PAYMENT:",
    "EXPRESS CHECKOUT PAYMENT:",
    "WITHDRAWAL FROM",
    "PAYMENT TO:",
]


def normalize_merchant_key(desc: str) -> str:
    s = desc.upper()
    # collapse stray newlines in source data
    s = re.sub(r"[\r\n]+", " ", s)
    # strip [debit]/[credit] tag
    s = re.sub(r"^\[(DEBIT|CREDIT)\]\s*", "", s)

    # Special pattern: "ORIG CO NAME:X CO ENTRY DESCR:..." -> extract X
    # (the real merchant/employer sits BETWEEN two markers here, not just
    # after a simple prefix, so it needs its own regex rather than a
    # startswith() check).
    orig_co_match = re.search(r"ORIG CO NAME:(.*?)CO ENTRY DESCR", s)
    if orig_co_match:
        s = orig_co_match.group(1).strip()
    else:
        # skip past a known processor prefix to reach the real merchant
        for prefix in PROCESSOR_PREFIXES:
            if s.startswith(prefix):
                s = s[len(prefix):].strip()
                break

    # strip trailing web-id noise
    s = re.sub(r"WEB ID.*$", "", s)
    # strip trailing ppd-id noise
    s = re.sub(r"PPD ID.*$", "", s)
    # strip all digits (store#, codes)
    s = re.sub(r"\d+", "", s)
    s = re.sub(r"[^A-Z\s&]", " ", s)                    # strip punctuation
    s = re.sub(r"\s+", " ", s).strip()
    # keep just the first 2 tokens as the candidate merchant key —
    # good enough for "REWE SAG" / "REWE MARKET" style variants
    tokens = s.split(" ")
    return " ".join(tokens[:2])


df["merchant_key"] = df["transaction_description"].apply(
    normalize_merchant_key)
unique_keys = df["merchant_key"].unique().tolist()
print(f"{len(unique_keys)} unique merchant keys before fuzzy clustering")

# ---------------------------------------------------------------------------
# 3. Fuzzy-cluster the unique keys
# ---------------------------------------------------------------------------
cluster_id_of = {}
next_cluster_id = 0

for key in unique_keys:
    if key in cluster_id_of:
        continue
    # start a new cluster with this key
    cluster_id_of[key] = next_cluster_id
    for other in unique_keys:
        if other in cluster_id_of:
            continue
        if fuzz.ratio(key, other) >= FUZZY_THRESHOLD:
            cluster_id_of[other] = next_cluster_id
    next_cluster_id += 1

print(f"Collapsed into {next_cluster_id} merchant clusters")
df["merchant_cluster"] = df["merchant_key"].map(cluster_id_of)

# ---------------------------------------------------------------------------
# 4. Hold out a random subset of merchant clusters entirely -> OOD
# ---------------------------------------------------------------------------
all_clusters = df["merchant_cluster"].unique()
rng = pd.Series(all_clusters).sample(
    frac=OOD_MERCHANT_FRACTION, random_state=RANDOM_SEED
)
ood_clusters = set(rng.tolist())

is_ood = df["merchant_cluster"].isin(ood_clusters)
df_ood = df[is_ood].copy()
df_remaining = df[~is_ood].copy()

print(
    f"\nOOD partition: {len(df_ood)} rows from {len(ood_clusters)} held-out merchant clusters")
print(df_ood["final_category"].value_counts())

# Sanity check: make sure no category was wiped out entirely by the OOD holdout
missing_in_remaining = set(df["final_category"].unique(
)) - set(df_remaining["final_category"].unique())
if missing_in_remaining:
    print(
        f"\nWARNING: these categories have NO rows left after OOD holdout: {missing_in_remaining}")
    print("Consider lowering OOD_MERCHANT_FRACTION or re-running with a different seed.")

# ---------------------------------------------------------------------------
# 5. Split the remaining (in-distribution) rows into train/val/test
# ---------------------------------------------------------------------------
train_df, temp_df = train_test_split(
    df_remaining,
    test_size=0.30,
    stratify=df_remaining["final_category"],
    random_state=RANDOM_SEED,
)
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["final_category"],
    random_state=RANDOM_SEED,
)

print(f"\nTrain: {len(train_df)} | Val: {len(val_df)} | Test (in-dist): {len(test_df)} | OOD: {len(df_ood)}")

# ---------------------------------------------------------------------------
# 6. Save
# ---------------------------------------------------------------------------
train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
val_df.to_csv(PROCESSED_DIR / "val.csv", index=False)
test_df.to_csv(PROCESSED_DIR / "test_in_distribution.csv", index=False)
df_ood.to_csv(PROCESSED_DIR / "test_ood.csv", index=False)

print("\nSaved train.csv, val.csv, test_in_distribution.csv, test_ood.csv to data/processed/")
