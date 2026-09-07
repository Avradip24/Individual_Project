"""
Remap both raw sources onto the project's 13-category
taxonomy, balance their scale, and write interim files.

Run from the project root (adjust paths if needed):
    uv run python build_interim_dataset.py
"""

from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")
INTERIM_DIR = Path("data/interim")
INTERIM_DIR.mkdir(parents=True, exist_ok=True)

# Cap per-category rows pulled from the much larger Mitulshah source so it
# matches US Bank v2's 4,000/category exactly — a balanced 50/50 blend,
# neither source dominates the merged dataset.
MITULSHAH_PER_CATEGORY_CAP = 4_000
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# 1. US Bank v2 (CSV)
# ---------------------------------------------------------------------------
us_bank_map = {
    "Insurance": "Insurance",
    "Entertainment": "Entertainment",
    "Utilities": "Utilities",
    "Groceries": "Groceries",
    "Shopping": "Shopping",
    "Healthcare": "Healthcare",
    "Transportation": "Transportation",
    "Education": "Education",
    "Restaurants": "Restaurants",
    "Travel": "Travel",
    "Mortgage": "Housing",
    "Rent": "Housing",
    "Income": "Income",
    "Personal Care": "Shopping",
    "Subscription": "Entertainment",
    "Fees": "Other",
    "Transfer": "Other",
}

df1 = pd.read_csv(RAW_DIR / "us_bank_transactions.csv")
df1["final_category"] = df1["category"].map(us_bank_map)
assert df1["final_category"].isna().sum(
) == 0, "Unmapped US Bank category found — check the dict."

df1_out = df1.rename(columns={"description": "transaction_description"})[
    ["transaction_description", "final_category"]
].copy()
df1_out["source"] = "us_bank_v2"
df1_out.to_csv(INTERIM_DIR / "us_bank_remapped.csv", index=False)
print(f"US Bank v2 remapped: {len(df1_out)} rows")
print(df1_out["final_category"].value_counts(), "\n")

# ---------------------------------------------------------------------------
# 2. Mitulshah (Parquet)
# ---------------------------------------------------------------------------
mitulshah_map = {
    "Food & Dining": "Restaurants",
    "Healthcare & Medical": "Healthcare",
    "Shopping & Retail": "Shopping",
    "Utilities & Services": "Utilities",
    "Entertainment & Recreation": "Entertainment",
    "Transportation": "Transportation",
    "Income": "Income",
    # Dropped — no home in the 13-category taxonomy, and keeping them would
    # bloat "Other" far past a reasonable share of the dataset:
    "Government & Legal": None,
    "Financial Services": None,
    "Charity & Donations": None,
}

df2 = pd.read_parquet(RAW_DIR / "mitulshah_transaction_categorization.parquet")
df2["final_category"] = df2["category"].map(mitulshah_map)
dropped = df2["final_category"].isna().sum()
print(f"Mitulshah: dropping {dropped} rows from unmapped categories "
      f"(Government & Legal / Financial Services / Charity & Donations)")

df2 = df2.dropna(subset=["final_category"])

# Subsample per category so this source doesn't dwarf US Bank v2
# Subsample per category so this source doesn't dwarf US Bank v2.
# (Using an explicit loop instead of groupby().apply() — some pandas
# versions silently drop the grouping column when apply() returns the
# group unchanged, which is what caused the KeyError.)
sampled_groups = []
for cat, group in df2.groupby("final_category"):
    n = min(len(group), MITULSHAH_PER_CATEGORY_CAP)
    sampled_groups.append(group.sample(n=n, random_state=RANDOM_SEED))
df2_sampled = pd.concat(sampled_groups, ignore_index=True)

df2_out = df2_sampled[
    ["transaction_description", "final_category", "country"]
].copy()
df2_out["source"] = "mitulshah"
df2_out.to_csv(INTERIM_DIR / "mitulshah_remapped.csv", index=False)
print(f"\nMitulshah remapped + subsampled: {len(df2_out)} rows")
print(df2_out["final_category"].value_counts(), "\n")

# ---------------------------------------------------------------------------
# 3. Merge (raw materials for the train/val/test/OOD split — do that split
#    in a separate step, not here, so this script stays a pure remap step)
# ---------------------------------------------------------------------------
merged = pd.concat(
    [
        df1_out.assign(country=None),
        df2_out,
    ],
    ignore_index=True,
).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
merged.to_csv(INTERIM_DIR / "merged_remapped.csv", index=False)
print(f"Merged total: {len(merged)} rows")
print(merged["final_category"].value_counts())
print("\nBy source:")
print(merged.groupby(["final_category", "source"]
                     ).size().unstack(fill_value=0))
