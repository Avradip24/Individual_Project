"""
SmartSpend — Classical v2: shared feature pipeline.

Implements the feature extraction the expose specifies for the classical
models: TF-IDF over words AND character n-grams, augmented with
hand-engineered features (string length, digit ratio, presence of merchant
cues). One feature pipeline is fit on the training split only and reused
by every classical model, so differences between models reflect the
algorithm, not the features.

The v1 baselines used word 1-2-grams only; `build_feature_union` can build
any subset ("word", "char", "hand") so the ablation script can show which
part of the new features actually helps.

Import note: scripts add the project root to sys.path and import this as
`src.models.classical_features`. Models pickled with this pipeline need the
same import path when they are loaded again.
"""

import re

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MaxAbsScaler

# Generic merchant / transaction cues. Each group becomes ONE binary feature.
# These are deliberately category-neutral words (legal suffixes, payment
# types, store types), not a hand-written keyword classifier.
CUE_GROUPS = {
    "cue_legal_suffix": ("llc", "inc", "ltd", "gmbh", "corp", "co.", "plc"),
    "cue_payroll": ("payroll", "salary", "direct dep", "wages", "paycheck"),
    "cue_transfer": ("transfer", "xfer", "zelle", "venmo", "paypal", "cash app"),
    "cue_fee": ("fee", "charge", "interest", "overdraft", "penalty"),
    "cue_housing": ("rent", "mortgage", "lease", "property", "apartments"),
    "cue_store": ("store", "market", "mart", "shop", "supermarket", "pharmacy"),
    "cue_food": ("restaurant", "cafe", "coffee", "pizza", "grill", "bar", "kitchen"),
    "cue_travel": ("airlines", "airways", "hotel", "inn", "resort", "air", "rail"),
    "cue_subscription": ("subscription", "membership", "monthly", "recurring"),
}

HAND_FEATURE_NAMES = [
    "length",
    "n_tokens",
    "digit_ratio",
    "upper_ratio",
    "punct_ratio",
    "has_star",
    "has_hash",
    "has_long_digit_run",
    "has_domain",
    "avg_token_len",
] + list(CUE_GROUPS)

_DIGITS = re.compile(r"\d+")
_LONG_DIGITS = re.compile(r"\d{4,}")
_DOMAIN = re.compile(r"\.(com|net|org|de|co|io)\b", re.IGNORECASE)
_PUNCT = re.compile(r"[^\w\s]")


def _contains_cue(text_lower: str, cues) -> bool:
    return any(cue in text_lower for cue in cues)


def hand_features_for(text: str) -> list:
    """Hand-engineered features for one transaction string (all >= 0)."""
    text = "" if text is None else str(text)
    n_chars = max(len(text), 1)
    tokens = text.split()
    n_tokens = len(tokens)
    letters = [c for c in text if c.isalpha()]
    lower = text.lower()

    row = [
        float(len(text)),
        float(n_tokens),
        sum(c.isdigit() for c in text) / n_chars,
        (sum(c.isupper() for c in letters) / len(letters)) if letters else 0.0,
        len(_PUNCT.findall(text)) / n_chars,
        float("*" in text),
        float("#" in text),
        float(bool(_LONG_DIGITS.search(text))),
        float(bool(_DOMAIN.search(text))),
        (sum(len(t) for t in tokens) / n_tokens) if n_tokens else 0.0,
    ]
    row += [float(_contains_cue(lower, cues)) for cues in CUE_GROUPS.values()]
    return row


class HandcraftedFeatures(BaseEstimator, TransformerMixin):
    """Stateless transformer: list/Series of strings -> sparse feature matrix."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = [hand_features_for(t) for t in X]
        return sparse.csr_matrix(np.asarray(rows, dtype=np.float64))

    def get_feature_names_out(self, input_features=None):
        return np.array(HAND_FEATURE_NAMES)


def _char_preprocessor(text: str) -> str:
    """Lowercase and collapse digit runs to '0' so store numbers and
    transaction codes don't create thousands of one-off character n-grams."""
    return _DIGITS.sub("0", str(text).lower())


def build_feature_union(parts=("word", "char", "hand")) -> FeatureUnion:
    """Return an UNFITTED feature pipeline for the requested parts.

    word: v1 settings (word 1-2-grams, min_df=2, 20k features)
    char: char_wb 2-5-grams on digit-normalized text, 60k features
    hand: hand-engineered features, scaled to [0, 1] with MaxAbsScaler so
          they are non-negative (needed by MultinomialNB) and comparable
          in scale to TF-IDF values.
    """
    unknown = set(parts) - {"word", "char", "hand"}
    if unknown or not parts:
        raise ValueError(
            f"parts must be a non-empty subset of word/char/hand, got {parts}")

    transformers = []
    if "word" in parts:
        transformers.append((
            "word",
            TfidfVectorizer(
                analyzer="word", ngram_range=(1, 2), min_df=2,
                max_features=20_000, lowercase=True,
            ),
        ))
    if "char" in parts:
        transformers.append((
            "char",
            TfidfVectorizer(
                analyzer="char_wb", ngram_range=(2, 5), min_df=3,
                max_features=60_000, sublinear_tf=True,
                preprocessor=_char_preprocessor,
            ),
        ))
    if "hand" in parts:
        transformers.append((
            "hand",
            Pipeline([("raw", HandcraftedFeatures()),
                     ("scale", MaxAbsScaler())]),
        ))
    return FeatureUnion(transformers)
