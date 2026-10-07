"""Tests for the classical v2 feature pipeline."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.classical_features import (  # noqa: E402  # pylint: disable=import-error,wrong-import-position
    HAND_FEATURE_NAMES,
    build_feature_union,
    hand_features_for,
)

TEXTS = [
    "STARBUCKS COFFEE #4471 SEATTLE WA",
    "PAYROLL ACME CORP DIRECT DEP",
    "AMAZON.COM*2K4LM9 AMZN.COM/BILL",
    "WALGREENS PHARMACY 10382",
    "SHELL OIL 5742 HOUSTON TX",
    "APARTMENTS RENT PMT WEB ID 1234567",
]


def test_hand_features_length_matches_names():
    assert len(hand_features_for("ANY TEXT 123")) == len(HAND_FEATURE_NAMES)


def test_hand_features_values():
    row = dict(
        zip(HAND_FEATURE_NAMES, hand_features_for("PAYROLL ACME CORP 1234")))
    assert row["length"] == len("PAYROLL ACME CORP 1234")
    assert row["n_tokens"] == 4
    assert row["has_long_digit_run"] == 1.0
    assert row["cue_payroll"] == 1.0
    assert row["cue_legal_suffix"] == 1.0
    assert row["cue_travel"] == 0.0
    assert row["upper_ratio"] == pytest.approx(1.0)


def test_hand_features_handles_empty_and_none():
    assert all(v == 0.0 for v in hand_features_for(""))
    assert len(hand_features_for(None)) == len(HAND_FEATURE_NAMES)


def test_hand_features_are_non_negative():
    for t in TEXTS:
        assert min(hand_features_for(t)) >= 0.0


@pytest.mark.parametrize("parts", [("word",), ("char",), ("hand",),
                                   ("word", "char"), ("word", "char", "hand")])
def test_union_fit_transform_shapes(parts):
    union = build_feature_union(parts)
    X = union.fit(TEXTS * 3).transform(TEXTS)
    assert X.shape[0] == len(TEXTS)
    assert X.shape[1] > 0
    assert X.min() >= 0.0  # MultinomialNB needs non-negative features


def test_hand_only_union_has_one_column_per_feature():
    X = build_feature_union(("hand",)).fit(TEXTS).transform(TEXTS)
    assert X.shape[1] == len(HAND_FEATURE_NAMES)


def test_union_is_deterministic():
    a = build_feature_union().fit(TEXTS * 3).transform(TEXTS)
    b = build_feature_union().fit(TEXTS * 3).transform(TEXTS)
    assert np.allclose(a.toarray(), b.toarray())


def test_char_features_ignore_store_numbers():
    union = build_feature_union(("char",)).fit(
        ["SHELL OIL 5742", "SHELL OIL 9981"] * 5)
    a, b = union.transform(["SHELL OIL 5742", "SHELL OIL 9981"])
    assert np.allclose(a.toarray(), b.toarray())


def test_invalid_parts_rejected():
    with pytest.raises(ValueError):
        build_feature_union(("bogus",))
    with pytest.raises(ValueError):
        build_feature_union(())
