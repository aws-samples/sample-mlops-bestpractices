"""Unit tests for `download_kaggle_dataset.py`'s column renaming and output
column order.

Covers:

- `KAGGLE_COLUMN_MAP` renames every raw Kaggle `creditcardfraud` column
  (`Time`, `V1`..`V28`, `Amount`, `Class`) to the project's business-friendly
  schema names.
- The transformed output CSV's column order is internally consistent
  (identifier + features + auxiliaries + target).

NOTE: These tests validate download_kaggle_dataset.py in isolation. The
live dataset_schema.yaml defaults to Bank Marketing, so these tests
verify internal consistency of the credit-card downloader rather than
cross-checking against the live schema.

No real network/Kaggle calls are made: `kagglehub.dataset_download` is
monkeypatched to point at a small in-memory-generated temp CSV fixture that
mimics the raw Kaggle file's shape.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.setup import download_kaggle_dataset as dk


# The credit-card schema that download_kaggle_dataset.py expects.
# These are NOT read from dataset_schema.yaml (which defaults to bank
# marketing) — they are the fixed expectations for the kaggle downloader.
EXPECTED_KAGGLE_TARGET = "is_fraud"
EXPECTED_KAGGLE_IDENTIFIER = "transaction_id"
EXPECTED_KAGGLE_TIMESTAMP = "transaction_timestamp"
EXPECTED_KAGGLE_FEATURES = [
    "transaction_hour", "transaction_day_of_week", "transaction_amount",
    "transaction_type_code", "customer_age", "customer_gender",
    "customer_tenure_months", "account_age_days", "distance_from_home_km",
    "distance_from_last_transaction_km", "time_since_last_transaction_min",
    "online_transaction", "international_transaction", "high_risk_country",
    "merchant_category_code", "merchant_reputation_score", "chip_transaction",
    "pin_used", "card_present", "cvv_match", "address_verification_match",
    "num_transactions_24h", "num_transactions_7days",
    "avg_transaction_amount_30days", "max_transaction_amount_30days",
    "velocity_score", "recurring_transaction", "previous_fraud_incidents",
    "credit_limit", "available_credit_ratio",
]
EXPECTED_KAGGLE_AUX_COLUMNS = ["fraud_prediction", "fraud_probability"]


def _raw_kaggle_columns() -> list[str]:
    """The exact raw column names present in the Kaggle creditcardfraud CSV."""
    return ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]


def _make_raw_kaggle_df(n: int = 5) -> pd.DataFrame:
    """A small DataFrame shaped like the real Kaggle creditcardfraud CSV."""
    rng = np.random.default_rng(0)
    data = {"Time": np.arange(n, dtype=float)}
    for i in range(1, 29):
        data[f"V{i}"] = rng.normal(size=n)
    data["Amount"] = rng.uniform(1, 500, size=n)
    data["Class"] = rng.integers(0, 2, size=n)
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# KAGGLE_COLUMN_MAP renames source columns correctly
# ---------------------------------------------------------------------------


def test_kaggle_column_map_keys_are_exactly_the_raw_kaggle_columns():
    """KAGGLE_COLUMN_MAP has one entry per raw Kaggle column, no more, no
    fewer."""
    assert set(dk.KAGGLE_COLUMN_MAP.keys()) == set(_raw_kaggle_columns())


def test_kaggle_column_map_renames_timestamp_and_target_sources():
    """The Kaggle columns that back the timestamp/target roles rename to
    the expected credit-card schema names."""
    assert dk.KAGGLE_COLUMN_MAP["Time"] == EXPECTED_KAGGLE_TIMESTAMP
    assert dk.KAGGLE_COLUMN_MAP["Amount"] == "transaction_amount"
    assert dk.KAGGLE_COLUMN_MAP["Class"] == EXPECTED_KAGGLE_TARGET


def test_kaggle_column_map_renames_v_columns_to_feature_names():
    """Every V1..V28 anonymized column renames to a name that is a known
    credit-card feature."""
    feature_set = set(EXPECTED_KAGGLE_FEATURES)
    v_columns = [f"V{i}" for i in range(1, 29)]
    for v_col in v_columns:
        renamed = dk.KAGGLE_COLUMN_MAP[v_col]
        assert renamed in feature_set, (
            f"{v_col} renamed to {renamed!r}, which is not a known credit-card feature"
        )
    assert dk.KAGGLE_COLUMN_MAP["V14"] == "num_transactions_24h"


def test_applying_column_map_produces_expected_renamed_columns():
    """Renaming an actual raw-shaped DataFrame with KAGGLE_COLUMN_MAP
    produces exactly the mapped column names."""
    raw = _make_raw_kaggle_df()
    renamed = raw.rename(columns=dk.KAGGLE_COLUMN_MAP)

    expected = {dk.KAGGLE_COLUMN_MAP[c] for c in raw.columns}
    assert set(renamed.columns) == expected
    # Values are unchanged by the rename (Time -> transaction_timestamp).
    assert list(renamed[EXPECTED_KAGGLE_TIMESTAMP]) == list(raw["Time"])


# ---------------------------------------------------------------------------
# CSV_COLUMN_ORDER is internally consistent
# ---------------------------------------------------------------------------


def test_csv_column_order_has_expected_structure():
    """CSV_COLUMN_ORDER should be: identifier + features + aux + target.
    This validates the download_kaggle_dataset module's own constant."""
    expected_order = (
        [EXPECTED_KAGGLE_IDENTIFIER]
        + EXPECTED_KAGGLE_FEATURES
        + EXPECTED_KAGGLE_AUX_COLUMNS
        + [EXPECTED_KAGGLE_TARGET]
    )
    assert dk.CSV_COLUMN_ORDER == expected_order


def test_download_and_transform_writes_csv_in_correct_column_order(tmp_path, monkeypatch):
    """End-to-end (no network): download_and_transform() writes a local CSV
    whose header equals the expected credit-card column order, using a fake
    kagglehub module and a small temp-file fixture standing in for the raw
    Kaggle download."""
    raw_df = _make_raw_kaggle_df(n=10)
    kaggle_download_dir = tmp_path / "kaggle_download"
    kaggle_download_dir.mkdir()
    raw_csv_path = kaggle_download_dir / "creditcard.csv"
    raw_df.to_csv(raw_csv_path, index=False)

    # Stand in for `import kagglehub` (imported lazily inside the function
    # under test) so no real network/Kaggle call is ever made.
    fake_kagglehub = types.ModuleType("kagglehub")
    fake_kagglehub.dataset_download = lambda handle: str(kaggle_download_dir)
    monkeypatch.setitem(sys.modules, "kagglehub", fake_kagglehub)

    local_csv_path = tmp_path / "output" / "creditcard_predictions_final.csv"
    monkeypatch.setattr(dk, "_DATA_DIR", local_csv_path.parent)
    monkeypatch.setattr(dk, "LOCAL_CSV", local_csv_path)

    result_path = dk.download_and_transform()

    assert result_path == local_csv_path
    assert local_csv_path.exists()

    written = pd.read_csv(local_csv_path)
    assert list(written.columns) == dk.CSV_COLUMN_ORDER
    assert len(written) == len(raw_df)
