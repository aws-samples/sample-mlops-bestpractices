#!/usr/bin/env python3
"""
Generate synthetic CSV datasets for the fraud detection pipeline.

Produces three files that mirror the structure and statistical properties
of the real datasets so the pipeline can run without the original data:

  - generated_creditcard_predictions_final.csv  (284,807 rows)
  - generated_creditcard_drifted.csv            (5,000 rows)
  - generated_creditcard_ground_truth.csv       (50,000 rows)

Usage:
    python data/generate_datasets.py                # generate all three
    python data/generate_datasets.py --predictions   # only predictions
    python data/generate_datasets.py --drifted       # only drifted
    python data/generate_datasets.py --ground-truth  # only ground truth

After generation, rename (drop the "generated_" prefix) to use them
in place of the originals, or update config.yaml / .env to point at them.
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
RANDOM_STATE = 42

# Row counts matching the real files
N_PREDICTIONS = 284_807
N_DRIFTED = 5_000
N_GROUND_TRUTH = 50_000

# Fraud rate in the original dataset (~0.17%)
FRAUD_RATE = 0.00173

# 30 feature columns (PCA-normalised in the original dataset)
FEATURE_COLUMNS = [
    "transaction_hour",
    "transaction_day_of_week",
    "customer_age",
    "account_age_days",
    "merchant_category_code",
    "distance_from_home_km",
    "distance_from_last_transaction_km",
    "online_transaction",
    "chip_transaction",
    "pin_used",
    "recurring_transaction",
    "international_transaction",
    "high_risk_country",
    "num_transactions_24h",
    "num_transactions_7days",
    "avg_transaction_amount_30days",
    "max_transaction_amount_30days",
    "card_present",
    "address_verification_match",
    "cvv_match",
    "velocity_score",
    "merchant_reputation_score",
    "time_since_last_transaction_min",
    "transaction_type_code",
    "customer_tenure_months",
    "credit_limit",
    "available_credit_ratio",
    "previous_fraud_incidents",
]

GENDERS = ["Male", "Female", "Other"]
GENDER_WEIGHTS = [0.45, 0.45, 0.10]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _generate_features(rng: np.random.Generator, n: int) -> dict:
    """Return a dict of feature arrays drawn from N(0,1), matching the
    PCA-normalised distributions in the original dataset."""
    return {col: rng.standard_normal(n) for col in FEATURE_COLUMNS}


def _generate_predictions_df(rng: np.random.Generator, n: int) -> pd.DataFrame:
    """Build a DataFrame that mirrors creditcard_predictions_final.csv."""
    features = _generate_features(rng, n)

    is_fraud = rng.random(n) < FRAUD_RATE
    fraud_prob = np.where(
        is_fraud,
        rng.uniform(0.5, 0.99, n),
        rng.uniform(0.01, 0.25, n),
    )
    fraud_prediction = fraud_prob > 0.5

    # transaction_amount: lognormal to mimic real dollar amounts (mean ~88)
    transaction_amount = np.round(rng.lognormal(mean=3.5, sigma=1.5, size=n), 2)
    # transaction_timestamp: uniform 0..172_800 (matches original range)
    transaction_timestamp = np.round(rng.uniform(0, 172_800, n), 1)

    gender = rng.choice(GENDERS, size=n, p=GENDER_WEIGHTS)

    df = pd.DataFrame(
        {
            "transaction_id": np.arange(n),
            "transaction_timestamp": transaction_timestamp,
            **features,
            "transaction_amount": transaction_amount,
            "fraud_prediction": fraud_prediction,
            "fraud_probability": np.round(fraud_prob, 16),
            "customer_gender": gender,
            "is_fraud": is_fraud,
        }
    )
    return df


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

def generate_predictions(out_dir: Path) -> Path:
    """Generate generated_creditcard_predictions_final.csv."""
    rng = np.random.default_rng(RANDOM_STATE)
    print(f"Generating predictions ({N_PREDICTIONS:,} rows) ...")
    df = _generate_predictions_df(rng, N_PREDICTIONS)
    path = out_dir / "generated_creditcard_predictions_final.csv"
    df.to_csv(path, index=False)
    print(f"  -> {path}  ({path.stat().st_size / 1024 / 1024:.1f} MB)")
    return path


def generate_drifted(out_dir: Path) -> Path:
    """Generate generated_creditcard_drifted.csv by sampling from a fresh
    predictions set and applying the same drift config used by
    src/drift_monitoring/generate_drift_dataset.py."""
    rng = np.random.default_rng(RANDOM_STATE + 1)
    print(f"Generating drifted dataset ({N_DRIFTED:,} rows) ...")

    df = _generate_predictions_df(rng, N_DRIFTED)

    # Apply drift (mirrors DRIFT_CONFIG in generate_drift_dataset.py)
    drift_rng = np.random.default_rng(123)
    n = len(df)

    # transaction_amount: +40% multiplicative
    df["transaction_amount"] *= drift_rng.uniform(1.26, 1.54, n)
    df["transaction_amount"] = np.round(df["transaction_amount"], 2)

    # transaction_timestamp: +50k additive
    df["transaction_timestamp"] += drift_rng.uniform(45_000, 55_000, n)

    # distance_from_home_km: 2x multiplicative
    df["distance_from_home_km"] *= drift_rng.uniform(1.4, 2.6, n)

    # velocity_score: 1.5x multiplicative
    df["velocity_score"] *= drift_rng.uniform(1.2, 1.8, n)

    # num_transactions_24h: +3 additive
    df["num_transactions_24h"] += drift_rng.uniform(2, 4, n)
    df["num_transactions_24h"] = np.round(df["num_transactions_24h"])

    path = out_dir / "generated_creditcard_drifted.csv"
    df.to_csv(path, index=False)
    print(f"  -> {path}  ({path.stat().st_size / 1024 / 1024:.1f} MB)")
    return path


def generate_ground_truth(out_dir: Path) -> Path:
    """Generate generated_creditcard_ground_truth.csv matching the schema
    of the real ground truth file (50,000 rows, 10 windows of 5,000)."""
    rng = np.random.default_rng(RANDOM_STATE + 2)
    print(f"Generating ground truth ({N_GROUND_TRUTH:,} rows) ...")

    num_windows = 10
    samples_per_window = N_GROUND_TRUTH // num_windows
    rows = []

    base_ts = datetime(2025, 11, 19, 16, 59, 39)

    for window_id in range(1, num_windows + 1):
        window_ts = base_ts + timedelta(hours=window_id)
        features = _generate_features(rng, samples_per_window)

        is_fraud = rng.random(samples_per_window) < FRAUD_RATE
        fraud_prob = np.where(
            is_fraud,
            rng.uniform(0.5, 0.99, samples_per_window),
            rng.uniform(0.01, 0.25, samples_per_window),
        )
        # observed_fraud: mostly matches ground_truth, with some noise
        observed = is_fraud.copy()
        flip_mask = rng.random(samples_per_window) < 0.05
        observed[flip_mask] = ~observed[flip_mask]

        transaction_timestamp = np.round(
            rng.uniform(50_000, 200_000, samples_per_window), 1
        )
        transaction_amount = np.round(
            rng.lognormal(mean=3.5, sigma=1.5, size=samples_per_window), 2
        )

        window_df = pd.DataFrame(
            {
                "transaction_id": [
                    f"TXN_{window_id}_{i:05d}"
                    for i in range(samples_per_window)
                ],
                "prediction_timestamp": str(window_ts),
                "window_id": window_id,
                "transaction_timestamp": transaction_timestamp,
                **features,
                "transaction_amount": transaction_amount,
                "ground_truth_fraud": is_fraud,
                "observed_fraud": observed,
                "fraud_probability": np.round(fraud_prob, 16),
            }
        )
        rows.append(window_df)

    df = pd.concat(rows, ignore_index=True)
    path = out_dir / "generated_creditcard_ground_truth.csv"
    df.to_csv(path, index=False)
    print(f"  -> {path}  ({path.stat().st_size / 1024 / 1024:.1f} MB)")
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic CSV datasets for the fraud detection pipeline."
    )
    parser.add_argument("--predictions", action="store_true", help="Generate predictions CSV only")
    parser.add_argument("--drifted", action="store_true", help="Generate drifted CSV only")
    parser.add_argument("--ground-truth", action="store_true", help="Generate ground truth CSV only")
    args = parser.parse_args()

    out_dir = SCRIPT_DIR
    generate_all = not (args.predictions or args.drifted or args.ground_truth)

    if generate_all or args.predictions:
        generate_predictions(out_dir)
    if generate_all or args.drifted:
        generate_drifted(out_dir)
    if generate_all or args.ground_truth:
        generate_ground_truth(out_dir)

    print("\nDone. To use these in the pipeline, either:")
    print("  1. Rename them (drop 'generated_' prefix), or")
    print("  2. Set CSV_TRAINING_DATA / CSV_GROUND_TRUTH / CSV_DRIFTED_DATA in .env")


if __name__ == "__main__":
    main()
