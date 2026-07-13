"""
Generate a drifted dataset for testing inference monitoring.

This script creates a new dataset with intentional feature drift to test
the MLflow inference monitoring system's ability to detect distribution changes.

All drift parameters (factor, noise, shift) are read from src/config/config.yaml
under the 'drift_generation.default_drift' section. No values are hardcoded.

To adjust drift amounts:
1. Edit src/config/config.yaml
2. Run this script to regenerate the drifted CSV
3. Test with the new drift levels

Example:
    python src/drift_monitoring/generate_drift_dataset.py
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config.config import (
    CSV_TRAINING_DATA,
    CSV_DRIFTED_DATA,
    S3_CSV_TRAINING_DATA,
    S3_CSV_DRIFTED_DATA,
    DRIFT_GEN_DEFAULT_CONFIG,
    DRIFT_GEN_NUM_SAMPLES,
    DRIFT_GEN_RANDOM_STATE,
)


def _resolve_source(local_path, s3_path, label: str) -> str:
    """Return the local file if present, else the S3 fallback.

    In a local checkout the CSV lives under data/. In CFN deployments the file
    is only uploaded to S3 (under its original name), so fall back to the
    account-agnostic S3 URI when the local file is absent.
    """
    if local_path and Path(local_path).exists():
        return str(local_path)
    if s3_path:
        print(f"  Local {label} not found, using S3 source: {s3_path}")
        return s3_path
    return str(local_path)


# Configuration (now read from config.yaml)
ORIGINAL_DATA_PATH = _resolve_source(
    CSV_TRAINING_DATA, S3_CSV_TRAINING_DATA, "training data"
)
DRIFTED_DATA_PATH = CSV_DRIFTED_DATA
NUM_SAMPLES = DRIFT_GEN_NUM_SAMPLES
RANDOM_STATE = DRIFT_GEN_RANDOM_STATE

# Drift parameters for key features (read from config.yaml drift_generation.default_drift)
DRIFT_CONFIG = DRIFT_GEN_DEFAULT_CONFIG


def apply_drift(df: pd.DataFrame, feature: str, config: dict) -> pd.DataFrame:
    """Apply drift to a specific feature based on configuration."""
    if feature not in df.columns:
        print(f"  Warning: Feature '{feature}' not found in dataset, skipping")
        return df

    original_values = df[feature].values

    drift_type = config.get("type", "")

    # Determine drift type from config keys if not explicitly set
    if not drift_type:
        if "factor" in config:
            drift_type = "multiplicative"
        elif "shift" in config:
            drift_type = "additive"
        else:
            print(f"  Warning: No drift type or factor/shift found for {feature}, skipping")
            return df

    if drift_type == "multiplicative":
        # Multiplicative drift: value = original * (factor ± noise)
        factor = config.get("factor", 1.0)
        noise = config.get("noise", 0)
        random_factors = np.random.uniform(
            factor - noise * factor,
            factor + noise * factor,
            size=len(df)
        )
        drifted_values = original_values * random_factors

    elif drift_type == "additive":
        # Additive drift: value = original + (shift ± noise)
        shift = config.get("shift", 0)
        noise = config.get("noise", 0)
        random_shifts = np.random.uniform(
            shift - noise,
            shift + noise,
            size=len(df)
        )
        drifted_values = original_values + random_shifts

    else:
        raise ValueError(f"Unknown drift type: {drift_type}")

    # Clip to zero: preserve non-negativity for any feature that was originally non-negative
    if (original_values >= 0).all():
        drifted_values = np.maximum(drifted_values, 0)

    # Preserve integer dtype if the original feature was integer-valued
    if np.issubdtype(df[feature].dtype, np.integer) or (original_values == np.floor(original_values)).all():
        drifted_values = np.round(drifted_values).astype(int)

    df[feature] = drifted_values

    # Print drift statistics
    original_mean = original_values.mean()
    drifted_mean = drifted_values.mean()
    pct_change = ((drifted_mean - original_mean) / original_mean) * 100 if original_mean != 0 else 0

    print(f"  {feature}:")
    print(f"    Original mean: {original_mean:.4f}")
    print(f"    Drifted mean: {drifted_mean:.4f}")
    print(f"    Change: {pct_change:+.2f}%")
    description = config.get("description", feature)
    print(f"    Description: {description}")

    return df


def generate_drifted_dataset():
    """Generate a drifted dataset for testing inference monitoring."""

    print("=" * 80)
    print("GENERATING DRIFTED DATASET")
    print("=" * 80)

    # Load original dataset
    print(f"\nLoading original dataset from: {ORIGINAL_DATA_PATH}")
    df_original = pd.read_csv(ORIGINAL_DATA_PATH)
    print(f"Original dataset shape: {df_original.shape}")

    # Sample random rows
    print(f"\nSampling {NUM_SAMPLES} random rows...")
    np.random.seed(RANDOM_STATE)
    df_drifted = df_original.sample(n=NUM_SAMPLES, random_state=RANDOM_STATE).copy()
    print(f"Sampled dataset shape: {df_drifted.shape}")

    # Apply drift to key features
    print("\nApplying feature drift:")
    print("-" * 80)
    for feature, config in DRIFT_CONFIG.items():
        df_drifted = apply_drift(df_drifted, feature, config)

    # Reset index
    df_drifted = df_drifted.reset_index(drop=True)

    # Save drifted dataset
    print("\n" + "-" * 80)
    print(f"Saving drifted dataset to: {DRIFTED_DATA_PATH}")
    df_drifted.to_csv(DRIFTED_DATA_PATH, index=False)
    print(f"Saved {len(df_drifted)} rows")

    # Summary statistics
    print("\n" + "=" * 80)
    print("DRIFT SUMMARY")
    print("=" * 80)
    print(f"Original dataset: {ORIGINAL_DATA_PATH}")
    print(f"Drifted dataset: {DRIFTED_DATA_PATH}")
    print(f"Number of samples: {NUM_SAMPLES}")
    print(f"\nClass distribution in drifted dataset:")
    if "is_fraud" in df_drifted.columns:
        fraud_counts = df_drifted["is_fraud"].value_counts()
        print(f"  Non-fraud: {fraud_counts.get(False, fraud_counts.get(0, 0))}")
        print(f"  Fraud: {fraud_counts.get(True, fraud_counts.get(1, 0))}")

    print("\nKey feature comparison:")
    print("-" * 80)
    for feature in DRIFT_CONFIG.keys():
        if feature in df_drifted.columns and feature in df_original.columns:
            original_mean = df_original[feature].mean()
            drifted_mean = df_drifted[feature].mean()
            pct_change = ((drifted_mean - original_mean) / original_mean) * 100
            print(f"{feature:30s} Original: {original_mean:10.2f}  Drifted: {drifted_mean:10.2f}  Change: {pct_change:+6.1f}%")

    print("\n" + "=" * 80)
    print("DRIFTED DATASET GENERATION COMPLETED")
    print("=" * 80)
    print(f"\nTo test with the drifted dataset, run:")
    print(f"  python main.py --mode test --endpoint-name <endpoint> --num-samples 100")
    print(f"\nThe test will automatically use data from: {ORIGINAL_DATA_PATH}")
    print(f"To use the drifted data, you'll need to modify test_endpoint.py or")
    print(f"manually specify the test data path.")
    print("=" * 80)


if __name__ == "__main__":
    generate_drifted_dataset()
