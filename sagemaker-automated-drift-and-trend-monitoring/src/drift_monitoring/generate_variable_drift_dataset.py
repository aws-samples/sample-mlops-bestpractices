"""
Generate datasets with time-varying drift patterns for realistic monitoring.

This creates multiple drifted datasets simulating different monitoring runs,
where specific features exhibit unique drift patterns at different times.

All drift patterns (run1-run6) are configured in src/config/config.yaml under
'drift_generation.variable_patterns'. No values are hardcoded.

To adjust drift amounts (e.g., reduce credit_limit spike from 8x to 1.5x):
1. Edit src/config/config.yaml → drift_generation.variable_patterns.run3
2. Run this script to regenerate the drifted CSVs
3. Test with the new drift levels

Usage:
    python src/drift_monitoring/generate_variable_drift_dataset.py

    This will generate multiple CSV files representing different time periods:
    - data/drifted_data_run1.csv  (baseline - minimal drift)
    - data/drifted_data_run2.csv  (moderate drift, distance_from_home spikes)
    - data/drifted_data_run3.csv  (high credit_limit drift - configurable)
    - data/drifted_data_run4.csv  (velocity & transaction spikes)
    - data/drifted_data_run5.csv  (return to normal with residual drift)
    - data/drifted_data_run6.csv  (account age anomaly)
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config.config import (
    CSV_TRAINING_DATA,
    DATA_DIR,
    DRIFT_GEN_VARIABLE_PATTERNS,
    DRIFT_GEN_NUM_SAMPLES_PER_RUN,
    DRIFT_GEN_RANDOM_STATE,
)

# Configuration (now read from config.yaml)
ORIGINAL_DATA_PATH = CSV_TRAINING_DATA
NUM_SAMPLES_PER_RUN = DRIFT_GEN_NUM_SAMPLES_PER_RUN
RANDOM_STATE_BASE = DRIFT_GEN_RANDOM_STATE

# Define drift patterns for different monitoring runs (now read from config.yaml)
# Each run simulates a different time period with unique drift characteristics
DRIFT_PATTERNS = {}

# Add default descriptions for each run
_RUN_DESCRIPTIONS = {
    "run1": "Baseline - Minimal drift (normal operations)",
    "run2": "Distance spike - Remote/travel transactions increase",
    "run3": "Credit limit anomaly - System changes or data quality issue",
    "run4": "High velocity period - Increased transaction frequency",
    "run5": "Recovery - Returning to normal with residual drift",
    "run6": "Account age anomaly - New user cohort or system change"
}

# Build DRIFT_PATTERNS from config
for run_name, features_config in DRIFT_GEN_VARIABLE_PATTERNS.items():
    DRIFT_PATTERNS[run_name] = {
        "description": _RUN_DESCRIPTIONS.get(run_name, f"Drift pattern: {run_name}"),
        "features": features_config
    }


def apply_drift_with_pattern(df: pd.DataFrame, feature: str, config: dict) -> pd.DataFrame:
    """Apply drift to a specific feature based on configuration."""
    if feature not in df.columns:
        print(f"    ⚠️  Feature '{feature}' not found, skipping")
        return df

    original_values = df[feature].values.copy()

    # Multiplicative drift (factor-based)
    if "factor" in config:
        factor = config["factor"]
        noise = config.get("noise", 0)
        random_factors = np.random.uniform(
            max(0.1, factor - noise),  # Ensure positive
            factor + noise,
            size=len(df)
        )
        drifted_values = original_values * random_factors

    # Additive drift (shift-based)
    elif "shift" in config:
        shift = config["shift"]
        noise = config.get("noise", 0)
        random_shifts = np.random.uniform(
            shift - noise,
            shift + noise,
            size=len(df)
        )
        drifted_values = original_values + random_shifts

    else:
        return df

    # Ensure non-negative for certain features
    if feature in ["transaction_amount", "distance_from_home_km", "velocity_score",
                   "num_transactions_24h", "account_age_days", "credit_limit"]:
        drifted_values = np.maximum(drifted_values, 0)

    # Round integer features
    if feature in ["num_transactions_24h", "merchant_category_code",
                   "account_age_days", "customer_tenure_days"]:
        drifted_values = np.round(drifted_values).astype(int)

    df[feature] = drifted_values

    # Print statistics
    original_mean = original_values.mean()
    drifted_mean = drifted_values.mean()
    pct_change = ((drifted_mean - original_mean) / original_mean) * 100 if original_mean != 0 else 0

    indicator = "🔥" if abs(pct_change) > 200 else "📈" if abs(pct_change) > 50 else "📊"
    print(f"    {indicator} {feature}: {original_mean:.2f} → {drifted_mean:.2f} ({pct_change:+.1f}%)")

    return df


def generate_run_dataset(df_original: pd.DataFrame, run_name: str, pattern: dict) -> pd.DataFrame:
    """Generate a drifted dataset for a specific run."""
    print(f"\n{'='*80}")
    print(f"GENERATING RUN: {run_name}")
    print(f"{'='*80}")
    print(f"Description: {pattern['description']}")
    print(f"Features to drift: {len(pattern['features'])}")

    # Sample random rows
    seed = RANDOM_STATE_BASE + int(run_name.replace("run", ""))
    np.random.seed(seed)
    df_run = df_original.sample(n=NUM_SAMPLES_PER_RUN, random_state=seed).copy()

    print(f"\nApplying drift to {len(pattern['features'])} features:")
    print("-" * 80)

    # Apply drift to each feature
    for feature, config in pattern["features"].items():
        df_run = apply_drift_with_pattern(df_run, feature, config)

    return df_run.reset_index(drop=True)


def generate_all_runs():
    """Generate drifted datasets for all runs."""
    print("=" * 80)
    print("VARIABLE DRIFT DATASET GENERATION")
    print("=" * 80)
    print(f"\nGenerating {len(DRIFT_PATTERNS)} monitoring run datasets")
    print(f"Samples per run: {NUM_SAMPLES_PER_RUN}")
    print(f"Output directory: {DATA_DIR}")

    # Load original dataset
    print(f"\nLoading training data: {ORIGINAL_DATA_PATH}")
    df_original = pd.read_csv(ORIGINAL_DATA_PATH)
    print(f"Original dataset shape: {df_original.shape}")
    print(f"Columns: {', '.join(df_original.columns[:10])}{'...' if len(df_original.columns) > 10 else ''}")

    # Generate each run
    output_files = []
    for run_name, pattern in DRIFT_PATTERNS.items():
        df_run = generate_run_dataset(df_original, run_name, pattern)

        # Save to CSV
        output_path = DATA_DIR / f"drifted_data_{run_name}.csv"
        df_run.to_csv(output_path, index=False)
        output_files.append(output_path)
        print(f"✅ Saved: {output_path}")

    # Summary
    print("\n" + "=" * 80)
    print("GENERATION COMPLETE")
    print("=" * 80)
    print(f"\nGenerated {len(output_files)} drift pattern datasets:")
    for i, (run_name, pattern) in enumerate(DRIFT_PATTERNS.items(), 1):
        print(f"  {i}. {run_name}: {pattern['description']}")

    print(f"\n📁 Files saved to: {DATA_DIR}")
    print("\n💡 Next Steps:")
    print("  1. Run batch transform with each dataset to simulate different time periods")
    print("  2. Run drift monitoring after each batch")
    print("  3. View timeline in QuickSight to see varying drift patterns")
    print("\n  Example commands:")
    for run_name in DRIFT_PATTERNS.keys():
        print(f"    # Process {run_name}")
        print(f"    python main.py --mode test --endpoint-name <endpoint> --test-data data/drifted_data_{run_name}.csv")
        print(f"    # Then run drift monitoring")
        print()

    print("=" * 80)


if __name__ == "__main__":
    generate_all_runs()
