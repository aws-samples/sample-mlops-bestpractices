"""
Generate datasets with time-varying drift patterns for realistic monitoring.

This creates multiple drifted datasets simulating different monitoring runs,
where specific features exhibit unique drift patterns at different times.

Usage:
    python src/drift_monitoring/generate_variable_drift_dataset.py

    This will generate multiple CSV files representing different time periods:
    - data/drifted_data_run1.csv  (baseline - minimal drift)
    - data/drifted_data_run2.csv  (moderate drift, distance_from_home spikes)
    - data/drifted_data_run3.csv  (high credit_limit drift)
    - data/drifted_data_run4.csv  (velocity & transaction spikes)
    - data/drifted_data_run5.csv  (return to normal with residual drift)
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config.config import CSV_TRAINING_DATA, DATA_DIR

# Configuration
ORIGINAL_DATA_PATH = CSV_TRAINING_DATA
NUM_SAMPLES_PER_RUN = 2000
RANDOM_STATE_BASE = 123

# Define drift patterns for different monitoring runs
# Each run simulates a different time period with unique drift characteristics
DRIFT_PATTERNS = {
    "run1": {
        "description": "Baseline - Minimal drift (normal operations)",
        "features": {
            "transaction_amount": {"factor": 1.05, "noise": 0.05},
            "distance_from_home_km": {"factor": 1.1, "noise": 0.1},
        }
    },
    "run2": {
        "description": "Distance spike - Remote/travel transactions increase",
        "features": {
            "transaction_amount": {"factor": 1.1, "noise": 0.1},
            "distance_from_home_km": {"factor": 3.5, "noise": 0.5},  # SPIKE!
            "velocity_score": {"factor": 1.3, "noise": 0.2},
            "num_transactions_24h": {"shift": 2, "noise": 1},
        }
    },
    "run3": {
        "description": "Credit limit anomaly - System changes or data quality issue",
        "features": {
            "transaction_amount": {"factor": 1.15, "noise": 0.1},
            "credit_limit": {"factor": 8.0, "noise": 2.0},  # MAJOR SPIKE!
            "distance_from_home_km": {"factor": 2.0, "noise": 0.3},
            "merchant_category_code": {"shift": 500, "noise": 200},
        }
    },
    "run4": {
        "description": "High velocity period - Increased transaction frequency",
        "features": {
            "transaction_amount": {"factor": 1.4, "noise": 0.2},
            "velocity_score": {"factor": 2.5, "noise": 0.4},  # SPIKE!
            "num_transactions_24h": {"shift": 5, "noise": 2},  # SPIKE!
            "distance_from_home_km": {"factor": 1.5, "noise": 0.2},
            "max_transaction_amount_30days": {"factor": 1.6, "noise": 0.3},
        }
    },
    "run5": {
        "description": "Recovery - Returning to normal with residual drift",
        "features": {
            "transaction_amount": {"factor": 1.2, "noise": 0.15},
            "distance_from_home_km": {"factor": 1.3, "noise": 0.2},
            "velocity_score": {"factor": 1.2, "noise": 0.15},
        }
    },
    "run6": {
        "description": "Account age anomaly - New user cohort or system change",
        "features": {
            "transaction_amount": {"factor": 1.1, "noise": 0.1},
            "account_age_days": {"factor": 0.3, "noise": 0.1},  # Younger accounts
            "distance_from_home_km": {"factor": 1.4, "noise": 0.2},
            "customer_tenure_days": {"factor": 0.4, "noise": 0.15},
        }
    },
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
