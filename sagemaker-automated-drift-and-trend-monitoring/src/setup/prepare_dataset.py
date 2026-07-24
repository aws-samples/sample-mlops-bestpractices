#!/usr/bin/env python3
"""
Generic dataset preprocessor — transforms a raw CSV into the pipeline-ready
format defined by dataset_schema.yaml, driven entirely by config.yaml.

This script handles:
  - Column renaming (raw source names → schema names)
  - Label-encoding categorical columns to numeric
  - Generating a synthetic identifier column (if not present in source)
  - Generating a synthetic timestamp column (if not present in source)
  - Generating synthetic prediction/probability auxiliary columns
  - Target column transformation (string → boolean mapping)
  - Reordering columns to match schema.csv_column_order()

All behaviour is driven by config.yaml's `preprocessing` section. No code
changes are needed to support a new dataset — just update the config.

Usage:
    # Preprocess a raw CSV (reads input path from config or auto-detects)
    python -m src.setup.prepare_dataset

    # Specify input file explicitly
    python -m src.setup.prepare_dataset --input data/raw/bank-additional-full.csv

    # From a notebook:
    from src.setup.prepare_dataset import prepare_dataset
    prepare_dataset()                         # uses config defaults
    prepare_dataset(input_csv="data/raw/my_data.csv")

Requires:
    pip install -e .  (installs pandas, scikit-learn, numpy via pyproject.toml)
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Make `src.config` importable when run as a script.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import schema  # noqa: E402
from src.config.config import (  # noqa: E402
    CSV_TRAINING_DATA,
    RANDOM_STATE,
    _yaml_cfg,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(_h)

# ---------------------------------------------------------------------------
# Load preprocessing config
# ---------------------------------------------------------------------------
_preproc_cfg = _yaml_cfg.get("preprocessing", {}) or {}

# Column renaming: raw_source_name → schema_name
COLUMN_RENAME_MAP: dict[str, str] = _preproc_cfg.get("column_rename_map", {})

# Categorical columns to label-encode (list of schema-side column names,
# i.e. names AFTER renaming)
CATEGORICAL_COLUMNS: list[str] = _preproc_cfg.get("categorical_columns", [])

# Target transformation
_target_cfg = _preproc_cfg.get("target_transform", {}) or {}
TARGET_SOURCE_COLUMN: str = _target_cfg.get("source_column", "")
TARGET_POSITIVE_VALUE: str = _target_cfg.get("positive_value", "")

# Identifier generation
_id_cfg = _preproc_cfg.get("identifier", {}) or {}
ID_PREFIX: str = _id_cfg.get("prefix", "id-")
ID_GENERATE: bool = _id_cfg.get("generate", True)

# Timestamp generation
_ts_cfg = _preproc_cfg.get("timestamp", {}) or {}
TS_GENERATE: bool = _ts_cfg.get("generate", True)
TS_LOOKBACK_DAYS: int = _ts_cfg.get("lookback_days", 730)

# Prediction/probability generation
_pred_cfg = _preproc_cfg.get("predictions", {}) or {}
PRED_GENERATE: bool = _pred_cfg.get("generate", True)
PRED_POSITIVE_HIGH: float = _pred_cfg.get("positive_probability_high", 0.99)
PRED_POSITIVE_LOW: float = _pred_cfg.get("positive_probability_low", 0.50)
PRED_NEGATIVE_HIGH: float = _pred_cfg.get("negative_probability_high", 0.40)
PRED_NEGATIVE_LOW: float = _pred_cfg.get("negative_probability_low", 0.01)

# Raw input file location
RAW_INPUT_CSV: str = _preproc_cfg.get("raw_input_csv", "")

# Output (from dataset_schema.yaml via schema module)
CSV_COLUMN_ORDER = schema.csv_column_order()
IDENTIFIER_COLUMN = schema.identifier_column()
TIMESTAMP_COLUMN = schema.timestamp_column()
TARGET_COLUMN = schema.target_column()
AUX_COLUMNS = [c.name for c in schema.auxiliary_columns()]


# ---------------------------------------------------------------------------
# Core preprocessing logic
# ---------------------------------------------------------------------------
def prepare_dataset(input_csv: str | Path | None = None) -> Path:
    """Preprocess a raw CSV into pipeline-ready format.

    Args:
        input_csv: Path to raw input CSV. If None, reads from config
                   (preprocessing.raw_input_csv).

    Returns:
        Path to the output CSV (same as CSV_TRAINING_DATA from config).
    """
    # Resolve input path
    if input_csv is None:
        if not RAW_INPUT_CSV:
            raise ValueError(
                "No input CSV specified. Either pass --input or set "
                "preprocessing.raw_input_csv in config.yaml"
            )
        input_path = _PROJECT_ROOT / RAW_INPUT_CSV
    else:
        input_path = Path(input_csv)
        if not input_path.is_absolute():
            input_path = _PROJECT_ROOT / input_path

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    logger.info("Reading raw data from %s …", input_path)
    # Auto-detect separator (semicolons common in UCI datasets)
    sep = _detect_separator(input_path)
    df = pd.read_csv(input_path, sep=sep)
    n = len(df)
    logger.info("  %d rows, %d columns (separator: %r)", n, len(df.columns), sep)

    rng = np.random.default_rng(RANDOM_STATE)

    # --- Step 1: Rename columns ---
    if COLUMN_RENAME_MAP:
        logger.info("Renaming %d columns…", len(COLUMN_RENAME_MAP))
        df = df.rename(columns=COLUMN_RENAME_MAP)

    # --- Step 2: Label-encode categorical columns ---
    if CATEGORICAL_COLUMNS:
        logger.info("Label-encoding %d categorical columns…", len(CATEGORICAL_COLUMNS))
        for col in CATEGORICAL_COLUMNS:
            if col not in df.columns:
                logger.warning("  Categorical column %r not found, skipping", col)
                continue
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str)).astype(float)

    # --- Step 3: Convert remaining numeric columns to float ---
    feature_names = schema.feature_names()
    for col in feature_names:
        if col in df.columns and col not in CATEGORICAL_COLUMNS:
            try:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
            except (ValueError, TypeError):
                pass  # leave non-numeric features as-is (e.g. string type in schema)

    # --- Step 4: Transform target column ---
    if TARGET_SOURCE_COLUMN and TARGET_POSITIVE_VALUE:
        logger.info(
            "Transforming target: %r == %r → %r (boolean)",
            TARGET_SOURCE_COLUMN, TARGET_POSITIVE_VALUE, TARGET_COLUMN,
        )
        if TARGET_SOURCE_COLUMN in df.columns:
            df[TARGET_COLUMN] = df[TARGET_SOURCE_COLUMN].astype(str) == TARGET_POSITIVE_VALUE
        else:
            logger.warning("  Target source column %r not found", TARGET_SOURCE_COLUMN)

    # --- Step 5: Generate identifier column ---
    if ID_GENERATE and IDENTIFIER_COLUMN not in df.columns:
        logger.info("Generating identifier column %r (prefix=%r)…", IDENTIFIER_COLUMN, ID_PREFIX)
        df.insert(0, IDENTIFIER_COLUMN, [f"{ID_PREFIX}{i+1:06d}" for i in range(n)])
    elif IDENTIFIER_COLUMN in df.columns:
        # Ensure it's the first column
        col_data = df.pop(IDENTIFIER_COLUMN)
        df.insert(0, IDENTIFIER_COLUMN, col_data)

    # --- Step 6: Generate timestamp column ---
    if TS_GENERATE and TIMESTAMP_COLUMN not in df.columns:
        logger.info("Generating timestamp column %r (lookback=%d days)…", TIMESTAMP_COLUMN, TS_LOOKBACK_DAYS)
        now = datetime.now()
        lookback_seconds = int(timedelta(days=TS_LOOKBACK_DAYS).total_seconds())
        random_offsets = rng.integers(0, lookback_seconds, size=n)
        timestamps = [
            (now - timedelta(seconds=int(offset))).strftime("%Y-%m-%d %H:%M:%S")
            for offset in random_offsets
        ]
        df[TIMESTAMP_COLUMN] = timestamps

    # --- Step 7: Generate prediction/probability auxiliary columns ---
    if PRED_GENERATE and AUX_COLUMNS:
        logger.info("Generating synthetic prediction/probability columns…")
        # Need target column to generate realistic predictions
        if TARGET_COLUMN in df.columns:
            target_bool = df[TARGET_COLUMN].astype(bool).values
            prob = np.where(
                target_bool,
                rng.uniform(PRED_POSITIVE_LOW, PRED_POSITIVE_HIGH, n),
                rng.uniform(PRED_NEGATIVE_LOW, PRED_NEGATIVE_HIGH, n),
            )
        else:
            prob = rng.uniform(0.01, 0.99, n)

        # First aux column = boolean prediction, second = probability
        if len(AUX_COLUMNS) >= 1:
            df[AUX_COLUMNS[0]] = prob > 0.5
        if len(AUX_COLUMNS) >= 2:
            df[AUX_COLUMNS[1]] = np.round(prob, 16)

    # --- Step 8: Select and reorder columns per schema ---
    missing = [c for c in CSV_COLUMN_ORDER if c not in df.columns]
    if missing:
        raise ValueError(
            f"After preprocessing, {len(missing)} columns required by "
            f"dataset_schema.yaml are missing from the data: {missing}"
        )
    df = df[CSV_COLUMN_ORDER]

    # --- Write output ---
    output_path = CSV_TRAINING_DATA
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(
        "✓ Wrote %s (%.1f MB, %d rows, %d columns)",
        output_path, output_path.stat().st_size / 1024**2, n, len(CSV_COLUMN_ORDER),
    )
    return output_path


def _detect_separator(path: Path) -> str:
    """Detect CSV separator by reading the first line."""
    with open(path, "r") as f:
        first_line = f.readline()
    if ";" in first_line and "," not in first_line:
        return ";"
    if "\t" in first_line and first_line.count("\t") > first_line.count(","):
        return "\t"
    return ","


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Preprocess a raw CSV into pipeline-ready format (config-driven)"
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="Path to raw input CSV (overrides preprocessing.raw_input_csv in config)",
    )
    args = parser.parse_args()
    prepare_dataset(input_csv=args.input)


if __name__ == "__main__":
    main()
