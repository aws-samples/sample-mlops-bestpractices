#!/usr/bin/env python3
"""
Download the UCI Bank Marketing dataset, transform to the project's
schema, and upload it to S3.

The Athena `training_data` table is created (empty) by the CloudFormation
lifecycle script and populated by the SageMaker pipeline's seed step —
this script does NOT touch Athena. Its only job is to make sure the
predictions CSV is sitting in S3 where the pipeline can read it.

Usage:
    # Force a full re-download
    python -m src.setup.download_dataset

    # From the notebook (recommended):
    from src.setup.download_dataset import ensure_training_data_downloaded
    ensure_training_data_downloaded()        # idempotent — skips if S3 already has it
    ensure_training_data_downloaded(force=True)  # always re-download

Requires:
    pip install -e .  (installs boto3, pandas, scikit-learn via pyproject.toml)
"""

from __future__ import annotations

import io
import logging
import sys
import urllib.request
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
from botocore.exceptions import ClientError
from sklearn.preprocessing import LabelEncoder

# Make `src.config.config` importable when this file is run as a script.
# Layout: src/setup/download_dataset.py — three parents to project root.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.config.config import (  # noqa: E402
    AWS_DEFAULT_REGION,
    DATA_S3_BUCKET,
    DATA_S3_PREFIX,
)
from src.config import schema  # noqa: E402

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(_h)


RANDOM_STATE = 42
# Local CSV is written to the project's data/ scratch directory.
_DATA_DIR = _PROJECT_ROOT / "data"
LOCAL_CSV = _DATA_DIR / "bank_marketing_predictions_final.csv"

# S3 keys the pipeline seed step reads from.
_PREDICTIONS_KEY = f"{DATA_S3_PREFIX}data/predictions/data.csv"
_ARCHIVE_KEY = f"{DATA_S3_PREFIX}data/bank_marketing_predictions_final.csv"

# UCI ML Repository download URL for the Bank Marketing dataset.
_UCI_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"

# Categorical columns to label-encode. Maps original CSV column name to the
# output column name (most stay the same; 'default' is renamed to avoid the
# Python keyword conflict).
_CATEGORICAL_COLUMNS = {
    "job": "job",
    "marital": "marital",
    "education": "education",
    "default": "credit_default",
    "housing": "housing",
    "loan": "loan",
    "contact": "contact",
    "month": "month",
    "day_of_week": "day_of_week",
    "poutcome": "poutcome",
}

# Dot-separated columns to rename (Athena/Python friendly names).
_RENAME_COLUMNS = {
    "emp.var.rate": "emp_var_rate",
    "cons.price.idx": "cons_price_idx",
    "cons.conf.idx": "cons_conf_idx",
}

# Canonical column order for the output CSV, driven by dataset_schema.yaml
# via src.config.schema.
CSV_COLUMN_ORDER = schema.csv_column_order()


# ---------------------------------------------------------------------------
# Step 1 — Download + transform UCI Bank Marketing data into the project's
#           schema
# ---------------------------------------------------------------------------
def download_and_transform() -> Path:
    """Download UCI Bank Marketing dataset, label-encode categoricals,
    add synthetic columns, and write local CSV."""
    logger.info("Downloading UCI Bank Marketing dataset from %s …", _UCI_URL)
    response = urllib.request.urlopen(_UCI_URL)  # noqa: S310
    zip_bytes = response.read()

    logger.info("Extracting bank-additional-full.csv from zip archive…")
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        # The zip contains bank-additional/bank-additional-full.csv
        target_name = None
        for name in zf.namelist():
            if name.endswith("bank-additional-full.csv"):
                target_name = name
                break
        if target_name is None:
            raise FileNotFoundError(
                "bank-additional-full.csv not found in the downloaded zip. "
                f"Archive contents: {zf.namelist()}"
            )
        with zf.open(target_name) as csv_file:
            df = pd.read_csv(csv_file, sep=";")

    logger.info("Transforming to project schema (%d rows)…", len(df))
    n = len(df)
    rng = np.random.default_rng(RANDOM_STATE)

    # --- Rename 'default' to 'credit_default' before encoding ---
    df = df.rename(columns={"default": "credit_default"})

    # --- Rename dot-separated columns ---
    df = df.rename(columns=_RENAME_COLUMNS)

    # --- Label-encode categorical columns ---
    # After the rename above, the df columns use the *output* names for
    # 'default' -> 'credit_default'. Build the list of columns to encode
    # using output names.
    cols_to_encode = list(_CATEGORICAL_COLUMNS.values())
    for col in cols_to_encode:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str)).astype(float)

    # --- Convert remaining numeric columns to float ---
    numeric_cols = [
        "age", "duration", "campaign", "pdays", "previous",
        "emp_var_rate", "cons_price_idx", "cons_conf_idx",
        "euribor3m", "nr_employed",
    ]
    for col in numeric_cols:
        df[col] = df[col].astype(float)

    # --- Synthetic client_id column ---
    df.insert(0, "client_id", [f"bm-{i+1:06d}" for i in range(n)])

    # --- Synthetic contact_timestamp column ---
    # Random timestamps within the last 2 years from now.
    now = datetime.now()
    two_years_seconds = int(timedelta(days=730).total_seconds())
    random_offsets = rng.integers(0, two_years_seconds, size=n)
    timestamps = [
        (now - timedelta(seconds=int(offset))).strftime("%Y-%m-%d %H:%M:%S")
        for offset in random_offsets
    ]
    df["contact_timestamp"] = timestamps

    # --- Synthetic prediction and probability_positive columns ---
    # Simulate model outputs: probability_positive is higher when target is
    # 'yes' (subscribed).
    target_bool = (df["y"] == "yes").values
    prob_positive = np.where(
        target_bool,
        rng.uniform(0.5, 0.99, n),
        rng.uniform(0.01, 0.40, n),
    )
    df["prediction"] = prob_positive > 0.5
    df["probability_positive"] = np.round(prob_positive, 16)

    # --- Transform target 'y' (yes/no) to boolean 'subscribed' ---
    df["subscribed"] = df["y"] == "yes"

    # --- Select and order columns per schema.csv_column_order() ---
    df = df[CSV_COLUMN_ORDER]

    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(LOCAL_CSV, index=False)
    logger.info(
        "Wrote %s (%.1f MB, %d rows)",
        LOCAL_CSV,
        LOCAL_CSV.stat().st_size / 1024**2,
        n,
    )
    return LOCAL_CSV


# ---------------------------------------------------------------------------
# Step 2 — Upload to S3 (both the canonical archive and the seeding location)
# ---------------------------------------------------------------------------
def upload_to_s3() -> None:
    """Upload the local CSV to S3.

    The pipeline seed step reads from `predictions/data.csv`; the
    `bank_marketing_predictions_final.csv` copy is a human-readable archive.
    """
    if not DATA_S3_BUCKET:
        raise RuntimeError(
            "DATA_S3_BUCKET is empty — check src/config/config.yaml (project.name) "
            "and your AWS credentials."
        )
    if not LOCAL_CSV.exists():
        raise FileNotFoundError(
            f"{LOCAL_CSV} not found — run download_and_transform() first."
        )

    s3 = boto3.client("s3", region_name=AWS_DEFAULT_REGION)
    for key in (_PREDICTIONS_KEY, _ARCHIVE_KEY):
        logger.info("Uploading to s3://%s/%s …", DATA_S3_BUCKET, key)
        s3.upload_file(str(LOCAL_CSV), DATA_S3_BUCKET, key)


def _s3_object_exists(bucket: str, key: str) -> bool:
    s3 = boto3.client("s3", region_name=AWS_DEFAULT_REGION)
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
            return False
        raise


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def ensure_training_data_downloaded(*, force: bool = False) -> dict:
    """Idempotent. Make sure the predictions CSV exists in S3.

    Skips when both S3 keys are already present unless ``force=True``.
    Athena seeding is handled by the SageMaker pipeline's seed step — this
    function does not touch Athena.
    """
    if not DATA_S3_BUCKET:
        raise RuntimeError(
            "DATA_S3_BUCKET is empty — check src/config/config.yaml and AWS credentials."
        )

    if not force and _s3_object_exists(DATA_S3_BUCKET, _PREDICTIONS_KEY):
        logger.info(
            "✓ s3://%s/%s already present — skipping download.",
            DATA_S3_BUCKET,
            _PREDICTIONS_KEY,
        )
        return {
            "downloaded": False,
            "bucket": DATA_S3_BUCKET,
            "predictions_key": _PREDICTIONS_KEY,
        }

    if not LOCAL_CSV.exists() or force:
        download_and_transform()
    upload_to_s3()

    logger.info(
        "✓ Predictions CSV ready at s3://%s/%s — pipeline seed step will load it into Athena.",
        DATA_S3_BUCKET,
        _PREDICTIONS_KEY,
    )
    return {
        "downloaded": True,
        "bucket": DATA_S3_BUCKET,
        "predictions_key": _PREDICTIONS_KEY,
    }


if __name__ == "__main__":
    ensure_training_data_downloaded(force=True)
