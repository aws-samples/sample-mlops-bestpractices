#!/usr/bin/env python3
"""
Download the Kaggle credit-card fraud dataset, transform to the project's
business-friendly schema, and upload it to S3.

The Athena `training_data` table is created (empty) by the CloudFormation
lifecycle script and populated by the SageMaker pipeline's seed step —
this script does NOT touch Athena. Its only job is to make sure the
predictions CSV is sitting in S3 where the pipeline can read it.

Usage:
    # Force a full re-download
    python -m src.setup.download_kaggle_dataset

    # From the notebook (recommended):
    from src.setup.download_kaggle_dataset import ensure_training_data_downloaded
    ensure_training_data_downloaded()        # idempotent — skips if S3 already has it
    ensure_training_data_downloaded(force=True)  # always re-download

Requires:
    pip install -e .  (installs kagglehub, boto3 via pyproject.toml)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
from botocore.exceptions import ClientError

# Make `src.config.config` importable when this file is run as a script.
# Layout: src/setup/download_kaggle_dataset.py — three parents to project root.
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
# Local CSV is written to the project's data/ scratch directory — the same
# location used by the drift-dataset generators (data/creditcard_drifted.csv,
# data/drifted_data_runN.csv) and read by notebook 2.
_DATA_DIR = _PROJECT_ROOT / "data"
LOCAL_CSV = _DATA_DIR / "creditcard_predictions_final.csv"

# S3 keys the pipeline seed step reads from.
_PREDICTIONS_KEY = f"{DATA_S3_PREFIX}data/predictions/data.csv"
_ARCHIVE_KEY = f"{DATA_S3_PREFIX}data/creditcard_predictions_final.csv"

GENDERS = ["Male", "Female", "Other"]
GENDER_WEIGHTS = [0.45, 0.45, 0.10]

# Kaggle V1..V28 are anonymized PCA components. We relabel them with
# business-friendly names; values are unchanged. The rename is purely cosmetic
# — V14 (renamed `num_transactions_24h`) keeps its real predictive power.
KAGGLE_COLUMN_MAP = {
    "Time": "transaction_timestamp",
    "V1": "transaction_hour",
    "V2": "transaction_day_of_week",
    "V3": "customer_age",
    "V4": "account_age_days",
    "V5": "merchant_category_code",
    "V6": "distance_from_home_km",
    "V7": "distance_from_last_transaction_km",
    "V8": "online_transaction",
    "V9": "chip_transaction",
    "V10": "pin_used",
    "V11": "recurring_transaction",
    "V12": "international_transaction",
    "V13": "high_risk_country",
    "V14": "num_transactions_24h",
    "V15": "num_transactions_7days",
    "V16": "avg_transaction_amount_30days",
    "V17": "max_transaction_amount_30days",
    "V18": "card_present",
    "V19": "address_verification_match",
    "V20": "cvv_match",
    "V21": "velocity_score",
    "V22": "merchant_reputation_score",
    "V23": "time_since_last_transaction_min",
    "V24": "transaction_type_code",
    "V25": "customer_tenure_months",
    "V26": "credit_limit",
    "V27": "available_credit_ratio",
    "V28": "previous_fraud_incidents",
    "Amount": "transaction_amount",
    "Class": "is_fraud",
}

# Canonical column order for creditcard_predictions_final.csv, driven by
# dataset_schema.yaml via src.config.schema. The pipeline seed step
# (seed_athena_tables.py) uses this exact same order to declare its
# staging table — both derive it from the same source, so they can never
# drift apart.
CSV_COLUMN_ORDER = schema.csv_column_order()


# ---------------------------------------------------------------------------
# Step 1 — Download + transform Kaggle data into the project's schema
# ---------------------------------------------------------------------------
def download_and_transform() -> Path:
    """Download Kaggle creditcardfraud, rename columns, write local CSV."""
    import kagglehub  # imported lazily so test environments don't need it

    logger.info("Downloading Kaggle credit-card fraud dataset (mlg-ulb/creditcardfraud)…")
    dataset_path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
    csv_path = Path(dataset_path) / "creditcard.csv"

    logger.info("Transforming to project schema…")
    df = pd.read_csv(csv_path)
    df = df.rename(columns=KAGGLE_COLUMN_MAP)

    rng = np.random.default_rng(RANDOM_STATE)
    n = len(df)
    is_fraud = df["is_fraud"].astype(bool).values

    fraud_prob = np.where(
        is_fraud,
        rng.uniform(0.5, 0.99, n),
        rng.uniform(0.01, 0.25, n),
    )

    df.insert(0, "transaction_id", np.arange(n))
    df["fraud_prediction"] = fraud_prob > 0.5
    df["fraud_probability"] = np.round(fraud_prob, 16)
    df["customer_gender"] = rng.choice(GENDERS, size=n, p=GENDER_WEIGHTS)

    df = df[CSV_COLUMN_ORDER]
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(LOCAL_CSV, index=False)
    logger.info("Wrote %s (%.1f MB, %d rows)", LOCAL_CSV, LOCAL_CSV.stat().st_size / 1024**2, n)
    return LOCAL_CSV


# ---------------------------------------------------------------------------
# Step 2 — Upload to S3 (both the canonical archive and the seeding location)
# ---------------------------------------------------------------------------
def upload_to_s3() -> None:
    """Upload the local CSV to s3://.../data/{predictions/data.csv,creditcard_predictions_final.csv}.

    The pipeline seed step reads from `predictions/data.csv`; the
    `creditcard_predictions_final.csv` copy is a human-readable archive.
    """
    if not DATA_S3_BUCKET:
        raise RuntimeError(
            "DATA_S3_BUCKET is empty — check src/config/config.yaml (project.name) "
            "and your AWS credentials."
        )
    if not LOCAL_CSV.exists():
        raise FileNotFoundError(f"{LOCAL_CSV} not found — run download_and_transform() first.")

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
            DATA_S3_BUCKET, _PREDICTIONS_KEY,
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
        DATA_S3_BUCKET, _PREDICTIONS_KEY,
    )
    return {
        "downloaded": True,
        "bucket": DATA_S3_BUCKET,
        "predictions_key": _PREDICTIONS_KEY,
    }


if __name__ == "__main__":
    ensure_training_data_downloaded(force=True)
