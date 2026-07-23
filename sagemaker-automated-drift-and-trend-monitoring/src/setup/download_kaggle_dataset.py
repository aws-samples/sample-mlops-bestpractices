#!/usr/bin/env python3
"""
Download the configured Kaggle dataset, transform it to the project's
schema, and upload it to S3.

The dataset slug, the raw-column-to-schema rename map, and the synthetic
demo columns all come from config.yaml (data_download section) and
dataset_schema.yaml — nothing is hardcoded to the credit-card fraud use
case, so a BYO-dataset user only edits those two YAML files.

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
    CSV_TRAINING_DATA,
    DATA_DOWNLOAD_RANDOM_STATE,
    KAGGLE_DATASET,
    KAGGLE_CSV_FILENAME,
    KAGGLE_COLUMN_MAP,
    SYNTHETIC_GENDER_COLUMN,
    SYNTHETIC_GENDER_CATEGORIES,
    SYNTHETIC_GENDER_WEIGHTS,
)
from src.config import schema  # noqa: E402

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(_h)


# Seed for the synthetic demo columns — from config.yaml data_download.random_state.
RANDOM_STATE = DATA_DOWNLOAD_RANDOM_STATE
# Local CSV path is the config-declared training-data path (config.yaml
# data.csv_training_data). The drift-dataset generators and notebook 2 read
# the same constant, so the filename lives in exactly one place.
LOCAL_CSV = CSV_TRAINING_DATA
_DATA_DIR = LOCAL_CSV.parent
_CSV_FILENAME = LOCAL_CSV.name

# S3 keys the pipeline seed step reads from. The seed step reads
# `data/predictions/data.csv` (a fixed pipeline convention); the archive copy
# keeps the human-readable CSV filename from the config path above.
_PREDICTIONS_KEY = f"{DATA_S3_PREFIX}data/predictions/data.csv"
_ARCHIVE_KEY = f"{DATA_S3_PREFIX}data/{_CSV_FILENAME}"

# Synthetic categorical column config (config.yaml data_download.synthetic_gender_*).
# The raw source has no categorical feature, so one is fabricated to exercise
# the string-feature path. Empty column name disables synthesis entirely.
GENDERS = SYNTHETIC_GENDER_CATEGORIES
GENDER_WEIGHTS = SYNTHETIC_GENDER_WEIGHTS

# Raw source column -> project schema column. The source uses anonymized
# names; this map (config.yaml data_download.kaggle_column_map) relabels them
# with the business-friendly names declared in dataset_schema.yaml. Values are
# unchanged — the rename is purely cosmetic.
#
# Schema-driven column roles (all from dataset_schema.yaml — nothing fraud-
# specific is hardcoded here):
_IDENTIFIER_COLUMN = schema.identifier_column()   # e.g. transaction_id
_TARGET_COLUMN = schema.target_column()           # e.g. is_fraud
# Auxiliary columns are the non-feature carry-through columns (a prior
# prediction + its probability, for audit). We fabricate them below, matching
# each by declared type: boolean aux <- prediction, double aux <- probability.
_AUX_COLUMNS = schema.auxiliary_columns()
_AUX_BOOL_COLUMN = next((c.name for c in _AUX_COLUMNS if c.type == "boolean"), None)
_AUX_PROB_COLUMN = next((c.name for c in _AUX_COLUMNS if c.type == "double"), None)

# Canonical column order for the predictions CSV, driven by dataset_schema.yaml
# via src.config.schema. The pipeline seed step (seed_athena_tables.py) uses
# this exact same order to declare its staging table — both derive it from the
# same source, so they can never drift apart.
CSV_COLUMN_ORDER = schema.csv_column_order()


# ---------------------------------------------------------------------------
# Step 1 — Download + transform Kaggle data into the project's schema
# ---------------------------------------------------------------------------
def download_and_transform() -> Path:
    """Download the configured dataset, rename columns, write local CSV."""
    import kagglehub  # imported lazily so test environments don't need it

    logger.info("Downloading Kaggle dataset (%s)…", KAGGLE_DATASET)
    dataset_path = kagglehub.dataset_download(KAGGLE_DATASET)
    csv_path = Path(dataset_path) / KAGGLE_CSV_FILENAME

    logger.info("Transforming to project schema…")
    df = pd.read_csv(csv_path)
    df = df.rename(columns=KAGGLE_COLUMN_MAP)

    rng = np.random.default_rng(RANDOM_STATE)
    n = len(df)

    # Insert the identifier column (schema.identifier_column()) as a simple
    # monotonic row id.
    df.insert(0, _IDENTIFIER_COLUMN, np.arange(n))

    # Fabricate the auxiliary "prior prediction" columns so the CSV carries a
    # realistic stored prediction + probability for auditability. The synthetic
    # probability correlates with the target so the demo prediction is roughly
    # accurate. Skipped cleanly if the schema declares no such aux columns.
    if _AUX_PROB_COLUMN or _AUX_BOOL_COLUMN:
        target_positive = df[_TARGET_COLUMN].astype(bool).values
        prob = np.where(
            target_positive,
            rng.uniform(0.5, 0.99, n),
            rng.uniform(0.01, 0.25, n),
        )
        if _AUX_PROB_COLUMN:
            df[_AUX_PROB_COLUMN] = np.round(prob, 16)
        if _AUX_BOOL_COLUMN:
            df[_AUX_BOOL_COLUMN] = prob > 0.5

    # Synthesize the categorical demo column, if configured. The raw source has
    # no string feature, so this fabricates one to exercise the categorical
    # path. Disabled when SYNTHETIC_GENDER_COLUMN is empty.
    if SYNTHETIC_GENDER_COLUMN and GENDERS:
        df[SYNTHETIC_GENDER_COLUMN] = rng.choice(GENDERS, size=n, p=GENDER_WEIGHTS)

    df = df[CSV_COLUMN_ORDER]
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(LOCAL_CSV, index=False)
    logger.info("Wrote %s (%.1f MB, %d rows)", LOCAL_CSV, LOCAL_CSV.stat().st_size / 1024**2, n)
    return LOCAL_CSV


# ---------------------------------------------------------------------------
# Step 2 — Upload to S3 (both the canonical archive and the seeding location)
# ---------------------------------------------------------------------------
def upload_to_s3() -> None:
    """Upload the local CSV to two S3 keys under data/.

    The pipeline seed step reads from `predictions/data.csv`; the second copy
    (named after the config CSV filename) is a human-readable archive.
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
