#!/usr/bin/env python3
"""
Create the Athena database and all Iceberg/external tables, driven
entirely by ``src/config/dataset_schema.yaml`` via ``src.config.schema``.

This is the ONE place table DDL is generated. It replaces the CloudFormation
lifecycle config's inline Python heredoc AND the old
``src.train_pipeline.athena.schema_definitions`` module — there is no
second source of table DDL anywhere else in the project.

Usage:
    # Normal setup (idempotent — skips tables that already exist)
    python -m src.setup.create_athena_tables

    # After editing dataset_schema.yaml: drop + recreate everything
    python -m src.setup.create_athena_tables --force-recreate

    # Just check current state, create nothing
    python -m src.setup.create_athena_tables --verify-only

The CloudFormation lifecycle config calls this script (no inline DDL of
its own) so a Studio Space launch and a manual run produce byte-identical
tables.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Union

import boto3

# Make `src.*` importable when run as a script.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import schema  # noqa: E402
from src.config.config import (  # noqa: E402
    AWS_DEFAULT_REGION,
    DATA_S3_BUCKET,
    DATA_S3_PREFIX,
    ATHENA_DATABASE,
    ATHENA_OUTPUT_S3,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# S3 prefix each table's data lives under, relative to DATA_S3_BUCKET.
# Matches the layout the seed script and drift monitor already expect.
_TABLE_S3_PREFIX = DATA_S3_PREFIX.rstrip("/") if DATA_S3_PREFIX else "fraud-detection"


def _run_athena_query(
    sql: str, *, timeout: int = 300, return_query_id: bool = False
) -> Union[bool, str, None]:
    """Run an Athena query to completion.

    By default returns True/False for success/failure (existing
    behavior, unchanged). When ``return_query_id`` is True, returns the
    query execution ID on success or None on failure/timeout — used by
    callers (e.g. ``_get_row_count``) that need to fetch query results
    afterwards via ``get_query_results``.
    """
    athena = boto3.client("athena", region_name=AWS_DEFAULT_REGION)
    qid = athena.start_query_execution(
        QueryString=sql,
        QueryExecutionContext={"Database": ATHENA_DATABASE},
        ResultConfiguration={"OutputLocation": ATHENA_OUTPUT_S3},
    )["QueryExecutionId"]

    deadline = time.time() + timeout
    while True:
        status = athena.get_query_execution(QueryExecutionId=qid)["QueryExecution"]["Status"]
        state = status["State"]
        if state == "SUCCEEDED":
            return qid if return_query_id else True
        if state in ("FAILED", "CANCELLED"):
            logger.error("Query %s: %s", state, status.get("StateChangeReason", ""))
            return None if return_query_id else False
        if time.time() > deadline:
            logger.error("Query timed out after %ss", timeout)
            return None if return_query_id else False
        time.sleep(2)


def _table_ddls() -> Dict[str, str]:
    """Generate CREATE TABLE DDL for all registered tables from the schema config.

    Column list construction mirrors the design used everywhere else in
    the project: identifier first, then all feature columns (types pulled
    from dataset_schema.yaml), then table-specific metadata columns.
    """
    bucket = DATA_S3_BUCKET
    prefix = _TABLE_S3_PREFIX
    feature_ddl = schema.athena_feature_ddl()
    id_col = schema.identifier_column()
    target_col = schema.target_column()
    target_athena_type = schema.ATHENA_TYPE_MAP[schema.target_type()]
    aux_ddl = ", ".join(
        f"{c.name} {c.athena_type}" for c in schema.auxiliary_columns()
    )
    aux_ddl_fragment = f"{aux_ddl}, " if aux_ddl else ""

    ddls: Dict[str, str] = {}

    ddls["training_data"] = f"""
CREATE TABLE IF NOT EXISTS {ATHENA_DATABASE}.training_data (
    {id_col} STRING,
    {feature_ddl},
    {aux_ddl_fragment}{target_col} {target_athena_type},
    data_version STRING, created_at TIMESTAMP, updated_at TIMESTAMP
)
LOCATION 's3://{bucket}/{prefix}/training_data/'
TBLPROPERTIES ('table_type' = 'ICEBERG', 'format' = 'parquet')
"""

    ddls["evaluation_data"] = f"""
CREATE TABLE IF NOT EXISTS {ATHENA_DATABASE}.evaluation_data (
    {id_col} STRING,
    {feature_ddl},
    {aux_ddl_fragment}{target_col} {target_athena_type},
    data_version STRING, created_at TIMESTAMP, updated_at TIMESTAMP
)
LOCATION 's3://{bucket}/{prefix}/evaluation_data/'
TBLPROPERTIES ('table_type' = 'ICEBERG', 'format' = 'parquet')
"""

    ddls["ground_truth"] = f"""
CREATE TABLE IF NOT EXISTS {ATHENA_DATABASE}.ground_truth (
    {id_col} STRING, prediction_timestamp TIMESTAMP, window_id INT,
    {feature_ddl},
    ground_truth_fraud BOOLEAN, observed_fraud BOOLEAN, fraud_probability DOUBLE,
    data_source STRING, ingestion_timestamp TIMESTAMP, batch_id STRING
)
PARTITIONED BY (day(prediction_timestamp))
LOCATION 's3://{bucket}/{prefix}/ground_truth/'
TBLPROPERTIES ('table_type' = 'ICEBERG', 'format' = 'parquet')
"""

    # prediction/ground_truth are DOUBLE (not INT): regression models produce
    # continuous predictions/targets that INT would truncate, making regression
    # drift meaningless. Binary/multiclass integer labels 0/1/2... still
    # round-trip losslessly through DOUBLE.
    #
    # MIGRATION: existing deployed Iceberg tables need the runbook in docs/
    # (ADD COLUMNS for monitoring_responses; prediction/ground_truth int→double
    # is NOT an Iceberg-safe promotion, so it needs table recreate + backfill).
    # Fresh deployments get this directly.
    ddls["inference_responses"] = f"""
CREATE TABLE IF NOT EXISTS {ATHENA_DATABASE}.inference_responses (
    inference_id STRING, request_timestamp TIMESTAMP, endpoint_name STRING,
    model_version STRING, mlflow_run_id STRING,
    input_features STRING,
    prediction DOUBLE, probability_fraud DOUBLE, probability_non_fraud DOUBLE, confidence_score DOUBLE,
    ground_truth DOUBLE, ground_truth_timestamp TIMESTAMP, ground_truth_source STRING, days_to_ground_truth DOUBLE,
    inference_latency_ms DOUBLE, model_load_time_ms DOUBLE, preprocessing_time_ms DOUBLE,
    {id_col} STRING, transaction_amount DOUBLE, customer_id STRING,
    is_high_confidence BOOLEAN, is_low_confidence BOOLEAN, prediction_bucket STRING,
    request_id STRING, response_time TIMESTAMP, error_message STRING, inference_mode STRING,
    monitoring_run_id STRING
)
-- NOTE: `monitoring_run_id` above is LEGACY (superseded by the
-- monitoring_run_inferences bridge table). It is retained for backward
-- compatibility with existing data but is no longer written by the drift
-- Lambda, and consumers (e.g. the governance dashboard) must join cohort
-- membership through the bridge, which records EXACT per-check membership.
PARTITIONED BY (day(request_timestamp), endpoint_name)
LOCATION 's3://{bucket}/{prefix}/inference_responses/'
TBLPROPERTIES ('table_type' = 'ICEBERG', 'format' = 'parquet')
"""

    ddls["drifted_data"] = f"""
CREATE EXTERNAL TABLE IF NOT EXISTS {ATHENA_DATABASE}.drifted_data (
    {id_col} STRING,
    {feature_ddl},
    {target_col} {target_athena_type}
)
STORED AS PARQUET
LOCATION 's3://{bucket}/{prefix}/drifted_data/'
TBLPROPERTIES ('parquet.compression'='SNAPPY')
"""

    ddls["ground_truth_updates"] = f"""
CREATE TABLE IF NOT EXISTS {ATHENA_DATABASE}.ground_truth_updates (
    {id_col} STRING, inference_id STRING,
    actual_fraud BOOLEAN, confirmation_timestamp TIMESTAMP, confirmation_source STRING,
    transaction_timestamp TIMESTAMP, prediction_timestamp TIMESTAMP,
    days_since_transaction DOUBLE, days_since_prediction DOUBLE,
    investigation_notes STRING, investigation_priority STRING,
    false_positive BOOLEAN, false_negative BOOLEAN,
    window_id INT,
    batch_id STRING, created_at TIMESTAMP, updated_at TIMESTAMP
)
PARTITIONED BY (day(confirmation_timestamp))
LOCATION 's3://{bucket}/{prefix}/ground_truth_updates/'
TBLPROPERTIES ('table_type' = 'ICEBERG', 'format' = 'parquet')
"""

    # model_package_arn / evaluation_snapshot_id / per_feature_drift_scores
    # are drift-monitor internals, unrelated to the dataset schema — kept
    # as literal columns, not schema-driven.
    #
    # ⚠️ KEEP IN SYNC: src/drift_monitoring/deploy_monitoring_writer.py has
    # a hardcoded `columns = [...]` list in its embedded Lambda code that
    # MUST mirror this DDL. Adding/removing columns requires editing BOTH
    # places. Order matters in the writer's parameterized INSERT.
    # The trailing 10 columns (problem_type ... model_drift_status) are nullable
    # and appended AFTER created_at so existing rows/positional appends stay
    # safe. They make monitoring_responses problem-type-aware: the classification
    # columns above (roc_auc_*, precision/recall/f1) only apply to binary runs,
    # while these carry multiclass/regression results (primary_metric +
    # baseline_primary/current_primary, mae/rmse/r2) and an explicit
    # execution_status for per-run success/skip/error reporting. The two
    # per-check statuses (data_drift_status, model_drift_status) record whether
    # each check ran (SUCCESS / NO_DATA / INSUFFICIENT_* / null-if-not-executed)
    # so a PARTIAL_SUCCESS run is distinguishable from a fully-successful one.
    #
    # MIGRATION: existing deployed Iceberg tables need the runbook in docs/
    # (ADD COLUMNS for monitoring_responses; prediction/ground_truth int→double
    # is NOT an Iceberg-safe promotion, so it needs table recreate + backfill).
    # Fresh deployments get this directly.
    ddls["monitoring_responses"] = f"""
CREATE TABLE IF NOT EXISTS {ATHENA_DATABASE}.monitoring_responses (
    monitoring_run_id STRING, monitoring_timestamp TIMESTAMP,
    endpoint_name STRING, model_version STRING, model_package_arn STRING,
    evaluation_snapshot_id STRING, training_snapshot_id STRING,
    data_drift_detected BOOLEAN, drifted_columns_count INT, drifted_columns_share DOUBLE,
    features_analyzed INT, data_sample_size INT, model_drift_detected BOOLEAN,
    baseline_roc_auc DOUBLE, current_roc_auc DOUBLE,
    roc_auc_degradation DOUBLE, roc_auc_degradation_pct DOUBLE,
    accuracy DOUBLE, precision DOUBLE, recall DOUBLE, f1_score DOUBLE,
    model_sample_size INT, per_feature_drift_scores STRING,
    evidently_report_s3_path STRING, mlflow_run_id STRING,
    alert_sent BOOLEAN, detection_engine STRING, created_at TIMESTAMP,
    problem_type STRING, primary_metric STRING,
    baseline_primary DOUBLE, current_primary DOUBLE,
    mae DOUBLE, rmse DOUBLE, r2 DOUBLE,
    execution_status STRING,
    data_drift_status STRING, model_drift_status STRING
)
PARTITIONED BY (day(monitoring_timestamp))
LOCATION 's3://{bucket}/{prefix}/monitoring_responses/'
TBLPROPERTIES ('table_type' = 'ICEBERG', 'format' = 'parquet')
"""

    # Run→inference membership bridge. Records the EXACT set of inference rows a
    # given monitoring run scored, per check, so cohort membership is precise
    # and reconstructable — not approximated by a timestamp window or a single
    # mutable foreign key on inference_responses.
    #
    # WHY a bridge instead of inference_responses.monitoring_run_id:
    #   * A run RE-SCORES an overlapping look-back window each time, drops
    #     malformed/unparseable rows, and caps at a LIMIT — so "rows in
    #     [lower, upper]" ≠ "rows actually scored". Only the exact ids do.
    #   * Data drift (request-time cohort) and model quality (ground-truth-
    #     arrival-time cohort) score DIFFERENT, overlapping row sets in the same
    #     run. One scalar FK per inference row cannot represent both; a bridge
    #     row per (run, inference, check_type) can.
    # check_type ∈ {'data_drift','model_drift'}. Written idempotently by the
    # drift Lambda (delete-by-run_id then insert) so a retry of the same
    # (now stable) monitoring_run_id re-materializes identical membership.
    ddls["monitoring_run_inferences"] = f"""
CREATE TABLE IF NOT EXISTS {ATHENA_DATABASE}.monitoring_run_inferences (
    monitoring_run_id STRING, inference_id STRING, check_type STRING,
    endpoint_name STRING, created_at TIMESTAMP
)
PARTITIONED BY (day(created_at))
LOCATION 's3://{bucket}/{prefix}/monitoring_run_inferences/'
TBLPROPERTIES ('table_type' = 'ICEBERG', 'format' = 'parquet')
"""

    # ------------------------------------------------------------------
    # monitoring_run_generations — atomic membership-completion markers
    # ------------------------------------------------------------------
    # One marker row per (monitoring_run_id, check_type). The drift Lambda
    # writes it ONLY after every monitoring_run_inferences row for that cohort
    # is durably inserted AND count-verified (see _write_generation_marker).
    # A consumer joins the bridge THROUGH a COMPLETE marker whose
    # expected_count == actual_count, so a crash mid-insert (partial bridge, no
    # marker) is never read as a whole cohort. Iceberg MERGE keeps it idempotent
    # across retries of the same (stable) run_id.
    ddls["monitoring_run_generations"] = f"""
CREATE TABLE IF NOT EXISTS {ATHENA_DATABASE}.monitoring_run_generations (
    monitoring_run_id STRING, check_type STRING,
    expected_count INT, actual_count INT,
    status STRING, created_at TIMESTAMP
)
PARTITIONED BY (day(created_at))
LOCATION 's3://{bucket}/{prefix}/monitoring_run_generations/'
TBLPROPERTIES ('table_type' = 'ICEBERG', 'format' = 'parquet')
"""

    # ------------------------------------------------------------------
    # monitoring_alerts — durable notification outbox (exactly-once-ish)
    # ------------------------------------------------------------------
    # One row per (monitoring_run_id, alert_type). The drift Lambda stakes a
    # PENDING claim before publishing to SNS, then flips it to SENT with the
    # returned MessageId (see notify_drift_alert). A retry of the same run_id
    # that sees SENT skips re-publishing, so an async EventBridge redelivery
    # doesn't duplicate a drift alert. Iceberg MERGE keeps the claim idempotent.
    ddls["monitoring_alerts"] = f"""
CREATE TABLE IF NOT EXISTS {ATHENA_DATABASE}.monitoring_alerts (
    monitoring_run_id STRING, alert_type STRING,
    subject STRING, message STRING,
    status STRING, sns_message_id STRING,
    created_at TIMESTAMP, sent_at TIMESTAMP
)
PARTITIONED BY (day(created_at))
LOCATION 's3://{bucket}/{prefix}/monitoring_alerts/'
TBLPROPERTIES ('table_type' = 'ICEBERG', 'format' = 'parquet')
"""

    return ddls


# KEEP IN SYNC: mirrors cloudformation/deploy-stack.sh's RESETTABLE_TABLES
# array. That script's --recreate-database flag drops the Athena database
# and clears these same tables' S3 data BEFORE a CFN deploy runs — it
# never calls this script directly. This script owns table CREATION;
# deploy-stack.sh owns the pre-deploy wipe. Keep both lists' table names
# identical.
ALL_TABLE_NAMES: List[str] = [
    "training_data", "evaluation_data", "ground_truth",
    "inference_responses", "drifted_data", "ground_truth_updates",
    "monitoring_responses", "monitoring_run_inferences",
    "monitoring_run_generations", "monitoring_alerts",
]

# KEEP IN SYNC: mirrors src/train_pipeline/athena/schema_definitions.py's
# get_iceberg_tables()/get_partitioned_tables() (that module is being
# retired — see .kiro/specs/config-driven-dataset-schema).
ICEBERG_TABLES: List[str] = [
    'training_data', 'evaluation_data', 'ground_truth',
    'inference_responses', 'ground_truth_updates', 'monitoring_responses',
    'monitoring_run_inferences', 'monitoring_run_generations', 'monitoring_alerts',
]
PARTITIONED_TABLES: List[str] = [
    'ground_truth', 'inference_responses', 'ground_truth_updates', 'monitoring_responses',
    'monitoring_run_inferences', 'monitoring_run_generations', 'monitoring_alerts',
]


def get_iceberg_tables() -> List[str]:
    """Iceberg-format tables (excludes the external `drifted_data` table)."""
    return ICEBERG_TABLES


def get_partitioned_tables() -> List[str]:
    """Tables declared with a PARTITIONED BY clause."""
    return PARTITIONED_TABLES


def create_s3_bucket() -> bool:
    """Create the data bucket if it doesn't exist. Idempotent."""
    s3 = boto3.client("s3", region_name=AWS_DEFAULT_REGION)
    try:
        s3.head_bucket(Bucket=DATA_S3_BUCKET)
        logger.info("✓ S3 bucket %s already exists", DATA_S3_BUCKET)
        return True
    except Exception:
        pass

    logger.info("Creating S3 bucket: %s", DATA_S3_BUCKET)
    if AWS_DEFAULT_REGION == "us-east-1":
        s3.create_bucket(Bucket=DATA_S3_BUCKET)
    else:
        s3.create_bucket(
            Bucket=DATA_S3_BUCKET,
            CreateBucketConfiguration={"LocationConstraint": AWS_DEFAULT_REGION},
        )
    s3.put_bucket_versioning(
        Bucket=DATA_S3_BUCKET, VersioningConfiguration={"Status": "Enabled"}
    )
    logger.info("✓ S3 bucket %s created", DATA_S3_BUCKET)
    return True


def create_database() -> bool:
    logger.info("Creating database (if not exists): %s", ATHENA_DATABASE)
    return _run_athena_query(f"CREATE DATABASE IF NOT EXISTS {ATHENA_DATABASE}")


def table_exists(glue, table_name: str) -> bool:
    try:
        glue.get_table(DatabaseName=ATHENA_DATABASE, Name=table_name)
        return True
    except glue.exceptions.EntityNotFoundException:
        return False


def drop_all_tables() -> None:
    """DROP TABLE + Glue force-purge every table this script owns.

    Used only under --force-recreate. DROP TABLE alone can leave orphaned
    Iceberg metadata in the Glue catalog (ICEBERG_MISSING_METADATA errors
    on next CREATE) — the Glue delete_table call force-purges the catalog
    entry regardless of Iceberg metadata state.
    """
    glue = boto3.client("glue", region_name=AWS_DEFAULT_REGION)
    for table in ALL_TABLE_NAMES:
        logger.info("Dropping table (if exists): %s", table)
        _run_athena_query(f"DROP TABLE IF EXISTS {ATHENA_DATABASE}.{table}")
        try:
            glue.delete_table(DatabaseName=ATHENA_DATABASE, Name=table)
        except glue.exceptions.EntityNotFoundException:
            pass


def create_all_tables(*, skip_existing: bool = True) -> Dict[str, bool]:
    glue = boto3.client("glue", region_name=AWS_DEFAULT_REGION)
    ddls = _table_ddls()
    results: Dict[str, bool] = {}

    for table_name in ALL_TABLE_NAMES:
        if skip_existing and table_exists(glue, table_name):
            logger.info("Table %s already exists, skipping", table_name)
            results[table_name] = True
            continue
        logger.info("Creating table: %s.%s", ATHENA_DATABASE, table_name)
        results[table_name] = _run_athena_query(ddls[table_name])

    success = sum(1 for v in results.values() if v)
    logger.info("Created/verified %d/%d tables", success, len(results))
    return results


def verify_all_tables() -> Dict[str, dict]:
    glue = boto3.client("glue", region_name=AWS_DEFAULT_REGION)
    results: Dict[str, dict] = {}
    for table_name in ALL_TABLE_NAMES:
        if not table_exists(glue, table_name):
            results[table_name] = {"exists": False}
            continue
        table_info = glue.get_table(DatabaseName=ATHENA_DATABASE, Name=table_name)["Table"]
        results[table_name] = {
            "exists": True,
            "location": table_info.get("StorageDescriptor", {}).get("Location", "unknown"),
            "table_type": table_info.get("TableType", "unknown"),
        }
    return results


def _get_row_count(table_name: str) -> int:
    """Return the row count for a table via Athena.

    Runs COUNT(*) and reads the scalar result back with
    get_query_results.
    """
    count_query = f"SELECT COUNT(*) as row_count FROM {ATHENA_DATABASE}.{table_name}"

    query_execution_id = _run_athena_query(count_query, return_query_id=True)
    if not query_execution_id:
        return 0

    athena = boto3.client("athena", region_name=AWS_DEFAULT_REGION)
    result = athena.get_query_results(QueryExecutionId=query_execution_id)

    # Row 0 is the header (column names); row 1 holds the value.
    rows = result["ResultSet"]["Rows"]
    if len(rows) < 2:
        return 0

    data = rows[1]["Data"]
    if not data or "VarCharValue" not in data[0]:
        return 0

    return int(data[0]["VarCharValue"])


def get_table_stats(table_name: str) -> Dict[str, Any]:
    """Get statistics for a table.

    Uses Athena for the row count and the Glue catalog for table
    metadata (location, table type).
    """
    logger.info("Getting stats for: %s.%s", ATHENA_DATABASE, table_name)

    row_count = _get_row_count(table_name)

    glue = boto3.client("glue", region_name=AWS_DEFAULT_REGION)
    table_metadata = glue.get_table(DatabaseName=ATHENA_DATABASE, Name=table_name).get("Table", {})
    storage = table_metadata.get("StorageDescriptor", {})

    return {
        "table_name": table_name,
        "database": ATHENA_DATABASE,
        "row_count": row_count,
        "location": storage.get("Location", "unknown"),
        "table_type": table_metadata.get("TableType", "unknown"),
        "is_iceberg": table_name in ICEBERG_TABLES,
        "is_partitioned": table_name in PARTITIONED_TABLES,
    }


def optimize_table(table_name: str) -> bool:
    """Optimize an Iceberg table by running compaction (best-effort).

    Runs the Iceberg OPTIMIZE ... REWRITE DATA USING BIN_PACK command.
    Only applies to Iceberg tables; not raised on failure since
    optimization is best-effort maintenance.
    """
    try:
        if table_name not in ICEBERG_TABLES:
            logger.warning("%s is not an Iceberg table, skipping optimization", table_name)
            return False

        logger.info("Optimizing table: %s.%s", ATHENA_DATABASE, table_name)
        sql = f"OPTIMIZE {ATHENA_DATABASE}.{table_name} REWRITE DATA USING BIN_PACK"
        success = _run_athena_query(sql)
        if success:
            logger.info("✓ Table %s optimized successfully", table_name)
        return bool(success)

    except Exception as e:
        logger.error("Error optimizing table %s: %s", table_name, e)
        return False


def vacuum_table(table_name: str, older_than_days: int = 7) -> bool:
    """Remove orphaned files from an Iceberg table (best-effort).

    Runs the Iceberg VACUUM command. Only applies to Iceberg tables; not
    raised on failure since vacuuming is best-effort maintenance.
    """
    try:
        if table_name not in ICEBERG_TABLES:
            logger.warning("%s is not an Iceberg table, skipping vacuum", table_name)
            return False

        logger.info("Vacuuming table: %s.%s", ATHENA_DATABASE, table_name)
        sql = f"""
        VACUUM {ATHENA_DATABASE}.{table_name}
        USING (older_than => TIMESTAMP '{older_than_days} days ago')
        """
        success = _run_athena_query(sql)
        if success:
            logger.info("✓ Table %s vacuumed successfully", table_name)
        return bool(success)

    except Exception as e:
        logger.error("Error vacuuming table %s: %s", table_name, e)
        return False


def expire_snapshots(table_name: str, older_than_days: int = 7) -> bool:
    """Expire old Iceberg snapshots for a table (best-effort).

    Calls the Iceberg expire_snapshots system procedure. Only applies to
    Iceberg tables; not raised on failure since this is best-effort
    maintenance.
    """
    try:
        if table_name not in ICEBERG_TABLES:
            logger.warning("%s is not an Iceberg table, skipping", table_name)
            return False

        logger.info("Expiring snapshots for: %s.%s", ATHENA_DATABASE, table_name)
        sql = f"""
        CALL {ATHENA_DATABASE}.system.expire_snapshots(
            table_name => '{table_name}',
            older_than => TIMESTAMP '{older_than_days} days ago'
        )
        """
        success = _run_athena_query(sql)
        if success:
            logger.info("✓ Snapshots expired for %s", table_name)
        return bool(success)

    except Exception as e:
        logger.error("Error expiring snapshots for %s: %s", table_name, e)
        return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create Athena database + all tables from dataset_schema.yaml"
    )
    parser.add_argument("--verify-only", action="store_true",
                        help="Only report current state; create nothing")
    parser.add_argument("--force-recreate", action="store_true",
                        help="Drop and recreate all tables against the CURRENT "
                             "dataset_schema.yaml. Destructive — all existing rows "
                             "in every table are lost.")
    parser.add_argument("--skip-s3", action="store_true",
                        help="Skip S3 bucket creation")
    args = parser.parse_args()

    print("=" * 80)
    print("ATHENA TABLES SETUP (schema-driven)")
    print("=" * 80)
    print(f"Database:    {ATHENA_DATABASE}")
    print(f"S3 Bucket:   {DATA_S3_BUCKET}")
    print(f"Region:      {AWS_DEFAULT_REGION}")
    print(f"Feature cnt: {len(schema.feature_names())} (from dataset_schema.yaml)")
    print("=" * 80 + "\n")

    if args.verify_only:
        results = verify_all_tables()
        for name, info in results.items():
            status = "✓ EXISTS" if info["exists"] else "✗ MISSING"
            print(f"  {status}  {name}")
        return 0 if all(r["exists"] for r in results.values()) else 1

    if not args.skip_s3:
        create_s3_bucket()

    if not create_database():
        logger.error("Failed to create database, aborting")
        return 1

    if args.force_recreate:
        logger.warning(
            "--force-recreate: dropping ALL %d tables — every existing row "
            "will be lost. Recreating against the current dataset_schema.yaml.",
            len(ALL_TABLE_NAMES),
        )
        drop_all_tables()

    results = create_all_tables(skip_existing=not args.force_recreate)

    verification = verify_all_tables()
    print("\n" + "=" * 80)
    print("VERIFICATION")
    print("=" * 80)
    for name, info in verification.items():
        status = "✓ EXISTS" if info["exists"] else "✗ MISSING"
        print(f"  {status}  {name}")

    all_ok = all(results.values()) and all(r["exists"] for r in verification.values())
    print("\n" + ("✓ SETUP COMPLETE" if all_ok else "✗ SETUP INCOMPLETE"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
