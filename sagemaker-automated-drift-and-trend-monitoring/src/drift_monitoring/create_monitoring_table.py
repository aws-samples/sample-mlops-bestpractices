#!/usr/bin/env python3
"""
Create monitoring_responses Iceberg table in Athena.

This table stores monitoring metrics and drift detection results
for long-term trend analysis and governance dashboards.

Kept in sync with src/setup/create_athena_tables.py (authoritative) and
src/drift_monitoring/deploy_monitoring_writer.py's INSERT column list — do not
diverge.

SAFETY
------
By default this script is NON-DESTRUCTIVE: it only issues
``CREATE TABLE IF NOT EXISTS`` and will never touch existing monitoring
metadata. The destructive ``DROP TABLE`` path is opt-in and gated behind the
explicit ``--recreate`` CLI flag (``recreate=True`` function parameter). When
requested, the exact ``database.table`` and S3 location to be dropped are
printed before the drop runs.

All of the database / bucket / location values default to the AUTHORITATIVE
deployment configuration (src/config/config.py — the same source
src/setup/create_athena_tables.py uses), so this standalone script can never
recreate the table in a divergent location or destroy metadata in the wrong
environment. Callers may pass explicit overrides, but the defaults are always
the shared config.
"""

import argparse
import sys
import time
from pathlib import Path

import boto3

# Make `src.*` importable when run as a script (mirror create_athena_tables.py).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.config.config import (  # noqa: E402
    AWS_DEFAULT_REGION,
    ATHENA_DATABASE,
    ATHENA_MONITORING_RESPONSES_TABLE,
    ATHENA_OUTPUT_S3,
    DATA_S3_BUCKET,
    DATA_S3_PREFIX,
)

# S3 prefix the table's data lives under, relative to DATA_S3_BUCKET. Matches
# create_athena_tables.py's _TABLE_S3_PREFIX so both scripts point the table at
# the identical location.
_TABLE_S3_PREFIX = DATA_S3_PREFIX.rstrip("/") if DATA_S3_PREFIX else "fraud-detection"


def _default_table_location(bucket: str) -> str:
    """Authoritative monitoring_responses location: matches create_athena_tables.py."""
    return f"s3://{bucket}/{_TABLE_S3_PREFIX}/{ATHENA_MONITORING_RESPONSES_TABLE}/"


def _wait_for_query(athena, execution_id):
    """Block until an Athena query reaches a terminal state; return the status dict."""
    while True:
        status = athena.get_query_execution(QueryExecutionId=execution_id)
        state = status['QueryExecution']['Status']['State']
        if state in ['SUCCEEDED', 'FAILED', 'CANCELLED']:
            return status, state
        time.sleep(1)


def create_monitoring_table(
    region=None,
    database=None,
    table=None,
    output_location=None,
    table_location=None,
    recreate=False,
):
    """Create the monitoring_responses Iceberg table.

    All configuration defaults are resolved from the authoritative deployment
    config (src/config/config.py). Any argument left as ``None`` falls back to
    that shared config — nothing is hardcoded to a specific environment.

    Args:
        region:          AWS region (default: config AWS_DEFAULT_REGION).
        database:        Athena database (default: config ATHENA_DATABASE).
        table:           Table name (default: config
                         ATHENA_MONITORING_RESPONSES_TABLE).
        output_location: Athena query-results S3 URI (default: config
                         ATHENA_OUTPUT_S3).
        table_location:  Iceberg table data S3 URI (default: derived from the
                         config data bucket + prefix, matching
                         create_athena_tables.py).
        recreate:        DESTRUCTIVE opt-in. When True, DROP the existing table
                         before recreating it. Defaults to False — the default
                         behavior is CREATE TABLE IF NOT EXISTS only.
    """
    # ---- Resolve config (authoritative defaults, explicit overrides win) ----
    region = region or AWS_DEFAULT_REGION
    database = database or ATHENA_DATABASE
    table = table or ATHENA_MONITORING_RESPONSES_TABLE

    # Get AWS account info (used for the info banner only).
    sts = boto3.client('sts', region_name=region)
    account_id = sts.get_caller_identity()['Account']

    bucket = DATA_S3_BUCKET
    if output_location is None:
        output_location = ATHENA_OUTPUT_S3
    if table_location is None:
        table_location = _default_table_location(bucket)

    if not output_location:
        print("  ❌ No Athena output location resolved. Set ATHENA_OUTPUT_S3 / "
              "DATA_S3_BUCKET in config (config.yaml / .env) or pass "
              "--output-location.")
        return False

    print(f"Creating monitoring table {database}.{table}")
    print(f"  Region: {region}")
    print(f"  Account: {account_id}")
    print(f"  Location: {table_location}")
    print(f"  Query results: {output_location}")
    print(f"  Mode: {'RECREATE (destructive)' if recreate else 'CREATE IF NOT EXISTS (safe)'}")
    print("")

    # Create Athena client
    athena = boto3.client('athena', region_name=region)

    # ------------------------------------------------------------------
    # Destructive path — opt-in only, with an explicit confirmation banner.
    # ------------------------------------------------------------------
    if recreate:
        print("╔════════════════════════════════════════════════════════════════════╗")
        print("║  ⚠️  DESTRUCTIVE: --recreate WILL DROP THE EXISTING TABLE          ║")
        print("╚════════════════════════════════════════════════════════════════════╝")
        print(f"  About to DROP: {database}.{table}")
        print(f"  Underlying S3 location: {table_location}")
        print("  All existing monitoring metadata in this table will be lost.")
        print("")
        print("[1/2] Dropping existing table...")
        drop_query = f"DROP TABLE IF EXISTS {database}.{table}"

        response = athena.start_query_execution(
            QueryString=drop_query,
            QueryExecutionContext={'Database': database},
            ResultConfiguration={'OutputLocation': output_location}
        )
        _, state = _wait_for_query(athena, response['QueryExecutionId'])

        if state == 'SUCCEEDED':
            print("  ✓ Old table dropped")
        else:
            print(f"  Note: Drop query state={state}")
        print("")
        print("[2/2] Creating Iceberg table...")
    else:
        print("[1/1] Creating Iceberg table (IF NOT EXISTS)...")

    # KEEP IN SYNC with src/setup/create_athena_tables.py's monitoring_responses
    # DDL (authoritative) and deploy_monitoring_writer.py's INSERT column list.
    # The trailing 10 columns (problem_type ... model_drift_status) are the
    # nullable problem-type-aware additions, appended after created_at. Column
    # names/types must match exactly — do not diverge.
    create_query = f"""
CREATE TABLE IF NOT EXISTS {database}.{table} (
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
LOCATION '{table_location}'
TBLPROPERTIES (
  'table_type' = 'ICEBERG',
  'format' = 'parquet',
  'write_compression' = 'snappy',
  'optimize_rewrite_delete_file_threshold' = '10'
)
"""

    response = athena.start_query_execution(
        QueryString=create_query,
        QueryExecutionContext={'Database': database},
        ResultConfiguration={'OutputLocation': output_location}
    )
    status, state = _wait_for_query(athena, response['QueryExecutionId'])

    if state == 'SUCCEEDED':
        print("  ✓ Table created successfully")
        print("")
        print("╔════════════════════════════════════════════════════════════════════╗")
        print("║  ✅ MONITORING TABLE CREATED                                       ║")
        print("╚════════════════════════════════════════════════════════════════════╝")
        print("")
        print(f"  Database: {database}")
        print(f"  Table: {table}")
        print(f"  Type: Iceberg (ACID-compliant)")
        print(f"  Location: {table_location}")
        print("")
        print("Next steps:")
        print(f"  1. Grant Lake Formation permissions")
        print(f"  2. Query: SELECT * FROM {database}.{table} LIMIT 10")
        print("")
        return True
    else:
        reason = status['QueryExecution']['Status'].get('StateChangeReason', 'Unknown error')
        print(f"  ❌ Table creation failed: {reason}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Create the monitoring_responses Iceberg table in Athena. "
                    "Non-destructive by default (CREATE TABLE IF NOT EXISTS)."
    )
    parser.add_argument(
        "region", nargs="?", default=None,
        help="AWS region (default: config AWS_DEFAULT_REGION)",
    )
    parser.add_argument(
        "--database", default=None,
        help="Athena database (default: config ATHENA_DATABASE)",
    )
    parser.add_argument(
        "--table", default=None,
        help="Table name (default: config ATHENA_MONITORING_RESPONSES_TABLE)",
    )
    parser.add_argument(
        "--output-location", default=None,
        help="Athena query-results S3 URI (default: config ATHENA_OUTPUT_S3)",
    )
    parser.add_argument(
        "--table-location", default=None,
        help="Iceberg table data S3 URI (default: derived from config data "
             "bucket + prefix, matching create_athena_tables.py)",
    )
    parser.add_argument(
        "--recreate", "--drop", dest="recreate", action="store_true",
        help="DESTRUCTIVE: DROP the existing table before recreating it. "
             "Off by default — omit this flag to run non-destructively.",
    )
    args = parser.parse_args()

    success = create_monitoring_table(
        region=args.region,
        database=args.database,
        table=args.table,
        output_location=args.output_location,
        table_location=args.table_location,
        recreate=args.recreate,
    )
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
