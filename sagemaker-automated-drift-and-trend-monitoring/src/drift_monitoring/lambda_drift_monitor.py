"""
Lambda function for automated drift detection and alerting.

Triggered by EventBridge on a schedule (e.g., daily).
Uses Evidently for data drift and model performance analysis.
Sends SNS alerts if thresholds exceeded.
Logs all metrics and Evidently HTML reports to MLflow for tracking.

Configuration via environment variables:
- DATA_DRIFT_LOOKBACK_DAYS: Days of inference data for data drift (default: 7)
- MODEL_DRIFT_LOOKBACK_DAYS: Days of inference data for model drift (default: 30)
- DATA_DRIFT_THRESHOLD: PSI threshold for data drift alerts (default: 0.2)
- MODEL_DRIFT_THRESHOLD: Performance degradation threshold (default: 0.05)
- MIN_SAMPLES: Minimum samples required for analysis (default: 100)

Time-based drift detection ensures fair comparison by using recent inference
data within a configurable time window, rather than all historical data.
"""

import json
import os
import uuid
import boto3
import time
from datetime import datetime, timedelta
import tempfile

import numpy as np
import pandas as pd

# Evidently-based reporting (used by check_data_drift / check_model_drift)
from src.drift_monitoring.evidently_reports import (
    run_data_drift_report,
    run_classification_report,
    run_regression_report,
)
from src.config import schema

# MLflow
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("⚠️ MLflow not available - skipping MLflow logging")

# AWS clients
athena = boto3.client('athena')
s3 = boto3.client('s3')
sns = boto3.client('sns')
sqs = boto3.client('sqs')
sagemaker_client = boto3.client('sagemaker')
cloudwatch = boto3.client('cloudwatch')

# Configuration from environment variables
ATHENA_DATABASE = os.getenv('ATHENA_DATABASE', 'fraud_detection')
ATHENA_OUTPUT_S3 = os.getenv('ATHENA_OUTPUT_S3', 's3://fraud-detection-data-lake/athena-query-results/')
ATHENA_EVALUATION_TABLE = os.getenv('ATHENA_EVALUATION_TABLE', 'evaluation_data')
# Run→inference membership bridge (see create_athena_tables.py). Records the
# EXACT inference rows each run scored per check_type; supersedes the legacy
# inference_responses.monitoring_run_id backfill.
ATHENA_RUN_INFERENCES_TABLE = os.getenv('ATHENA_RUN_INFERENCES_TABLE', 'monitoring_run_inferences')
# Membership generation-completion markers (Finding 4). A run writes ONE COMPLETE
# marker per (monitoring_run_id, check_type), carrying expected/actual row counts,
# ONLY after every bridge row for that cohort is durably inserted AND count-
# verified. Consumers MUST gate cohort trust on a COMPLETE marker with
# expected_count == actual_count — a crash mid-insert leaves the bridge partially
# populated but writes NO marker, so partial membership is never read as complete.
ATHENA_RUN_GENERATIONS_TABLE = os.getenv('ATHENA_RUN_GENERATIONS_TABLE', 'monitoring_run_generations')
# Durable alert outbox (Finding 6). One claim row per (monitoring_run_id,
# alert_type): the alert is INSERTed PENDING, published to SNS, then flipped to
# SENT. A redelivered scheduled tick (same run_id) sees SENT and skips publishing
# (idempotent); a publish failure leaves the claim PENDING and is re-raised so a
# retry re-delivers rather than silently dropping the alert.
ATHENA_ALERTS_TABLE = os.getenv('ATHENA_ALERTS_TABLE', 'monitoring_alerts')
MODEL_PACKAGE_GROUP = os.getenv('MODEL_PACKAGE_GROUP', 'fraud-detection')
SNS_TOPIC_ARN = os.getenv('SNS_TOPIC_ARN')
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI')
MONITORING_SQS_QUEUE_URL = os.getenv('MONITORING_SQS_QUEUE_URL', '')

# Base S3 location for durable per-run Evidently HTML reports. When unset it is
# derived from the Athena output bucket (see _reports_s3_base). A deterministic
# key per run (<base>/<run_id>/<check>.html) lets a persisted monitoring row
# correlate to a durable artifact instead of a random /tmp path that MLflow
# unlinks after logging.
MONITORING_REPORTS_S3 = os.getenv('MONITORING_REPORTS_S3', '')

# CloudWatch namespace for drift metrics published on every scheduled run.
# MUST match the namespace create_cloudwatch_monitoring.py builds its alarms
# and dashboard against (CLOUDWATCH_NAMESPACE there defaults to the same
# value) — otherwise the alarms/dashboard show "no data".
CLOUDWATCH_NAMESPACE = os.getenv('CLOUDWATCH_NAMESPACE', 'FraudDetection/DriftMonitoring')

# Thresholds
DATA_DRIFT_THRESHOLD = float(os.getenv('DATA_DRIFT_THRESHOLD', '0.2'))  # PSI threshold
KS_PVALUE_THRESHOLD = float(os.getenv('KS_PVALUE_THRESHOLD', '0.05'))  # KS p-value threshold
MODEL_DRIFT_THRESHOLD = float(os.getenv('MODEL_DRIFT_THRESHOLD', '0.05'))  # 5% degradation
MIN_SAMPLES = int(os.getenv('MIN_SAMPLES', '100'))  # Minimum samples for analysis

# Lookback periods (from config or environment)
DATA_DRIFT_LOOKBACK_DAYS = int(os.getenv('DATA_DRIFT_LOOKBACK_DAYS', '7'))  # Days of data for drift comparison
MODEL_DRIFT_LOOKBACK_DAYS = int(os.getenv('MODEL_DRIFT_LOOKBACK_DAYS', '30'))  # Days of data for model performance

# Training features — sourced from the configured dataset schema
# (src/config/dataset_schema.yaml) rather than hardcoded here, so drift
# monitoring stays correct after a customer changes their feature set.
TRAINING_FEATURES = schema.feature_names()

# Target and prediction-output column names — schema/config-driven so the
# baseline SELECT and Evidently model-drift comparison work for BYO datasets
# without needing to grep the module for hardcoded `is_fraud`/`probability_fraud`.
# Env-var overrides let scripts/deploy_lambda_container.sh set these from
# config.py at deploy time; hardcoded fallbacks match the fraud-detection
# reference implementation.
TARGET_COLUMN = os.getenv('TARGET_COLUMN', schema.target_column())
PREDICTION_COLUMN = os.getenv('PREDICTION_COLUMN', 'prediction')
PROBABILITY_COLUMN = os.getenv('PROBABILITY_COLUMN', 'probability_fraud')

# ML problem type — drives which metrics / Evidently preset the model-quality
# check uses. ALWAYS resolve through schema.problem_type(): it already reads
# the PROBLEM_TYPE env var FIRST and canonicalizes shorthands/casing
# ('multiclass' → 'multiclass_classification', 'Regression' → 'regression',
# etc.) via its _CANON map, then falls back to the schema/target-type
# inference. Reading os.getenv('PROBLEM_TYPE') directly here would bypass that
# canonicalization — a documented value like PROBLEM_TYPE=multiclass would be
# stored verbatim and never match the 'multiclass_classification' checks
# below, silently running the binary branch. One of:
#   'binary_classification' | 'multiclass_classification' | 'regression'
PROBLEM_TYPE = schema.problem_type()


# =========================================================================
# Cohort anchor time (Finding 3) — retry determinism.
#
# The drift checks used to window on (datetime.now() - lookback). EventBridge
# invokes this function ASYNCHRONOUSLY and retries the SAME scheduled tick on
# failure; a now()-relative window means each retry scores a DIFFERENT cohort
# (later now() → newer rows, shifted lower bound), so "first-writer-wins" gave
# nondeterministic membership across retries. Anchoring both window bounds to
# the tick's immutable `time` makes every retry re-score the SAME
# [as_of - lookback, as_of] cohort, so re-materialized membership and reports
# are byte-identical. Falls back to wall-clock now() for manual invocations
# that carry no event time.
# =========================================================================
_RUN_AS_OF = None


def _derive_run_as_of(event):
    """Parse the scheduled event's ``time`` into a naive-UTC datetime anchor."""
    t = (event or {}).get('time')
    if t:
        try:
            # EventBridge `time` is RFC3339 (e.g. '2026-08-24T12:00:00Z'). Store
            # naive UTC to match the naive TIMESTAMPs in inference_responses.
            dt = datetime.fromisoformat(str(t).replace('Z', '+00:00'))
            return dt.replace(tzinfo=None)
        except Exception:
            print(f"⚠️ Could not parse event time {t!r}; anchoring cohort on now()")
    return datetime.now()


def _run_as_of():
    """This run's cohort anchor time (see _RUN_AS_OF / _derive_run_as_of)."""
    return _RUN_AS_OF or datetime.now()


def execute_athena_query(sql, wait=True):
    """Execute Athena query and return results as dict."""
    response = athena.start_query_execution(
        QueryString=sql,
        QueryExecutionContext={'Database': ATHENA_DATABASE},
        ResultConfiguration={'OutputLocation': ATHENA_OUTPUT_S3}
    )
    execution_id = response['QueryExecutionId']

    if not wait:
        return execution_id

    # Wait for completion
    while True:
        status = athena.get_query_execution(QueryExecutionId=execution_id)
        state = status['QueryExecution']['Status']['State']

        if state in ['SUCCEEDED', 'FAILED', 'CANCELLED']:
            break
        time.sleep(1)

    if state != 'SUCCEEDED':
        raise Exception(f"Query failed: {state}")

    # Get results
    result_s3_path = status['QueryExecution']['ResultConfiguration']['OutputLocation']
    bucket, key = result_s3_path.replace('s3://', '').split('/', 1)
    obj = s3.get_object(Bucket=bucket, Key=key)

    # Parse CSV results
    import csv
    lines = obj['Body'].read().decode('utf-8').splitlines()
    reader = csv.DictReader(lines)
    return list(reader)


# =========================================================================
# Baseline lookup — resolves the ModelPackage actually serving the
# endpoint, then loads its frozen baseline.json. The chain is
#
#     endpoint → endpoint config → variant.ModelName → describe_model
#         → Containers[].ModelPackageName → describe_model_package
#         → ModelStatistics.S3Uri → baseline.json
#
# This answers "what's running NOW", not "what we built last." The two
# diverge during canaries, rollbacks, or pending approvals — answering
# the wrong question is the #1 cause of false drift alerts.
# =========================================================================

ENDPOINT_NAME = os.getenv('ENDPOINT_NAME', '')
_BASELINE_CACHE = {}

# Execution-status diagnostics for the current invocation, reset at the top of
# lambda_handler. Each drift check records WHY it did or didn't produce a result
# so the run can persist an explicit execution_status — the equivalent of
# SageMaker Model Monitor's per-execution Completed / NoData / Failed states.
# Without it, a run that scored zero rows and a run that scored thousands both
# just "succeed", and an operator can't tell a stalled pipeline from a
# healthy-but-empty window. Valid per-check values:
#   'SUCCESS'               — produced a drift verdict
#   'NO_DATA'               — the query returned zero rows
#   'INSUFFICIENT_SAMPLES'  — some rows, but below MIN_SAMPLES / unusable
#   'INSUFFICIENT_CLASSES'  — enough rows, but the ground-truth window held a
#                             single class so the metric (e.g. ROC-AUC) is undefined
# A check that was NOT executed leaves its slot None (NULL) and is NOT coerced
# into a real status/verdict — see _overall_execution_status.
RUN_DIAGNOSTICS = {'data': None, 'model': None}


def _record_status(check, status):
    """Record a per-check execution status; see RUN_DIAGNOSTICS."""
    RUN_DIAGNOSTICS[check] = status


def _overall_execution_status():
    """Collapse the per-check statuses into one run-level execution_status.

    A partial/skipped run must NOT read as full success. Rules:
      * SUCCESS         — every EXECUTED check produced a verdict (none is
                          NO_DATA / INSUFFICIENT_*).
      * PARTIAL_SUCCESS — at least one executed check produced a verdict AND at
                          least one other executed check is NO_DATA /
                          INSUFFICIENT_* (e.g. data drift scored but no ground
                          truth landed for model drift).
      * INSUFFICIENT_*  — no check produced a verdict, but a specific reason is
                          available (preferred over NO_DATA so a thin/degenerate
                          window is distinguishable from a truly empty one).
      * NO_DATA         — nothing executed, or every executed check found nothing.

    Checks that were not executed contribute None and are ignored. Fatal errors
    are recorded as 'ERROR' directly by the handler, not here.
    """
    statuses = [s for s in RUN_DIAGNOSTICS.values() if s]
    if not statuses:
        return 'NO_DATA'
    if all(s == 'SUCCESS' for s in statuses):
        return 'SUCCESS'
    if any(s == 'SUCCESS' for s in statuses):
        return 'PARTIAL_SUCCESS'
    # No verdict produced — surface the most specific non-success reason.
    for s in statuses:
        if s.startswith('INSUFFICIENT'):
            return s
    return 'NO_DATA'


def _endpoint_predicate():
    """SQL predicate scoping a query to this endpoint's rows.

    inference_responses is PARTITIONED BY (..., endpoint_name), so without this
    a multi-endpoint data lake would mix another endpoint's traffic into this
    monitor's baseline/current windows. Empty string when ENDPOINT_NAME is unset
    (single-endpoint dev deployments) so the SQL stays valid.
    """
    return f"AND endpoint_name = '{ENDPOINT_NAME}'" if ENDPOINT_NAME else ''


class BaselineResolutionError(Exception):
    """Baseline could not be resolved for the endpoint being monitored.

    Raised instead of silently substituting an unrelated approved package (which
    could be undeployed or belong to a different model), so the run can FAIL
    CLOSED and record a specific execution_status rather than emitting a drift
    verdict against the wrong baseline.
    """
    status = 'BASELINE_RESOLUTION_FAILED'


class MultiVariantUnsupportedError(BaselineResolutionError):
    """Endpoint serves >1 production variant.

    Inference rows carry no variant identity, so blended traffic can't be
    correctly scored against any single variant's baseline. Rather than guess
    (arbitrarily picking variants[0]) we fail closed; the handler records
    'MULTI_VARIANT_UNSUPPORTED' and skips scoring. Building per-variant
    partitioning is intentionally out of scope.
    """
    status = 'MULTI_VARIANT_UNSUPPORTED'


def _resolve_model_package_arn_from_endpoint(endpoint_name: str) -> str | None:
    """Walk the SageMaker objects to find the ModelPackage backing an endpoint.

    Returns the ARN, or None if any link in the chain is missing (e.g., the
    endpoint serves a Model that was built directly from artifacts rather
    than from a registered package).
    """
    try:
        ep = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
        cfg = sagemaker_client.describe_endpoint_config(
            EndpointConfigName=ep['EndpointConfigName']
        )
        variants = cfg.get('ProductionVariants', [])
        if not variants:
            print(f"⚠️ Endpoint {endpoint_name} has no ProductionVariants")
            return None
        if len(variants) > 1:
            # Multi-variant endpoints (A/B, canary, shadow) serve more than one
            # model at once, each potentially with its own baseline. Inference
            # rows carry no variant identity, so we can't correctly attribute
            # blended traffic to one arm's baseline. FAIL CLOSED rather than
            # guess variants[0] and emit a wrong verdict.
            names = [v.get('ModelName') for v in variants]
            raise MultiVariantUnsupportedError(
                f"Endpoint {endpoint_name} has {len(variants)} ProductionVariants "
                f"({names}); per-variant baselines are unsupported."
            )
        model_name = variants[0]['ModelName']
        model = sagemaker_client.describe_model(ModelName=model_name)
        for container in model.get('Containers', []) or [model.get('PrimaryContainer', {})]:
            arn = container.get('ModelPackageName')
            if arn:
                return arn
        print(f"⚠️ Model {model_name} was not built from a registered ModelPackage")
        return None
    except BaselineResolutionError:
        # Propagate the fail-closed signal; the handler maps it to a status.
        raise
    except Exception as e:
        print(f"⚠️ Could not resolve ModelPackage from endpoint {endpoint_name}: {e}")
        return None


def _latest_approved_model_package_arn() -> str | None:
    """Fallback for first-ever monitor runs (no endpoint yet)."""
    try:
        resp = sagemaker_client.list_model_packages(
            ModelPackageGroupName=MODEL_PACKAGE_GROUP,
            ModelApprovalStatus='Approved',
            SortBy='CreationTime',
            SortOrder='Descending',
            MaxResults=1,
        )
        packages = resp.get('ModelPackageSummaryList', [])
        return packages[0]['ModelPackageArn'] if packages else None
    except Exception as e:
        print(f"⚠️ list_model_packages fallback failed: {e}")
        return None


def _expected_primary_metric_key() -> str:
    """The baseline metric key this monitor scores against, per PROBLEM_TYPE."""
    if PROBLEM_TYPE == 'regression':
        return 'rmse'
    if PROBLEM_TYPE == 'multiclass_classification':
        return 'accuracy'
    return 'roc_auc'


def _validate_baseline_content(baseline) -> list:
    """Return a list of reasons a baseline is unusable for scoring (Finding 2).

    Corrupt or incomplete baseline content used to slip through: a run would
    fall back to env/live/prior-window defaults and still emit an
    authoritative-looking verdict. When an endpoint is configured, ANY problem
    here is treated as fatal (BASELINE_RESOLUTION_FAILED) so the run fails closed
    instead of scoring live traffic against a reference it can't trust.

    Validated: schema version, feature schema, reference-table identity, a
    non-empty metrics block, and problem-type / expected-primary-metric
    consistency with what THIS monitor scores (PROBLEM_TYPE).
    """
    problems = []
    if not isinstance(baseline, dict):
        return ['baseline is not a JSON object']
    if not baseline.get('schema_version'):
        problems.append('missing schema_version')
    if not baseline.get('feature_schema'):
        problems.append('missing feature_schema')
    if not (baseline.get('training_table') or baseline.get('evaluation_table')):
        problems.append('missing training_table/evaluation_table identity')
    metrics = baseline.get('metrics')
    if not isinstance(metrics, dict) or not metrics:
        problems.append('missing/empty metrics')
    bpt = baseline.get('problem_type')
    if not bpt:
        problems.append('missing problem_type')
    elif bpt != PROBLEM_TYPE:
        # A baseline built for a different task than the one being monitored
        # would compare apples to oranges — fail closed rather than score.
        problems.append(f'problem_type {bpt!r} != monitored {PROBLEM_TYPE!r}')
    elif isinstance(metrics, dict):
        expected_key = _expected_primary_metric_key()
        if expected_key not in metrics:
            problems.append(f'metrics missing expected primary key {expected_key!r}')
    return problems


def load_baseline_from_registry() -> dict | None:
    """Return the baseline.json registered with the model serving the endpoint.

    Resolution order:
      1. Endpoint walk (the deployed model — correct answer)
      2. Latest Approved ModelPackage in MODEL_PACKAGE_GROUP (only valid
         on first-ever monitor runs before any endpoint exists)

    Cached per warm Lambda container, KEYED BY THE RESOLVED ModelPackage ARN.
    Keying by ARN (not a single 'value' slot) means a model rollout mid-container
    busts the cache automatically: the next run resolves the new ARN, misses the
    cache, and reloads that model's baseline instead of comparing new traffic to
    the retired model's "normal". Failures are NOT cached — a transient
    describe/S3 error on one run must not pin None for the container's lifetime.

    Returns the parsed baseline.json with ``model_package_arn`` added,
    or ``None`` if no baseline can be resolved (the caller then falls
    back to env-based defaults — see check_data_drift / check_model_drift).
    """
    # Re-resolve the serving ARN every call (cheap describe_* calls) so a
    # rollout is detected; the expensive part (S3 baseline.json fetch + parse)
    # is what the per-ARN cache actually saves.
    #
    # FAIL CLOSED when an endpoint is configured: if we can't resolve the
    # ModelPackage actually serving that endpoint we must NOT fall back to the
    # latest-Approved package — that package may be undeployed or unrelated, and
    # scoring live traffic against it produces meaningless drift verdicts. The
    # latest-Approved lookup is valid ONLY on first-ever runs before any
    # endpoint exists (ENDPOINT_NAME unset). _resolve_model_package_arn_from_endpoint
    # may raise MultiVariantUnsupportedError — let it propagate to the handler.
    arn = None
    if ENDPOINT_NAME:
        arn = _resolve_model_package_arn_from_endpoint(ENDPOINT_NAME)
        if not arn:
            raise BaselineResolutionError(
                f"Could not resolve the ModelPackage serving endpoint "
                f"{ENDPOINT_NAME}; failing closed rather than substituting an "
                f"unrelated approved package."
            )
    else:
        arn = _latest_approved_model_package_arn()
    if not arn:
        print(f"⚠️ No ModelPackage available (endpoint={ENDPOINT_NAME or '<unset>'}, "
              f"group={MODEL_PACKAGE_GROUP})")
        return None

    if arn in _BASELINE_CACHE:
        return _BASELINE_CACHE[arn]

    # FAIL CLOSED when an endpoint is configured (Finding 2): a missing
    # ModelStatistics URI, an S3/JSON error, or incomplete/invalid content must
    # NOT degrade to env/live/prior-window fallbacks and emit an authoritative-
    # looking verdict. Each of those cases raises BaselineResolutionError below;
    # the handler maps it to BASELINE_RESOLUTION_FAILED and skips scoring. When
    # NO endpoint is configured (endpoint-less first-run/dev), the old permissive
    # behavior is preserved: log and return None so the checks fall back.
    try:
        pkg = sagemaker_client.describe_model_package(ModelPackageName=arn)
        # SageMaker's describe-model-package returns model statistics under
        # ModelMetrics.ModelQuality.Statistics.S3Uri (per the boto3 schema).
        # The legacy key ModelMetrics.ModelStatistics.S3Uri is kept as a
        # fallback in case older SDK versions populate it.
        metrics = pkg.get('ModelMetrics', {})
        s3_uri = (
            metrics.get('ModelQuality', {}).get('Statistics', {}).get('S3Uri')
            or metrics.get('ModelStatistics', {}).get('S3Uri')
        )
        if not s3_uri:
            msg = f"ModelPackage {arn} has no ModelStatistics URI"
            if ENDPOINT_NAME:
                raise BaselineResolutionError(f"{msg}; failing closed.")
            print(f"⚠️ {msg} — skipping baseline")
            return None

        bucket, key = s3_uri.replace('s3://', '').split('/', 1)
        body = s3.get_object(Bucket=bucket, Key=key)['Body'].read()
        baseline = json.loads(body)
        baseline['model_package_arn'] = arn
    except BaselineResolutionError:
        # Already a fail-closed signal (e.g. the no-URI branch) — propagate.
        raise
    except Exception as e:
        if ENDPOINT_NAME:
            # A corrupt/unfetchable baseline for a monitored endpoint is fatal:
            # fail closed rather than score against nothing. NOT cached — a
            # transient S3/JSON error must not pin failure for the warm container.
            raise BaselineResolutionError(
                f"Could not load/parse baseline.json for {arn}: {e}; failing closed."
            ) from e
        print(f"⚠️ Could not load baseline.json for {arn}: {e}")
        # Deliberately NOT cached — a transient error must not pin None for the
        # rest of the warm container's life; the next run retries.
        return None

    # Content validation (Finding 2). Incomplete/mismatched content fails closed
    # for a monitored endpoint; endpoint-less dev runs log and proceed.
    problems = _validate_baseline_content(baseline)
    if problems:
        if ENDPOINT_NAME:
            raise BaselineResolutionError(
                f"Baseline for {arn} is invalid/incomplete: {problems}; failing closed."
            )
        print(f"⚠️ Baseline for {arn} incomplete ({problems}); proceeding with "
              f"fallbacks (no endpoint configured)")

    print(
        f"✓ Loaded baseline from {s3_uri}\n"
        f"  ModelPackage:        {arn}\n"
        f"  Baseline ROC-AUC:    {baseline.get('metrics', {}).get('roc_auc', '?')}\n"
        f"  Evaluation table:    {baseline.get('evaluation_table', '?')}"
        f"  (snapshot {baseline.get('evaluation_snapshot_id') or 'live'})"
    )
    _BASELINE_CACHE[arn] = baseline
    return baseline


def _safe_load_baseline() -> dict | None:
    """load_baseline_from_registry() that swallows fail-closed errors.

    For diagnostic/record-writing paths (MLflow tagging, monitoring row) a
    baseline-resolution failure has ALREADY been surfaced as an execution_status
    by the handler; those paths just want whatever immutable references they can
    get and must not re-raise. Scoring paths call load_baseline_from_registry()
    directly so the failure still propagates there.
    """
    try:
        return load_baseline_from_registry()
    except BaselineResolutionError:
        return None


def _reports_s3_base() -> str:
    """Base S3 URI (no trailing slash) for durable per-run Evidently reports."""
    if MONITORING_REPORTS_S3:
        return MONITORING_REPORTS_S3.rstrip('/')
    bucket = ATHENA_OUTPUT_S3.replace('s3://', '').split('/', 1)[0]
    return f"s3://{bucket}/monitoring/evidently_reports"


def _upload_evidently_reports(run_id, data_drift_result, model_drift_result):
    """Upload each check's local HTML report to a deterministic per-run S3 key.

    Key layout: <base>/<run_id>/<check>.html — so a persisted monitoring row
    (evidently_report_s3_path) correlates to a durable artifact instead of the
    random /tmp path MLflow logs then unlinks. Sets 'evidently_report_s3_path'
    on each result dict (the run-level folder). Best-effort: an upload failure
    is logged, never raised.
    """
    base = _reports_s3_base()
    run_folder = f"{base}/{run_id}/"

    def _upload(local_path, check):
        if not local_path or not os.path.exists(local_path):
            return None
        key_uri = f"{base}/{run_id}/{check}.html"
        try:
            bucket, key = key_uri.replace('s3://', '').split('/', 1)
            s3.upload_file(local_path, bucket, key)
            return run_folder
        except Exception as e:
            print(f"⚠️ Failed to upload {check} report to {key_uri}: {e}")
            return None

    if data_drift_result:
        path = _upload(data_drift_result.get('html_report_path'), 'data_drift')
        if path:
            data_drift_result['evidently_report_s3_path'] = path
    if model_drift_result:
        path = _upload(model_drift_result.get('html_report_path'), 'model_drift')
        if path:
            model_drift_result['evidently_report_s3_path'] = path


def _confirm_persisted(run_id, tries=5, delay=2):
    """Poll Athena until the monitoring row for run_id is durably queryable.

    The SQS→writer→Athena path is asynchronous, so before we backfill inference
    rows with run_id we CONFIRM the history row actually landed — otherwise a
    dropped/failed write leaves orphaned inference rows pointing at a
    nonexistent monitoring_run_id. Bounded retry with short backoff; returns
    True once a row is found, False if it never appears within the budget.
    """
    sql = (
        f"SELECT 1 FROM {ATHENA_DATABASE}.monitoring_responses "
        f"WHERE monitoring_run_id = '{run_id}' LIMIT 1"
    )
    for attempt in range(tries):
        try:
            rows = execute_athena_query(sql)
        except Exception as e:
            print(f"  Persistence check attempt {attempt + 1}/{tries} errored: {e}")
            rows = None
        if rows:
            return True
        if attempt < tries - 1:
            time.sleep(delay)
    return False


def _write_run_inferences(run_id, endpoint_name, check_type, inference_ids,
                          created_at, batch_size=500):
    """Record EXACT run→inference membership in the monitoring_run_inferences bridge.

    Idempotent at the run level: first DELETEs any existing rows for
    (monitoring_run_id, check_type), then re-INSERTs the current membership in
    batches. Because run_id is now stable across retries (see _derive_run_id),
    a redelivered event re-materializes identical membership instead of
    duplicating it. Rows are inserted in bounded VALUES batches so the Athena
    query string stays well under the ~262 KB limit even for a full 10k cohort.

    Returns the number of membership rows written (0 if there were none).
    """
    ids = [str(i) for i in (inference_ids or []) if i]
    db = ATHENA_DATABASE
    table = ATHENA_RUN_INFERENCES_TABLE
    ct = str(check_type).replace("'", "''")
    rid = str(run_id).replace("'", "''")
    ep = str(endpoint_name or '').replace("'", "''")
    ts = str(created_at).replace("'", "''")

    # Idempotency: clear any prior membership for this (run, check) first so a
    # retry doesn't accumulate duplicate bridge rows. A genuine DELETE failure
    # now PROPAGATES (Finding 4): silently continuing to INSERT after a failed
    # clear could leave duplicated membership. execute_athena_query raises on a
    # non-SUCCEEDED state, so a real failure surfaces to the caller, which then
    # withholds the completion marker and lets a retry re-materialize.
    delete_sql = (
        f"DELETE FROM {db}.{table} "
        f"WHERE monitoring_run_id = '{rid}' AND check_type = '{ct}'"
    )
    execute_athena_query(delete_sql, wait=True)

    if not ids:
        return 0

    written = 0
    for start in range(0, len(ids), batch_size):
        chunk = ids[start:start + batch_size]
        rows = ", ".join(
            f"('{rid}', '{i.replace(chr(39), chr(39) * 2)}', '{ct}', '{ep}', TIMESTAMP '{ts}')"
            for i in chunk
        )
        insert_sql = (
            f"INSERT INTO {db}.{table} "
            f"(monitoring_run_id, inference_id, check_type, endpoint_name, created_at) "
            f"VALUES {rows}"
        )
        execute_athena_query(insert_sql, wait=True)
        written += len(chunk)

    # Verify the bridge actually holds EXACTLY the membership we intended
    # (Finding 4). A silent short-write (dropped batch, engine hiccup) would
    # otherwise leave incomplete-but-populated membership that still looks whole.
    # Count-check and raise on mismatch so the caller withholds the COMPLETE
    # marker and a retry re-materializes the full cohort.
    verify_sql = (
        f"SELECT COUNT(*) AS c FROM {db}.{table} "
        f"WHERE monitoring_run_id = '{rid}' AND check_type = '{ct}'"
    )
    vrows = execute_athena_query(verify_sql)
    actual = int(vrows[0]['c']) if vrows and vrows[0].get('c') is not None else 0
    if actual != len(ids):
        raise RuntimeError(
            f"Bridge membership count mismatch for {rid}/{ct}: "
            f"expected {len(ids)}, found {actual}"
        )

    print(f"✓ Bridge: recorded {written} {check_type} membership rows for run {run_id}")
    return written


def _write_generation_marker(run_id, check_type, expected, actual,
                             created_at, status='COMPLETE'):
    """Write the atomic membership-completion marker for (run_id, check_type).

    This single-row MERGE is the COMMIT POINT for a cohort's bridge membership
    (Finding 4): it is written ONLY after every bridge row was inserted AND
    count-verified by _write_run_inferences, so a consumer that joins the bridge
    through a COMPLETE marker (expected_count == actual_count) never sees a
    partially-written cohort. Idempotent + keyed on (monitoring_run_id,
    check_type): a retry that re-materializes identical membership UPDATEs the
    counts in place rather than duplicating the marker.
    """
    db = ATHENA_DATABASE
    table = ATHENA_RUN_GENERATIONS_TABLE
    rid = str(run_id).replace("'", "''")
    ct = str(check_type).replace("'", "''")
    ts = str(created_at).replace("'", "''")
    st = str(status).replace("'", "''")
    query = f"""
    MERGE INTO {db}.{table} AS t
    USING (SELECT '{rid}' AS monitoring_run_id, '{ct}' AS check_type,
                  {int(expected)} AS expected_count, {int(actual)} AS actual_count,
                  '{st}' AS status, TIMESTAMP '{ts}' AS created_at) AS s
    ON t.monitoring_run_id = s.monitoring_run_id AND t.check_type = s.check_type
    WHEN MATCHED THEN UPDATE SET
        expected_count = s.expected_count, actual_count = s.actual_count,
        status = s.status, created_at = s.created_at
    WHEN NOT MATCHED THEN INSERT
        (monitoring_run_id, check_type, expected_count, actual_count, status, created_at)
        VALUES (s.monitoring_run_id, s.check_type, s.expected_count,
                s.actual_count, s.status, s.created_at)
    """
    execute_athena_query(query, wait=True)


# =========================================================================
# Legacy statistical functions (kept for reference)
#
# These demonstrate how to compute PSI and KS drift statistics explicitly
# without Evidently. The active Lambda flow now uses Evidently's
# DataDriftPreset and ClassificationPreset via evidently_reports.py.
# =========================================================================

def calculate_psi(baseline_values, current_values, bins=10):
    """Calculate Population Stability Index (PSI).

    LEGACY — This is the manual implementation of PSI using numpy.
    Kept to show what it takes to compute PSI without Evidently.
    The active drift check now delegates to ``run_data_drift_report()``
    which uses Evidently's DataDriftPreset internally.
    """
    baseline_values = np.array(baseline_values, dtype=float)
    current_values = np.array(current_values, dtype=float)

    # Create bins from baseline percentiles
    breakpoints = np.percentile(baseline_values, np.linspace(0, 100, bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    # Histogram
    baseline_hist, _ = np.histogram(baseline_values, bins=breakpoints)
    current_hist, _ = np.histogram(current_values, bins=breakpoints)

    # Convert to percentages
    baseline_pct = baseline_hist / len(baseline_values)
    current_pct = current_hist / len(current_values)

    # Add floor to avoid log(0)
    baseline_pct = np.where(baseline_pct == 0, 0.0001, baseline_pct)
    current_pct = np.where(current_pct == 0, 0.0001, current_pct)

    # Calculate PSI
    psi = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))

    return float(psi)


def calculate_ks_statistic(baseline_values, current_values):
    """Calculate Kolmogorov-Smirnov test statistic.

    LEGACY — This is the manual implementation of the two-sample KS test
    using scipy.stats. Kept to show what it takes to compute KS without
    Evidently. The active drift check now delegates to
    ``run_data_drift_report()`` which uses Evidently's DataDriftPreset
    (which includes KS as one of its statistical tests).

    The KS test measures the maximum distance between the cumulative
    distribution functions (CDFs) of two samples. It's particularly
    sensitive to changes in distribution tails, making it ideal for
    fraud detection.

    Args:
        baseline_values: List of baseline (training) values
        current_values: List of current (inference) values

    Returns:
        tuple: (ks_statistic, p_value)
            - ks_statistic: 0-1 (0 = identical, 1 = completely different)
            - p_value: Probability that difference is random (< 0.05 = significant)
    """
    from scipy import stats

    baseline_values = np.array(baseline_values, dtype=float)
    current_values = np.array(current_values, dtype=float)

    # Remove NaN values
    baseline_values = baseline_values[~np.isnan(baseline_values)]
    current_values = current_values[~np.isnan(current_values)]

    if len(baseline_values) == 0 or len(current_values) == 0:
        return 0.0, 1.0

    # Perform two-sample KS test
    ks_stat, p_value = stats.ks_2samp(baseline_values, current_values)

    return float(ks_stat), float(p_value)


# =========================================================================
# Active drift detection — powered by Evidently
# =========================================================================

def check_data_drift():
    """Check for data drift using Evidently DataDriftPreset.

    Queries recent inference data and baseline training data from Athena,
    builds DataFrames, and runs Evidently's DataDriftPreset report.

    Returns:
        dict with drift results or None if insufficient data.
    """
    print("🔍 Checking data drift (Evidently)...")

    # Get recent inference data (using configured lookback period). Anchor the
    # window on THIS run's as-of time (the immutable EventBridge event `time`),
    # NOT wall-clock now(): a retry/late async re-run then re-materializes the
    # SAME [lower, upper] cohort deterministically instead of sliding forward.
    as_of = _run_as_of()
    as_of_str = as_of.strftime('%Y-%m-%d %H:%M:%S')
    lookback_start = (as_of - timedelta(days=DATA_DRIFT_LOOKBACK_DAYS)).strftime('%Y-%m-%d %H:%M:%S')
    print(f"  Querying inference data from last {DATA_DRIFT_LOOKBACK_DAYS} days "
          f"(since {lookback_start}, as-of {as_of_str})")

    # Deterministic cohort: ORDER BY request_timestamp BEFORE LIMIT so the set
    # of scored rows is well-defined (an unordered LIMIT lets the engine return
    # a different arbitrary subset each run). Ascending from lookback_start makes
    # the analyzed window a contiguous [lower, upper] slice, which lets the
    # backfill re-scope to EXACTLY the rows this run scored (see watermarks
    # below + write_monitoring_results). The upper bound (request_timestamp <=
    # as_of) fixes the window's ceiling to the run anchor so rows that arrive
    # AFTER the run started can never leak into (or shift) this cohort on retry.
    # Also SELECT request_timestamp so we can capture the true upper watermark.
    # SELECT inference_id too: the EXACT ids of the rows we actually score
    # (post-parse) are recorded in the monitoring_run_inferences bridge so
    # cohort membership is precise, not a timestamp-window approximation.
    recent_data_sql = f"""
    SELECT inference_id, input_features, request_timestamp
    FROM {ATHENA_DATABASE}.inference_responses
    WHERE request_timestamp >= TIMESTAMP '{lookback_start}'
      AND request_timestamp <= TIMESTAMP '{as_of_str}'
      {_endpoint_predicate()}
    ORDER BY request_timestamp, inference_id
    LIMIT 10000
    """

    recent_data = execute_athena_query(recent_data_sql)

    if len(recent_data) == 0:
        print(f"⚠️ No inference rows in the last {DATA_DRIFT_LOOKBACK_DAYS} days")
        _record_status('data', 'NO_DATA')
        return None
    if len(recent_data) < MIN_SAMPLES:
        print(f"⚠️ Not enough recent samples ({len(recent_data)} < {MIN_SAMPLES})")
        _record_status('data', 'INSUFFICIENT_SAMPLES')
        return None

    print(f"✓ Found {len(recent_data)} recent inference samples")

    # Window watermarks scoping the analyzed cohort. lower = lookback_start (the
    # WHERE bound); upper = the max request_timestamp actually pulled (the newest
    # row inside the LIMIT). The backfill re-uses exactly [lower, upper] so the
    # rows it tags are a subset of what we analyzed — never "everything through
    # now()", which would tag rows newer than (and excluded from) this cohort.
    ts_values = [r.get('request_timestamp') for r in recent_data if r.get('request_timestamp')]
    window_lower = lookback_start
    window_upper = max(ts_values) if ts_values else None

    # Parse JSON features into a DataFrame. Track the inference_id of every row
    # that yields a usable parsed sample so the bridge records EXACTLY the rows
    # scored — malformed/unparseable rows are skipped here and therefore never
    # claimed as members of this run's cohort.
    current_rows = []
    scored_inference_ids = []
    for row in recent_data:
        try:
            features = json.loads(row['input_features'])
            parsed = {}
            for feat in TRAINING_FEATURES:
                if feat in features:
                    parsed[feat] = float(features[feat])
            if parsed:
                current_rows.append(parsed)
                inf_id = row.get('inference_id')
                if inf_id:
                    scored_inference_ids.append(inf_id)
        except Exception:
            continue

    if len(current_rows) < MIN_SAMPLES:
        print(f"⚠️ Not enough parseable samples ({len(current_rows)} < {MIN_SAMPLES})")
        _record_status('data', 'INSUFFICIENT_SAMPLES')
        return None

    current_df = pd.DataFrame(current_rows)

    # Industry-standard data-drift baseline: training_data (the distribution
    # the model was TRAINED on). Pin the exact Iceberg snapshot the training
    # job used (carried in baseline.json as training_snapshot_id) so
    # re-seeding training_data later doesn't retroactively change "what
    # this model considers normal". Model drift uses evaluation_data
    # (the labeled held-out set) — see check_model_drift below.
    baseline = load_baseline_from_registry()
    baseline_table = (baseline or {}).get('training_table') or 'training_data'
    train_snapshot = (baseline or {}).get('training_snapshot_id') or ''

    if train_snapshot:
        from_clause = (
            f"{ATHENA_DATABASE}.{baseline_table} "
            f"FOR VERSION AS OF {train_snapshot}"
        )
        snapshot_log = f"snapshot {train_snapshot}"
    else:
        from_clause = f"{ATHENA_DATABASE}.{baseline_table}"
        snapshot_log = "live table (no snapshot pinned)"

    # LIMIT 5000 — NOT a coverage gap, deliberate cost/perf cap.
    #
    # We're characterizing a *distribution* for Evidently's KS / PSI tests,
    # not enumerating rows. Both tests are stable well below 5K samples
    # for the 30 input features here — additional rows stop moving the
    # KS p-value or PSI meaningfully past ~2K. Evidently's own docs cap
    # the recommended reference size at ~10K.
    #
    # On a ~56K-row eval slice this is ~9% sampling, which still gives
    # ~8 fraud-class rows on average (fraud ≈ 0.17%). That's fine here
    # because drift detection is UNSUPERVISED — we compare feature
    # distributions, not label-conditioned ones. Baseline-side classification
    # metrics live in baseline.json (computed on the FULL eval slice at
    # train time), so we never need to recompute them from this sample.
    #
    # If we ever add supervised drift checks, switch to a stratified pull:
    # all fraud rows UNION ALL 5000 random non-fraud rows.
    #
    # ORDER BY RANDOM() is fine at this scale; for tables >10M rows
    # consider TABLESAMPLE BERNOULLI to avoid a full-table sort.
    # Filter for rows with a valid target label. Using the schema-driven
    # TARGET_COLUMN instead of a hardcoded `is_fraud` so this Lambda works
    # against any BYO dataset whose target column has a different name.
    baseline_sql = f"""
    SELECT {', '.join(TRAINING_FEATURES)}
    FROM {from_clause}
    WHERE {TARGET_COLUMN} IS NOT NULL
    ORDER BY RANDOM()
    LIMIT 5000
    """

    baseline_data = execute_athena_query(baseline_sql)
    print(f"✓ Loaded {len(baseline_data)} baseline samples from {baseline_table} ({snapshot_log})")

    baseline_df = pd.DataFrame(baseline_data)
    # customer_gender is stored as STRING in the Athena training_data table
    # ('Male'/'Female') because the raw table is pre-preprocessing. Mirror
    # the label-encoding that preprocessing_pyspark.py:298-318 does at train
    # time (sorted lowercased categories -> 0..N-1) so the encoded value here
    # matches what the model was trained on. current_df is already numeric
    # because inference inputs are post-preprocessing (parsed via
    # float(features[feat]) above), so no transform on that side.
    categorical_cols = {'customer_gender'}
    for col in baseline_df.columns:
        if col in categorical_cols:
            cats = sorted(baseline_df[col].dropna().astype(str).str.lower().unique())
            mapping = {c: i for i, c in enumerate(cats)}
            baseline_df[col] = (
                baseline_df[col].astype(str).str.lower().map(mapping).fillna(-1)
            )
        else:
            baseline_df[col] = pd.to_numeric(baseline_df[col], errors='coerce')

    # Use only columns present in both DataFrames
    common_cols = sorted(set(baseline_df.columns) & set(current_df.columns))
    if not common_cols:
        print("⚠️ No common columns between baseline and current data")
        _record_status('data', 'INSUFFICIENT_SAMPLES')
        return None

    baseline_df = baseline_df[common_cols]
    current_df = current_df[common_cols]

    # Save Evidently HTML report to /tmp for MLflow artifact logging
    html_path = tempfile.NamedTemporaryFile(
        suffix='.html', prefix='data_drift_', delete=False, dir='/tmp'
    ).name

    # Run Evidently data drift report
    drift_result = run_data_drift_report(
        baseline_df=baseline_df,
        current_df=current_df,
        output_path=html_path,
    )

    # Build per-column summary for SNS alert
    drifted_features = []
    per_column = drift_result.get('per_column', {})
    for col, info in per_column.items():
        if info.get('drifted'):
            drifted_features.append({
                'feature': col,
                'drift_score': info.get('drift_score', 0),
                'drift_magnitude': info.get('drift_magnitude', 0),
                'method': info.get('method', ''),
                'threshold': info.get('threshold', 0),
            })

    # Sort by drift_magnitude descending (higher = more drifted). Magnitude is
    # test-agnostic and bounded because evidently_reports forces the
    # jensenshannon distance metric for every column, so this ranks features by
    # drift severity rather than by raw p-value precision (see the
    # evidently_reports module docstring for why the metric is pinned).
    drifted_features.sort(key=lambda x: x['drift_magnitude'], reverse=True)

    features_analyzed = len(per_column)
    drifted_count = drift_result['drifted_columns_count']
    drift_share = drift_result['drifted_columns_share']

    print(f"  Evidently: {drifted_count}/{features_analyzed} features drifted ({drift_share:.1%})")
    if drift_result['drift_detected']:
        print("  🚨 Overall data drift DETECTED")
    else:
        print("  ✓ No overall data drift detected")

    _record_status('data', 'SUCCESS')
    return {
        'detected': drift_result['drift_detected'],
        'features_analyzed': features_analyzed,
        'drifted_features_count': drifted_count,
        'drift_percentage': drift_share * 100,
        'drifted_columns_share': drift_share,
        'drifted_features': drifted_features[:5],  # Top 5 — used for the SNS alert only
        # Full per-column result (every feature, drifted or not). Persisted to
        # monitoring_responses.per_feature_drift_scores so the governance
        # dashboard sees the same complete feature set the notebook writes —
        # not just the top-5 alert subset.
        'per_column': per_column,
        'sample_size': len(current_rows),
        'html_report_path': html_path,
        # EXACT membership: the inference_ids actually scored by THIS run's data
        # -drift check. Written to the monitoring_run_inferences bridge with
        # check_type='data_drift' (see write_monitoring_results). Supersedes the
        # old timestamp-window backfill onto inference_responses.monitoring_run_id.
        'scored_inference_ids': scored_inference_ids,
        # Analyzed-cohort watermarks (in-memory only, kept for logging/diagnostics).
        'window_lower': window_lower,
        'window_upper': window_upper,
    }


def check_model_drift():
    """Check for model performance drift using Evidently ClassificationPreset.

    Queries recent predictions with ground truth from Athena, builds a
    baseline comparison DataFrame, and runs Evidently's ClassificationPreset.

    Returns:
        dict with model drift results or None if insufficient data.
    """
    # Regression models take a completely different metric path (MAE/RMSE/R2
    # instead of ROC-AUC/precision/recall) — dispatch early.
    if PROBLEM_TYPE == 'regression':
        return _check_regression_drift()

    is_multiclass = PROBLEM_TYPE == 'multiclass_classification'
    print(f"🔍 Checking model drift (Evidently, {PROBLEM_TYPE})...")

    # Get recent predictions with ground truth (using configured lookback
    # period). Anchor on THIS run's as-of time (immutable EventBridge event
    # `time`), not wall-clock now(), so a retry re-materializes the same cohort.
    as_of = _run_as_of()
    as_of_str = as_of.strftime('%Y-%m-%d %H:%M:%S')
    lookback_start = (as_of - timedelta(days=MODEL_DRIFT_LOOKBACK_DAYS)).strftime('%Y-%m-%d %H:%M:%S')
    print(f"  Querying predictions with ground truth from last {MODEL_DRIFT_LOOKBACK_DAYS} days "
          f"(since {lookback_start}, as-of {as_of_str})")

    # SELECT the config-driven prediction/probability columns and alias them
    # to stable in-Python names so the rest of this function keeps using
    # `prediction` and `probability_fraud` regardless of the actual Athena
    # column names in a BYO deployment.
    # Window on WHEN THE LABEL LANDED, not when the request was served. Ground
    # truth for fraud arrives days-to-weeks late (chargebacks, investigations),
    # so a request served just before the window but labeled inside it belongs
    # in this run's model-quality check. COALESCE falls back to request_timestamp
    # when ground_truth_timestamp isn't populated, so older rows still count.
    # SELECT inference_id too so the EXACT scored rows are recorded in the
    # monitoring_run_inferences bridge (check_type='model_drift'). This cohort
    # is a DIFFERENT (ground-truth-arrival-time) set than the data-drift
    # (request-time) cohort, which is precisely why one mutable FK on
    # inference_responses cannot represent both.
    performance_sql = f"""
    SELECT
        inference_id,
        {PREDICTION_COLUMN} AS prediction,
        {PROBABILITY_COLUMN} AS probability_fraud,
        ground_truth
    FROM {ATHENA_DATABASE}.inference_responses
    WHERE ground_truth IS NOT NULL
      AND COALESCE(ground_truth_timestamp, request_timestamp) >= TIMESTAMP '{lookback_start}'
      AND COALESCE(ground_truth_timestamp, request_timestamp) <= TIMESTAMP '{as_of_str}'
      {_endpoint_predicate()}
    ORDER BY COALESCE(ground_truth_timestamp, request_timestamp), inference_id
    LIMIT 10000
    """

    recent_performance = execute_athena_query(performance_sql)

    if len(recent_performance) == 0:
        print(f"⚠️ No labeled predictions in the last {MODEL_DRIFT_LOOKBACK_DAYS} days")
        _record_status('model', 'NO_DATA')
        return None
    if len(recent_performance) < MIN_SAMPLES:
        print(f"⚠️ Not enough samples with ground truth ({len(recent_performance)} < {MIN_SAMPLES})")
        _record_status('model', 'INSUFFICIENT_SAMPLES')
        return None

    print(f"✓ Found {len(recent_performance)} samples with ground truth")

    # Exact scored cohort for the bridge; drop inference_id from the frame so
    # the metric/Evidently code below sees the same schema it always has.
    scored_inference_ids = [r.get('inference_id') for r in recent_performance if r.get('inference_id')]

    # Build current DataFrame
    current_df = pd.DataFrame(recent_performance)
    if 'inference_id' in current_df.columns:
        current_df = current_df.drop(columns=['inference_id'])
    current_df['ground_truth'] = current_df['ground_truth'].astype(int)
    current_df['prediction'] = current_df['prediction'].astype(int)
    # probability_fraud is only meaningful for a binary score. For multiclass
    # a single probability column can't feed ROC-AUC, so parse it best-effort
    # and don't rely on it below.
    current_df['probability_fraud'] = pd.to_numeric(
        current_df['probability_fraud'], errors='coerce'
    )

    # Compute sklearn metrics for the SNS alert / response payload.
    # Binary averages the positive class (default); multiclass uses macro
    # averaging so every class counts equally regardless of support.
    from sklearn.metrics import (
        roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
    )

    y_true = current_df['ground_truth'].values
    y_pred = current_df['prediction'].values
    y_prob = current_df['probability_fraud'].values

    # A single-class ground-truth window is NORMAL on imbalanced data (fraud can
    # go a whole window with zero positives). roc_auc_score raises in that case,
    # which previously turned a benign empty-of-one-class window into a fatal
    # run/retry. Check class diversity BEFORE computing ROC-AUC and record
    # INSUFFICIENT_CLASSES instead of crashing. (Multiclass uses accuracy as its
    # primary metric, so it isn't affected by this.)
    if not is_multiclass and len(np.unique(y_true)) < 2:
        only = np.unique(y_true).tolist()
        print(f"⚠️ Ground-truth window has a single class {only}; ROC-AUC is "
              f"undefined — recording INSUFFICIENT_CLASSES and skipping model drift.")
        _record_status('model', 'INSUFFICIENT_CLASSES')
        return None

    avg = 'macro' if is_multiclass else 'binary'
    current_accuracy = accuracy_score(y_true, y_pred)
    current_precision = precision_score(y_true, y_pred, average=avg, zero_division=0)
    current_recall = recall_score(y_true, y_pred, average=avg, zero_division=0)
    # Macro-F1 for multiclass (harmonic mean of macro-precision/recall is NOT
    # macro-F1); binary averages the positive class. Computed from labels, not
    # re-derived from precision/recall downstream.
    current_f1 = float(f1_score(y_true, y_pred, average=avg, zero_division=0))

    # ROC-AUC only applies to the binary single-probability case. For
    # multiclass we set it to None and use accuracy as the primary metric.
    if is_multiclass:
        current_roc_auc = None
        print(f"  Current Accuracy: {current_accuracy:.4f}")
        print(f"  Current Precision (macro): {current_precision:.4f}")
        print(f"  Current Recall (macro): {current_recall:.4f}")
    else:
        current_roc_auc = roc_auc_score(y_true, y_prob)
        print(f"  Current ROC-AUC: {current_roc_auc:.4f}")
        print(f"  Current Accuracy: {current_accuracy:.4f}")
        print(f"  Current Precision: {current_precision:.4f}")
        print(f"  Current Recall: {current_recall:.4f}")

    # Compare to baseline. Source of truth is baseline.json registered on
    # the latest Approved ModelPackage — that anchors the baseline to the
    # exact slice (evaluation_data) the model was scored on at training.
    # Falls back to BASELINE_ROC_AUC env var only when the registry lookup
    # fails (e.g., first ever monitor run before any model is approved).
    # Primary degradation metric depends on the problem type:
    #   binary     → ROC-AUC (baseline.json 'roc_auc', else BASELINE_ROC_AUC)
    #   multiclass → accuracy (baseline.json 'accuracy', else BASELINE_ACCURACY),
    #                since a single probability column can't produce ROC-AUC.
    baseline = load_baseline_from_registry()
    baseline_metrics = (baseline or {}).get('metrics', {}) if baseline else {}

    if is_multiclass:
        primary_metric = 'accuracy'
        current_primary = current_accuracy
        if 'accuracy' in baseline_metrics:
            baseline_primary = float(baseline_metrics['accuracy'])
            baseline_source = baseline.get('model_package_arn', 'registered baseline.json')
        else:
            baseline_primary = float(os.getenv('BASELINE_ACCURACY', '0.90'))
            baseline_source = 'env:BASELINE_ACCURACY (no registered baseline.json found)'
        # ROC-AUC is not defined for the multiclass single-probability case.
        baseline_roc_auc = None
    else:
        primary_metric = 'roc_auc'
        current_primary = current_roc_auc
        if 'roc_auc' in baseline_metrics:
            baseline_primary = float(baseline_metrics['roc_auc'])
            baseline_source = baseline.get('model_package_arn', 'registered baseline.json')
        else:
            baseline_primary = float(os.getenv('BASELINE_ROC_AUC', '0.92'))
            baseline_source = 'env:BASELINE_ROC_AUC (no registered baseline.json found)'
        baseline_roc_auc = baseline_primary

    degradation = baseline_primary - current_primary
    degradation_pct = (degradation / baseline_primary) * 100 if baseline_primary else 0.0

    print(f"  Baseline {primary_metric}: {baseline_primary:.4f}  ← {baseline_source}")
    print(f"  Current {primary_metric}:  {current_primary:.4f}")
    print(f"  Degradation: {degradation:.4f} ({degradation_pct:.1f}%)")

    # Build a synthetic baseline DataFrame with the same schema so Evidently
    # can compare reference vs current classification performance.
    # In production you'd load actual baseline predictions from S3/Athena.
    # Alias config-driven column names to stable in-Python identifiers
    # (same aliasing pattern as the current-window SELECT above).
    baseline_sql = f"""
    SELECT
        {PREDICTION_COLUMN} AS prediction,
        {PROBABILITY_COLUMN} AS probability_fraud,
        ground_truth
    FROM {ATHENA_DATABASE}.inference_responses
    WHERE ground_truth IS NOT NULL
      AND COALESCE(ground_truth_timestamp, request_timestamp) < TIMESTAMP '{lookback_start}'
      {_endpoint_predicate()}
    ORDER BY RANDOM()
    LIMIT 10000
    """

    try:
        baseline_data = execute_athena_query(baseline_sql)
        if len(baseline_data) >= MIN_SAMPLES:
            baseline_df = pd.DataFrame(baseline_data)
            baseline_df['ground_truth'] = baseline_df['ground_truth'].astype(int)
            baseline_df['prediction'] = baseline_df['prediction'].astype(int)
            baseline_df['probability_fraud'] = baseline_df['probability_fraud'].astype(float)
        else:
            # Fall back: duplicate current as baseline (report still generates)
            baseline_df = current_df.copy()
    except Exception:
        baseline_df = current_df.copy()

    # Save Evidently HTML report to /tmp
    html_path = tempfile.NamedTemporaryFile(
        suffix='.html', prefix='model_perf_', delete=False, dir='/tmp'
    ).name

    # Evidently's ClassificationPreset needs BOTH classes (0 AND 1) in BOTH
    # `ground_truth` and `prediction` of BOTH datasets — otherwise it raises
    # KeyError: '0' deep inside ClassificationQualityByClass. On highly
    # imbalanced data (fraud ≈ 0.2%) it's common for the model to never
    # predict the minority class on a small sample. Skip the report
    # generation in that case rather than failing the whole Lambda run.
    sides = [
        ('baseline.ground_truth', baseline_df['ground_truth']),
        ('baseline.prediction',   baseline_df['prediction']),
        ('current.ground_truth',  current_df['ground_truth']),
        ('current.prediction',    current_df['prediction']),
    ]
    degenerate = [name for name, col in sides if col.nunique() < 2]
    classification_result = None
    if degenerate:
        print(f"  ⚠ Skipping Evidently classification report — single-class column(s): {degenerate}")
        print(f"    Likely cause: model never predicted the minority class on this sample.")
        print(f"    Numeric metrics (ROC-AUC, precision, recall) above are still valid.")
    else:
        try:
            classification_result = run_classification_report(
                baseline_df=baseline_df,
                current_df=current_df,
                target_column='ground_truth',
                prediction_column='prediction',
                output_path=html_path,
                multiclass=is_multiclass,
            )
        except ValueError as e:
            # Pre-flight in run_classification_report caught a degenerate case
            # we didn't (e.g., NaN-filtering inside Evidently). Log and proceed.
            print(f"  ⚠ Skipping Evidently classification report: {e}")

    detected = degradation_pct >= (MODEL_DRIFT_THRESHOLD * 100)
    if detected:
        print("  🚨 Model performance drift DETECTED")
    else:
        print("  ✓ No model performance drift detected")

    _record_status('model', 'SUCCESS')
    return {
        'detected': detected,
        'problem_type': PROBLEM_TYPE,
        'primary_metric': primary_metric,
        'baseline_primary': baseline_primary,
        'current_primary': current_primary,
        'baseline_roc_auc': baseline_roc_auc,
        'current_roc_auc': current_roc_auc,
        'degradation': degradation,
        'degradation_pct': degradation_pct,
        'accuracy': current_accuracy,
        'precision': current_precision,
        'recall': current_recall,
        # True F1 (macro for multiclass, positive-class for binary) computed from
        # labels — persisted as f1_score by write_monitoring_results.
        'f1': current_f1,
        'sample_size': len(recent_performance),
        'html_report_path': html_path if classification_result else None,
        'evidently_metrics': classification_result.get('metrics', []) if classification_result else [],
        # EXACT model-quality cohort — written to the bridge with
        # check_type='model_drift' by write_monitoring_results.
        'scored_inference_ids': scored_inference_ids,
    }


def _check_regression_drift():
    """Model-quality drift for a regression target (MAE / RMSE / R²).

    The regression analogue of the classification path in
    ``check_model_drift``: pulls recent predictions + numeric ground truth,
    computes error metrics, compares RMSE against a prior-window baseline,
    and generates an Evidently RegressionPreset report. RMSE is the primary
    metric — degradation is a RELATIVE INCREASE in RMSE (higher = worse), so
    ``degradation_pct >= MODEL_DRIFT_THRESHOLD * 100`` flags drift, matching
    the "percent worse than baseline" semantics the classification path uses.

    Returns a result dict shaped for the same downstream consumers
    (send_sns_alert / write_monitoring_results / publish_cloudwatch_metrics /
    log_to_mlflow) with classification-only fields set to None.
    """
    print(f"🔍 Checking model drift (Evidently, {PROBLEM_TYPE})...")

    # Anchor on THIS run's as-of time (immutable EventBridge event `time`), not
    # wall-clock now(), so a retry re-materializes the same cohort.
    as_of = _run_as_of()
    as_of_str = as_of.strftime('%Y-%m-%d %H:%M:%S')
    lookback_start = (as_of - timedelta(days=MODEL_DRIFT_LOOKBACK_DAYS)).strftime('%Y-%m-%d %H:%M:%S')
    print(f"  Querying predictions with ground truth from last {MODEL_DRIFT_LOOKBACK_DAYS} days "
          f"(since {lookback_start}, as-of {as_of_str})")

    # SELECT inference_id so the EXACT scored rows (post-NaN-drop) are recorded
    # in the monitoring_run_inferences bridge with check_type='model_drift'.
    performance_sql = f"""
    SELECT
        inference_id,
        {PREDICTION_COLUMN} AS prediction,
        ground_truth
    FROM {ATHENA_DATABASE}.inference_responses
    WHERE ground_truth IS NOT NULL
      AND COALESCE(ground_truth_timestamp, request_timestamp) >= TIMESTAMP '{lookback_start}'
      AND COALESCE(ground_truth_timestamp, request_timestamp) <= TIMESTAMP '{as_of_str}'
      {_endpoint_predicate()}
    ORDER BY COALESCE(ground_truth_timestamp, request_timestamp), inference_id
    LIMIT 10000
    """
    recent_performance = execute_athena_query(performance_sql)

    if len(recent_performance) == 0:
        print(f"⚠️ No labeled predictions in the last {MODEL_DRIFT_LOOKBACK_DAYS} days")
        _record_status('model', 'NO_DATA')
        return None
    if len(recent_performance) < MIN_SAMPLES:
        print(f"⚠️ Not enough samples with ground truth ({len(recent_performance)} < {MIN_SAMPLES})")
        _record_status('model', 'INSUFFICIENT_SAMPLES')
        return None

    print(f"✓ Found {len(recent_performance)} samples with ground truth")

    current_df = pd.DataFrame(recent_performance)
    current_df['prediction'] = pd.to_numeric(current_df['prediction'], errors='coerce')
    current_df['ground_truth'] = pd.to_numeric(current_df['ground_truth'], errors='coerce')
    current_df = current_df.dropna(subset=['prediction', 'ground_truth'])

    if len(current_df) < MIN_SAMPLES:
        print(f"⚠️ Not enough numeric samples after parsing ({len(current_df)} < {MIN_SAMPLES})")
        _record_status('model', 'INSUFFICIENT_SAMPLES')
        return None

    # Exact scored cohort = rows surviving the NaN drop; capture ids then drop
    # the column so the Evidently regression report sees its usual schema.
    scored_inference_ids = (
        current_df['inference_id'].dropna().astype(str).tolist()
        if 'inference_id' in current_df.columns else []
    )
    if 'inference_id' in current_df.columns:
        current_df = current_df.drop(columns=['inference_id'])

    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_true = current_df['ground_truth'].values
    y_pred = current_df['prediction'].values

    current_mae = float(mean_absolute_error(y_true, y_pred))
    current_rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    current_r2 = float(r2_score(y_true, y_pred))

    print(f"  Current MAE:  {current_mae:.4f}")
    print(f"  Current RMSE: {current_rmse:.4f}")
    print(f"  Current R²:   {current_r2:.4f}")

    # Baseline RMSE: baseline.json 'rmse' if present, else BASELINE_RMSE env,
    # else the prior-window RMSE computed the same way.
    baseline = load_baseline_from_registry()
    baseline_metrics = (baseline or {}).get('metrics', {}) if baseline else {}

    baseline_rmse = None
    baseline_source = None
    if 'rmse' in baseline_metrics:
        baseline_rmse = float(baseline_metrics['rmse'])
        baseline_source = baseline.get('model_package_arn', 'registered baseline.json')

    # Prior-window baseline DataFrame (for the Evidently report + RMSE fallback).
    baseline_sql = f"""
    SELECT
        {PREDICTION_COLUMN} AS prediction,
        ground_truth
    FROM {ATHENA_DATABASE}.inference_responses
    WHERE ground_truth IS NOT NULL
      AND COALESCE(ground_truth_timestamp, request_timestamp) < TIMESTAMP '{lookback_start}'
      {_endpoint_predicate()}
    ORDER BY RANDOM()
    LIMIT 10000
    """
    try:
        baseline_data = execute_athena_query(baseline_sql)
        baseline_df = pd.DataFrame(baseline_data)
        if len(baseline_df) >= MIN_SAMPLES:
            baseline_df['prediction'] = pd.to_numeric(baseline_df['prediction'], errors='coerce')
            baseline_df['ground_truth'] = pd.to_numeric(baseline_df['ground_truth'], errors='coerce')
            baseline_df = baseline_df.dropna(subset=['prediction', 'ground_truth'])
        if len(baseline_df) < MIN_SAMPLES:
            baseline_df = current_df.copy()
    except Exception:
        baseline_df = current_df.copy()

    if baseline_rmse is None:
        # Fall back to the prior-window RMSE (or env, or current as last resort).
        env_rmse = os.getenv('BASELINE_RMSE')
        if env_rmse is not None:
            baseline_rmse = float(env_rmse)
            baseline_source = 'env:BASELINE_RMSE'
        elif len(baseline_df) >= MIN_SAMPLES:
            baseline_rmse = float(np.sqrt(mean_squared_error(
                baseline_df['ground_truth'].values, baseline_df['prediction'].values
            )))
            baseline_source = 'prior-window RMSE'
        else:
            baseline_rmse = current_rmse
            baseline_source = 'current-window RMSE (no baseline available)'

    # Degradation = relative INCREASE in RMSE (higher RMSE = worse).
    degradation = current_rmse - baseline_rmse
    degradation_pct = (degradation / baseline_rmse) * 100 if baseline_rmse else 0.0

    print(f"  Baseline RMSE: {baseline_rmse:.4f}  ← {baseline_source}")
    print(f"  RMSE increase: {degradation:.4f} ({degradation_pct:.1f}%)")

    # Evidently RegressionPreset report.
    html_path = tempfile.NamedTemporaryFile(
        suffix='.html', prefix='model_reg_', delete=False, dir='/tmp'
    ).name
    regression_result = None
    try:
        regression_result = run_regression_report(
            baseline_df=baseline_df,
            current_df=current_df,
            target_column='ground_truth',
            prediction_column='prediction',
            output_path=html_path,
        )
    except ValueError as e:
        print(f"  ⚠ Skipping Evidently regression report: {e}")

    detected = degradation_pct >= (MODEL_DRIFT_THRESHOLD * 100)
    if detected:
        print("  🚨 Model performance drift DETECTED (RMSE increase)")
    else:
        print("  ✓ No model performance drift detected")

    _record_status('model', 'SUCCESS')
    return {
        'detected': detected,
        'problem_type': PROBLEM_TYPE,
        'primary_metric': 'rmse',
        'baseline_primary': baseline_rmse,
        'current_primary': current_rmse,
        # Classification-only fields left None so downstream consumers that
        # read them (write_monitoring_results, publish, alert) degrade cleanly.
        'baseline_roc_auc': None,
        'current_roc_auc': None,
        'accuracy': None,
        'precision': None,
        'recall': None,
        'f1': None,
        'degradation': degradation,
        'degradation_pct': degradation_pct,
        'mae': current_mae,
        'rmse': current_rmse,
        'r2': current_r2,
        'sample_size': len(current_df),
        'html_report_path': html_path if regression_result else None,
        'evidently_metrics': regression_result.get('metrics', []) if regression_result else [],
        # EXACT model-quality cohort — written to the bridge with
        # check_type='model_drift' by write_monitoring_results.
        'scored_inference_ids': scored_inference_ids,
    }


def _build_alert_message(data_drift_result, model_drift_result):
    """Render the (subject, message) for a drift alert, or (None, None) when
    no drift was detected.

    Extracted from send_sns_alert so both the legacy fire-and-forget path and
    the durable-outbox notifier (notify_drift_alert) format IDENTICAL alert
    bodies from one place.
    """
    data_drift_detected = data_drift_result and data_drift_result.get('detected', False)
    model_drift_detected = model_drift_result and model_drift_result.get('detected', False)

    if not data_drift_detected and not model_drift_detected:
        return None, None

    subject = "🚨 ML Model Drift Alert - Fraud Detection"

    message_lines = [
        "=" * 80,
        "ML MODEL DRIFT ALERT",
        "=" * 80,
        f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Detection Engine: Evidently AI",
        "",
    ]

    if data_drift_detected:
        message_lines.extend([
            "🔴 DATA DRIFT DETECTED (Evidently DataDriftPreset)",
            "=" * 80,
            f"Features Analyzed: {data_drift_result['features_analyzed']}",
            f"Drifted Features: {data_drift_result['drifted_features_count']} "
            f"({data_drift_result['drift_percentage']:.1f}%)",
            f"Drifted Columns Share: {data_drift_result['drifted_columns_share']:.1%}",
            "",
            "Top Drifted Features (by drift score):",
        ])

        for feat_info in data_drift_result.get('drifted_features', []):
            message_lines.append(
                f"  - {feat_info['feature']}: "
                f"drift_score={feat_info['drift_score']:.4f}"
            )

        message_lines.append("")

    if model_drift_detected:
        # The alert body adapts to the problem type: binary reports ROC-AUC +
        # accuracy/precision/recall, multiclass reports accuracy (its primary)
        # + macro precision/recall, and regression reports RMSE/MAE/R². Every
        # value is formatted through _fmt so a None (e.g. ROC-AUC on a
        # regression run) renders as "n/a" instead of crashing on ``:.4f``.
        def _fmt(val, spec='.4f'):
            if val is None:
                return "n/a"
            try:
                return format(float(val), spec)
            except (TypeError, ValueError):
                return str(val)

        problem_type = model_drift_result.get('problem_type', 'binary_classification')
        primary_metric = model_drift_result.get('primary_metric', 'roc_auc')
        preset = 'RegressionPreset' if problem_type == 'regression' else 'ClassificationPreset'

        message_lines.extend([
            f"🔴 MODEL PERFORMANCE DRIFT DETECTED (Evidently {preset})",
            "=" * 80,
            f"Problem Type: {problem_type}",
            f"Primary Metric: {primary_metric.upper()}",
            f"Baseline {primary_metric.upper()}: {_fmt(model_drift_result.get('baseline_primary'))}",
            f"Current {primary_metric.upper()}:  {_fmt(model_drift_result.get('current_primary'))}",
            f"Degradation: {_fmt(model_drift_result.get('degradation'))} "
            f"({_fmt(model_drift_result.get('degradation_pct'), '.1f')}%)",
            f"Threshold: {MODEL_DRIFT_THRESHOLD * 100:.1f}%",
            "",
        ])

        if problem_type == 'regression':
            message_lines.extend([
                f"Current MAE:  {_fmt(model_drift_result.get('mae'))}",
                f"Current RMSE: {_fmt(model_drift_result.get('rmse'))}",
                f"Current R²:   {_fmt(model_drift_result.get('r2'))}",
                "",
            ])
        else:
            # Binary shows ROC-AUC; multiclass has it as None → renders "n/a".
            message_lines.extend([
                f"Current ROC-AUC: {_fmt(model_drift_result.get('current_roc_auc'))}",
                f"Current Accuracy: {_fmt(model_drift_result.get('accuracy'))}",
                f"Current Precision: {_fmt(model_drift_result.get('precision'))}",
                f"Current Recall: {_fmt(model_drift_result.get('recall'))}",
                "",
            ])

    message_lines.extend([
        "=" * 80,
        "RECOMMENDED ACTIONS:",
        "=" * 80,
        "1. Review Evidently HTML reports in MLflow monitoring experiment",
        "2. Investigate root cause of drift (data quality, population shift, etc.)",
        "3. Consider retraining model with recent data",
        "4. Review and adjust decision thresholds if needed",
        "",
        "View detailed Evidently reports in MLflow artifacts or 3_inference_monitoring.ipynb",
        "=" * 80,
    ])

    message = "\n".join(message_lines)
    return subject, message


def send_sns_alert(data_drift_result, model_drift_result):
    """Send SNS notification if drift detected (fire-and-forget).

    Retained for callers/tests that don't need outbox de-duplication. The
    scheduled handler uses notify_drift_alert (durable outbox) instead.
    """
    if not SNS_TOPIC_ARN:
        print("⚠️ SNS_TOPIC_ARN not configured, skipping notification")
        return

    subject, message = _build_alert_message(data_drift_result, model_drift_result)
    if subject is None:
        print("✓ No drift detected, no alert sent")
        return

    # Send SNS notification
    try:
        response = sns.publish(
            TopicArn=SNS_TOPIC_ARN,
            Subject=subject,
            Message=message
        )
        print(f"✓ SNS alert sent: {response['MessageId']}")
    except Exception as e:
        print(f"❌ Failed to send SNS alert: {e}")


# =========================================================================
# Durable alert outbox — exactly-once-ish drift notifications (Finding 6)
#
# EventBridge → Lambda is asynchronous and retried, and the handler itself
# now re-raises on membership failure, so the same (stable) run_id can be
# processed more than once. A naive sns.publish() on every pass would send
# duplicate drift alerts. The monitoring_alerts table is an outbox keyed on
# (monitoring_run_id, alert_type): a PENDING claim is inserted once, the SNS
# publish happens, then the row is flipped to SENT. A retry that sees SENT
# skips publishing; a retry that sees PENDING (publish previously failed
# mid-flight) re-attempts. Publish failures re-raise so the run is retried.
# =========================================================================

def _alert_status(run_id, alert_type):
    """Return the current outbox status for (run_id, alert_type), or None if
    no claim row exists yet."""
    db = ATHENA_DATABASE
    table = ATHENA_ALERTS_TABLE
    rid = str(run_id).replace("'", "''")
    at = str(alert_type).replace("'", "''")
    rows = execute_athena_query(
        f"SELECT status FROM {db}.{table} "
        f"WHERE monitoring_run_id = '{rid}' AND alert_type = '{at}' LIMIT 1"
    )
    if rows and rows[0].get('status'):
        return rows[0]['status']
    return None


def _insert_alert_claim(run_id, alert_type, subject, message, created_at):
    """Idempotently stake a PENDING outbox claim for (run_id, alert_type).

    WHEN NOT MATCHED INSERT only — an existing row (PENDING or SENT) is left
    untouched so a retry never resets a SENT alert back to PENDING.
    """
    db = ATHENA_DATABASE
    table = ATHENA_ALERTS_TABLE
    rid = str(run_id).replace("'", "''")
    at = str(alert_type).replace("'", "''")
    subj = str(subject).replace("'", "''")
    msg = str(message).replace("'", "''")
    ts = str(created_at).replace("'", "''")
    execute_athena_query(f"""
    MERGE INTO {db}.{table} AS t
    USING (SELECT '{rid}' AS monitoring_run_id, '{at}' AS alert_type,
                  '{subj}' AS subject, '{msg}' AS message,
                  'PENDING' AS status, TIMESTAMP '{ts}' AS created_at) AS s
    ON t.monitoring_run_id = s.monitoring_run_id AND t.alert_type = s.alert_type
    WHEN NOT MATCHED THEN INSERT
        (monitoring_run_id, alert_type, subject, message, status, created_at)
        VALUES (s.monitoring_run_id, s.alert_type, s.subject, s.message,
                s.status, s.created_at)
    """, wait=True)


def _mark_alert_sent(run_id, alert_type, sns_message_id, sent_at):
    """Flip a PENDING outbox claim to SENT (records the SNS MessageId).

    WHEN MATCHED AND status='PENDING' only — an already-SENT row is left
    untouched, keeping the first send's MessageId/sent_at authoritative.
    """
    db = ATHENA_DATABASE
    table = ATHENA_ALERTS_TABLE
    rid = str(run_id).replace("'", "''")
    at = str(alert_type).replace("'", "''")
    mid = str(sns_message_id).replace("'", "''")
    ts = str(sent_at).replace("'", "''")
    execute_athena_query(f"""
    MERGE INTO {db}.{table} AS t
    USING (SELECT '{rid}' AS monitoring_run_id, '{at}' AS alert_type,
                  '{mid}' AS sns_message_id, TIMESTAMP '{ts}' AS sent_at) AS s
    ON t.monitoring_run_id = s.monitoring_run_id AND t.alert_type = s.alert_type
    WHEN MATCHED AND t.status = 'PENDING' THEN UPDATE SET
        status = 'SENT', sns_message_id = s.sns_message_id, sent_at = s.sent_at
    """, wait=True)


def notify_drift_alert(run_id, data_drift_result, model_drift_result):
    """Send the drift alert exactly-once-ish via the monitoring_alerts outbox.

    Called AFTER the monitoring row + bridge membership are durably persisted,
    so an alert never fires for a run whose results didn't land. Idempotent
    across EventBridge retries of the same run_id: a SENT claim short-circuits,
    a PENDING claim (previous publish failed) re-attempts. A publish failure
    re-raises so the handler surfaces it and the run is retried.
    """
    if not SNS_TOPIC_ARN:
        print("⚠️ SNS_TOPIC_ARN not configured, skipping notification")
        return

    subject, message = _build_alert_message(data_drift_result, model_drift_result)
    if subject is None:
        print("✓ No drift detected, no alert sent")
        return

    alert_type = 'drift'
    # Skip if this run's drift alert already went out (retry / duplicate).
    if _alert_status(run_id, alert_type) == 'SENT':
        print(f"✓ Drift alert for run {run_id} already SENT; skipping duplicate")
        return

    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # Stake the claim before publishing so the intent is durable even if the
    # publish (or this invocation) dies immediately after.
    _insert_alert_claim(run_id, alert_type, subject, message, now_str)

    response = sns.publish(TopicArn=SNS_TOPIC_ARN, Subject=subject, Message=message)
    message_id = response.get('MessageId', '')
    print(f"✓ SNS alert sent: {message_id}")

    sent_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    _mark_alert_sent(run_id, alert_type, message_id, sent_str)


# =========================================================================
# Legacy chart functions (kept for reference)
#
# These show how to build custom matplotlib visualizations for drift
# analysis. The active Lambda flow now logs Evidently's interactive HTML
# reports as MLflow artifacts instead.
# =========================================================================

def create_psi_chart(drift_results):
    """Create PSI bar chart visualization.

    LEGACY — Replaced by Evidently HTML data drift report logged as an
    MLflow artifact. Kept to demonstrate custom matplotlib charting.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if not drift_results:
        return None

    # Sort by PSI value
    sorted_results = sorted(drift_results, key=lambda x: x['psi'], reverse=True)[:15]

    features = [r['feature'] for r in sorted_results]
    psi_values = [r['psi'] for r in sorted_results]
    colors = ['red' if r['drifted'] else 'green' for r in sorted_results]

    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(features, psi_values, color=colors, alpha=0.7)

    ax.axvline(x=DATA_DRIFT_THRESHOLD, color='orange', linestyle='--',
               linewidth=2, label=f'Threshold ({DATA_DRIFT_THRESHOLD})')

    ax.set_xlabel('Population Stability Index (PSI)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Features', fontsize=12, fontweight='bold')
    ax.set_title('Data Drift Analysis - PSI by Feature', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)

    for i, (bar, val) in enumerate(zip(bars, psi_values)):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=9)

    plt.tight_layout()

    temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    plt.savefig(temp_file.name, dpi=150, bbox_inches='tight')
    plt.close()

    return temp_file.name


def create_model_performance_chart(model_drift_result):
    """Create model performance comparison chart.

    LEGACY — Replaced by Evidently HTML classification report logged as
    an MLflow artifact. Kept to demonstrate custom matplotlib charting.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if not model_drift_result:
        return None

    metrics = ['ROC-AUC', 'Accuracy', 'Precision', 'Recall']
    baseline_values = [
        model_drift_result['baseline_roc_auc'],
        0.95,
        0.90,
        0.85
    ]
    current_values = [
        model_drift_result['current_roc_auc'],
        model_drift_result['accuracy'],
        model_drift_result['precision'],
        model_drift_result['recall']
    ]

    x = range(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar([i - width/2 for i in x], baseline_values, width,
                    label='Baseline', alpha=0.8, color='green')
    bars2 = ax.bar([i + width/2 for i in x], current_values, width,
                    label='Current', alpha=0.8, color='blue')

    ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.1)

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    plt.savefig(temp_file.name, dpi=150, bbox_inches='tight')
    plt.close()

    return temp_file.name


# =========================================================================
# MLflow logging — logs Evidently HTML reports as artifacts
# =========================================================================

def log_to_mlflow(data_drift_result, model_drift_result):
    """Log drift metrics and Evidently HTML reports to MLflow.

    Returns:
        str: The MLflow run ID, or None if logging failed/skipped
    """
    if not MLFLOW_AVAILABLE:
        print("⚠️ MLflow not available - skipping MLflow logging")
        return None

    if not MLFLOW_TRACKING_URI:
        print("⚠️ MLFLOW_TRACKING_URI not configured - skipping MLflow logging")
        return None

    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment("fraud-detection-drift_monitoring")

        with mlflow.start_run(run_name=f"drift-check-{datetime.now().strftime('%Y%m%d-%H%M%S')}") as run:
            # Capture the run ID
            run_id = run.info.run_id

            # Tag the run with the immutable references from baseline.json.
            # These let the MLflow UI group/filter drift runs by code version
            # (tags.code_commit_sha) and by deployed model (tags.model_package_arn)
            # — answering "which drift checks ran against model X?" with a
            # single MLflow filter.
            baseline = _safe_load_baseline() or {}
            mlflow.set_tags({
                'run_type': 'drift_check',
                'detection_engine': 'evidently',
                'endpoint_name': ENDPOINT_NAME or 'unknown',
                'model_package_arn': baseline.get('model_package_arn') or 'unresolved',
                'code_commit_sha': baseline.get('code_commit_sha') or 'unknown',
                'evaluation_snapshot_id': baseline.get('evaluation_snapshot_id') or 'live',
                'feature_schema_version': str(baseline.get('feature_schema_version') or 0),
            })

            # Log configuration parameters
            mlflow.log_param("detection_engine", "evidently")
            mlflow.log_param("model_drift_threshold", MODEL_DRIFT_THRESHOLD)
            mlflow.log_param("min_samples", MIN_SAMPLES)

            # --- Data drift metrics + Evidently report ---
            if data_drift_result:
                mlflow.log_metric("data_drift_detected", 1 if data_drift_result['detected'] else 0)
                mlflow.log_metric("features_analyzed", data_drift_result['features_analyzed'])
                mlflow.log_metric("drifted_features_count", data_drift_result['drifted_features_count'])
                mlflow.log_metric("drift_percentage", data_drift_result['drift_percentage'])
                mlflow.log_metric("drifted_columns_share", data_drift_result['drifted_columns_share'])
                mlflow.log_metric("data_sample_size", data_drift_result['sample_size'])

                # Log per-feature drift scores
                for feat_info in data_drift_result.get('drifted_features', []):
                    mlflow.log_metric(
                        f"drift_score_{feat_info['feature']}",
                        feat_info['drift_score'],
                    )

                # Log Evidently HTML report as artifact
                html_path = data_drift_result.get('html_report_path')
                if html_path and os.path.exists(html_path):
                    mlflow.log_artifact(html_path, "evidently_reports")
                    os.unlink(html_path)

            # --- Model drift metrics + Evidently report ---
            if model_drift_result:
                # mlflow.log_metric rejects None, so only log metrics the run
                # actually produced. A regression run has None for ROC-AUC /
                # accuracy / precision / recall and instead carries mae/rmse/r2;
                # a multiclass run has None for ROC-AUC. _log_metric skips any
                # None so the same call list works for every problem type.
                def _log_metric(name, value):
                    if value is None:
                        return
                    try:
                        mlflow.log_metric(name, float(value))
                    except (TypeError, ValueError):
                        pass

                mlflow.log_metric("model_drift_detected", 1 if model_drift_result['detected'] else 0)
                # Problem-type-agnostic primary metric (roc_auc | accuracy | rmse).
                mlflow.log_param("model_primary_metric", model_drift_result.get('primary_metric', 'roc_auc'))
                mlflow.log_param("model_problem_type", model_drift_result.get('problem_type', 'binary_classification'))
                _log_metric("baseline_primary", model_drift_result.get('baseline_primary'))
                _log_metric("current_primary", model_drift_result.get('current_primary'))
                _log_metric("baseline_roc_auc", model_drift_result.get('baseline_roc_auc'))
                _log_metric("current_roc_auc", model_drift_result.get('current_roc_auc'))
                # Degradation columns keep their historical roc_auc_* names for
                # dashboard continuity even though they now track whichever
                # primary metric the problem type uses.
                _log_metric("roc_auc_degradation", model_drift_result.get('degradation'))
                _log_metric("roc_auc_degradation_pct", model_drift_result.get('degradation_pct'))
                _log_metric("current_accuracy", model_drift_result.get('accuracy'))
                _log_metric("current_precision", model_drift_result.get('precision'))
                _log_metric("current_recall", model_drift_result.get('recall'))
                # Regression-only quality metrics.
                _log_metric("current_mae", model_drift_result.get('mae'))
                _log_metric("current_rmse", model_drift_result.get('rmse'))
                _log_metric("current_r2", model_drift_result.get('r2'))
                _log_metric("model_sample_size", model_drift_result.get('sample_size'))

                # Log Evidently classification metrics (accuracy, F1, etc.)
                for m in model_drift_result.get('evidently_metrics', []):
                    name = m.get('metric_name', '')
                    value = m.get('value')
                    if isinstance(value, (int, float)):
                        import re
                        # Strip parenthesized args and sanitize for MLflow
                        safe_name = re.sub(r'\([^)]*\)', '', name)
                        safe_name = safe_name.replace('::', '_').replace(' ', '_').lower().strip('_')
                        safe_name = re.sub(r'[^a-z0-9_\-\. /:]', '', safe_name)
                        if safe_name:
                            mlflow.log_metric(f"evidently_{safe_name}", value)

                # Log Evidently HTML report as artifact
                html_path = model_drift_result.get('html_report_path')
                if html_path and os.path.exists(html_path):
                    mlflow.log_artifact(html_path, "evidently_reports")
                    os.unlink(html_path)

            # Log drift summary as JSON artifact
            summary = {
                'timestamp': datetime.now().isoformat(),
                'detection_engine': 'evidently',
                'data_drift': {
                    k: v for k, v in (data_drift_result or {}).items()
                    if k not in ('html_report_path',)
                },
                'model_drift': {
                    k: v for k, v in (model_drift_result or {}).items()
                    if k not in ('html_report_path', 'evidently_metrics')
                },
                'alert_sent': (
                    (data_drift_result and data_drift_result.get('detected', False)) or
                    (model_drift_result and model_drift_result.get('detected', False))
                ),
            }

            summary_file = tempfile.NamedTemporaryFile(
                mode='w', suffix='.json', delete=False
            )
            json.dump(summary, summary_file, indent=2, default=str)
            summary_file.close()
            mlflow.log_artifact(summary_file.name, "drift_reports")
            os.unlink(summary_file.name)

            print("✓ Successfully logged Evidently reports and metrics to MLflow")
            print(f"  MLflow Run ID: {run_id}")

            return run_id

    except Exception as e:
        print(f"⚠️ Failed to log to MLflow: {e}")
        import traceback
        traceback.print_exc()
        return None


# =========================================================================
# Write monitoring results to SQS → Athena monitoring_responses table
# =========================================================================

def write_monitoring_results(data_drift_result, model_drift_result, mlflow_run_id=None,
                             execution_status='SUCCESS', run_id=None):
    """Send monitoring results to SQS for writing to Athena monitoring_responses table.

    A row is written on EVERY run, including no-data runs (both results None) —
    the row still carries the run id, timestamp, and execution_status so the
    monitoring table doubles as an execution history (Model-Monitor parity).

    run_id is a collision-resistant UUID (not a second-resolution timestamp, which
    two runs in the same second would share). It is passed in from the handler so
    the SAME id ties together the persisted monitoring row, the uploaded Evidently
    report prefix, and the inference-row backfill; if omitted (best-effort ERROR
    path) one is generated here.
    """
    if not MONITORING_SQS_QUEUE_URL:
        print("⚠️ MONITORING_SQS_QUEUE_URL not configured - skipping Athena write")
        return

    now = datetime.now()
    if not run_id:
        run_id = f"drift-{uuid.uuid4().hex}"

    # Build per-feature drift scores JSON. Nested per-feature object so the
    # governance dashboard can plot the test-agnostic magnitude alongside the
    # raw score. The feature_drift_detail Athena view parses this shape
    # (MAP(VARCHAR, JSON)).
    #
    # Write EVERY analyzed feature (drifted or not) from the full per_column
    # result, matching what notebook 3 (cell "Write the monitoring run")
    # persists. The `drifted_features` list is filtered + top-5 capped for the
    # SNS alert; using it here would drop non-drifted and beyond-top-5 features
    # from the dashboard, so Lambda-generated runs would show a thinner feature
    # set than notebook-generated ones. Iterate per_column instead.
    per_feature = {}
    if data_drift_result:
        for col, info in data_drift_result.get('per_column', {}).items():
            per_feature[col] = {
                'score': info.get('drift_score', 0),
                'magnitude': info.get('drift_magnitude', 0),
                'method': info.get('method', ''),
                'threshold': info.get('threshold', 0),
            }

    # True F1 comes from the check (macro-F1 for multiclass, positive-class for
    # binary), NOT recomputed here as the harmonic mean of macro-precision and
    # macro-recall — that harmonic mean is NOT macro-F1 and was wrong for the
    # multiclass path. Regression runs carry f1=None.
    f1 = model_drift_result.get('f1') if model_drift_result else None

    # A check that did NOT execute (result is None) leaves its verdict NULL — it
    # must not read as a real "no drift" verdict. Only a check that actually ran
    # contributes a boolean.
    data_detected = data_drift_result.get('detected') if data_drift_result else None
    model_detected = model_drift_result.get('detected') if model_drift_result else None

    # Per-check execution statuses (from RUN_DIAGNOSTICS) so the record carries
    # WHY each check did/didn't produce a verdict, not just the collapsed
    # overall status. NOTE: added SQS record keys (data_drift_status /
    # model_drift_status) — the current writer/DDL have no column for them, so
    # they round-trip through the SQS body but are not persisted to Athena
    # without a new column (flagged in the report).
    data_status = RUN_DIAGNOSTICS.get('data')
    model_status = RUN_DIAGNOSTICS.get('model')

    # Binary-only ROC-AUC-named metrics. For multiclass/regression these must
    # stay None so a regression RMSE increase never leaks into roc_auc_degradation
    # etc.; the generic primary_metric / baseline_primary / current_primary /
    # mae/rmse/r2 fields carry those values instead.
    is_binary = bool(
        model_drift_result
        and model_drift_result.get('problem_type', 'binary_classification') == 'binary_classification'
    )

    # Stamp the resolved ModelPackage ARN + Iceberg snapshot ID. These are
    # the immutable references that let you query monitoring_responses per
    # model version — joining on a human-readable label like model_version
    # silently mixes results across rollouts. Uses the fail-closed-safe loader:
    # a baseline-resolution failure was already surfaced as an execution_status
    # by the handler, so this record-writing path must not re-raise.
    baseline = _safe_load_baseline()
    model_package_arn = (baseline or {}).get('model_package_arn')
    evaluation_snapshot_id = (baseline or {}).get('evaluation_snapshot_id')
    # training_snapshot_id was previously used internally for the FOR VERSION
    # AS OF baseline pull but NOT persisted per-run. Persisting it makes the
    # Model Lineage dashboard sheet able to answer "which training-data
    # snapshot underlay every drift verdict" without joining back to
    # baseline.json in the model registry.
    training_snapshot_id = (baseline or {}).get('training_snapshot_id')

    record = {
        'monitoring_run_id': run_id,
        'monitoring_timestamp': now.strftime('%Y-%m-%d %H:%M:%S'),
        'endpoint_name': ENDPOINT_NAME or os.getenv('ENDPOINT_NAME', 'fraud-detector-endpoint'),
        'model_version': os.getenv('MODEL_VERSION', 'latest'),
        'model_package_arn': model_package_arn,
        'evaluation_snapshot_id': evaluation_snapshot_id,
        'training_snapshot_id': training_snapshot_id,
        'data_drift_detected': data_detected,
        'drifted_columns_count': data_drift_result.get('drifted_features_count', 0) if data_drift_result else None,
        'drifted_columns_share': data_drift_result.get('drifted_columns_share', 0) if data_drift_result else None,
        'features_analyzed': data_drift_result.get('features_analyzed', 0) if data_drift_result else None,
        'data_sample_size': data_drift_result.get('sample_size', 0) if data_drift_result else None,
        'model_drift_detected': model_detected,
        # ROC-AUC-named columns are BINARY-ONLY. For multiclass/regression they
        # stay None (the generic primary_metric/baseline_primary/current_primary
        # and mae/rmse/r2 below carry the real values) so a non-binary
        # degradation never leaks into a ROC-AUC-labeled column.
        'baseline_roc_auc': model_drift_result.get('baseline_roc_auc') if is_binary else None,
        'current_roc_auc': model_drift_result.get('current_roc_auc') if is_binary else None,
        'roc_auc_degradation': model_drift_result.get('degradation') if is_binary else None,
        'roc_auc_degradation_pct': model_drift_result.get('degradation_pct') if is_binary else None,
        'accuracy': model_drift_result.get('accuracy') if model_drift_result else None,
        'precision': model_drift_result.get('precision') if model_drift_result else None,
        'recall': model_drift_result.get('recall') if model_drift_result else None,
        'f1_score': f1,
        # Problem-type-aware fields — persisted to monitoring_responses so a
        # multiclass or regression deployment's runs are fully queryable in
        # Athena, not just observable in CloudWatch/MLflow. These map to the 10
        # columns appended to the monitoring_responses DDL (see
        # create_athena_tables.py + deploy_monitoring_writer.py). For a binary
        # run mae/rmse/r2 are None; for a regression run accuracy/precision/
        # recall/roc_auc are None. degradation (RMSE increase for regression)
        # also lands in the generic roc_auc_degradation column above.
        'problem_type': model_drift_result.get('problem_type') if model_drift_result else None,
        'primary_metric': model_drift_result.get('primary_metric') if model_drift_result else None,
        'baseline_primary': model_drift_result.get('baseline_primary') if model_drift_result else None,
        'current_primary': model_drift_result.get('current_primary') if model_drift_result else None,
        'mae': model_drift_result.get('mae') if model_drift_result else None,
        'rmse': model_drift_result.get('rmse') if model_drift_result else None,
        'r2': model_drift_result.get('r2') if model_drift_result else None,
        # Explicit per-run execution state (SUCCESS | PARTIAL_SUCCESS | NO_DATA |
        # INSUFFICIENT_SAMPLES | INSUFFICIENT_CLASSES | ERROR |
        # BASELINE_RESOLUTION_FAILED | MULTI_VARIANT_UNSUPPORTED) — lets the table
        # answer "did the monitor actually run and score data?" the way Model
        # Monitor's execution status does, instead of inferring it from NULL
        # metric columns.
        'execution_status': execution_status,
        # Per-check execution statuses — persisted to the data_drift_status /
        # model_drift_status columns (last two of the appended 10) so a
        # PARTIAL_SUCCESS run's data-vs-model breakdown is queryable in Athena,
        # not just carried in the SQS body.
        'data_drift_status': data_status,
        'model_drift_status': model_status,
        'model_sample_size': model_drift_result.get('sample_size') if model_drift_result else None,
        'per_feature_drift_scores': json.dumps(per_feature) if per_feature else None,
        # Durable per-run Evidently report location (uploaded by the handler
        # before MLflow unlinks the temp files). Prefer the data-drift report,
        # fall back to the model-drift report; both live under the same run
        # folder anyway.
        'evidently_report_s3_path': (
            (data_drift_result or {}).get('evidently_report_s3_path')
            or (model_drift_result or {}).get('evidently_report_s3_path')
        ),
        'mlflow_run_id': mlflow_run_id,
        'alert_sent': bool(data_detected) or bool(model_detected),
        'detection_engine': 'evidently',
        'created_at': now.strftime('%Y-%m-%d %H:%M:%S'),
    }

    # PERSIST-BEFORE-BACKFILL. The history row (monitoring_responses) and the
    # inference-row backfill must be ONE recoverable sequence: if we tag
    # inference rows with a run_id whose history row never landed, those rows are
    # orphaned (they point at a nonexistent monitoring_run_id) AND their NULL
    # slot is consumed so a healthy retry skips them. So: send the record, then
    # CONFIRM it is durably queryable in Athena before backfilling. If the send
    # raises, nothing was persisted — do NOT backfill.
    try:
        sqs.send_message(
            QueueUrl=MONITORING_SQS_QUEUE_URL,
            MessageBody=json.dumps(record, default=str),
        )
        print(f"✓ Monitoring results sent to SQS: {run_id}")
    except Exception as e:
        # PERSIST_UNCONFIRMED — the row was never enqueued, so the backfill is
        # skipped and the inference rows stay untagged for a later (idempotent)
        # retry. Logged rather than raised so a transient SQS blip doesn't fail
        # the whole run.
        print(f"❌ Failed to send monitoring results to SQS (PERSIST_UNCONFIRMED): {e}")
        return

    # EXACT-membership recording (replaces the legacy timestamp-window backfill
    # onto inference_responses.monitoring_run_id). Each check contributes the
    # precise inference_ids it actually scored to the monitoring_run_inferences
    # bridge, tagged by check_type — so the data-drift (request-time) cohort and
    # the model-quality (ground-truth-arrival-time) cohort are recorded
    # SEPARATELY and exactly, and malformed / over-limit / boundary rows are
    # never claimed. Only run when analysis actually happened.
    if not (ENDPOINT_NAME and execution_status in ('SUCCESS', 'PARTIAL_SUCCESS')):
        return

    data_ids = (data_drift_result or {}).get('scored_inference_ids') or []
    model_ids = (model_drift_result or {}).get('scored_inference_ids') or []
    if not data_ids and not model_ids:
        return

    # Confirm the history row is durably persisted before recording membership
    # so bridge rows never reference a monitoring_run_id whose summary row never
    # landed (which would orphan them). If unconfirmed, skip — a later retry of
    # the same (stable) run_id re-records membership idempotently.
    if not _confirm_persisted(run_id):
        print(f"⚠️ Monitoring row {run_id} not confirmed persisted after retries "
              f"(PERSIST_UNCONFIRMED); skipping membership write — recorded on a "
              f"later run.")
        return

    created_at = now.strftime('%Y-%m-%d %H:%M:%S')
    membership_errors = []
    for check_type, ids in (('data_drift', data_ids), ('model_drift', model_ids)):
        if not ids:
            continue
        try:
            # Insert + count-verify the membership, THEN stamp the completion
            # marker. The marker is the cohort's commit point (Finding 4): it is
            # written ONLY after _write_run_inferences confirmed exactly len(ids)
            # rows landed, so a consumer that gates on a COMPLETE marker with
            # expected_count == actual_count never joins a half-written cohort.
            written = _write_run_inferences(run_id, ENDPOINT_NAME, check_type, ids, created_at)
            _write_generation_marker(run_id, check_type, len(ids), written, created_at)
        except Exception as e:
            # A membership-write failure now PROPAGATES (Finding 4): the marker
            # was NOT stamped, so the cohort is incomplete. Re-raising lets the
            # handler surface the error and EventBridge retry the (stable) run_id,
            # which re-materializes the deterministic cohort and re-records
            # membership idempotently (delete → insert → verify → marker).
            print(f"⚠️ Bridge write failed for {check_type} (run {run_id}): {e}")
            membership_errors.append(f"{check_type}: {e}")

    if membership_errors:
        raise RuntimeError(
            f"Bridge membership incomplete for run {run_id}: "
            + "; ".join(membership_errors)
        )


# =========================================================================
# CloudWatch metrics — published on every scheduled run
#
# The scheduled Lambda already fans results out to MLflow (artifacts),
# Athena (monitoring_responses), and SNS (alerts). This adds the fourth
# sink SageMaker Model Monitor gave you for free: automatic per-run
# CloudWatch metrics, so the alarms + dashboard built by
# create_cloudwatch_monitoring.py light up from the schedule rather than
# only when that script is run by hand.
#
# Metric names are chosen to match the widgets/alarms in
# create_cloudwatch_monitoring.py. All metrics carry an `Endpoint`
# dimension so multiple endpoints publishing to the same namespace stay
# separable.
# =========================================================================

def publish_cloudwatch_metrics(data_drift_result, model_drift_result):
    """Publish drift metrics to CloudWatch for the current run.

    Uses the in-memory result dicts (not a re-read from Athena) so the
    numbers are exactly what this run computed. Safe to call with either
    result being None — only the metrics that exist are published.

    Returns the number of metric datums published (0 if nothing to publish
    or the put_metric_data call failed — failures are logged, never raised,
    so a CloudWatch hiccup can't fail an otherwise-successful drift run).
    """
    endpoint = ENDPOINT_NAME or os.getenv('ENDPOINT_NAME', 'fraud-detector-endpoint')
    dimensions = [{'Name': 'Endpoint', 'Value': endpoint}]
    metric_data = []

    def _add(name, value, unit='None'):
        # Guard against None / non-finite values — CloudWatch rejects NaN/inf.
        if value is None:
            return
        try:
            fval = float(value)
        except (TypeError, ValueError):
            return
        if not np.isfinite(fval):
            return
        metric_data.append({
            'MetricName': name,
            'Value': fval,
            'Unit': unit,
            'Dimensions': dimensions,
        })

    # Heartbeat: emitted on EVERY run, including no-data runs where both
    # results are None. Without this a stalled schedule and a run that simply
    # found nothing to analyze look identical in CloudWatch (both "no data").
    # A "missing data" alarm on this metric distinguishes "the monitor stopped
    # running" from "the monitor ran and had nothing to score".
    _add('DriftRunExecuted', 1, unit='Count')

    if data_drift_result:
        _add('DriftedColumnsShare', data_drift_result.get('drifted_columns_share'))
        _add('DriftedColumnsCount', data_drift_result.get('drifted_features_count'), unit='Count')
        _add('DataDriftDetected', 1 if data_drift_result.get('detected') else 0)

    if model_drift_result:
        # PROBLEM-TYPE-AGNOSTIC degradation metric — this is what the model-drift
        # alarm compares against. degradation_pct is the RELATIVE degradation of
        # the primary metric (ROC-AUC / accuracy drop, or RMSE increase) as a
        # percentage; /100 makes it a fraction (0.10 == 10% relative
        # degradation) so a single threshold (MODEL_DRIFT_THRESHOLD, also a
        # fraction) is directly comparable across binary/multiclass/regression.
        # This replaces the old alarm on the ABSOLUTE `ROCAUCDegradation` metric,
        # whose scale never matched the fractional threshold.
        deg_pct = model_drift_result.get('degradation_pct')
        if deg_pct is not None:
            _add('PrimaryMetricDegradationRatio', float(deg_pct) / 100.0)
        # ROC-AUC-named metrics are BINARY-ONLY. For multiclass/regression the
        # ROC-AUC values are None (and a regression `degradation` is an RMSE
        # increase, NOT a ROC-AUC drop) — publishing them under these names
        # would mislabel the metric. Gate on problem_type so a non-binary
        # degradation never lands on ROCAUCDegradation.
        is_binary = model_drift_result.get('problem_type', 'binary_classification') == 'binary_classification'
        if is_binary:
            _add('BaselineROCAUC', model_drift_result.get('baseline_roc_auc'))
            _add('CurrentROCAUC', model_drift_result.get('current_roc_auc'))
            # ABSOLUTE ROC-AUC degradation — binary-only informational context on
            # the dashboard. NOT alarmed on (see PrimaryMetricDegradationRatio).
            _add('ROCAUCDegradation', model_drift_result.get('degradation'))
        _add('Accuracy', model_drift_result.get('accuracy'))
        _add('Precision', model_drift_result.get('precision'))
        _add('Recall', model_drift_result.get('recall'))
        # Regression runs surface MAE/RMSE/R2 instead of the classification
        # metrics above; publish whichever the result carries.
        _add('MAE', model_drift_result.get('mae'))
        _add('RMSE', model_drift_result.get('rmse'))
        _add('R2', model_drift_result.get('r2'))
        _add('ModelDriftDetected', 1 if model_drift_result.get('detected') else 0)

    if not metric_data:
        print("⚠️ No CloudWatch metrics to publish (no drift results this run)")
        return 0

    try:
        # CloudWatch caps put_metric_data at 1000 datums / 20 per request in
        # practice; we send well under that, but batch by 20 to stay safe.
        for i in range(0, len(metric_data), 20):
            cloudwatch.put_metric_data(
                Namespace=CLOUDWATCH_NAMESPACE,
                MetricData=metric_data[i:i + 20],
            )
        print(f"✓ Published {len(metric_data)} metrics to CloudWatch "
              f"namespace {CLOUDWATCH_NAMESPACE} (Endpoint={endpoint})")
        return len(metric_data)
    except Exception as e:
        print(f"⚠️ Failed to publish CloudWatch metrics: {e}")
        return 0


# =========================================================================
# Lambda entry point
# =========================================================================


def _derive_run_id(event):
    """Derive a STABLE per-scheduled-event run id (idempotent across retries).

    EventBridge invokes this function asynchronously and each scheduled tick
    carries a unique, immutable top-level ``id``. That id is preserved verbatim
    when EventBridge retries delivery AND when Lambda re-runs the same async
    event after a handler exception, so keying the run on it makes retries and
    duplicate deliveries collapse onto the SAME monitoring_run_id. The Athena
    writer MERGEs on monitoring_run_id, so a redelivered run is then a no-op
    instead of a duplicate history row / report folder / metric / alert.

    Falls back to a random UUID for manual invocations (console test, local
    ``__main__``, or any event without an ``id``), which have no natural dedup
    key and should each get a fresh identity.
    """
    event_id = (event or {}).get('id')
    if event_id:
        # Sanitize: ids are already URL/path-safe, but guard the S3 report
        # prefix against any stray characters just in case.
        safe = ''.join(c if (c.isalnum() or c in '-_') else '-' for c in str(event_id))
        return f"drift-{safe}"
    return f"drift-{uuid.uuid4().hex}"

def lambda_handler(event, context):
    """Lambda handler for EventBridge scheduled drift monitoring."""
    print("=" * 80)
    print(f"Drift Monitoring Check (Evidently) - {datetime.now()}")
    print("=" * 80)

    # Reset per-invocation execution diagnostics — warm Lambda containers reuse
    # module globals, so a prior run's status must not leak into this one.
    RUN_DIAGNOSTICS['data'] = None
    RUN_DIAGNOSTICS['model'] = None

    # Anchor this run's cohort windows on the immutable EventBridge event `time`
    # (falls back to now() for manual invokes). Warm containers reuse globals,
    # so set it every invocation — see _derive_run_as_of / _run_as_of.
    global _RUN_AS_OF
    _RUN_AS_OF = _derive_run_as_of(event)

    # Stable per-SCHEDULED-EVENT id shared by the monitoring row, the Evidently
    # report S3 prefix, and the inference-row backfill. Derived from the
    # EventBridge event id so retries / duplicate deliveries of the SAME tick
    # reuse it and dedup at the Athena MERGE (see _derive_run_id); manual
    # invocations without an id get a fresh UUID.
    run_id = _derive_run_id(event)

    try:
        # FAIL CLOSED before scoring: resolve the baseline for the configured
        # endpoint up front so a multi-variant or unresolvable endpoint records a
        # specific status and skips scoring rather than emitting a verdict against
        # an arbitrary/unrelated baseline. Only meaningful when an endpoint is
        # configured; the endpoint-less first-run path keeps its existing
        # latest-Approved behavior and is resolved lazily inside the checks.
        if ENDPOINT_NAME:
            try:
                load_baseline_from_registry()
            except BaselineResolutionError as e:
                status = getattr(e, 'status', 'BASELINE_RESOLUTION_FAILED')
                print(f"❌ Baseline resolution failed ({status}): {e}")
                print("   Skipping scoring — recording status and heartbeat only.")
                try:
                    publish_cloudwatch_metrics(None, None)
                except Exception:
                    pass
                try:
                    write_monitoring_results(None, None, None,
                                             execution_status=status, run_id=run_id)
                except Exception:
                    pass
                return {
                    'statusCode': 200,
                    'body': json.dumps({
                        'timestamp': datetime.now().isoformat(),
                        'detection_engine': 'evidently',
                        'execution_status': status,
                        'data_drift': None,
                        'model_drift': None,
                        'alert_sent': False,
                    }, indent=2, default=str),
                }

        # Check data drift (Evidently DataDriftPreset)
        data_drift_result = check_data_drift()

        # Check model drift (Evidently ClassificationPreset)
        model_drift_result = check_model_drift()

        # Collapse the per-check outcomes into one run-level status.
        execution_status = _overall_execution_status()
        print(f"Execution status: {execution_status}")

        # Upload the Evidently HTML reports to a deterministic per-run S3 prefix
        # BEFORE MLflow logs-and-unlinks the temp files, so the persisted row's
        # evidently_report_s3_path points at a durable artifact.
        _upload_evidently_reports(run_id, data_drift_result, model_drift_result)

        # Log Evidently reports and metrics to MLflow (captures run ID)
        mlflow_run_id = log_to_mlflow(data_drift_result, model_drift_result)

        # Publish this run's metrics to CloudWatch so the alarms + dashboard
        # created by create_cloudwatch_monitoring.py update automatically on
        # every scheduled run (not just when that script is run by hand).
        publish_cloudwatch_metrics(data_drift_result, model_drift_result)

        # DURABLE PERSIST BEFORE ALERT (Finding 6): write the monitoring row +
        # bridge membership first. write_monitoring_results now re-raises if the
        # bridge membership can't be completed, so a persistence failure aborts
        # the run BEFORE any alert fires — a drift alert never goes out for a run
        # whose results didn't land, and the retry re-materializes deterministically.
        write_monitoring_results(data_drift_result, model_drift_result, mlflow_run_id,
                                 execution_status=execution_status, run_id=run_id)

        # Alert LAST, via the durable outbox so retries of the same run_id don't
        # re-send. notify_drift_alert re-raises on publish failure → run retried.
        notify_drift_alert(run_id, data_drift_result, model_drift_result)

        # Prepare response (exclude local file paths)
        def _clean(result):
            if result is None:
                return None
            return {
                k: v for k, v in result.items()
                if k not in ('html_report_path', 'evidently_metrics')
            }

        response = {
            'timestamp': datetime.now().isoformat(),
            'detection_engine': 'evidently',
            'execution_status': execution_status,
            'data_drift': _clean(data_drift_result),
            'model_drift': _clean(model_drift_result),
            'alert_sent': (
                (data_drift_result and data_drift_result.get('detected', False)) or
                (model_drift_result and model_drift_result.get('detected', False))
            ),
        }

        print("=" * 80)
        print(f"Drift monitoring check completed ({execution_status})")
        print("=" * 80)

        return {
            'statusCode': 200,
            'body': json.dumps(response, indent=2, default=str)
        }

    except Exception as e:
        print(f"❌ Error during drift monitoring: {e}")
        import traceback
        traceback.print_exc()

        # Record the failed execution best-effort (an ERROR row in
        # monitoring_responses + the CloudWatch heartbeat) so the schedule
        # failure is VISIBLE, then RE-RAISE. Returning a 200/500 body makes an
        # EventBridge-triggered (async) invocation look successful, suppressing
        # the automatic retries and DLQ routing that surface a broken monitor.
        # Model Monitor marks such executions Failed; re-raising does the same.
        try:
            publish_cloudwatch_metrics(None, None)  # emits the DriftRunExecuted heartbeat
        except Exception:
            pass
        try:
            # Reuse the SAME run_id as the (possibly partial) run above so a
            # retry of this event MERGEs onto the existing ERROR row instead of
            # writing a new one. run_id is assigned before the try, so it is
            # always bound here.
            write_monitoring_results(None, None, None,
                                     execution_status='ERROR', run_id=run_id)
        except Exception:
            pass
        raise


if __name__ == '__main__':
    # For local testing
    lambda_handler({}, {})
