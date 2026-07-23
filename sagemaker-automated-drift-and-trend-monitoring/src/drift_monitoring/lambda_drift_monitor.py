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
import boto3
import time
from datetime import datetime, timedelta
import tempfile

import numpy as np
import pandas as pd

# Evidently-based reporting (used by check_data_drift / check_model_drift)
from src.drift_monitoring.evidently_reports import run_data_drift_report, run_classification_report
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

# Configuration from environment variables
ATHENA_DATABASE = os.getenv('ATHENA_DATABASE', 'fraud_detection')
ATHENA_OUTPUT_S3 = os.getenv('ATHENA_OUTPUT_S3', 's3://fraud-detection-data-lake/athena-query-results/')
ATHENA_EVALUATION_TABLE = os.getenv('ATHENA_EVALUATION_TABLE', 'evaluation_data')
MODEL_PACKAGE_GROUP = os.getenv('MODEL_PACKAGE_GROUP', 'fraud-detection')
SNS_TOPIC_ARN = os.getenv('SNS_TOPIC_ARN')
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI')
MONITORING_SQS_QUEUE_URL = os.getenv('MONITORING_SQS_QUEUE_URL', '')

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
        model_name = variants[0]['ModelName']
        model = sagemaker_client.describe_model(ModelName=model_name)
        for container in model.get('Containers', []) or [model.get('PrimaryContainer', {})]:
            arn = container.get('ModelPackageName')
            if arn:
                return arn
        print(f"⚠️ Model {model_name} was not built from a registered ModelPackage")
        return None
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


def load_baseline_from_registry() -> dict | None:
    """Return the baseline.json registered with the model serving the endpoint.

    Resolution order:
      1. Endpoint walk (the deployed model — correct answer)
      2. Latest Approved ModelPackage in MODEL_PACKAGE_GROUP (only valid
         on first-ever monitor runs before any endpoint exists)

    Cached per warm Lambda container.

    Returns the parsed baseline.json with ``model_package_arn`` added,
    or ``None`` if no baseline can be resolved (the caller then falls
    back to env-based defaults — see check_data_drift / check_model_drift).
    """
    if 'value' in _BASELINE_CACHE:
        return _BASELINE_CACHE['value']

    arn = None
    if ENDPOINT_NAME:
        arn = _resolve_model_package_arn_from_endpoint(ENDPOINT_NAME)
    if not arn:
        if ENDPOINT_NAME:
            print(f"  Falling back to latest-Approved lookup in group {MODEL_PACKAGE_GROUP}")
        arn = _latest_approved_model_package_arn()
    if not arn:
        print(f"⚠️ No ModelPackage available (endpoint={ENDPOINT_NAME or '<unset>'}, "
              f"group={MODEL_PACKAGE_GROUP})")
        _BASELINE_CACHE['value'] = None
        return None

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
            print(f"⚠️ ModelPackage {arn} has no ModelStatistics URI — skipping baseline")
            _BASELINE_CACHE['value'] = None
            return None

        bucket, key = s3_uri.replace('s3://', '').split('/', 1)
        body = s3.get_object(Bucket=bucket, Key=key)['Body'].read()
        baseline = json.loads(body)
        baseline['model_package_arn'] = arn
        print(
            f"✓ Loaded baseline from {s3_uri}\n"
            f"  ModelPackage:        {arn}\n"
            f"  Baseline ROC-AUC:    {baseline.get('metrics', {}).get('roc_auc', '?')}\n"
            f"  Evaluation table:    {baseline.get('evaluation_table', '?')}"
            f"  (snapshot {baseline.get('evaluation_snapshot_id') or 'live'})"
        )
        _BASELINE_CACHE['value'] = baseline
        return baseline
    except Exception as e:
        print(f"⚠️ Could not load baseline.json for {arn}: {e}")
        _BASELINE_CACHE['value'] = None
        return None


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

    # Get recent inference data (using configured lookback period)
    lookback_start = (datetime.now() - timedelta(days=DATA_DRIFT_LOOKBACK_DAYS)).strftime('%Y-%m-%d %H:%M:%S')
    print(f"  Querying inference data from last {DATA_DRIFT_LOOKBACK_DAYS} days (since {lookback_start})")

    recent_data_sql = f"""
    SELECT input_features
    FROM {ATHENA_DATABASE}.inference_responses
    WHERE request_timestamp >= TIMESTAMP '{lookback_start}'
    LIMIT 10000
    """

    recent_data = execute_athena_query(recent_data_sql)

    if len(recent_data) < MIN_SAMPLES:
        print(f"⚠️ Not enough recent samples ({len(recent_data)} < {MIN_SAMPLES})")
        return None

    print(f"✓ Found {len(recent_data)} recent inference samples")

    # Parse JSON features into a DataFrame
    current_rows = []
    for row in recent_data:
        try:
            features = json.loads(row['input_features'])
            parsed = {}
            for feat in TRAINING_FEATURES:
                if feat in features:
                    parsed[feat] = float(features[feat])
            if parsed:
                current_rows.append(parsed)
        except Exception:
            continue

    if len(current_rows) < MIN_SAMPLES:
        print(f"⚠️ Not enough parseable samples ({len(current_rows)} < {MIN_SAMPLES})")
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

    # Sort by drift score ascending (lower p-value = more drifted)
    drifted_features.sort(key=lambda x: x['drift_score'])

    features_analyzed = len(per_column)
    drifted_count = drift_result['drifted_columns_count']
    drift_share = drift_result['drifted_columns_share']

    print(f"  Evidently: {drifted_count}/{features_analyzed} features drifted ({drift_share:.1%})")
    if drift_result['drift_detected']:
        print("  🚨 Overall data drift DETECTED")
    else:
        print("  ✓ No overall data drift detected")

    return {
        'detected': drift_result['drift_detected'],
        'features_analyzed': features_analyzed,
        'drifted_features_count': drifted_count,
        'drift_percentage': drift_share * 100,
        'drifted_columns_share': drift_share,
        'drifted_features': drifted_features[:5],  # Top 5
        'sample_size': len(current_rows),
        'html_report_path': html_path,
    }


def check_model_drift():
    """Check for model performance drift using Evidently ClassificationPreset.

    Queries recent predictions with ground truth from Athena, builds a
    baseline comparison DataFrame, and runs Evidently's ClassificationPreset.

    Returns:
        dict with model drift results or None if insufficient data.
    """
    print("🔍 Checking model drift (Evidently)...")

    # Get recent predictions with ground truth (using configured lookback period)
    lookback_start = (datetime.now() - timedelta(days=MODEL_DRIFT_LOOKBACK_DAYS)).strftime('%Y-%m-%d %H:%M:%S')
    print(f"  Querying predictions with ground truth from last {MODEL_DRIFT_LOOKBACK_DAYS} days (since {lookback_start})")

    # SELECT the config-driven prediction/probability columns and alias them
    # to stable in-Python names so the rest of this function keeps using
    # `prediction` and `probability_fraud` regardless of the actual Athena
    # column names in a BYO deployment.
    performance_sql = f"""
    SELECT
        {PREDICTION_COLUMN} AS prediction,
        {PROBABILITY_COLUMN} AS probability_fraud,
        ground_truth
    FROM {ATHENA_DATABASE}.inference_responses
    WHERE ground_truth IS NOT NULL
      AND request_timestamp >= TIMESTAMP '{lookback_start}'
    LIMIT 10000
    """

    recent_performance = execute_athena_query(performance_sql)

    if len(recent_performance) < MIN_SAMPLES:
        print(f"⚠️ Not enough samples with ground truth ({len(recent_performance)} < {MIN_SAMPLES})")
        return None

    print(f"✓ Found {len(recent_performance)} samples with ground truth")

    # Build current DataFrame
    current_df = pd.DataFrame(recent_performance)
    current_df['ground_truth'] = current_df['ground_truth'].astype(int)
    current_df['prediction'] = current_df['prediction'].astype(int)
    current_df['probability_fraud'] = current_df['probability_fraud'].astype(float)

    # Compute sklearn metrics for the SNS alert / response payload
    from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

    y_true = current_df['ground_truth'].values
    y_pred = current_df['prediction'].values
    y_prob = current_df['probability_fraud'].values

    current_roc_auc = roc_auc_score(y_true, y_prob)
    current_accuracy = accuracy_score(y_true, y_pred)
    current_precision = precision_score(y_true, y_pred, zero_division=0)
    current_recall = recall_score(y_true, y_pred, zero_division=0)

    print(f"  Current ROC-AUC: {current_roc_auc:.4f}")
    print(f"  Current Accuracy: {current_accuracy:.4f}")
    print(f"  Current Precision: {current_precision:.4f}")
    print(f"  Current Recall: {current_recall:.4f}")

    # Compare to baseline. Source of truth is baseline.json registered on
    # the latest Approved ModelPackage — that anchors the baseline to the
    # exact slice (evaluation_data) the model was scored on at training.
    # Falls back to BASELINE_ROC_AUC env var only when the registry lookup
    # fails (e.g., first ever monitor run before any model is approved).
    baseline = load_baseline_from_registry()
    if baseline and 'metrics' in baseline and 'roc_auc' in baseline['metrics']:
        baseline_roc_auc = float(baseline['metrics']['roc_auc'])
        baseline_source = baseline.get('model_package_arn', 'registered baseline.json')
    else:
        baseline_roc_auc = float(os.getenv('BASELINE_ROC_AUC', '0.92'))
        baseline_source = 'env:BASELINE_ROC_AUC (no registered baseline.json found)'

    degradation = baseline_roc_auc - current_roc_auc
    degradation_pct = (degradation / baseline_roc_auc) * 100

    print(f"  Baseline ROC-AUC: {baseline_roc_auc:.4f}  ← {baseline_source}")
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
      AND request_timestamp < TIMESTAMP '{lookback_start}'
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

    return {
        'detected': detected,
        'baseline_roc_auc': baseline_roc_auc,
        'current_roc_auc': current_roc_auc,
        'degradation': degradation,
        'degradation_pct': degradation_pct,
        'accuracy': current_accuracy,
        'precision': current_precision,
        'recall': current_recall,
        'sample_size': len(recent_performance),
        'html_report_path': html_path if classification_result else None,
        'evidently_metrics': classification_result.get('metrics', []) if classification_result else [],
    }


def send_sns_alert(data_drift_result, model_drift_result):
    """Send SNS notification if drift detected."""
    if not SNS_TOPIC_ARN:
        print("⚠️ SNS_TOPIC_ARN not configured, skipping notification")
        return

    data_drift_detected = data_drift_result and data_drift_result.get('detected', False)
    model_drift_detected = model_drift_result and model_drift_result.get('detected', False)

    if not data_drift_detected and not model_drift_detected:
        print("✓ No drift detected, no alert sent")
        return

    # Build alert message
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
        message_lines.extend([
            "🔴 MODEL PERFORMANCE DRIFT DETECTED (Evidently ClassificationPreset)",
            "=" * 80,
            f"Baseline ROC-AUC: {model_drift_result['baseline_roc_auc']:.4f}",
            f"Current ROC-AUC: {model_drift_result['current_roc_auc']:.4f}",
            f"Degradation: {model_drift_result['degradation']:.4f} ({model_drift_result['degradation_pct']:.1f}%)",
            f"Threshold: {MODEL_DRIFT_THRESHOLD * 100:.1f}%",
            "",
            f"Current Accuracy: {model_drift_result['accuracy']:.4f}",
            f"Current Precision: {model_drift_result['precision']:.4f}",
            f"Current Recall: {model_drift_result['recall']:.4f}",
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
            baseline = load_baseline_from_registry() or {}
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
                mlflow.log_metric("model_drift_detected", 1 if model_drift_result['detected'] else 0)
                mlflow.log_metric("baseline_roc_auc", model_drift_result['baseline_roc_auc'])
                mlflow.log_metric("current_roc_auc", model_drift_result['current_roc_auc'])
                mlflow.log_metric("roc_auc_degradation", model_drift_result['degradation'])
                mlflow.log_metric("roc_auc_degradation_pct", model_drift_result['degradation_pct'])
                mlflow.log_metric("current_accuracy", model_drift_result['accuracy'])
                mlflow.log_metric("current_precision", model_drift_result['precision'])
                mlflow.log_metric("current_recall", model_drift_result['recall'])
                mlflow.log_metric("model_sample_size", model_drift_result['sample_size'])

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

def write_monitoring_results(data_drift_result, model_drift_result, mlflow_run_id=None):
    """Send monitoring results to SQS for writing to Athena monitoring_responses table."""
    if not MONITORING_SQS_QUEUE_URL:
        print("⚠️ MONITORING_SQS_QUEUE_URL not configured - skipping Athena write")
        return

    now = datetime.now()
    run_id = f"drift-{now.strftime('%Y%m%d-%H%M%S')}"

    # Build per-feature drift scores JSON. Nested per-feature object so the
    # governance dashboard can plot the test-agnostic magnitude alongside the
    # raw score. The feature_drift_detail Athena view parses this shape
    # (MAP(VARCHAR, JSON)); legacy flat rows still read via its TRY_CAST path.
    per_feature = {}
    if data_drift_result:
        for feat_info in data_drift_result.get('drifted_features', []):
            per_feature[feat_info['feature']] = {
                'score': feat_info.get('drift_score', 0),
                'magnitude': feat_info.get('drift_magnitude', 0),
                'method': feat_info.get('method', ''),
                'threshold': feat_info.get('threshold', 0),
            }

    # Compute F1 from precision and recall if model drift available
    f1 = None
    if model_drift_result:
        p = model_drift_result.get('precision', 0)
        r = model_drift_result.get('recall', 0)
        if p + r > 0:
            f1 = 2 * p * r / (p + r)

    data_detected = data_drift_result.get('detected', False) if data_drift_result else False
    model_detected = model_drift_result.get('detected', False) if model_drift_result else False

    # Stamp the resolved ModelPackage ARN + Iceberg snapshot ID. These are
    # the immutable references that let you query monitoring_responses per
    # model version — joining on a human-readable label like model_version
    # silently mixes results across rollouts.
    baseline = load_baseline_from_registry()
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
        'baseline_roc_auc': model_drift_result.get('baseline_roc_auc') if model_drift_result else None,
        'current_roc_auc': model_drift_result.get('current_roc_auc') if model_drift_result else None,
        'roc_auc_degradation': model_drift_result.get('degradation') if model_drift_result else None,
        'roc_auc_degradation_pct': model_drift_result.get('degradation_pct') if model_drift_result else None,
        'accuracy': model_drift_result.get('accuracy') if model_drift_result else None,
        'precision': model_drift_result.get('precision') if model_drift_result else None,
        'recall': model_drift_result.get('recall') if model_drift_result else None,
        'f1_score': f1,
        'model_sample_size': model_drift_result.get('sample_size') if model_drift_result else None,
        'per_feature_drift_scores': json.dumps(per_feature) if per_feature else None,
        'evidently_report_s3_path': None,  # Populated if reports uploaded to S3
        'mlflow_run_id': mlflow_run_id,
        'alert_sent': data_detected or model_detected,
        'detection_engine': 'evidently',
        'created_at': now.strftime('%Y-%m-%d %H:%M:%S'),
    }

    try:
        sqs.send_message(
            QueueUrl=MONITORING_SQS_QUEUE_URL,
            MessageBody=json.dumps(record, default=str),
        )
        print(f"✓ Monitoring results sent to SQS: {run_id}")
    except Exception as e:
        print(f"❌ Failed to send monitoring results to SQS: {e}")

    # Backfill monitoring_run_id onto the inference rows this run scored.
    # The `monitoring_run_id IS NULL` guard makes this naturally delta-shaped:
    # each scheduled run only tags predictions never tagged by any prior run.
    # Same id is now in monitoring_responses (above) and inference_responses
    # (here) → QuickSight can join the two tables on monitoring_run_id.
    if ENDPOINT_NAME:
        backfill_sql = f"""
        UPDATE {ATHENA_DATABASE}.inference_responses
        SET monitoring_run_id = '{run_id}'
        WHERE endpoint_name = '{ENDPOINT_NAME}'
          AND monitoring_run_id IS NULL
          AND request_timestamp <= TIMESTAMP '{now.strftime('%Y-%m-%d %H:%M:%S')}'
        """
        try:
            execute_athena_query(backfill_sql, wait=True)
            print(f"✓ Backfilled monitoring_run_id={run_id} onto inference_responses (delta since last run)")
        except Exception as e:
            # Athena UPDATE manifest parse may raise but the UPDATE still succeeded.
            # Treat real failures distinctly from the parse warning.
            msg = str(e)
            if 'Query failed' in msg:
                print(f"⚠️ Backfill UPDATE failed: {e}")
            else:
                print(f"✓ Backfilled monitoring_run_id={run_id} (result-parse warning ignored: {msg[:80]})")


# =========================================================================
# Lambda entry point
# =========================================================================

def lambda_handler(event, context):
    """Lambda handler for EventBridge scheduled drift monitoring."""
    print("=" * 80)
    print(f"Drift Monitoring Check (Evidently) - {datetime.now()}")
    print("=" * 80)

    try:
        # Check data drift (Evidently DataDriftPreset)
        data_drift_result = check_data_drift()

        # Check model drift (Evidently ClassificationPreset)
        model_drift_result = check_model_drift()

        # Log Evidently reports and metrics to MLflow (captures run ID)
        mlflow_run_id = log_to_mlflow(data_drift_result, model_drift_result)

        # Send alert if drift detected
        send_sns_alert(data_drift_result, model_drift_result)

        # Write monitoring results to SQS → Athena (with MLflow run ID)
        write_monitoring_results(data_drift_result, model_drift_result, mlflow_run_id)

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
            'data_drift': _clean(data_drift_result),
            'model_drift': _clean(model_drift_result),
            'alert_sent': (
                (data_drift_result and data_drift_result.get('detected', False)) or
                (model_drift_result and model_drift_result.get('detected', False))
            ),
        }

        print("=" * 80)
        print("Drift monitoring check completed successfully")
        print("=" * 80)

        return {
            'statusCode': 200,
            'body': json.dumps(response, indent=2, default=str)
        }

    except Exception as e:
        print(f"❌ Error during drift monitoring: {e}")
        import traceback
        traceback.print_exc()

        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }


if __name__ == '__main__':
    # For local testing
    lambda_handler({}, {})
