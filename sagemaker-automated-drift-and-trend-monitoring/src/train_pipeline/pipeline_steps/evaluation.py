"""
Model evaluation script for SageMaker Pipeline.

This script runs as an EvaluationStep and:
- Loads trained model and test data
- Calculates comprehensive evaluation metrics
- Generates evaluation report for quality gates
- Saves metrics to property file for ConditionStep
- Optionally logs results to MLflow (if available)
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score,
    recall_score, f1_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)

# MLflow is optional - XGBoost container doesn't have it by default
# Need sagemaker-mlflow for ARN URI support
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    print("MLflow not found, attempting to install...")
    try:
        import subprocess
        # Install sagemaker-mlflow which includes mlflow + AWS SageMaker integration
        print("Installing sagemaker-mlflow for ARN URI support...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'sagemaker-mlflow>=0.1.0'])
        import mlflow
        MLFLOW_AVAILABLE = True
        print("✓ SageMaker MLflow installed successfully")
    except Exception as e:
        print(f"⚠ Could not install MLflow: {e}")
        MLFLOW_AVAILABLE = False

# Matplotlib/seaborn for visualizations
# Install if not available (XGBoost 1.7-1 container supports it)
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
    print("✓ Visualization libraries available")
except ImportError:
    print("Matplotlib not found, attempting to install...")
    try:
        import subprocess
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'matplotlib', 'seaborn'])
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
        VISUALIZATION_AVAILABLE = True
        print("✓ Visualization libraries installed successfully")
    except Exception as e:
        print(f"⚠ Could not install visualization libraries: {e}")
        VISUALIZATION_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model(model_dir: str) -> xgb.Booster:
    """
    Load trained XGBoost model.

    Args:
        model_dir: Directory containing model artifacts

    Returns:
        Loaded XGBoost Booster
    """
    model_dir_path = Path(model_dir)
    
    # SageMaker XGBoost saves model as model.tar.gz containing xgboost-model
    # Check for tar.gz first
    tar_path = model_dir_path / "model.tar.gz"
    if tar_path.exists():
        import tarfile
        logger.info(f"Extracting model from {tar_path}")
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(model_dir_path)
    
    # Try different model file names
    possible_names = ['xgboost-model', 'xgboost-model.json', 'model.json', 'model']
    model_path = None
    
    for name in possible_names:
        candidate = model_dir_path / name
        if candidate.exists():
            model_path = candidate
            break
    
    if model_path is None:
        # List directory contents for debugging
        contents = list(model_dir_path.iterdir()) if model_dir_path.exists() else []
        raise FileNotFoundError(
            f"Model not found in {model_dir}. "
            f"Tried: {possible_names}. "
            f"Directory contents: {[f.name for f in contents]}"
        )

    logger.info(f"Loading model from {model_path}")
    model = xgb.Booster()
    model.load_model(str(model_path))
    logger.info("✓ Model loaded successfully")

    return model


def load_test_data(data_dir: str, target_column: str) -> tuple:
    """
    Load test data with feature names from metadata.

    Args:
        data_dir: Directory containing test.csv and feature_metadata.json
        target_column: Name of target column

    Returns:
        Tuple of (X_test, y_test)
    """
    test_path = Path(data_dir) / "test.csv"
    metadata_path = Path(data_dir) / "feature_metadata.json"

    if not test_path.exists():
        raise FileNotFoundError(f"Test data not found: {test_path}")

    logger.info(f"Loading test data from {test_path}")

    # Load feature names from metadata (created by preprocessing step)
    if metadata_path.exists():
        logger.info(f"Loading feature metadata from {metadata_path}")
        with open(metadata_path, 'r') as f:
            feature_metadata = json.load(f)
        feature_names = feature_metadata['feature_names']
        logger.info(f"✓ Loaded {len(feature_names)} actual feature names from Athena")
        logger.info(f"  Feature names: {feature_names[:5]}...")
    else:
        # Fallback to generated names if metadata not found
        logger.warning(f"Feature metadata not found at {metadata_path}")
        logger.warning("Falling back to numeric column names - this may cause feature mismatch!")
        feature_names = None

    # XGBoost training data has no header, target column is first
    # Preprocessing already filtered to numeric columns only
    test_df = pd.read_csv(test_path, header=None)
    logger.info(f"✓ Loaded {len(test_df):,} test samples with {len(test_df.columns)} columns")

    # First column is target, rest are features
    y_test = test_df.iloc[:, 0].astype(float)
    X_test = test_df.iloc[:, 1:].astype(float)

    # Assign feature names to match training
    if feature_names is not None:
        X_test.columns = feature_names
    else:
        # Generate sequential names as fallback
        X_test.columns = [str(i+1) for i in range(X_test.shape[1])]

    # Fill any NaN values
    X_test = X_test.fillna(0)

    logger.info(f"  Target column (first): {len(y_test)} values, unique: {y_test.unique()}")
    logger.info(f"  Feature columns: {X_test.shape[1]}")
    logger.info(f"  Column names: {list(X_test.columns[:5])}...")

    return X_test, y_test


def evaluate_model(
    model: xgb.Booster,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Evaluate model comprehensively.

    Args:
        model: Trained XGBoost model
        X_test: Test features
        y_test: Test target
        threshold: Classification threshold

    Returns:
        Dictionary of evaluation metrics and report
    """
    logger.info("Evaluating model performance...")
    logger.info(f"  X_test shape: {X_test.shape}")
    logger.info(f"  y_test distribution: {y_test.value_counts().to_dict()}")

    # Make predictions
    dtest = xgb.DMatrix(X_test)
    y_pred_proba = model.predict(dtest)
    
    logger.info(f"  Prediction range: [{y_pred_proba.min():.4f}, {y_pred_proba.max():.4f}]")
    logger.info(f"  Prediction mean: {y_pred_proba.mean():.4f}")
    
    y_pred = (y_pred_proba >= threshold).astype(int)
    logger.info(f"  Predicted distribution (threshold={threshold}): {pd.Series(y_pred).value_counts().to_dict()}")

    # Core metrics
    metrics = {
        'roc_auc': float(roc_auc_score(y_test, y_pred_proba)),
        'pr_auc': float(average_precision_score(y_test, y_pred_proba)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1_score': float(f1_score(y_test, y_pred, zero_division=0)),
    }

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics.update({
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp),
        'accuracy': float((tp + tn) / (tp + tn + fp + fn)),
        'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        'false_positive_rate': float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        'false_negative_rate': float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0,
    })

    # Sample counts
    metrics['test_samples'] = len(y_test)
    metrics['positive_samples'] = int(y_test.sum())
    metrics['negative_samples'] = int(len(y_test) - y_test.sum())
    metrics['threshold'] = threshold

    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    metrics['classification_report'] = report

    # Feature importance (top 20)
    importance = model.get_score(importance_type='gain')
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:20]
    metrics['feature_importance'] = {k: float(v) for k, v in sorted_importance}

    # Log results
    logger.info("=" * 80)
    logger.info("Model Evaluation Results")
    logger.info("=" * 80)
    logger.info(f"ROC-AUC:           {metrics['roc_auc']:.4f}")
    logger.info(f"PR-AUC:            {metrics['pr_auc']:.4f}")
    logger.info(f"Precision:         {metrics['precision']:.4f}")
    logger.info(f"Recall:            {metrics['recall']:.4f}")
    logger.info(f"F1-Score:          {metrics['f1_score']:.4f}")
    logger.info(f"Accuracy:          {metrics['accuracy']:.4f}")
    logger.info(f"Specificity:       {metrics['specificity']:.4f}")
    logger.info("")
    logger.info("Confusion Matrix:")
    logger.info(f"  TN: {tn:6,}  |  FP: {fp:6,}")
    logger.info(f"  FN: {fn:6,}  |  TP: {tp:6,}")
    logger.info("=" * 80)

    return metrics


def create_evaluation_visualizations(
    model: xgb.Booster,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    metrics: Dict[str, Any],
    output_dir: str
) -> Dict[str, Any]:
    """
    Create evaluation visualizations following MLflow best practices.

    Returns figure objects that can be logged directly with mlflow.log_figure().
    This approach prevents file corruption issues when logging to MLflow.

    Args:
        model: Trained XGBoost model
        X_test: Test features
        y_test: Test labels
        metrics: Evaluation metrics dictionary
        output_dir: Directory to save visualizations (for backward compatibility)

    Returns:
        Dictionary of figure objects keyed by visualization name
    """
    if not VISUALIZATION_AVAILABLE:
        logger.warning("⚠ Visualization libraries not available, skipping visualizations")
        return {}

    logger.info("Creating evaluation visualizations...")
    figures = {}

    # Get predictions for visualizations
    dtest = xgb.DMatrix(X_test)
    y_pred_proba = model.predict(dtest)
    y_pred = (y_pred_proba >= 0.5).astype(int)

    try:
        # 1. Confusion Matrix Heatmap
        logger.info("  Creating confusion matrix...")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Not Fraud', 'Fraud'],
                   yticklabels=['Not Fraud', 'Fraud'])
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title('Confusion Matrix - Model Evaluation', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.close(fig)
        figures['confusion_matrix'] = fig
        logger.info("    ✓ Confusion matrix created")
    except Exception as e:
        logger.error(f"    ✗ Failed to create confusion matrix: {e}")

    try:
        # 2. ROC Curve
        logger.info("  Creating ROC curve...")
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = metrics['roc_auc']

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(fpr, tpr, color='darkorange', lw=2,
               label=f'ROC curve (AUC = {roc_auc:.4f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curve - Model Evaluation', fontsize=14, fontweight='bold')
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.close(fig)
        figures['roc_curve'] = fig
        logger.info("    ✓ ROC curve created")
    except Exception as e:
        logger.error(f"    ✗ Failed to create ROC curve: {e}")

    try:
        # 3. Precision-Recall Curve
        logger.info("  Creating precision-recall curve...")
        precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = metrics['pr_auc']

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(recall_vals, precision_vals, color='blue', lw=2,
               label=f'PR curve (AUC = {pr_auc:.4f})')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title('Precision-Recall Curve - Model Evaluation', fontsize=14, fontweight='bold')
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.close(fig)
        figures['precision_recall_curve'] = fig
        logger.info("    ✓ Precision-recall curve created")
    except Exception as e:
        logger.error(f"    ✗ Failed to create precision-recall curve: {e}")

    try:
        # 4. Metrics Summary
        logger.info("  Creating metrics summary...")
        metric_names = ['ROC-AUC', 'PR-AUC', 'F1-Score', 'Precision', 'Recall', 'Accuracy']
        metric_values = [
            metrics['roc_auc'],
            metrics['pr_auc'],
            metrics['f1_score'],
            metrics['precision'],
            metrics['recall'],
            metrics['accuracy']
        ]

        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.barh(metric_names, metric_values, color='steelblue')

        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, metric_values)):
            ax.text(value + 0.01, i, f'{value:.4f}',
                   va='center', fontsize=10, fontweight='bold')

        ax.set_xlim([0, 1.15])
        ax.set_xlabel('Score', fontsize=12)
        ax.set_title('Evaluation Metrics Summary', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.close(fig)
        figures['metrics_summary'] = fig
        logger.info("    ✓ Metrics summary created")
    except Exception as e:
        logger.error(f"    ✗ Failed to create metrics summary: {e}")

    logger.info(f"✓ Created {len(figures)} evaluation visualizations")
    return figures


def check_quality_gates(
    metrics: Dict[str, Any],
    min_roc_auc: float = 0.85,
    min_pr_auc: float = 0.50
) -> Dict[str, Any]:
    """
    Check if model passes quality gates.

    Args:
        metrics: Evaluation metrics
        min_roc_auc: Minimum ROC-AUC threshold
        min_pr_auc: Minimum PR-AUC threshold

    Returns:
        Dictionary with quality gate results
    """
    logger.info("Checking quality gates...")

    results = {
        'passed': True,
        'checks': [],
        'failures': []
    }

    # ROC-AUC check
    roc_auc = metrics['roc_auc']
    roc_check = {
        'metric': 'roc_auc',
        'value': roc_auc,
        'threshold': min_roc_auc,
        'passed': roc_auc >= min_roc_auc
    }
    results['checks'].append(roc_check)

    if not roc_check['passed']:
        results['passed'] = False
        results['failures'].append(f"ROC-AUC {roc_auc:.4f} < {min_roc_auc:.4f}")
        logger.warning(f"✗ ROC-AUC check failed: {roc_auc:.4f} < {min_roc_auc:.4f}")
    else:
        logger.info(f"✓ ROC-AUC check passed: {roc_auc:.4f} >= {min_roc_auc:.4f}")

    # PR-AUC check
    pr_auc = metrics['pr_auc']
    pr_check = {
        'metric': 'pr_auc',
        'value': pr_auc,
        'threshold': min_pr_auc,
        'passed': pr_auc >= min_pr_auc
    }
    results['checks'].append(pr_check)

    if not pr_check['passed']:
        results['passed'] = False
        results['failures'].append(f"PR-AUC {pr_auc:.4f} < {min_pr_auc:.4f}")
        logger.warning(f"✗ PR-AUC check failed: {pr_auc:.4f} < {min_pr_auc:.4f}")
    else:
        logger.info(f"✓ PR-AUC check passed: {pr_auc:.4f} >= {min_pr_auc:.4f}")

    if results['passed']:
        logger.info("✓ All quality gates passed")
    else:
        logger.warning(f"✗ Quality gates failed: {results['failures']}")

    return results


def _current_snapshot_id(
    athena_client, database: str, table: str, output_s3: str
) -> str:
    """Return the most recent Iceberg snapshot ID for ``database.table``.

    Iceberg exposes every table's snapshot history via the ``$snapshots``
    metadata table. The most recent snapshot is the one preprocessing
    just read — pinning it in baseline.json freezes the drift-monitor
    reference, so re-seeding the table cannot retroactively change what
    "the data this model was scored on" means.

    Returns "" if the query fails so eval can still finish (with a less
    rigorous baseline).
    """
    sql = (
        f'SELECT CAST(snapshot_id AS VARCHAR) AS sid '
        f'FROM "{database}"."{table}$snapshots" '
        f"ORDER BY committed_at DESC LIMIT 1"
    )
    try:
        qid = athena_client.start_query_execution(
            QueryString=sql,
            QueryExecutionContext={"Database": database},
            ResultConfiguration={"OutputLocation": output_s3},
        )["QueryExecutionId"]
        import time as _time
        for _ in range(60):
            st = athena_client.get_query_execution(QueryExecutionId=qid)["QueryExecution"]["Status"]
            if st["State"] == "SUCCEEDED":
                break
            if st["State"] in ("FAILED", "CANCELLED"):
                logger.warning(f"snapshot query for {table} {st['State']}: {st.get('StateChangeReason', '')}")
                return ""
            _time.sleep(1)
        else:
            logger.warning(f"snapshot query for {table} timed out")
            return ""
        rows = athena_client.get_query_results(QueryExecutionId=qid)["ResultSet"]["Rows"]
        if len(rows) >= 2:
            return rows[1]["Data"][0].get("VarCharValue", "")
        return ""
    except Exception as e:
        logger.warning(f"could not read $snapshots for {table}: {e}")
        return ""


def save_evaluation_report(
    metrics: Dict[str, Any],
    quality_gates: Dict[str, Any],
    output_dir: str,
    *,
    evaluation_table: str,
    training_table: str,
    model_package_group: str,
    code_commit_sha: str,
    feature_schema_version: int,
    feature_schema: list,
    athena_database: str,
    athena_output_s3: str,
) -> None:
    """
    Save evaluation outputs to output directory:
      - evaluation_report.json — full report (metrics + quality gates)
      - evaluation.json        — property file consumed by ConditionStep
      - baseline.json          — frozen drift-monitor baseline. Contains the
        same metrics on the evaluation_data slice that registered this model,
        plus the immutable references (Iceberg snapshot IDs, code commit SHA,
        feature schema version) needed to reproduce the experiment exactly.
        The drift Lambda uses these to read the *frozen* reference rows via
        Iceberg time travel — re-seeding the table cannot corrupt historical
        baselines.

    Args:
        metrics: Evaluation metrics
        quality_gates: Quality gate results
        output_dir: Output directory path
        evaluation_table: Athena table the metrics were computed on.
        training_table: Athena table the model was trained on (captured for
            full lineage; not strictly required by the drift monitor).
        model_package_group: Model Registry group this baseline applies to.
        code_commit_sha: Git commit SHA of the pipeline code that produced
            this baseline (passed in from the pipeline parameter — captured
            on the orchestrator host, not inside the eval container).
        feature_schema_version: Bumped whenever the feature list changes
            (schema breaks force retraining).
        feature_schema: Ordered list of feature names + dtypes the model was
            scored on. Stored so the monitor can detect schema drift.
        athena_database: Athena database to query for $snapshots.
        athena_output_s3: S3 output location for the snapshot queries.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Combine metrics and quality gates
    report = {
        'metrics': metrics,
        'quality_gates': quality_gates,
        'timestamp': pd.Timestamp.now().isoformat()
    }

    # Save full report
    report_path = output_path / "evaluation_report.json"
    logger.info(f"Saving evaluation report to {report_path}")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    # Save property file for ConditionStep
    # This file is used by SageMaker Pipelines to access metrics in conditions
    property_path = output_path / "evaluation.json"
    property_data = {
        'regression_metrics': {},  # Empty for classification
        'binary_classification_metrics': {
            'roc_auc': {'value': metrics['roc_auc']},
            'pr_auc': {'value': metrics['pr_auc']},
            'precision': {'value': metrics['precision']},
            'recall': {'value': metrics['recall']},
            'f1_score': {'value': metrics['f1_score']},
            'accuracy': {'value': metrics['accuracy']},
        }
    }

    logger.info(f"Saving property file to {property_path}")
    with open(property_path, 'w') as f:
        json.dump(property_data, f, indent=2)

    # Pin Iceberg snapshot IDs so the drift Lambda can time-travel the
    # reference slice via FOR VERSION AS OF — see load_baseline_from_registry
    # in src/drift_monitoring/lambda_drift_monitor.py. We query $snapshots
    # AFTER preprocessing/training/eval have read the tables; with no
    # concurrent writers in a pipeline run, the latest snapshot is the one
    # this model was actually scored on.
    import boto3
    athena_client = boto3.client(
        "athena",
        region_name=os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("AWS_REGION"),
    )
    train_snapshot = _current_snapshot_id(
        athena_client, athena_database, training_table, athena_output_s3
    )
    eval_snapshot = _current_snapshot_id(
        athena_client, athena_database, evaluation_table, athena_output_s3
    )
    if not train_snapshot or not eval_snapshot:
        logger.warning(
            "Could not capture one or both Iceberg snapshot IDs "
            "(train='%s', eval='%s'). Drift monitor will fall back to the "
            "live table.",
            train_snapshot, eval_snapshot,
        )

    # Save drift-monitor baseline. The drift Lambda reads this artifact via
    # the registered ModelPackage's ModelStatistics URI — see
    # _create_register_model_step in pipeline.py. Schema_version is bumped
    # when fields change; older Lambdas should fall back gracefully on
    # unknown versions.
    baseline_path = output_path / "baseline.json"
    baseline_data = {
        # v3 adds problem_type + primary_metric so the drift Lambda can
        # validate baseline content and pick the right primary metric without
        # inferring it from which metric keys happen to be present. This
        # pipeline trains a binary fraud classifier whose primary metric is
        # ROC-AUC (matches _expected_primary_metric_key in the drift Lambda).
        'schema_version': 3,
        'problem_type': 'binary_classification',
        'primary_metric': 'roc_auc',
        'created_at': pd.Timestamp.now().isoformat(),
        'model_package_group': model_package_group,
        'code_commit_sha': code_commit_sha,
        'evaluation_table': evaluation_table,
        'training_table': training_table,
        'evaluation_snapshot_id': eval_snapshot,
        'training_snapshot_id': train_snapshot,
        'feature_schema_version': int(feature_schema_version),
        'feature_schema': feature_schema,
        'metrics': {
            'roc_auc': float(metrics['roc_auc']),
            'pr_auc': float(metrics['pr_auc']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'f1_score': float(metrics['f1_score']),
            'accuracy': float(metrics['accuracy']),
        },
        'sample_size': int(metrics.get('test_samples', 0)),
        'positive_samples': int(metrics.get('positive_samples', 0)),
        'negative_samples': int(metrics.get('negative_samples', 0)),
        'threshold': float(metrics.get('threshold', 0.5)),
    }
    logger.info(f"Saving drift baseline to {baseline_path}")
    with open(baseline_path, 'w') as f:
        json.dump(baseline_data, f, indent=2)

    logger.info("✓ Evaluation report saved successfully")


def log_figure_to_mlflow(fig, artifact_name: str) -> None:
    """
    Log a matplotlib figure to MLflow ensuring proper binary PNG encoding.

    This ensures the PNG is saved as a proper binary file (not base64-encoded)
    so it renders correctly in the MLflow UI.

    Args:
        fig: Matplotlib figure object
        artifact_name: Name for the artifact (should end with .png)
    """
    import io
    import tempfile

    try:
        # Method 1: Use mlflow.log_figure() directly (preferred method)
        # MLflow handles the encoding internally and should produce binary PNG
        mlflow.log_figure(fig, artifact_name)
        logger.info(f"  ✓ Logged {artifact_name}")

    except Exception as e1:
        # Method 2: Fallback - manually save as binary PNG then log as artifact
        logger.warning(f"  ⚠ mlflow.log_figure() failed for {artifact_name}, trying manual save: {e1}")
        try:
            # Save to temporary file ensuring binary PNG format
            with tempfile.NamedTemporaryFile(mode='wb', suffix='.png', delete=False) as tmp:
                # Save figure as binary PNG (not base64)
                fig.savefig(tmp.name, format='png', dpi=150, bbox_inches='tight')
                tmp_path = tmp.name

            # Verify it's a proper binary PNG (starts with PNG magic number)
            with open(tmp_path, 'rb') as f:
                magic_bytes = f.read(4)
                if magic_bytes != b'\x89PNG':
                    raise ValueError(f"Generated file is not a valid binary PNG (got {magic_bytes.hex()})")

            # Log as artifact
            mlflow.log_artifact(tmp_path, artifact_path='')
            logger.info(f"  ✓ Logged {artifact_name} (via artifact)")

            # Clean up temp file
            import os
            os.unlink(tmp_path)

        except Exception as e2:
            logger.error(f"  ✗ Failed to log {artifact_name}: {e2}")


def log_to_mlflow(
    metrics: Dict[str, Any],
    quality_gates: Dict[str, Any],
    figures: Dict[str, Any] = None,
    *,
    code_commit_sha: str = "unknown",
    model_package_group: str = "",
    feature_schema_version: int = 0,
) -> None:
    """
    Log evaluation results to MLflow.

    Ensures images are saved as proper binary PNG files (not base64-encoded)
    for MLflow UI rendering.

    Args:
        metrics: Evaluation metrics
        quality_gates: Quality gate results
        figures: Dictionary of matplotlib figure objects to log
        code_commit_sha: Git SHA of the pipeline code that produced this run.
            Logged as an MLflow tag so the UI can group/filter runs by code
            version. Comes from the CodeCommitSha pipeline parameter.
        model_package_group: Model Registry group this run will register
            into. Logged as a tag for the same grouping purpose.
        feature_schema_version: Bumped when the feature list changes —
            useful for spotting incompatible model families in the UI.
    """
    logger.info("="*80)
    logger.info("EVALUATION STEP - MLflow Logging")
    logger.info("="*80)

    # Check if MLflow is available
    if not MLFLOW_AVAILABLE:
        logger.warning("⚠ MLflow not installed, skipping MLflow logging")
        logger.info("   (This is expected in XGBoost container without mlflow)")
        return

    logger.info("✓ MLflow is available")

    # Set MLflow tracking URI
    mlflow_tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
    if not mlflow_tracking_uri or mlflow_tracking_uri == '':
        logger.warning("⚠ MLflow tracking URI not set, skipping MLflow logging")
        return

    logger.info(f"✓ MLflow tracking URI: {mlflow_tracking_uri}")

    try:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        logger.info("✓ MLflow tracking URI configured")

        # Set experiment
        experiment_name = os.getenv('MLFLOW_EXPERIMENT_NAME', 'credit-card-fraud-detection-evaluation')
        mlflow.set_experiment(experiment_name)
        logger.info(f"✓ MLflow experiment set: {experiment_name}")

        # Start MLflow run
        logger.info("Starting MLflow run...")
        with mlflow.start_run() as run:
            logger.info(f"✓ MLflow run started: {run.info.run_id}")

            # Log core metrics
            logger.info("Logging evaluation metrics...")
            mlflow.log_metric('eval_roc_auc', metrics['roc_auc'])
            mlflow.log_metric('eval_pr_auc', metrics['pr_auc'])
            mlflow.log_metric('eval_precision', metrics['precision'])
            mlflow.log_metric('eval_recall', metrics['recall'])
            mlflow.log_metric('eval_f1_score', metrics['f1_score'])
            mlflow.log_metric('eval_accuracy', metrics['accuracy'])

            # Log confusion matrix values
            mlflow.log_metric('eval_true_positives', metrics['true_positives'])
            mlflow.log_metric('eval_true_negatives', metrics['true_negatives'])
            mlflow.log_metric('eval_false_positives', metrics['false_positives'])
            mlflow.log_metric('eval_false_negatives', metrics['false_negatives'])
            logger.info("✓ Metrics logged")

            # Log quality gate results
            mlflow.log_param('quality_gates_passed', quality_gates['passed'])

            # Log tags. code_commit_sha is the MLflow-side anchor that pairs
            # with baseline.json's same field — letting MLflow's UI group
            # runs by code version (Filter: tags.code_commit_sha = "...").
            mlflow.set_tags({
                'pipeline_step': 'evaluation',
                'quality_gates_status': 'passed' if quality_gates['passed'] else 'failed',
                'code_commit_sha': code_commit_sha,
                'model_package_group': model_package_group,
                'feature_schema_version': str(feature_schema_version),
            })
            logger.info("✓ Tags logged (code_commit_sha=%s)", code_commit_sha[:12])

            # Log visualizations ensuring proper binary PNG format for MLflow UI
            if figures:
                logger.info(f"Logging {len(figures)} visualizations...")
                for fig_name, fig in figures.items():
                    if fig is not None:
                        log_figure_to_mlflow(fig, f"{fig_name}.png")
                logger.info("✓ All visualizations logged")

            logger.info("="*80)
            logger.info(f"✅ SUCCESS! Logged evaluation to MLflow run: {run.info.run_id}")
            logger.info(f"   ROC-AUC: {metrics['roc_auc']:.4f}")
            logger.info(f"   Quality Gates: {'PASSED' if quality_gates['passed'] else 'FAILED'}")
            logger.info("="*80)

    except Exception as e:
        logger.error("="*80)
        logger.error(f"❌ FAILED TO LOG TO MLFLOW: {e}")
        logger.error(f"   Error type: {type(e).__name__}")
        logger.error("="*80)
        import traceback
        traceback.print_exc()
        # Don't raise - evaluation should still succeed even if MLflow fails


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate model in SageMaker Pipeline")

    # Input arguments
    parser.add_argument('--model-dir', type=str, default='/opt/ml/processing/model',
                       help='Directory containing trained model')
    parser.add_argument('--test-data-dir', type=str, default='/opt/ml/processing/test',
                       help='Directory containing test data')
    parser.add_argument('--target-column', type=str, default='is_fraud',
                       help='Target column name')

    # Evaluation parameters
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Classification threshold')
    parser.add_argument('--min-roc-auc', type=float, default=0.85,
                       help='Minimum ROC-AUC for quality gate')
    parser.add_argument('--min-pr-auc', type=float, default=0.50,
                       help='Minimum PR-AUC for quality gate')

    # Drift-baseline anchoring (written into baseline.json).
    # These are pure pass-through — the eval container is the wrong place to
    # discover them; pipeline.py resolves the values from config + a pipeline
    # parameter (commit SHA) and forwards them here.
    parser.add_argument('--evaluation-table', type=str, default='evaluation_data',
                       help='Athena table the metrics were computed on')
    parser.add_argument('--training-table', type=str, default='training_data',
                       help='Athena table the model was trained on '
                            '(captured for lineage; not required by drift monitor)')
    parser.add_argument('--model-package-group', type=str, default='fraud-detection',
                       help='Model Registry group this baseline applies to')
    parser.add_argument('--code-commit-sha', type=str, default='unknown',
                       help='Git commit SHA of the pipeline code that produced this baseline')
    parser.add_argument('--feature-schema-version', type=int, default=1,
                       help='Bump when feature list changes (forces retrain to '
                            'pass schema check in monitor)')
    parser.add_argument('--athena-database', type=str,
                       default=os.environ.get('ATHENA_DATABASE', 'fraud_detection'),
                       help='Athena database for $snapshots metadata queries')
    parser.add_argument('--athena-output-s3', type=str,
                       default=os.environ.get('ATHENA_OUTPUT_S3', ''),
                       help='S3 location for Athena query results')

    # Output arguments
    parser.add_argument('--output-dir', type=str, default='/opt/ml/processing/evaluation',
                       help='Directory to save evaluation report')

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("Model Evaluation for SageMaker Pipeline")
    logger.info("=" * 80)
    logger.info(f"Model directory: {args.model_dir}")
    logger.info(f"Test data directory: {args.test_data_dir}")
    logger.info(f"Quality gates: ROC-AUC >= {args.min_roc_auc}, PR-AUC >= {args.min_pr_auc}")
    logger.info("")

    try:
        # Step 1: Load model
        model = load_model(args.model_dir)

        # Step 2: Load test data
        X_test, y_test = load_test_data(args.test_data_dir, args.target_column)

        # Step 3: Evaluate model
        metrics = evaluate_model(model, X_test, y_test, args.threshold)

        # Step 4: Create visualizations
        figures = create_evaluation_visualizations(model, X_test, y_test, metrics, args.output_dir)

        # Step 5: Check quality gates
        quality_gates = check_quality_gates(
            metrics,
            min_roc_auc=args.min_roc_auc,
            min_pr_auc=args.min_pr_auc
        )

        # Load the feature schema preprocessing wrote alongside test.csv.
        # If absent, fall back to "unknown" so eval still completes.
        feature_schema = []
        metadata_path = Path(args.test_data_dir) / "feature_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as _mf:
                _fm = json.load(_mf)
            feature_schema = [
                {"name": n, "dtype": "double"} for n in _fm.get("feature_names", [])
            ]
        else:
            logger.warning("feature_metadata.json not found — baseline.feature_schema will be empty")

        # Step 6: Save evaluation report (+ baseline.json for the drift monitor)
        save_evaluation_report(
            metrics, quality_gates, args.output_dir,
            evaluation_table=args.evaluation_table,
            training_table=args.training_table,
            model_package_group=args.model_package_group,
            code_commit_sha=args.code_commit_sha,
            feature_schema_version=args.feature_schema_version,
            feature_schema=feature_schema,
            athena_database=args.athena_database,
            athena_output_s3=args.athena_output_s3,
        )

        # Step 7: Log to MLflow (if available and configured)
        if MLFLOW_AVAILABLE and os.getenv('MLFLOW_TRACKING_URI'):
            log_to_mlflow(
                metrics, quality_gates, figures,
                code_commit_sha=args.code_commit_sha,
                model_package_group=args.model_package_group,
                feature_schema_version=args.feature_schema_version,
            )

        logger.info("=" * 80)
        logger.info("✓ Evaluation completed successfully")
        logger.info(f"Quality gates: {'PASSED' if quality_gates['passed'] else 'FAILED'}")
        logger.info("=" * 80)

        # Exit with appropriate code for quality gates
        if not quality_gates['passed']:
            logger.warning("Model failed quality gates")
            # Don't exit with error - let ConditionStep handle this
            # sys.exit(1)

    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
