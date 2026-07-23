"""
Evidently-based drift detection and model performance reporting.

Replaces custom matplotlib visualizations with Evidently's built-in
interactive HTML reports for data drift and classification metrics.

Drift test selection
---------------------
We pin ONE statistical test for every column rather than letting Evidently
auto-select per column. By default Evidently chooses the test based on sample
size and column type: small samples get p-value tests (Kolmogorov-Smirnov,
Chi-square) while large samples get distance/divergence tests (Wasserstein,
PSI, Jensen-Shannon). That per-column auto-selection breaks a governance
dashboard in two ways:

  1. Drift direction is not uniform. p-value tests flag drift when score <
     threshold (lower = more drift); distance tests flag when score >
     threshold (higher = more drift). Mixing both in one run means the raw
     `drift_score` column has no single "higher = worse" reading, so it can't
     be sorted or aggregated across features.
  2. Magnitude is unbounded for p-value tests. `drift_magnitude` for a p-value
     test is `threshold / p_value`, which explodes as the p-value approaches
     zero — and at the 5k–10k sample sizes this Lambda pulls, the KS test is
     hypersensitive, so a trivial distribution difference yields a near-zero
     p-value and a huge magnitude. A feature that merely drew the smallest
     p-value would rank as the most drifted, turning p-value precision into
     apparent drift severity.

So we force a single bounded distance metric for every column: the test is
deterministic (not sample-size-dependent), drift direction is uniform
(higher = more drift for all features), and magnitude is bounded and
comparable across features. We use Jensen-Shannon distance (range [0, 1])
for both numeric and categorical columns. See DRIFT_NUM_METHOD /
DRIFT_CAT_METHOD below and `run_data_drift_report`.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from evidently import Report
from evidently.core.datasets import BinaryClassification, DataDefinition, Dataset
from evidently.metrics import DriftedColumnsCount, ValueDrift
from evidently.presets import ClassificationPreset, DataDriftPreset

logger = logging.getLogger(__name__)

# Forced drift test — see module docstring. Jensen-Shannon is a bounded
# distance metric ([0, 1], higher = more drift) applied uniformly to every
# column so test choice never depends on sample size. Both numeric and
# categorical columns use it so drift_magnitude is comparable across all
# features. `DRIFT_THRESHOLD` is the JS distance above which a column is
# flagged as drifted (Evidently's documented default for jensenshannon is
# 0.1); `drift_magnitude` = score / threshold, so 1.0 = at threshold.
DRIFT_NUM_METHOD = "jensenshannon"
DRIFT_CAT_METHOD = "jensenshannon"
DRIFT_THRESHOLD = 0.1
# Share of columns that must drift before the dataset is flagged overall.
DRIFT_SHARE_THRESHOLD = 0.5


def run_data_drift_report(
    baseline_df: pd.DataFrame,
    current_df: pd.DataFrame,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run Evidently DataDriftPreset report comparing baseline vs current data.

    Args:
        baseline_df: Reference/training data (numeric features only).
        current_df: Current/inference data (same columns as baseline).
        output_path: If provided, saves the HTML report to this path.

    Returns:
        Dictionary with:
            - 'snapshot': The Evidently report snapshot (call .save_html() etc.)
            - 'drift_detected': bool, whether overall drift was detected
            - 'drifted_columns_count': number of drifted columns
            - 'drifted_columns_share': share of drifted columns
            - 'per_column': dict mapping column name -> {'drift_score': float, 'drifted': bool}
    """
    # Force a single bounded distance metric for every column instead of
    # letting Evidently auto-pick per column by sample size (see the module
    # docstring for why the auto-pick breaks cross-feature ranking).
    # num_method/cat_method pin the test; the matching thresholds make
    # `drift_magnitude = score / threshold` comparable across all features.
    report = Report(metrics=[DataDriftPreset(
        num_method=DRIFT_NUM_METHOD,
        cat_method=DRIFT_CAT_METHOD,
        num_threshold=DRIFT_THRESHOLD,
        cat_threshold=DRIFT_THRESHOLD,
        drift_share=DRIFT_SHARE_THRESHOLD,
    )])
    snapshot = report.run(reference_data=baseline_df, current_data=current_df)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        snapshot.save_html(output_path)
        logger.info(f"Data drift report saved to {output_path}")

    # Extract structured results from the report dict
    metrics_list = snapshot.dict().get("metrics", [])
    result: Dict[str, Any] = {
        "snapshot": snapshot,
        "drift_detected": False,
        "drifted_columns_count": 0,
        "drifted_columns_share": 0.0,
        "per_column": {},
    }

    drift_share_threshold = 0.5  # default from DriftedColumnsCount

    for m in metrics_list:
        name = m.get("metric_name", "")
        config = m.get("config", {})
        value = m.get("value")

        if "DriftedColumnsCount" in name:
            count = value.get("count", 0) if isinstance(value, dict) else 0
            share = value.get("share", 0.0) if isinstance(value, dict) else 0.0
            drift_share_threshold = config.get("drift_share", 0.5)
            result["drifted_columns_count"] = int(count)
            result["drifted_columns_share"] = float(share)
            result["drift_detected"] = share >= drift_share_threshold

        elif "ValueDrift" in name:
            col = config.get("column", "unknown")
            threshold = float(config.get("threshold", 0.05))
            method = config.get("method", "")
            drift_score = float(value) if value is not None else 1.0

            # We force a distance metric (jensenshannon) for every column via
            # DataDriftPreset above, so drift is uniformly "score > threshold"
            # (higher = more drift). The `is_p_value` handling below keeps the
            # drift direction correct for any test family, so a different
            # DRIFT_NUM_METHOD / DRIFT_CAT_METHOD still reads correctly:
            #   * p-value tests (KS, Chi-Square)   → drift when score < threshold
            #   * distance / divergence tests      → drift when score > threshold
            #     (Wasserstein, PSI, Jensen-Shannon, Hellinger, TVD, ...)
            # Evidently exposes the chosen test in `config.method` but has no
            # boolean drift flag in the per-metric dict — we compute it here.
            is_p_value = "p_value" in method.lower() or "p-value" in method.lower()
            drifted = drift_score < threshold if is_p_value else drift_score > threshold

            # drift_magnitude is a test-agnostic "how far past the threshold":
            #   1.0 = at threshold, >1.0 = drifted, higher = more drifted
            # Callers can sort by this descending to get "top N drifted" without
            # caring which test was used for which column. With the forced
            # jensenshannon metric this is bounded (score in [0, 1], threshold
            # 0.1 → magnitude in [0, 10]). The p-value branch uses the inverse
            # ratio only so a p-value-based DRIFT_*_METHOD stays directionally
            # correct; it is intentionally not the configured default because it
            # is unbounded (see the module docstring).
            if is_p_value:
                # p-values: smaller = more drifted → invert ratio
                drift_magnitude = (threshold / drift_score) if drift_score > 0 else float("inf")
            else:
                drift_magnitude = (drift_score / threshold) if threshold > 0 else float("inf")

            result["per_column"][col] = {
                "drift_score": drift_score,
                "drifted": drifted,
                "method": method,
                "threshold": threshold,
                "drift_magnitude": drift_magnitude,
            }

    logger.info(
        f"Data drift report: {result['drifted_columns_count']} drifted columns "
        f"({result['drifted_columns_share']:.1%}), overall drift: {result['drift_detected']}"
    )
    return result


def run_classification_report(
    baseline_df: pd.DataFrame,
    current_df: pd.DataFrame,
    target_column: str = "target",
    prediction_column: str = "prediction",
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run Evidently ClassificationPreset report comparing baseline vs current model performance.

    Both DataFrames must contain the target and prediction columns.

    ⚠ Evidently quirk — both class labels (0 AND 1) must appear in BOTH
    ``baseline_df`` and ``current_df``. If either side has only one class
    (common when sampling a highly-imbalanced eval set with a flat LIMIT),
    ClassificationPreset raises ``KeyError: '0'`` deep inside Evidently's
    metric parser. Stratify the baseline by class label (e.g. UNION ALL
    of N fraud + N non-fraud rows) to guarantee both classes appear.
    https://github.com/evidentlyai/evidently/issues  (see ClassMetric parsing)

    Args:
        baseline_df: Reference data with target and prediction columns. Must
            contain BOTH class labels (0 and 1).
        current_df: Current data with target and prediction columns. Must
            also contain BOTH class labels.
        target_column: Name of the ground-truth label column.
        prediction_column: Name of the predicted label column.
        output_path: If provided, saves the HTML report to this path.

    Returns:
        Dictionary with:
            - 'snapshot': The Evidently report snapshot
            - 'metrics': Raw metrics dict extracted from the report

    Raises:
        ValueError: If either dataframe lacks both class labels (caught
            early before Evidently crashes with a confusing KeyError).
    """
    # Pre-flight check — Evidently's KeyError is unhelpful; fail loudly here.
    # Both `target` AND `prediction` columns must contain BOTH classes (0 AND 1)
    # in BOTH datasets. Evidently's `ClassificationQualityByClass` internally
    # calls sklearn's `classification_report` without `labels=` — sklearn then
    # omits classes that never appear as predictions, and Evidently crashes
    # with `KeyError: '0'` when it tries to read the omitted class.
    for label, df in (("baseline_df", baseline_df), ("current_df", current_df)):
        for col in (target_column, prediction_column):
            unique = sorted(df[col].dropna().unique().tolist())
            if len(unique) < 2:
                raise ValueError(
                    f"{label}.{col} has only {len(unique)} unique value(s): {unique}. "
                    f"Evidently ClassificationPreset needs BOTH 0 and 1 in BOTH "
                    f"target and prediction columns of BOTH datasets. "
                    f"If the model never predicted the minority class on this sample, "
                    f"either increase the sample size, lower the decision threshold, "
                    f"or stratify the upstream query to guarantee class diversity."
                )

    data_def = DataDefinition(
        classification=[
            BinaryClassification(
                target=target_column,
                prediction_labels=prediction_column,
            )
        ]
    )

    ref_dataset = Dataset.from_pandas(baseline_df, data_definition=data_def)
    cur_dataset = Dataset.from_pandas(current_df, data_definition=data_def)

    report = Report(metrics=[ClassificationPreset()])
    snapshot = report.run(reference_data=ref_dataset, current_data=cur_dataset)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        snapshot.save_html(output_path)
        logger.info(f"Classification report saved to {output_path}")

    result: Dict[str, Any] = {
        "snapshot": snapshot,
        "metrics": snapshot.dict().get("metrics", []),
    }

    logger.info("Classification report generated successfully")
    return result

