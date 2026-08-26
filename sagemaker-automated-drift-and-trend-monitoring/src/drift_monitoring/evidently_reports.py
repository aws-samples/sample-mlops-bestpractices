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
from evidently.core.datasets import (
    BinaryClassification,
    DataDefinition,
    Dataset,
    MulticlassClassification,
    Regression,
)
from evidently.metrics import DriftedColumnsCount, ValueDrift
from evidently.presets import ClassificationPreset, DataDriftPreset, RegressionPreset

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
    multiclass: bool = False,
) -> Dict[str, Any]:
    """
    Run Evidently ClassificationPreset report comparing baseline vs current model performance.

    Both DataFrames must contain the target and prediction columns.

    Set ``multiclass=True`` for targets with more than two classes: the
    Evidently data definition switches from ``BinaryClassification`` to
    ``MulticlassClassification`` and the "both 0 AND 1 must appear" pre-flight
    relaxes to "at least 2 distinct classes must appear". Leave it False (the
    default) for the binary case, which is unchanged.

    ⚠ Multiclass class-set consistency — the ">= 2 distinct classes" check is
    NOT sufficient for the multiclass path. Evidently computes per-class metrics
    for the CURRENT frame and then looks each of those classes up in the
    REFERENCE result, while sklearn silently omits any class that never appears.
    So on top of the "2+ classes" loop, multiclass adds two checks:

      * every class present in ``current_df[target]`` must also be predicted at
        least once in ``current_df[prediction]`` (else sklearn drops it and
        Evidently raises a bare ``KeyError``); and
      * every class present in ``current_df[target]`` must also appear in
        ``baseline_df[target]`` (else the reference result has no entry to align
        against and Evidently raises a bare ``KeyError``).

    Passing ``labels=`` to ``MulticlassClassification`` does NOT prevent either
    KeyError (verified on evidently 0.7.21), so both are validated up front and
    raise a clear ``ValueError`` instead. The fix is to increase the sample size
    or stratify the upstream query so every actual class is represented in the
    predictions and covered by the baseline.

    ⚠ Evidently quirk — both class labels (0 AND 1) must appear in BOTH
    ``baseline_df`` and ``current_df``. If either side has only one class
    (common when sampling a highly-imbalanced eval set with a flat LIMIT),
    ClassificationPreset raises ``KeyError: '0'`` deep inside Evidently's
    metric parser. Stratify the baseline by class label (e.g. UNION ALL
    of N fraud + N non-fraud rows) to guarantee both classes appear.
    https://github.com/evidentlyai/evidently/issues  (see ClassMetric parsing)

    Args:
        baseline_df: Reference data with target and prediction columns. Must
            contain BOTH class labels (0 and 1). The target/prediction columns
            may be any numeric dtype (int, nullable Int64, float) — they are
            normalized to int internally so the caller need not cast them.
        current_df: Current data with target and prediction columns. Must
            also contain BOTH class labels. Same dtype handling as baseline_df.
        target_column: Name of the ground-truth label column.
        prediction_column: Name of the predicted label column.
        output_path: If provided, saves the HTML report to this path.

    Returns:
        Dictionary with:
            - 'snapshot': The Evidently report snapshot
            - 'metrics': Raw metrics dict extracted from the report

    Raises:
        ValueError: If either dataframe lacks both class labels (caught
            early before Evidently crashes with a confusing KeyError). When
            ``multiclass=True``, also raised if a class in ``current_df[target]``
            is never predicted in ``current_df[prediction]``, or if a class in
            ``current_df[target]`` never appears in ``baseline_df[target]`` —
            both would otherwise surface as a bare Evidently ``KeyError``.
    """
    # Normalize label dtypes FIRST — this must happen before the class check
    # and before building the Evidently Datasets. Evidently's
    # `ClassificationQualityByClass` builds its class set from
    # `str(k) for k in y_true.unique()` and then looks those keys up in
    # sklearn's `classification_report` dict. If `target` and `prediction`
    # arrive with different dtypes (e.g. one column is numpy int64 → labels
    # "0"/"1", while the other is pandas nullable Int64 or float64 → labels
    # "0.0"/"1.0"), the string keys don't match and Evidently raises a bare
    # `KeyError: '0'`. This happens whenever a caller mixes a hand-built
    # DataFrame (dtype int) with one read from Athena via awswrangler (which
    # returns INTEGER columns as nullable Int64/float). Casting both columns
    # in both frames to plain Python-int labels removes the mismatch. We copy
    # so the caller's DataFrames are left untouched.
    label_cols = (target_column, prediction_column)
    baseline_df = baseline_df.dropna(subset=list(label_cols)).copy()
    current_df = current_df.dropna(subset=list(label_cols)).copy()
    for df in (baseline_df, current_df):
        for col in label_cols:
            # round() guards against float labels like 0.9999; .astype(int)
            # then yields consistent "0"/"1" string keys for every caller.
            # (NaNs were dropped above so the int cast can't raise.)
            df[col] = df[col].astype(float).round().astype(int)

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
                    f"Evidently ClassificationPreset needs at least 2 classes in BOTH "
                    f"target and prediction columns of BOTH datasets. "
                    f"If the model never predicted a class on this sample, "
                    f"either increase the sample size, lower the decision threshold, "
                    f"or stratify the upstream query to guarantee class diversity."
                )

    # Multiclass-only pre-flight — the "< 2 unique" loop above guarantees each
    # column has 2+ classes, but for MULTICLASS that is not enough. Evidently's
    # ClassificationQualityByClass computes per-class metrics for the CURRENT
    # frame and then, when rendering the reference column, looks each of those
    # classes up in the REFERENCE result (see evidently
    # core/metric_types.py::get_default_render_ref → `ref_result.values[k]`).
    # sklearn's classification_report omits any class that never appears, so two
    # situations raise a bare `KeyError` deep inside Evidently:
    #   1. a class occurs in current[target] but the model never predicted it in
    #      current[prediction] — sklearn drops it from the current per-class
    #      report, then Evidently KeyErrors reading the missing class; and
    #   2. a class occurs in current[target] but never in baseline[target] — the
    #      reference result has no entry for it, so the ref-alignment lookup
    #      KeyErrors.
    # Passing `labels=` to MulticlassClassification does NOT prevent either
    # KeyError (verified on evidently 0.7.21), so we validate and fail loudly.
    # The binary path is untouched: it keeps the {0, 1}-both-present guarantee
    # from the loop above and never enters this block.
    if multiclass:
        base_target = set(baseline_df[target_column].dropna().unique().tolist())
        cur_target = set(current_df[target_column].dropna().unique().tolist())
        cur_pred = set(current_df[prediction_column].dropna().unique().tolist())

        # (1) Model must predict every class that actually occurs in current
        #     ground truth, or sklearn omits it and Evidently KeyErrors.
        unpredicted = cur_target - cur_pred
        if unpredicted:
            raise ValueError(
                f"current_df: class(es) {sorted(unpredicted)} occur in "
                f"{target_column!r} but were never predicted in "
                f"{prediction_column!r} (current target classes "
                f"{sorted(cur_target)}, predicted classes {sorted(cur_pred)}). "
                f"Evidently's multiclass ClassificationPreset scores every class "
                f"present in current[{target_column!r}], but sklearn omits classes "
                f"that never appear as a prediction, so Evidently crashes with a "
                f"confusing KeyError on the missing class. "
                f"Increase the sample size, lower the decision threshold, or "
                f"stratify the upstream query so every actual class is represented "
                f"in the predictions."
            )

        # (2) Every class in current ground truth must also appear in the
        #     baseline ground truth, or the reference result has nothing to align
        #     the per-class metric against and Evidently KeyErrors.
        missing_in_baseline = cur_target - base_target
        if missing_in_baseline:
            raise ValueError(
                f"current_df: class(es) {sorted(missing_in_baseline)} occur in "
                f"{target_column!r} but never appear in baseline_df[{target_column!r}] "
                f"(baseline target classes {sorted(base_target)}, current target "
                f"classes {sorted(cur_target)}). "
                f"Evidently aligns per-class metrics between baseline and current, "
                f"and a class present only in current has no reference entry, so it "
                f"crashes with a confusing KeyError. "
                f"Stratify the upstream query (or widen the baseline sample) so the "
                f"baseline and current cover the same set of classes."
            )

    classification = (
        MulticlassClassification(
            target=target_column,
            prediction_labels=prediction_column,
        )
        if multiclass
        else BinaryClassification(
            target=target_column,
            prediction_labels=prediction_column,
        )
    )
    data_def = DataDefinition(classification=[classification])

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


def run_regression_report(
    baseline_df: pd.DataFrame,
    current_df: pd.DataFrame,
    target_column: str = "target",
    prediction_column: str = "prediction",
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run Evidently RegressionPreset comparing baseline vs current model performance.

    The regression analogue of ``run_classification_report``: use it when the
    model predicts a continuous value rather than a class label. Both
    DataFrames must contain the target (actual) and prediction columns as
    numeric values.

    Unlike the classification path there is no "both classes present"
    pre-flight — the only requirement is that the target/prediction columns
    are numeric and non-empty. Rows with a NaN in either column are dropped
    (Evidently's regression metrics can't score a row with a missing actual
    or prediction).

    Args:
        baseline_df: Reference data with numeric target and prediction columns.
        current_df: Current data with numeric target and prediction columns.
        target_column: Name of the ground-truth (actual value) column.
        prediction_column: Name of the predicted value column.
        output_path: If provided, saves the HTML report to this path.

    Returns:
        Dictionary with:
            - 'snapshot': The Evidently report snapshot
            - 'metrics': Raw metrics dict extracted from the report

    Raises:
        ValueError: If either dataframe has no rows left after dropping NaNs
            in the target/prediction columns.
    """
    label_cols = [target_column, prediction_column]
    baseline_df = baseline_df.dropna(subset=label_cols).copy()
    current_df = current_df.dropna(subset=label_cols).copy()
    for df in (baseline_df, current_df):
        for col in label_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    baseline_df = baseline_df.dropna(subset=label_cols)
    current_df = current_df.dropna(subset=label_cols)

    for label, df in (("baseline_df", baseline_df), ("current_df", current_df)):
        if len(df) == 0:
            raise ValueError(
                f"{label} has no numeric rows for regression scoring after "
                f"dropping NaNs in {label_cols}. Check that {target_column!r} "
                f"and {prediction_column!r} are populated numeric columns."
            )

    data_def = DataDefinition(
        regression=[
            Regression(target=target_column, prediction=prediction_column)
        ]
    )

    ref_dataset = Dataset.from_pandas(baseline_df, data_definition=data_def)
    cur_dataset = Dataset.from_pandas(current_df, data_definition=data_def)

    report = Report(metrics=[RegressionPreset()])
    snapshot = report.run(reference_data=ref_dataset, current_data=cur_dataset)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        snapshot.save_html(output_path)
        logger.info(f"Regression report saved to {output_path}")

    result: Dict[str, Any] = {
        "snapshot": snapshot,
        "metrics": snapshot.dict().get("metrics", []),
    }

    logger.info("Regression report generated successfully")
    return result

