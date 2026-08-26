"""Unit tests for the two drift-monitor gaps closed on this branch:

1. CloudWatch auto-publish — ``publish_cloudwatch_metrics()`` emits the
   drift metrics to CloudWatch on every scheduled run, so the alarms and
   dashboard built by ``create_cloudwatch_monitoring.py`` update from the
   schedule rather than only when that script is run by hand. This is the
   sink SageMaker Model Monitor gave for free.

2. Automatic problem-type handling — the model-quality path adapts to
   binary / multiclass / regression. These tests pin the two behaviors that
   would otherwise crash on the non-binary paths:
     * ``publish_cloudwatch_metrics`` publishes only finite metrics and
       skips ``None`` (regression carries None ROC-AUC / accuracy; binary
       carries None MAE / RMSE / R²), never raising on a CloudWatch error.
     * ``send_sns_alert`` renders a None primary metric as "n/a" instead of
       crashing on ``:.4f`` formatting.

All AWS clients are monkeypatched — nothing here touches a live endpoint.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.drift_monitoring import lambda_drift_monitor as ldm


# ---------------------------------------------------------------------------
# publish_cloudwatch_metrics
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_cloudwatch(monkeypatch):
    """Replace the module-level CloudWatch client with a MagicMock and
    return it so tests can inspect the put_metric_data calls."""
    cw = MagicMock()
    monkeypatch.setattr(ldm, "cloudwatch", cw)
    monkeypatch.setattr(ldm, "CLOUDWATCH_NAMESPACE", "Test/Drift")
    monkeypatch.setattr(ldm, "ENDPOINT_NAME", "test-endpoint")
    return cw


def _published_metric_names(cw_mock):
    """Flatten every MetricData datum across all put_metric_data calls into
    a {name: value} dict for easy assertions."""
    published = {}
    for call in cw_mock.put_metric_data.call_args_list:
        for datum in call.kwargs["MetricData"]:
            published[datum["MetricName"]] = datum["Value"]
    return published


def test_publish_binary_metrics(mock_cloudwatch):
    data_drift = {
        "drifted_columns_share": 0.25,
        "drifted_features_count": 3,
        "detected": True,
    }
    model_drift = {
        "baseline_roc_auc": 0.92,
        "current_roc_auc": 0.88,
        "degradation": 0.04,
        "accuracy": 0.95,
        "precision": 0.80,
        "recall": 0.70,
        "detected": False,
        "mae": None,
        "rmse": None,
        "r2": None,
    }

    count = ldm.publish_cloudwatch_metrics(data_drift, model_drift)
    published = _published_metric_names(mock_cloudwatch)

    assert published["DriftedColumnsShare"] == 0.25
    assert published["DriftedColumnsCount"] == 3
    assert published["DataDriftDetected"] == 1
    assert published["BaselineROCAUC"] == 0.92
    assert published["CurrentROCAUC"] == 0.88
    assert published["ROCAUCDegradation"] == 0.04
    assert published["Accuracy"] == 0.95
    assert published["ModelDriftDetected"] == 0
    # Regression-only metrics were None → must NOT be published.
    assert "MAE" not in published
    assert "RMSE" not in published
    assert "R2" not in published
    assert count == len(published)


def test_publish_regression_metrics_skips_none_classification(mock_cloudwatch):
    # A regression result carries None for every classification metric and
    # real values for MAE/RMSE/R². Only the finite ones should be published.
    model_drift = {
        "problem_type": "regression",
        "primary_metric": "rmse",
        "baseline_roc_auc": None,
        "current_roc_auc": None,
        "accuracy": None,
        "precision": None,
        "recall": None,
        "degradation": 1.5,
        "mae": 3.2,
        "rmse": 4.1,
        "r2": 0.78,
        "detected": True,
    }

    ldm.publish_cloudwatch_metrics(None, model_drift)
    published = _published_metric_names(mock_cloudwatch)

    assert published["MAE"] == 3.2
    assert published["RMSE"] == 4.1
    assert published["R2"] == 0.78
    assert published["ModelDriftDetected"] == 1
    # A regression `degradation` is an RMSE INCREASE, not a ROC-AUC drop — it
    # must NOT be republished under the ROC-AUC-named metric. ROC-AUC metrics are
    # binary-only.
    assert "ROCAUCDegradation" not in published
    # None classification metrics must be dropped, not sent as None/NaN.
    for skipped in ("BaselineROCAUC", "CurrentROCAUC", "Accuracy", "Precision", "Recall"):
        assert skipped not in published


def test_publish_multiclass_omits_roc_auc_named_metrics(mock_cloudwatch):
    # Multiclass has no single-probability ROC-AUC; its degradation is an
    # accuracy drop. None of the ROC-AUC-named metrics should be published.
    model_drift = {
        "problem_type": "multiclass_classification",
        "primary_metric": "accuracy",
        "baseline_roc_auc": None,
        "current_roc_auc": None,
        "accuracy": 0.80,
        "precision": 0.78,
        "recall": 0.75,
        "degradation": 0.10,
        "degradation_pct": 11.1,
        "detected": True,
    }
    ldm.publish_cloudwatch_metrics(None, model_drift)
    published = _published_metric_names(mock_cloudwatch)

    assert published["Accuracy"] == 0.80
    # Primary degradation still surfaces under the generic normalized metric.
    assert round(published["PrimaryMetricDegradationRatio"], 5) == round(0.111, 5)
    for skipped in ("BaselineROCAUC", "CurrentROCAUC", "ROCAUCDegradation"):
        assert skipped not in published


def test_publish_skips_non_finite_values(mock_cloudwatch):
    # NaN / inf are rejected by CloudWatch — _add must filter them.
    model_drift = {
        "baseline_roc_auc": float("nan"),
        "current_roc_auc": float("inf"),
        "degradation": 0.01,
        "accuracy": 0.9,
        "detected": True,
    }
    ldm.publish_cloudwatch_metrics(None, model_drift)
    published = _published_metric_names(mock_cloudwatch)

    assert "BaselineROCAUC" not in published
    assert "CurrentROCAUC" not in published
    assert published["ROCAUCDegradation"] == 0.01
    assert published["Accuracy"] == 0.9


def test_publish_with_no_results_still_emits_heartbeat(mock_cloudwatch):
    # A run that scored nothing (empty/thin window) must still emit the
    # DriftRunExecuted heartbeat so the schedule-stall alarm can tell a
    # no-data run apart from a Lambda that never fired. Only the heartbeat
    # is published in that case.
    assert ldm.publish_cloudwatch_metrics(None, None) == 1
    mock_cloudwatch.put_metric_data.assert_called_once()
    published = _published_metric_names(mock_cloudwatch)
    assert published == {"DriftRunExecuted": 1}


def test_publish_never_raises_on_cloudwatch_error(mock_cloudwatch):
    # A CloudWatch outage must not fail an otherwise-successful drift run.
    mock_cloudwatch.put_metric_data.side_effect = RuntimeError("throttled")
    result = ldm.publish_cloudwatch_metrics({"drifted_columns_share": 0.1, "detected": False}, None)
    assert result == 0


def test_publish_uses_configured_namespace_and_endpoint_dimension(mock_cloudwatch):
    ldm.publish_cloudwatch_metrics({"drifted_columns_share": 0.1, "detected": False}, None)
    call = mock_cloudwatch.put_metric_data.call_args
    assert call.kwargs["Namespace"] == "Test/Drift"
    for datum in call.kwargs["MetricData"]:
        assert {"Name": "Endpoint", "Value": "test-endpoint"} in datum["Dimensions"]


# ---------------------------------------------------------------------------
# send_sns_alert — must not crash on None primary metrics (regression/multiclass)
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_sns(monkeypatch):
    sns = MagicMock()
    sns.publish.return_value = {"MessageId": "test-id"}
    monkeypatch.setattr(ldm, "sns", sns)
    monkeypatch.setattr(ldm, "SNS_TOPIC_ARN", "arn:aws:sns:us-east-1:123:test")
    return sns


def _published_message(sns_mock):
    return sns_mock.publish.call_args.kwargs["Message"]


def test_alert_regression_renders_none_roc_auc_as_na(mock_sns):
    model_drift = {
        "problem_type": "regression",
        "primary_metric": "rmse",
        "baseline_primary": 4.0,
        "current_primary": 5.0,
        "baseline_roc_auc": None,
        "current_roc_auc": None,
        "accuracy": None,
        "precision": None,
        "recall": None,
        "degradation": 1.0,
        "degradation_pct": 25.0,
        "mae": 3.2,
        "rmse": 5.0,
        "r2": 0.6,
        "detected": True,
    }

    # Must not raise despite the None ROC-AUC / accuracy fields.
    ldm.send_sns_alert(None, model_drift)
    msg = _published_message(mock_sns)

    assert "RegressionPreset" in msg
    assert "regression" in msg
    assert "RMSE" in msg
    assert "Current RMSE: 5.0000" in msg
    assert "Current MAE:  3.2000" in msg


def test_alert_multiclass_renders_none_roc_auc_as_na(mock_sns):
    model_drift = {
        "problem_type": "multiclass_classification",
        "primary_metric": "accuracy",
        "baseline_primary": 0.90,
        "current_primary": 0.80,
        "baseline_roc_auc": None,
        "current_roc_auc": None,
        "accuracy": 0.80,
        "precision": 0.78,
        "recall": 0.75,
        "degradation": 0.10,
        "degradation_pct": 11.1,
        "detected": True,
    }

    ldm.send_sns_alert(None, model_drift)
    msg = _published_message(mock_sns)

    assert "ClassificationPreset" in msg
    # ROC-AUC is undefined for the multiclass single-probability case → "n/a".
    assert "Current ROC-AUC: n/a" in msg
    assert "Current Accuracy: 0.8000" in msg


def test_alert_binary_still_reports_roc_auc(mock_sns):
    model_drift = {
        "problem_type": "binary_classification",
        "primary_metric": "roc_auc",
        "baseline_primary": 0.92,
        "current_primary": 0.85,
        "baseline_roc_auc": 0.92,
        "current_roc_auc": 0.85,
        "accuracy": 0.95,
        "precision": 0.80,
        "recall": 0.70,
        "degradation": 0.07,
        "degradation_pct": 7.6,
        "detected": True,
    }

    ldm.send_sns_alert(None, model_drift)
    msg = _published_message(mock_sns)

    assert "Current ROC-AUC: 0.8500" in msg
    assert "n/a" not in msg


def test_alert_not_sent_when_no_drift(mock_sns):
    ldm.send_sns_alert(
        {"detected": False},
        {"detected": False, "problem_type": "binary_classification"},
    )
    mock_sns.publish.assert_not_called()
