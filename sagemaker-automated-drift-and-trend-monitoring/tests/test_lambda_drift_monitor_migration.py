"""Unit tests for the drift-monitor migration changes in
``src/drift_monitoring/lambda_drift_monitor.py``.

Locks down the behaviors introduced/altered on this branch:

1. HEARTBEAT — ``publish_cloudwatch_metrics(None, None)`` still emits the
   ``DriftRunExecuted`` metric (value 1) so a scored-nothing run is
   distinguishable from a stalled schedule. (Contrast with the drift-results
   case covered in test_lambda_drift_monitor_cloudwatch.py.)
2. NORMALIZED DEGRADATION — a ``degradation_pct`` publishes as
   ``PrimaryMetricDegradationRatio == degradation_pct / 100``, the absolute
   ``ROCAUCDegradation`` is still published, and the ratio is dropped when
   ``degradation_pct`` is None.
3. EXECUTION STATUS collapse — ``_overall_execution_status()``.
4. ENDPOINT PREDICATE — ``_endpoint_predicate()``.
5. WRITE RECORD — new problem-type fields round-trip through the SQS body.
6. BACKFILL GUARD — the inference_responses UPDATE runs only on SUCCESS.
7. HANDLER RE-RAISES — a fatal error re-raises (no 200/500 body) after a
   best-effort ERROR row is recorded.

All AWS clients are monkeypatched — nothing here touches a live endpoint.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from src.drift_monitoring import lambda_drift_monitor as ldm


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _published_metric_names(cw_mock):
    """Flatten every MetricData datum across all put_metric_data calls into a
    {name: value} dict for easy assertions."""
    published = {}
    for call in cw_mock.put_metric_data.call_args_list:
        for datum in call.kwargs["MetricData"]:
            published[datum["MetricName"]] = datum["Value"]
    return published


@pytest.fixture
def mock_cloudwatch(monkeypatch):
    cw = MagicMock()
    monkeypatch.setattr(ldm, "cloudwatch", cw)
    monkeypatch.setattr(ldm, "CLOUDWATCH_NAMESPACE", "Test/Drift")
    monkeypatch.setattr(ldm, "ENDPOINT_NAME", "test-endpoint")
    return cw


# ---------------------------------------------------------------------------
# 1. HEARTBEAT — DriftRunExecuted always published, even with no results
# ---------------------------------------------------------------------------


def test_heartbeat_published_with_no_results(mock_cloudwatch):
    # A run that scored nothing must STILL emit the heartbeat metric so a
    # missing-data alarm can tell "monitor stalled" from "monitor ran, empty".
    count = ldm.publish_cloudwatch_metrics(None, None)
    published = _published_metric_names(mock_cloudwatch)

    assert "DriftRunExecuted" in published
    assert published["DriftRunExecuted"] == 1
    # put_metric_data WAS called (the heartbeat alone is enough to publish).
    mock_cloudwatch.put_metric_data.assert_called()
    assert count == 1


def test_heartbeat_present_alongside_drift_results(mock_cloudwatch):
    ldm.publish_cloudwatch_metrics({"drifted_columns_share": 0.1, "detected": False}, None)
    published = _published_metric_names(mock_cloudwatch)
    assert published["DriftRunExecuted"] == 1


# ---------------------------------------------------------------------------
# 2. NORMALIZED DEGRADATION — PrimaryMetricDegradationRatio == pct / 100
# ---------------------------------------------------------------------------


def test_degradation_ratio_is_pct_over_100(mock_cloudwatch):
    model_drift = {
        "degradation": 0.04,
        "degradation_pct": 25.0,
        "detected": True,
    }
    ldm.publish_cloudwatch_metrics(None, model_drift)
    published = _published_metric_names(mock_cloudwatch)

    assert published["PrimaryMetricDegradationRatio"] == 0.25
    # The absolute degradation metric is still published from `degradation`.
    assert published["ROCAUCDegradation"] == 0.04


def test_degradation_ratio_not_published_when_pct_none(mock_cloudwatch):
    model_drift = {
        "degradation": 0.04,
        "degradation_pct": None,
        "detected": True,
    }
    ldm.publish_cloudwatch_metrics(None, model_drift)
    published = _published_metric_names(mock_cloudwatch)

    assert "PrimaryMetricDegradationRatio" not in published
    # Absolute degradation still lands.
    assert published["ROCAUCDegradation"] == 0.04


# ---------------------------------------------------------------------------
# 3. EXECUTION STATUS collapse — _overall_execution_status()
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "data_status, model_status, expected",
    [
        (None, None, "NO_DATA"),
        # Both executed checks produced a verdict → full SUCCESS.
        ("SUCCESS", "SUCCESS", "SUCCESS"),
        # Only one check executed and it succeeded → still SUCCESS (the other was
        # not executed, so it contributes nothing rather than a false verdict).
        ("SUCCESS", None, "SUCCESS"),
        # One check scored, the other found no data → PARTIAL_SUCCESS (previously
        # this incorrectly collapsed to SUCCESS).
        ("SUCCESS", "NO_DATA", "PARTIAL_SUCCESS"),
        ("SUCCESS", "INSUFFICIENT_SAMPLES", "PARTIAL_SUCCESS"),
        ("SUCCESS", "INSUFFICIENT_CLASSES", "PARTIAL_SUCCESS"),
        # No verdict at all → surface the most specific reason.
        ("INSUFFICIENT_SAMPLES", "NO_DATA", "INSUFFICIENT_SAMPLES"),
        ("INSUFFICIENT_CLASSES", None, "INSUFFICIENT_CLASSES"),
        ("NO_DATA", "NO_DATA", "NO_DATA"),
    ],
)
def test_overall_execution_status(monkeypatch, data_status, model_status, expected):
    monkeypatch.setattr(ldm, "RUN_DIAGNOSTICS", {"data": data_status, "model": model_status})
    assert ldm._overall_execution_status() == expected


# ---------------------------------------------------------------------------
# 4. ENDPOINT PREDICATE — _endpoint_predicate()
# ---------------------------------------------------------------------------


def test_endpoint_predicate_with_name(monkeypatch):
    monkeypatch.setattr(ldm, "ENDPOINT_NAME", "ep1")
    pred = ldm._endpoint_predicate()
    assert pred.startswith("AND")
    assert "endpoint_name = 'ep1'" in pred


def test_endpoint_predicate_empty_when_unset(monkeypatch):
    monkeypatch.setattr(ldm, "ENDPOINT_NAME", "")
    assert ldm._endpoint_predicate() == ""


# ---------------------------------------------------------------------------
# 5. WRITE RECORD — new problem-type fields round-trip through the SQS body
# ---------------------------------------------------------------------------


def _sent_record(sqs_mock):
    return json.loads(sqs_mock.send_message.call_args.kwargs["MessageBody"])


def test_write_monitoring_results_round_trips_new_fields(monkeypatch):
    sqs = MagicMock()
    monkeypatch.setattr(ldm, "sqs", sqs)
    monkeypatch.setattr(ldm, "MONITORING_SQS_QUEUE_URL", "https://sqs.local/queue")
    # No endpoint → no backfill UPDATE (that guard is exercised separately).
    monkeypatch.setattr(ldm, "ENDPOINT_NAME", "")
    monkeypatch.setattr(ldm, "load_baseline_from_registry", lambda: {})

    # Regression-style model dict: mae/rmse/r2 set, classification fields None.
    model_drift = {
        "detected": False,
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
        "sample_size": 500,
    }
    data_drift = {
        "detected": True,
        "drifted_features_count": 2,
        "drifted_columns_share": 0.1,
        "features_analyzed": 20,
        "sample_size": 300,
        "per_column": {},
    }

    ldm.write_monitoring_results(data_drift, model_drift, "mlflow-1", execution_status="SUCCESS")

    record = _sent_record(sqs)
    assert record["execution_status"] == "SUCCESS"
    # New problem-type-aware fields exist and round-trip.
    for key in ("problem_type", "primary_metric", "baseline_primary",
                "current_primary", "mae", "rmse", "r2"):
        assert key in record
    assert record["problem_type"] == "regression"
    assert record["primary_metric"] == "rmse"
    assert record["baseline_primary"] == 4.0
    assert record["current_primary"] == 5.0
    assert record["mae"] == 3.2
    assert record["rmse"] == 5.0
    assert record["r2"] == 0.6
    assert record["mlflow_run_id"] == "mlflow-1"


def test_unexecuted_check_verdict_is_null_and_per_check_status_round_trips(monkeypatch):
    # A check that did NOT execute (result None) must leave its verdict NULL —
    # not coerce to a real "no drift" (False) verdict — and the per-check
    # execution statuses must round-trip through the SQS body.
    sqs = MagicMock()
    monkeypatch.setattr(ldm, "sqs", sqs)
    monkeypatch.setattr(ldm, "MONITORING_SQS_QUEUE_URL", "https://sqs.local/queue")
    monkeypatch.setattr(ldm, "ENDPOINT_NAME", "")  # no backfill
    monkeypatch.setattr(ldm, "load_baseline_from_registry", lambda: {})
    monkeypatch.setattr(ldm, "RUN_DIAGNOSTICS", {"data": "SUCCESS", "model": None})

    data_drift = {
        "detected": True,
        "drifted_features_count": 1,
        "drifted_columns_share": 0.05,
        "features_analyzed": 10,
        "sample_size": 200,
        "per_column": {},
    }
    # model_drift_result is None → model check never ran.
    ldm.write_monitoring_results(data_drift, None, None, execution_status="PARTIAL_SUCCESS")

    record = _sent_record(sqs)
    assert record["execution_status"] == "PARTIAL_SUCCESS"
    assert record["data_drift_detected"] is True
    # Unexecuted model check → NULL verdict, NOT False.
    assert record["model_drift_detected"] is None
    # Per-check statuses round-trip.
    assert record["data_drift_status"] == "SUCCESS"
    assert record["model_drift_status"] is None
    # alert_sent stays a real bool even though one verdict is NULL.
    assert record["alert_sent"] is True


# ---------------------------------------------------------------------------
# 6. MEMBERSHIP BRIDGE — monitoring_run_inferences rows are written only on
#    SUCCESS/PARTIAL_SUCCESS, only after the summary row is confirmed persisted,
#    and record the EXACT scored inference_ids per check_type.
# ---------------------------------------------------------------------------


def _athena_side_effect(bridge_count):
    """Build an execute_athena_query side_effect for the membership path.

    The bridge writer now COUNT-verifies its inserts (Finding 4), so the mock
    must answer the verification SELECT with a matching count. Returns:
      * the persistence-confirmation SELECT (monitoring_responses) → truthy
      * the bridge COUNT(*) verification → [{'c': bridge_count}]
      * everything else (DELETE / INSERT / MERGE marker) → []
    """
    def _fake(sql, *args, **kwargs):
        if "COUNT(*)" in sql and "monitoring_run_inferences" in sql:
            return [{"c": str(bridge_count)}]
        if "monitoring_responses" in sql and "SELECT 1" in sql:
            return [{"_col0": "1"}]
        return []
    return _fake


def _bridge_insert_issued(athena_mock):
    """True if any execute_athena_query call INSERTed into the bridge table."""
    for call in athena_mock.call_args_list:
        sql = call.args[0] if call.args else call.kwargs.get("sql", "")
        if "INSERT INTO" in sql and "monitoring_run_inferences" in sql:
            return True
    return False


def _bridge_insert_sqls(athena_mock):
    """All INSERT statements aimed at the bridge table."""
    out = []
    for call in athena_mock.call_args_list:
        sql = call.args[0] if call.args else call.kwargs.get("sql", "")
        if "INSERT INTO" in sql and "monitoring_run_inferences" in sql:
            out.append(sql)
    return out


def _data_drift_result(**overrides):
    base = {
        "detected": False,
        "drifted_features_count": 1,
        "drifted_columns_share": 0.05,
        "features_analyzed": 10,
        "sample_size": 200,
        "per_column": {},
        # EXACT scored cohort — drives the bridge write (not watermarks).
        "scored_inference_ids": ["inf-1", "inf-2", "inf-3"],
        # Watermarks are retained for diagnostics but no longer drive membership.
        "window_lower": "2026-08-17 00:00:00",
        "window_upper": "2026-08-24 00:00:00",
    }
    base.update(overrides)
    return base


def test_membership_skipped_when_not_success(monkeypatch):
    athena = MagicMock()
    sqs = MagicMock()
    monkeypatch.setattr(ldm, "execute_athena_query", athena)
    monkeypatch.setattr(ldm, "sqs", sqs)
    monkeypatch.setattr(ldm, "MONITORING_SQS_QUEUE_URL", "https://sqs.local/queue")
    monkeypatch.setattr(ldm, "ENDPOINT_NAME", "ep1")
    monkeypatch.setattr(ldm, "load_baseline_from_registry", lambda: {})

    ldm.write_monitoring_results(None, None, None, execution_status="NO_DATA")

    assert not _bridge_insert_issued(athena), "bridge INSERT must NOT run for a non-SUCCESS status"


def test_membership_written_on_success_with_exact_ids(monkeypatch):
    # execute_athena_query serves the persistence-confirmation SELECT (truthy →
    # confirmed on first poll), the bridge DELETE/INSERT, and the COUNT(*)
    # verification (3 scored ids → count 3). The bridge write is driven by the
    # check's scored_inference_ids, tagged with check_type.
    athena = MagicMock(side_effect=_athena_side_effect(3))
    sqs = MagicMock()
    monkeypatch.setattr(ldm, "execute_athena_query", athena)
    monkeypatch.setattr(ldm, "sqs", sqs)
    monkeypatch.setattr(ldm, "MONITORING_SQS_QUEUE_URL", "https://sqs.local/queue")
    monkeypatch.setattr(ldm, "ENDPOINT_NAME", "ep1")
    monkeypatch.setattr(ldm, "load_baseline_from_registry", lambda: {})

    ldm.write_monitoring_results(
        _data_drift_result(), None, None, execution_status="SUCCESS"
    )

    inserts = _bridge_insert_sqls(athena)
    assert inserts, "bridge INSERT must run for a SUCCESS status"
    joined = "\n".join(inserts)
    # Exact scored ids recorded, tagged as the data-drift cohort.
    assert "inf-1" in joined and "inf-2" in joined and "inf-3" in joined
    assert "data_drift" in joined
    # No stale window-based UPDATE onto inference_responses anymore. (The
    # generation-marker MERGE legitimately UPDATEs monitoring_run_generations,
    # so scope the check to the legacy inference_responses backfill.)
    assert not any(
        "UPDATE" in c.args[0] and "inference_responses" in c.args[0]
        for c in athena.call_args_list if c.args
    )


def test_membership_skipped_when_persistence_unconfirmed(monkeypatch):
    # If the summary row never becomes queryable, membership must NOT be written
    # so bridge rows never reference a missing monitoring_run_id.
    athena = MagicMock(return_value=[])  # confirmation SELECT always empty
    sqs = MagicMock()
    monkeypatch.setattr(ldm, "execute_athena_query", athena)
    monkeypatch.setattr(ldm, "sqs", sqs)
    monkeypatch.setattr(ldm, "MONITORING_SQS_QUEUE_URL", "https://sqs.local/queue")
    monkeypatch.setattr(ldm, "ENDPOINT_NAME", "ep1")
    monkeypatch.setattr(ldm, "load_baseline_from_registry", lambda: {})
    monkeypatch.setattr(ldm.time, "sleep", lambda *_a, **_k: None)

    ldm.write_monitoring_results(
        _data_drift_result(), None, None, execution_status="SUCCESS"
    )

    assert not _bridge_insert_issued(athena), "bridge INSERT must NOT run when persistence is unconfirmed"


def test_membership_skipped_when_send_raises(monkeypatch):
    # If the SQS send itself raises, nothing was persisted → no membership write.
    athena = MagicMock(return_value=[{"_col0": "1"}])
    sqs = MagicMock()
    sqs.send_message.side_effect = RuntimeError("sqs down")
    monkeypatch.setattr(ldm, "execute_athena_query", athena)
    monkeypatch.setattr(ldm, "sqs", sqs)
    monkeypatch.setattr(ldm, "MONITORING_SQS_QUEUE_URL", "https://sqs.local/queue")
    monkeypatch.setattr(ldm, "ENDPOINT_NAME", "ep1")
    monkeypatch.setattr(ldm, "load_baseline_from_registry", lambda: {})

    ldm.write_monitoring_results(
        _data_drift_result(), None, None, execution_status="SUCCESS"
    )

    assert not _bridge_insert_issued(athena), "bridge INSERT must NOT run when the SQS send failed"


def test_write_run_inferences_batches_and_deletes_first(monkeypatch):
    # 1200 ids at batch_size=500 → one idempotency DELETE + three INSERT batches
    # (500 + 500 + 200), all keyed on run_id + check_type. The COUNT(*) verify
    # returns the matching 1200 so the write is accepted.
    athena = MagicMock(side_effect=_athena_side_effect(1200))
    monkeypatch.setattr(ldm, "execute_athena_query", athena)

    ids = [f"inf-{n}" for n in range(1200)]
    written = ldm._write_run_inferences(
        "drift-abc", "ep1", "model_drift", ids, "2026-08-25 00:00:00", batch_size=500
    )

    assert written == 1200
    sqls = [c.args[0] for c in athena.call_args_list if c.args]
    deletes = [s for s in sqls if s.strip().startswith("DELETE")]
    inserts = [s for s in sqls if "INSERT INTO" in s]
    assert len(deletes) == 1, "must clear prior membership exactly once (idempotency)"
    assert "check_type = 'model_drift'" in deletes[0]
    assert len(inserts) == 3, "1200 ids / 500 per batch → 3 INSERT batches"
    assert "drift-abc" in inserts[0] and "model_drift" in inserts[0]


def test_write_run_inferences_empty_only_deletes(monkeypatch):
    # No ids → still clears any stale membership (idempotent re-run) but writes
    # no INSERT, and reports zero rows.
    athena = MagicMock(return_value=[])
    monkeypatch.setattr(ldm, "execute_athena_query", athena)

    written = ldm._write_run_inferences(
        "drift-abc", "ep1", "data_drift", [], "2026-08-25 00:00:00"
    )

    assert written == 0
    sqls = [c.args[0] for c in athena.call_args_list if c.args]
    assert any(s.strip().startswith("DELETE") for s in sqls)
    assert not any("INSERT INTO" in s for s in sqls)


def test_write_run_inferences_escapes_single_quotes(monkeypatch):
    # A stray apostrophe in an id must be SQL-escaped, not break the INSERT.
    athena = MagicMock(side_effect=_athena_side_effect(1))
    monkeypatch.setattr(ldm, "execute_athena_query", athena)

    ldm._write_run_inferences(
        "drift-abc", "ep1", "data_drift", ["inf-o'brien"], "2026-08-25 00:00:00"
    )

    insert = next(c.args[0] for c in athena.call_args_list if c.args and "INSERT INTO" in c.args[0])
    assert "inf-o''brien" in insert


def test_write_run_inferences_raises_on_count_mismatch(monkeypatch):
    # The COUNT(*) verification returns FEWER rows than were inserted (a silent
    # short-write). _write_run_inferences must RAISE so the caller withholds the
    # completion marker and a retry re-materializes the full cohort (Finding 4).
    def _fake(sql, *a, **k):
        if "COUNT(*)" in sql:
            return [{"c": "2"}]   # only 2 landed, but 3 ids were requested
        return []
    monkeypatch.setattr(ldm, "execute_athena_query", MagicMock(side_effect=_fake))

    with pytest.raises(RuntimeError, match="count mismatch"):
        ldm._write_run_inferences(
            "drift-abc", "ep1", "data_drift", ["a", "b", "c"], "2026-08-25 00:00:00"
        )


def test_generation_marker_written_after_verified_insert(monkeypatch):
    # On a SUCCESS run, the completion marker MERGE into monitoring_run_generations
    # is issued only AFTER the count-verified bridge insert (Finding 4).
    athena = MagicMock(side_effect=_athena_side_effect(3))
    sqs = MagicMock()
    monkeypatch.setattr(ldm, "execute_athena_query", athena)
    monkeypatch.setattr(ldm, "sqs", sqs)
    monkeypatch.setattr(ldm, "MONITORING_SQS_QUEUE_URL", "https://sqs.local/queue")
    monkeypatch.setattr(ldm, "ENDPOINT_NAME", "ep1")
    monkeypatch.setattr(ldm, "load_baseline_from_registry", lambda: {})

    ldm.write_monitoring_results(
        _data_drift_result(), None, None, execution_status="SUCCESS"
    )

    sqls = [c.args[0] for c in athena.call_args_list if c.args]
    markers = [s for s in sqls if "monitoring_run_generations" in s and "MERGE INTO" in s]
    assert markers, "a completion marker MERGE must be issued for the scored cohort"
    assert "data_drift" in markers[0]
    # The marker comes AFTER the bridge COUNT(*) verification.
    verify_idx = next(i for i, s in enumerate(sqls)
                      if "COUNT(*)" in s and "monitoring_run_inferences" in s)
    marker_idx = next(i for i, s in enumerate(sqls) if s in markers)
    assert marker_idx > verify_idx


def test_membership_failure_reraises(monkeypatch):
    # A bridge write that fails its count check must make write_monitoring_results
    # re-raise (marker withheld → EventBridge retry), NOT swallow (Finding 4).
    def _fake(sql, *a, **k):
        if "monitoring_responses" in sql and "SELECT 1" in sql:
            return [{"_col0": "1"}]     # persistence confirmed
        if "COUNT(*)" in sql:
            return [{"c": "0"}]         # nothing landed → mismatch
        return []
    monkeypatch.setattr(ldm, "execute_athena_query", MagicMock(side_effect=_fake))
    sqs = MagicMock()
    monkeypatch.setattr(ldm, "sqs", sqs)
    monkeypatch.setattr(ldm, "MONITORING_SQS_QUEUE_URL", "https://sqs.local/queue")
    monkeypatch.setattr(ldm, "ENDPOINT_NAME", "ep1")
    monkeypatch.setattr(ldm, "load_baseline_from_registry", lambda: {})

    with pytest.raises(RuntimeError, match="membership incomplete"):
        ldm.write_monitoring_results(
            _data_drift_result(), None, None, execution_status="SUCCESS"
        )


# ---------------------------------------------------------------------------
# 6B. ALERT OUTBOX — notify_drift_alert is idempotent across retries (Finding 6)
# ---------------------------------------------------------------------------


def _drift_detected():
    return {
        "detected": True,
        "features_analyzed": 10,
        "drifted_features_count": 2,
        "drift_percentage": 20.0,
        "drifted_columns_share": 0.2,
        "drifted_features": [],
        "per_column": {},
    }


def test_notify_drift_alert_claims_publishes_and_marks_sent(monkeypatch):
    # First delivery: no prior claim → INSERT PENDING, publish, flip to SENT.
    def _fake(sql, *a, **k):
        if "SELECT status" in sql:
            return []   # no existing claim
        return []
    athena = MagicMock(side_effect=_fake)
    sns = MagicMock()
    sns.publish.return_value = {"MessageId": "msg-123"}
    monkeypatch.setattr(ldm, "execute_athena_query", athena)
    monkeypatch.setattr(ldm, "sns", sns)
    monkeypatch.setattr(ldm, "SNS_TOPIC_ARN", "arn:aws:sns:local:topic")

    ldm.notify_drift_alert("drift-run-1", _drift_detected(), None)

    sns.publish.assert_called_once()
    sqls = [c.args[0] for c in athena.call_args_list if c.args]
    assert any("WHEN NOT MATCHED THEN INSERT" in s and "'PENDING'" in s for s in sqls)
    assert any("status = 'SENT'" in s and "msg-123" in s for s in sqls)


def test_notify_drift_alert_skips_when_already_sent(monkeypatch):
    # A retry that sees a SENT claim must NOT publish again (dedup).
    def _fake(sql, *a, **k):
        if "SELECT status" in sql:
            return [{"status": "SENT"}]
        return []
    athena = MagicMock(side_effect=_fake)
    sns = MagicMock()
    monkeypatch.setattr(ldm, "execute_athena_query", athena)
    monkeypatch.setattr(ldm, "sns", sns)
    monkeypatch.setattr(ldm, "SNS_TOPIC_ARN", "arn:aws:sns:local:topic")

    ldm.notify_drift_alert("drift-run-1", _drift_detected(), None)

    sns.publish.assert_not_called()


def test_notify_drift_alert_noop_when_no_drift(monkeypatch):
    # No drift detected → no claim, no publish.
    athena = MagicMock(return_value=[])
    sns = MagicMock()
    monkeypatch.setattr(ldm, "execute_athena_query", athena)
    monkeypatch.setattr(ldm, "sns", sns)
    monkeypatch.setattr(ldm, "SNS_TOPIC_ARN", "arn:aws:sns:local:topic")

    ldm.notify_drift_alert("drift-run-1", {"detected": False}, {"detected": False})

    sns.publish.assert_not_called()
    athena.assert_not_called()


# ---------------------------------------------------------------------------
# 3B. COHORT ANCHOR — windows are anchored on the event `time`, not now()
# ---------------------------------------------------------------------------


def test_derive_run_as_of_parses_event_time():
    # The immutable EventBridge `time` field anchors the cohort window so a
    # retry re-materializes the SAME [lower, upper] slice.
    dt = ldm._derive_run_as_of({"time": "2026-08-24T12:00:00Z"})
    assert dt.year == 2026 and dt.month == 8 and dt.day == 24
    assert dt.hour == 12 and dt.minute == 0
    assert dt.tzinfo is None   # naive, to match Athena TIMESTAMP literals


def test_derive_run_as_of_falls_back_to_now_on_missing_or_bad_time(monkeypatch):
    # Manual invoke (no `time`) or an unparseable value → anchor on now().
    assert ldm._derive_run_as_of({}) is not None
    assert ldm._derive_run_as_of({"time": "not-a-timestamp"}) is not None


# ---------------------------------------------------------------------------
# 7. HANDLER RE-RAISES — fatal error re-raises after best-effort ERROR record
# ---------------------------------------------------------------------------


def test_handler_reraises_and_records_error(monkeypatch):
    def _boom():
        raise RuntimeError("boom")

    monkeypatch.setattr(ldm, "check_data_drift", _boom)
    # Best-effort recording must not hit AWS.
    pub = MagicMock()
    write = MagicMock()
    monkeypatch.setattr(ldm, "publish_cloudwatch_metrics", pub)
    monkeypatch.setattr(ldm, "write_monitoring_results", write)

    with pytest.raises(RuntimeError, match="boom"):
        ldm.lambda_handler({}, None)

    # The best-effort ERROR row was recorded.
    assert write.called
    assert write.call_args.kwargs.get("execution_status") == "ERROR"
    # The ERROR row must be keyed on THIS event's run_id (not a fresh, unrelated
    # UUID) so a retry MERGEs onto it instead of writing a second ERROR row.
    assert write.call_args.kwargs.get("run_id", "").startswith("drift-")


def test_derive_run_id_is_stable_for_same_event():
    """Retries/duplicate deliveries of the same scheduled event → same run_id.

    EventBridge preserves the event ``id`` across delivery retries and Lambda
    async re-runs, so deriving the run_id from it makes those retries dedup at
    the Athena MERGE instead of creating duplicate history rows.
    """
    event = {"id": "abcd-1234-EF", "source": "aws.events"}
    first = ldm._derive_run_id(event)
    second = ldm._derive_run_id(event)
    assert first == second
    assert first == "drift-abcd-1234-EF"


def test_derive_run_id_falls_back_to_uuid_for_manual_invoke():
    """Manual invocations (no event id) each get a fresh, unique identity."""
    a = ldm._derive_run_id({})
    b = ldm._derive_run_id({})
    assert a.startswith("drift-") and b.startswith("drift-")
    assert a != b


def test_derive_run_id_sanitizes_unsafe_chars():
    """A stray character in the id must not break the S3 report prefix."""
    rid = ldm._derive_run_id({"id": "a/b c:d"})
    assert rid == "drift-a-b-c-d"
