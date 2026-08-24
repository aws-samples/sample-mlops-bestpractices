"""Unit tests for `src/drift_monitoring/manage_drift_lambda.py`.

Covers:
  - `bootstrap_drift_lambda_role()` is idempotent when the role already
    exists (no create_role call) vs. creates it when missing
    (NoSuchEntityException -> create_role called).
  - `update_drift_thresholds()` builds the correct SNS ARN and env var
    payload, and the `merge_with_existing=True` path preserves an existing
    unrelated env var while overriding the threshold ones.
  - `set_drift_schedule_state()` raises ValueError on an invalid state
    string and calls the right enable/disable method for valid ones.
  - `get_drift_lambda_logs()` handles the ResourceNotFoundException case
    gracefully without raising.
  - `deploy_drift_lambda_container()` raises ValueError on a malformed
    email before touching subprocess, and raises RuntimeError when the
    subprocess call fails.

All boto3 clients and subprocess calls are mocked — no real AWS calls, no
real shell execution.
"""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from src.drift_monitoring import manage_drift_lambda as mdl


class _NoSuchEntityException(Exception):
    pass


def _make_iam_client(role_exists: bool):
    iam_client = MagicMock()
    iam_client.exceptions.NoSuchEntityException = _NoSuchEntityException
    if role_exists:
        iam_client.get_role.return_value = {"Role": {"RoleName": "some-role"}}
    else:
        iam_client.get_role.side_effect = _NoSuchEntityException()
    return iam_client


def _sts_client(account_id: str = "123456789012"):
    sts_client = MagicMock()
    sts_client.get_caller_identity.return_value = {"Account": account_id}
    return sts_client


# ---------------------------------------------------------------------------
# bootstrap_drift_lambda_role()
# ---------------------------------------------------------------------------


def test_bootstrap_drift_lambda_role_idempotent_when_role_exists():
    """get_role succeeds -> create_role must NOT be called."""
    iam_client = _make_iam_client(role_exists=True)
    sts_client = _sts_client()

    result = mdl.bootstrap_drift_lambda_role(
        lambda_exec_role="arn:aws:iam::123456789012:role/my-drift-role",
        iam_client=iam_client,
        sts_client=sts_client,
        propagation_wait_seconds=0,
    )

    iam_client.get_role.assert_called_once_with(RoleName="my-drift-role")
    iam_client.create_role.assert_not_called()
    iam_client.put_role_policy.assert_called_once()
    assert result == {
        "role_name": "my-drift-role",
        "role_arn": "arn:aws:iam::123456789012:role/my-drift-role",
    }


def test_bootstrap_drift_lambda_role_creates_when_missing():
    """NoSuchEntityException on get_role -> create_role must be called."""
    iam_client = _make_iam_client(role_exists=False)
    sts_client = _sts_client()

    with patch("time.sleep") as mock_sleep:
        result = mdl.bootstrap_drift_lambda_role(
            lambda_exec_role="arn:aws:iam::123456789012:role/my-drift-role",
            iam_client=iam_client,
            sts_client=sts_client,
            propagation_wait_seconds=0,
        )

    iam_client.create_role.assert_called_once()
    _, kwargs = iam_client.create_role.call_args
    assert kwargs["RoleName"] == "my-drift-role"
    assert result["role_name"] == "my-drift-role"
    # 5s IAM-propagation sleep after create_role is still expected.
    mock_sleep.assert_any_call(5)


def test_bootstrap_drift_lambda_role_inline_policy_grants_put_metric_data():
    """The inline policy attached by bootstrap must allow
    cloudwatch:PutMetricData — without it the scheduled Lambda's
    publish_cloudwatch_metrics() call is denied and the CloudWatch alarms
    and dashboard stay on "no data". This is the IAM half of the
    CloudWatch auto-publish fix.
    """
    iam_client = _make_iam_client(role_exists=True)
    sts_client = _sts_client()

    mdl.bootstrap_drift_lambda_role(
        lambda_exec_role="arn:aws:iam::123456789012:role/my-drift-role",
        iam_client=iam_client,
        sts_client=sts_client,
        propagation_wait_seconds=0,
    )

    iam_client.put_role_policy.assert_called_once()
    _, kwargs = iam_client.put_role_policy.call_args
    import json as _json

    policy = _json.loads(kwargs["PolicyDocument"])
    granted_actions = set()
    for statement in policy["Statement"]:
        actions = statement["Action"]
        if isinstance(actions, str):
            actions = [actions]
        granted_actions.update(actions)

    assert "cloudwatch:PutMetricData" in granted_actions, (
        "bootstrap inline policy must grant cloudwatch:PutMetricData — "
        "otherwise publish_cloudwatch_metrics() is denied at runtime and "
        "the alarms/dashboard never receive data."
    )


def test_bootstrap_drift_lambda_role_defaults_role_name_when_no_exec_role():
    iam_client = _make_iam_client(role_exists=True)
    sts_client = _sts_client()

    result = mdl.bootstrap_drift_lambda_role(
        lambda_exec_role="",
        iam_client=iam_client,
        sts_client=sts_client,
        propagation_wait_seconds=0,
    )

    assert result["role_name"] == "fraud-detection-drift-monitor-role"


def test_bootstrap_drift_lambda_role_swallows_already_attached_but_warns_on_other_errors():
    iam_client = _make_iam_client(role_exists=True)
    sts_client = _sts_client()

    already_attached_error = ClientError(
        error_response={"Error": {"Code": "EntityAlreadyExists", "Message": "already attached"}},
        operation_name="AttachRolePolicy",
    )
    unexpected_error = ClientError(
        error_response={"Error": {"Code": "AccessDenied", "Message": "nope"}},
        operation_name="AttachRolePolicy",
    )
    iam_client.attach_role_policy.side_effect = [already_attached_error, unexpected_error]

    # Should not raise even though one attach call hits an unexpected error —
    # unexpected errors are logged via logger.warning, not swallowed silently
    # and not propagated (matching the notebook's "attach best-effort" behavior).
    result = mdl.bootstrap_drift_lambda_role(
        lambda_exec_role="arn:aws:iam::123456789012:role/my-drift-role",
        iam_client=iam_client,
        sts_client=sts_client,
        propagation_wait_seconds=0,
    )

    assert result["role_name"] == "my-drift-role"
    assert iam_client.attach_role_policy.call_count == 2


# ---------------------------------------------------------------------------
# update_drift_thresholds()
# ---------------------------------------------------------------------------


def test_update_drift_thresholds_builds_correct_sns_arn_and_payload():
    lambda_client = MagicMock()
    sts_client = _sts_client("999988887777")

    result = mdl.update_drift_thresholds(
        lambda_name="fraud-detection-drift-monitor",
        region="us-west-2",
        data_drift_threshold=0.15,
        model_drift_threshold=0.03,
        lambda_client=lambda_client,
        sts_client=sts_client,
    )

    expected_sns_arn = f"arn:aws:sns:us-west-2:999988887777:{mdl.SNS_TOPIC_NAME}"
    assert result["sns_topic_arn"] == expected_sns_arn
    assert result["data_drift_threshold"] == 0.15
    assert result["model_drift_threshold"] == 0.03

    lambda_client.update_function_configuration.assert_called_once()
    _, kwargs = lambda_client.update_function_configuration.call_args
    assert kwargs["FunctionName"] == "fraud-detection-drift-monitor"
    env_vars = kwargs["Environment"]["Variables"]
    assert env_vars["SNS_TOPIC_ARN"] == expected_sns_arn
    assert env_vars["DATA_DRIFT_THRESHOLD"] == "0.15"
    assert env_vars["MODEL_DRIFT_THRESHOLD"] == "0.03"
    assert env_vars["BASELINE_ROC_AUC"] == "0.92"
    assert "ATHENA_DATABASE" in env_vars
    assert "ATHENA_OUTPUT_S3" in env_vars


def test_update_drift_thresholds_replaces_env_by_default_not_merging():
    """merge_with_existing=False (default) must NOT call
    get_function_configuration, and the payload contains only the 5
    (+ extras) keys — matching notebook parity."""
    lambda_client = MagicMock()
    sts_client = _sts_client()

    mdl.update_drift_thresholds(
        lambda_name="fraud-detection-drift-monitor",
        sns_topic_arn="arn:aws:sns:us-west-2:123456789012:some-topic",
        lambda_client=lambda_client,
        sts_client=sts_client,
    )

    lambda_client.get_function_configuration.assert_not_called()
    _, kwargs = lambda_client.update_function_configuration.call_args
    env_vars = kwargs["Environment"]["Variables"]
    # BYO-dataset knobs (TARGET_COLUMN / PREDICTION_COLUMN /
    # PROBABILITY_COLUMN) get pushed to the Lambda too so its baseline
    # SELECT and model-drift comparison target the correct columns for
    # datasets whose target isn't named `is_fraud`. See Refactor 2 in
    # the schema-agnostic pipeline work — dropping any of these would
    # silently break BYO deployments.
    assert set(env_vars.keys()) == {
        "ATHENA_DATABASE",
        "ATHENA_OUTPUT_S3",
        "SNS_TOPIC_ARN",
        "DATA_DRIFT_THRESHOLD",
        "MODEL_DRIFT_THRESHOLD",
        "BASELINE_ROC_AUC",
        "TARGET_COLUMN",
        "PREDICTION_COLUMN",
        "PROBABILITY_COLUMN",
    }


def test_update_drift_thresholds_merge_preserves_existing_unrelated_var():
    """merge_with_existing=True must preserve ENDPOINT_NAME (an existing,
    unrelated env var) while overriding the threshold-related keys."""
    lambda_client = MagicMock()
    lambda_client.get_function_configuration.return_value = {
        "Environment": {
            "Variables": {
                "ENDPOINT_NAME": "fraud-detector-endpoint",
                "MODEL_PACKAGE_GROUP": "fraud-detection",
                "DATA_DRIFT_THRESHOLD": "0.20",  # will be overridden
            }
        }
    }
    sts_client = _sts_client()

    mdl.update_drift_thresholds(
        lambda_name="fraud-detection-drift-monitor",
        data_drift_threshold=0.10,
        model_drift_threshold=0.02,
        sns_topic_arn="arn:aws:sns:us-west-2:123456789012:some-topic",
        merge_with_existing=True,
        lambda_client=lambda_client,
        sts_client=sts_client,
    )

    lambda_client.get_function_configuration.assert_called_once_with(
        FunctionName="fraud-detection-drift-monitor"
    )
    _, kwargs = lambda_client.update_function_configuration.call_args
    env_vars = kwargs["Environment"]["Variables"]

    # Preserved from existing config.
    assert env_vars["ENDPOINT_NAME"] == "fraud-detector-endpoint"
    assert env_vars["MODEL_PACKAGE_GROUP"] == "fraud-detection"
    # Overridden by this call.
    assert env_vars["DATA_DRIFT_THRESHOLD"] == "0.1"
    assert env_vars["MODEL_DRIFT_THRESHOLD"] == "0.02"


def test_update_drift_thresholds_raises_on_client_error():
    lambda_client = MagicMock()
    lambda_client.update_function_configuration.side_effect = ClientError(
        error_response={"Error": {"Code": "ResourceNotFoundException", "Message": "not found"}},
        operation_name="UpdateFunctionConfiguration",
    )
    sts_client = _sts_client()

    with pytest.raises(ClientError):
        mdl.update_drift_thresholds(
            lambda_name="fraud-detection-drift-monitor",
            sns_topic_arn="arn:aws:sns:us-west-2:123456789012:some-topic",
            lambda_client=lambda_client,
            sts_client=sts_client,
        )


# ---------------------------------------------------------------------------
# set_drift_schedule_state()
# ---------------------------------------------------------------------------


def test_set_drift_schedule_state_raises_on_invalid_state():
    with pytest.raises(ValueError):
        mdl.set_drift_schedule_state("PAUSED", events_client=MagicMock())


def test_set_drift_schedule_state_enables():
    events_client = MagicMock()
    events_client.describe_rule.return_value = {
        "State": "ENABLED",
        "ScheduleExpression": "cron(0 2 * * ? *)",
    }

    result = mdl.set_drift_schedule_state(
        "ENABLED", rule_name="my-rule", events_client=events_client
    )

    events_client.enable_rule.assert_called_once_with(Name="my-rule")
    events_client.disable_rule.assert_not_called()
    assert result == {
        "rule_name": "my-rule",
        "state": "ENABLED",
        "schedule_expression": "cron(0 2 * * ? *)",
    }


def test_set_drift_schedule_state_disables():
    events_client = MagicMock()
    events_client.describe_rule.return_value = {
        "State": "DISABLED",
        "ScheduleExpression": "cron(0 2 * * ? *)",
    }

    result = mdl.set_drift_schedule_state(
        "DISABLED", rule_name="my-rule", events_client=events_client
    )

    events_client.disable_rule.assert_called_once_with(Name="my-rule")
    events_client.enable_rule.assert_not_called()
    assert result["state"] == "DISABLED"


# ---------------------------------------------------------------------------
# get_drift_lambda_logs()
# ---------------------------------------------------------------------------


class _ResourceNotFoundException(Exception):
    pass


def test_get_drift_lambda_logs_handles_resource_not_found_gracefully():
    logs_client = MagicMock()
    logs_client.exceptions.ResourceNotFoundException = _ResourceNotFoundException
    logs_client.describe_log_streams.side_effect = _ResourceNotFoundException()

    result = mdl.get_drift_lambda_logs(
        lambda_name="fraud-detection-drift-monitor", logs_client=logs_client
    )

    assert result["log_group"] == "/aws/lambda/fraud-detection-drift-monitor"
    assert result["log_stream"] is None
    assert result["events"] == []
    assert "error" in result


def test_get_drift_lambda_logs_handles_no_streams_yet():
    logs_client = MagicMock()
    logs_client.exceptions.ResourceNotFoundException = _ResourceNotFoundException
    logs_client.describe_log_streams.return_value = {"logStreams": []}

    result = mdl.get_drift_lambda_logs(
        lambda_name="fraud-detection-drift-monitor", logs_client=logs_client
    )

    assert result["log_stream"] is None
    assert result["events"] == []
    assert "error" not in result
    logs_client.get_log_events.assert_not_called()


def test_get_drift_lambda_logs_returns_events_when_stream_exists():
    logs_client = MagicMock()
    logs_client.exceptions.ResourceNotFoundException = _ResourceNotFoundException
    logs_client.describe_log_streams.return_value = {
        "logStreams": [{"logStreamName": "2024/01/01/[$LATEST]abc123"}]
    }
    logs_client.get_log_events.return_value = {
        "events": [
            {"timestamp": 1704110400000, "message": "Starting drift check\n"},
        ]
    }

    result = mdl.get_drift_lambda_logs(
        lambda_name="fraud-detection-drift-monitor", logs_client=logs_client
    )

    assert result["log_stream"] == "2024/01/01/[$LATEST]abc123"
    assert len(result["events"]) == 1
    assert result["events"][0]["message"] == "Starting drift check"


# ---------------------------------------------------------------------------
# deploy_drift_lambda_container()
# ---------------------------------------------------------------------------


def test_deploy_drift_lambda_container_raises_valueerror_on_bad_email():
    with patch("subprocess.run") as mock_run:
        with pytest.raises(ValueError):
            mdl.deploy_drift_lambda_container(alert_email="not-an-email")
        mock_run.assert_not_called()


def test_deploy_drift_lambda_container_raises_runtimeerror_on_subprocess_failure():
    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.CalledProcessError(returncode=1, cmd=["bash"])
        with pytest.raises(RuntimeError):
            mdl.deploy_drift_lambda_container(alert_email="user@example.com")


def test_deploy_drift_lambda_container_returns_deployed_on_success():
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        result = mdl.deploy_drift_lambda_container(alert_email="user@example.com")

    assert result == {"status": "deployed", "alert_email": "user@example.com"}
    mock_run.assert_called_once()
    args, kwargs = mock_run.call_args
    cmd = args[0]
    assert cmd[0] == "bash"
    assert cmd[2] == "user@example.com"
    assert kwargs["check"] is True
