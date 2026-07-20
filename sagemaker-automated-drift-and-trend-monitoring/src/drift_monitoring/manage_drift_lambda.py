"""
Library of drift-monitor Lambda management operations.

Functions:
- bootstrap_drift_lambda_role  — create/verify the Lambda's IAM execution role
- invoke_drift_lambda          — manually invoke the drift-monitor Lambda
- get_drift_lambda_logs        — fetch recent CloudWatch logs for the Lambda
- update_drift_thresholds      — update drift threshold env vars without redeploying
- set_drift_schedule_state     — enable/disable the EventBridge schedule
- deploy_drift_lambda_container — shell out to scripts/deploy_lambda_container.sh

Ported from `notebooks/3_inference_monitoring.ipynb` (Section 7: "Deploy &
manage the drift-monitor Lambda") so this operational logic is importable
and testable outside a notebook. Consumed by the project's user-facing CLI
in `main.py` (subcommand `monitoring`). This module is intentionally
library-only — no `__main__` block — so there is one entry point
(`main.py`) for command-line use, matching `deploy_endpoint.py`.
"""

import json
import logging
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import boto3
from botocore.exceptions import ClientError

from src.config import config
from src.config.config import (
    AWS_DEFAULT_REGION,
    ATHENA_DATABASE,
    ATHENA_OUTPUT_S3,
    DATA_DRIFT_THRESHOLD,
    DRIFT_LAMBDA_NAME,
    ECR_REPO_NAME,
    EVENTBRIDGE_RULE_NAME,
    LAMBDA_EXEC_ROLE,
    MODEL_DRIFT_THRESHOLD,
    PREDICTION_COLUMN,
    PROBABILITY_COLUMN,
    SNS_TOPIC_NAME,
)
from src.config import schema

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Resolve project root (three levels up from this file: src/drift_monitoring/manage_drift_lambda.py)
# — matches the pattern used by create_athena_tables.py / pipeline.py.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def bootstrap_drift_lambda_role(
    lambda_exec_role: Optional[str] = None,
    region: Optional[str] = None,
    iam_client: Optional[Any] = None,
    sts_client: Optional[Any] = None,
    propagation_wait_seconds: int = 10,
) -> Dict[str, Any]:
    """
    Create (or verify) the IAM execution role for the drift-monitor Lambda.

    Idempotent: get_role -> if missing, create_role (then wait 5s for IAM
    propagation) -> re-attach managed policies -> re-put the inline
    SQS/S3 policy. Safe to call repeatedly.

    Args:
        lambda_exec_role: Role ARN or name to derive the role name from
            (only the last path segment after '/' is used). Defaults to
            config.LAMBDA_EXEC_ROLE; if that's empty, falls back to the
            literal 'fraud-detection-drift-monitor-role'.
        region: AWS region. Defaults to config.AWS_DEFAULT_REGION.
        iam_client: Optional boto3 IAM client (constructed if not provided)
        sts_client: Optional boto3 STS client (constructed if not provided)
        propagation_wait_seconds: Seconds to sleep at the end of the
            function for IAM propagation. Defaults to 10 (matching the
            notebook). Pass 0 in tests to skip the wait.

    Returns:
        Dict with keys: 'role_name', 'role_arn'.
    """
    resolved_region = region or AWS_DEFAULT_REGION
    resolved_lambda_exec_role = (
        lambda_exec_role if lambda_exec_role is not None else LAMBDA_EXEC_ROLE
    )

    if iam_client is None:
        iam_client = boto3.client('iam', region_name=resolved_region)
    if sts_client is None:
        sts_client = boto3.client('sts', region_name=resolved_region)

    role_name = (
        resolved_lambda_exec_role.split('/')[-1]
        if resolved_lambda_exec_role
        else 'fraud-detection-drift-monitor-role'
    )

    trust_policy = {
        'Version': '2012-10-17',
        'Statement': [{
            'Effect': 'Allow',
            'Principal': {'Service': 'lambda.amazonaws.com'},
            'Action': 'sts:AssumeRole',
        }],
    }

    # Create role if it doesn't exist.
    try:
        iam_client.get_role(RoleName=role_name)
        logger.info(f"✓ Role {role_name} already exists")
    except iam_client.exceptions.NoSuchEntityException:
        iam_client.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description='Lambda role for fraud detection drift monitoring',
        )
        logger.info(f"✓ Created role {role_name}")
        time.sleep(5)  # Wait for IAM propagation

    # Attach managed policies. Only the "already attached" case is
    # swallowed — any other failure is surfaced via logger.warning so a
    # real permissions problem doesn't silently disappear.
    managed_policies = [
        'arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole',
        'arn:aws:iam::aws:policy/AmazonAthenaFullAccess',
    ]
    for policy_arn in managed_policies:
        try:
            iam_client.attach_role_policy(RoleName=role_name, PolicyArn=policy_arn)
            logger.info(f"✓ Attached {policy_arn.split('/')[-1]}")
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code == 'EntityAlreadyExists':
                logger.info(f"  {policy_arn.split('/')[-1]} already attached")
            else:
                logger.warning(f"Failed to attach {policy_arn}: {e}")

    # Inline policy for SQS + S3 access.
    inline_policy = {
        'Version': '2012-10-17',
        'Statement': [
            {
                'Effect': 'Allow',
                'Action': ['sqs:SendMessage', 'sqs:GetQueueAttributes'],
                'Resource': '*',
            },
            {
                'Effect': 'Allow',
                'Action': [
                    's3:GetObject',
                    's3:PutObject',
                    's3:ListBucket',
                    's3:GetBucketLocation',
                ],
                'Resource': '*',
            },
        ],
    }
    iam_client.put_role_policy(
        RoleName=role_name,
        PolicyName='SQS-S3-Access',
        PolicyDocument=json.dumps(inline_policy),
    )
    logger.info("✓ Attached inline SQS + S3 policy")

    account_id = sts_client.get_caller_identity()['Account']
    role_arn = f'arn:aws:iam::{account_id}:role/{role_name}'

    logger.info(f"✓ Role ready: {role_name}")
    if propagation_wait_seconds:
        logger.info(f"  Waiting {propagation_wait_seconds}s for IAM propagation...")
        time.sleep(propagation_wait_seconds)

    return {'role_name': role_name, 'role_arn': role_arn}


def repoint_drift_lambda_to_latest(
    lambda_name: Optional[str] = None,
    repository: Optional[str] = None,
    region: Optional[str] = None,
    lambda_client: Optional[Any] = None,
    ecr_client: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Point the drift-monitor Lambda at the current ``:latest`` ECR image.

    A container Lambda pins whatever digest ``:latest`` resolved to at its
    last ``update_function_code`` call and never re-resolves it on invocation.
    So after a new image is pushed to ``:latest`` (by CI, a rebuild, or a
    teammate) the function keeps running the OLD digest until something
    repoints it. This resolves ``:latest`` to its immutable digest, updates
    the function BY DIGEST (explicit + verifiable, not a race against the
    mutable tag), waits for the update, and confirms the function landed on
    it. Use this to pick up a freshly pushed image without a Docker rebuild.

    Args:
        lambda_name: Name of the drift-monitor Lambda. Defaults to config.DRIFT_LAMBDA_NAME.
        repository: ECR repository name. Defaults to config.ECR_REPO_NAME.
        region: AWS region. Defaults to config.AWS_DEFAULT_REGION.
        lambda_client: Optional boto3 Lambda client (constructed if not provided).
        ecr_client: Optional boto3 ECR client (constructed if not provided).

    Returns:
        Dict with keys: 'lambda_name', 'repository', 'previous_image_uri',
        'deployed_image_uri', 'digest', 'changed' (bool — whether the
        function's resolved image actually moved).

    Raises:
        RuntimeError: If ``:latest`` has no image in ECR, or the function did
            not land on the intended digest after the update.
    """
    resolved_region = region or AWS_DEFAULT_REGION
    resolved_lambda_name = lambda_name or DRIFT_LAMBDA_NAME
    resolved_repository = repository or ECR_REPO_NAME

    if lambda_client is None:
        lambda_client = boto3.client('lambda', region_name=resolved_region)
    if ecr_client is None:
        ecr_client = boto3.client('ecr', region_name=resolved_region)

    # Resolve :latest -> immutable digest.
    try:
        images = ecr_client.describe_images(
            repositoryName=resolved_repository,
            imageIds=[{'imageTag': 'latest'}],
        )['imageDetails']
    except ClientError as e:
        raise RuntimeError(
            f"Could not resolve {resolved_repository}:latest in ECR "
            f"({resolved_region}): {e}. Build/push the image first."
        ) from e
    if not images:
        raise RuntimeError(
            f"No image tagged ':latest' in ECR repo '{resolved_repository}'. "
            "Build/push the drift-monitor image before repointing."
        )
    digest = images[0]['imageDigest']

    account_id = boto3.client('sts', region_name=resolved_region).get_caller_identity()['Account']
    image_uri = f"{account_id}.dkr.ecr.{resolved_region}.amazonaws.com/{resolved_repository}@{digest}"

    previous = lambda_client.get_function(
        FunctionName=resolved_lambda_name,
    )['Code'].get('ResolvedImageUri', '')

    logger.info(f"Repointing {resolved_lambda_name} to {image_uri}")
    lambda_client.update_function_code(
        FunctionName=resolved_lambda_name,
        ImageUri=image_uri,
    )
    lambda_client.get_waiter('function_updated_v2').wait(FunctionName=resolved_lambda_name)

    deployed = lambda_client.get_function(
        FunctionName=resolved_lambda_name,
    )['Code'].get('ResolvedImageUri', '')
    if deployed.split('@')[-1] != digest:
        raise RuntimeError(
            f"{resolved_lambda_name} resolved to {deployed.split('@')[-1]} "
            f"but expected {digest} — the function did not pick up the image."
        )
    logger.info(f"✓ {resolved_lambda_name} is now running {digest}")

    return {
        'lambda_name': resolved_lambda_name,
        'repository': resolved_repository,
        'previous_image_uri': previous,
        'deployed_image_uri': deployed,
        'digest': digest,
        'changed': previous != deployed,
    }


def invoke_drift_lambda(
    lambda_name: Optional[str] = None,
    region: Optional[str] = None,
    lambda_client: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Manually invoke the drift-monitor Lambda synchronously.

    Args:
        lambda_name: Name of the drift-monitor Lambda. Defaults to config.DRIFT_LAMBDA_NAME.
        region: AWS region. Defaults to config.AWS_DEFAULT_REGION.
        lambda_client: Optional boto3 Lambda client (constructed if not provided)

    Returns:
        Dict with keys: 'status_code' (the HTTP-style Lambda invoke status
        code), 'payload' (the parsed JSON response body from the function).
    """
    resolved_region = region or AWS_DEFAULT_REGION
    resolved_lambda_name = lambda_name or DRIFT_LAMBDA_NAME

    if lambda_client is None:
        lambda_client = boto3.client('lambda', region_name=resolved_region)

    response = lambda_client.invoke(
        FunctionName=resolved_lambda_name,
        InvocationType='RequestResponse',
    )
    payload = json.loads(response['Payload'].read())

    return {'status_code': response['StatusCode'], 'payload': payload}


def get_drift_lambda_logs(
    lambda_name: Optional[str] = None,
    region: Optional[str] = None,
    limit: int = 50,
    logs_client: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Fetch the most recent CloudWatch log events for the drift-monitor Lambda.

    Args:
        lambda_name: Name of the drift-monitor Lambda. Defaults to config.DRIFT_LAMBDA_NAME.
        region: AWS region. Defaults to config.AWS_DEFAULT_REGION.
        limit: Max number of log events to return from the latest log stream.
        logs_client: Optional boto3 CloudWatch Logs client (constructed if not provided)

    Returns:
        Dict with keys: 'log_group', 'log_stream', 'events'. 'events' is a
        list of {'timestamp': iso_string, 'message': msg}. If the log group
        doesn't exist yet (Lambda never deployed/invoked), returns
        'log_stream': None, 'events': [], and an 'error' key explaining why,
        rather than raising. If the log group exists but has no streams yet,
        returns 'log_stream': None with an empty 'events' list (no 'error' key).
    """
    resolved_region = region or AWS_DEFAULT_REGION
    resolved_lambda_name = lambda_name or DRIFT_LAMBDA_NAME
    log_group = f'/aws/lambda/{resolved_lambda_name}'

    if logs_client is None:
        logs_client = boto3.client('logs', region_name=resolved_region)

    try:
        streams = logs_client.describe_log_streams(
            logGroupName=log_group,
            orderBy='LastEventTime',
            descending=True,
            limit=1,
        )
    except logs_client.exceptions.ResourceNotFoundException:
        logger.info(f"Log group {log_group} not found")
        return {
            'log_group': log_group,
            'log_stream': None,
            'events': [],
            'error': 'Log group not found — deploy and invoke the Lambda first.',
        }

    if not streams['logStreams']:
        logger.info(f"No log streams found in {log_group} yet")
        return {'log_group': log_group, 'log_stream': None, 'events': []}

    stream_name = streams['logStreams'][0]['logStreamName']
    events_response = logs_client.get_log_events(
        logGroupName=log_group,
        logStreamName=stream_name,
        limit=limit,
    )

    events = [
        {
            'timestamp': datetime.fromtimestamp(event['timestamp'] / 1000).isoformat(),
            'message': event['message'].rstrip(),
        }
        for event in events_response['events']
    ]

    return {'log_group': log_group, 'log_stream': stream_name, 'events': events}


def update_drift_thresholds(
    lambda_name: Optional[str] = None,
    region: Optional[str] = None,
    data_drift_threshold: Optional[float] = None,
    model_drift_threshold: Optional[float] = None,
    sns_topic_arn: Optional[str] = None,
    extra_env_vars: Optional[Dict[str, str]] = None,
    merge_with_existing: bool = False,
    lambda_client: Optional[Any] = None,
    sts_client: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Update the drift-monitor Lambda's threshold env vars without redeploying.

    ⚠️  IMPORTANT — REPLACE, NOT MERGE, BY DEFAULT: `update_function_configuration`
    REPLACES the Lambda's entire environment variable set; it does not merge.
    With `merge_with_existing=False` (the default, matching the notebook this
    was ported from), this call sets ONLY the env vars named below and DROPS
    every other env var the Lambda had — including things set at container-deploy
    time such as MODEL_PACKAGE_GROUP, ENDPOINT_NAME, MODEL_VERSION,
    MLFLOW_TRACKING_URI, MONITORING_SQS_QUEUE_URL, KS_PVALUE_THRESHOLD,
    DATA_DRIFT_LOOKBACK_DAYS, MODEL_DRIFT_LOOKBACK_DAYS, etc. Calling this
    against a Lambda that depends on those other vars will break it until
    it's redeployed via `deploy_drift_lambda_container()`.

    Pass `merge_with_existing=True` for a safer update that first reads the
    Lambda's current env vars via `get_function_configuration` and merges
    them with the new ones — the new values below always win over anything
    with the same key, but every other pre-existing var is preserved.

    Args:
        lambda_name: Name of the drift-monitor Lambda. Defaults to config.DRIFT_LAMBDA_NAME.
        region: AWS region. Defaults to config.AWS_DEFAULT_REGION.
        data_drift_threshold: New data-drift threshold. Defaults to config.DATA_DRIFT_THRESHOLD.
        model_drift_threshold: New model-drift threshold. Defaults to config.MODEL_DRIFT_THRESHOLD.
        sns_topic_arn: SNS topic ARN for alerts. If not passed, resolved as
            f"arn:aws:sns:{region}:{account_id}:{config.SNS_TOPIC_NAME}".
        extra_env_vars: Additional env vars to set/override (merged in last,
            so these take precedence over the 5 defaults below).
        merge_with_existing: If True, fetch the Lambda's existing env vars
            first and merge (existing vars are preserved unless explicitly
            overridden by this call). Defaults to False, matching the
            notebook's original replace-the-whole-set behavior.
        lambda_client: Optional boto3 Lambda client (constructed if not provided)
        sts_client: Optional boto3 STS client (constructed if not provided)

    Returns:
        Dict with keys: 'lambda_name', 'data_drift_threshold',
        'model_drift_threshold', 'sns_topic_arn'.

    Raises:
        ClientError: Re-raised with additional context logged (unlike the
            notebook, which swallowed this with a bare print) so a CLI
            caller gets a real, non-zero exit code.
    """
    resolved_region = region or AWS_DEFAULT_REGION
    resolved_lambda_name = lambda_name or DRIFT_LAMBDA_NAME
    resolved_data_drift_threshold = (
        data_drift_threshold if data_drift_threshold is not None else DATA_DRIFT_THRESHOLD
    )
    resolved_model_drift_threshold = (
        model_drift_threshold if model_drift_threshold is not None else MODEL_DRIFT_THRESHOLD
    )

    if lambda_client is None:
        lambda_client = boto3.client('lambda', region_name=resolved_region)
    if sts_client is None:
        sts_client = boto3.client('sts', region_name=resolved_region)

    resolved_sns_topic_arn = sns_topic_arn
    if not resolved_sns_topic_arn:
        account_id = sts_client.get_caller_identity()['Account']
        resolved_sns_topic_arn = f"arn:aws:sns:{resolved_region}:{account_id}:{SNS_TOPIC_NAME}"

    new_vars = {
        'ATHENA_DATABASE': ATHENA_DATABASE,
        'ATHENA_OUTPUT_S3': ATHENA_OUTPUT_S3,
        'SNS_TOPIC_ARN': resolved_sns_topic_arn,
        'DATA_DRIFT_THRESHOLD': str(resolved_data_drift_threshold),
        'MODEL_DRIFT_THRESHOLD': str(resolved_model_drift_threshold),
        'BASELINE_ROC_AUC': '0.92',
        # BYO-dataset knobs — let the Lambda's baseline SELECT and
        # model-drift comparison use the correct target/prediction column
        # names without hardcoded `is_fraud` / `probability_fraud` in its
        # source. Fallbacks in lambda_drift_monitor.py match these defaults.
        'TARGET_COLUMN': schema.target_column(),
        'PREDICTION_COLUMN': PREDICTION_COLUMN,
        'PROBABILITY_COLUMN': PROBABILITY_COLUMN,
        **(extra_env_vars or {}),
    }

    env_vars = new_vars
    if merge_with_existing:
        existing_config = lambda_client.get_function_configuration(
            FunctionName=resolved_lambda_name
        )
        existing_vars = existing_config.get('Environment', {}).get('Variables', {})
        env_vars = {**existing_vars, **new_vars}

    logger.info(f"Updating drift thresholds for Lambda: {resolved_lambda_name}")
    logger.info(f"  SNS Topic ARN: {resolved_sns_topic_arn}")

    try:
        lambda_client.update_function_configuration(
            FunctionName=resolved_lambda_name,
            Environment={'Variables': env_vars},
        )
    except ClientError as e:
        logger.error(f"Failed to update configuration for {resolved_lambda_name}: {e}")
        raise ClientError(
            error_response=e.response,
            operation_name=e.operation_name,
        ) from e

    logger.info(
        f"✓ Configuration updated: DATA_DRIFT_THRESHOLD={resolved_data_drift_threshold}, "
        f"MODEL_DRIFT_THRESHOLD={resolved_model_drift_threshold}"
    )

    return {
        'lambda_name': resolved_lambda_name,
        'data_drift_threshold': resolved_data_drift_threshold,
        'model_drift_threshold': resolved_model_drift_threshold,
        'sns_topic_arn': resolved_sns_topic_arn,
    }


def set_drift_schedule_state(
    desired_state: str,
    rule_name: Optional[str] = None,
    region: Optional[str] = None,
    events_client: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Enable or disable the EventBridge schedule that triggers the drift-monitor Lambda.

    Args:
        desired_state: Must be 'ENABLED' or 'DISABLED'.
        rule_name: Name of the EventBridge rule. Defaults to config.EVENTBRIDGE_RULE_NAME.
        region: AWS region. Defaults to config.AWS_DEFAULT_REGION.
        events_client: Optional boto3 EventBridge client (constructed if not provided)

    Returns:
        Dict with keys: 'rule_name', 'state', 'schedule_expression'.

    Raises:
        ValueError: If `desired_state` isn't 'ENABLED' or 'DISABLED'.
    """
    if desired_state not in ('ENABLED', 'DISABLED'):
        raise ValueError(
            f"desired_state must be 'ENABLED' or 'DISABLED', got {desired_state!r}"
        )

    resolved_region = region or AWS_DEFAULT_REGION
    resolved_rule_name = rule_name or EVENTBRIDGE_RULE_NAME

    if events_client is None:
        events_client = boto3.client('events', region_name=resolved_region)

    if desired_state == 'DISABLED':
        events_client.disable_rule(Name=resolved_rule_name)
        logger.info("⏸️ Drift monitoring DISABLED")
    else:
        events_client.enable_rule(Name=resolved_rule_name)
        logger.info("▶️ Drift monitoring ENABLED")

    rule = events_client.describe_rule(Name=resolved_rule_name)
    logger.info(f"Current status: {rule['State']}")
    logger.info(f"Schedule: {rule['ScheduleExpression']}")

    return {
        'rule_name': resolved_rule_name,
        'state': rule['State'],
        'schedule_expression': rule['ScheduleExpression'],
    }


def deploy_drift_lambda_container(
    alert_email: str,
    data_drift_threshold: Optional[float] = None,
    model_drift_threshold: Optional[float] = None,
    region: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Deploy the drift-monitor Lambda as a container image.

    Shells out to `scripts/deploy_lambda_container.sh` rather than
    reimplementing its SNS/IAM/ECR/EventBridge/Lambda orchestration in
    Python — that script is the single source of truth for the deploy
    sequence. The subprocess's stdout/stderr are streamed live (not
    captured) since the script takes ~5-8 minutes and prints progress a
    CLI user needs to see in real time.

    Args:
        alert_email: Email address to subscribe to drift alerts. Must
            contain '@'.
        data_drift_threshold: Data-drift threshold to pass to the script.
            Defaults to config.DATA_DRIFT_THRESHOLD.
        model_drift_threshold: Model-drift threshold to pass to the script.
            Defaults to config.MODEL_DRIFT_THRESHOLD.
        region: Unused directly (the script resolves region itself via
            config/_read_config.sh); accepted for API symmetry with the
            other functions in this module.

    Returns:
        Dict with keys: 'status' ('deployed'), 'alert_email'.

    Raises:
        ValueError: If `alert_email` doesn't contain '@'.
        RuntimeError: If the deploy script exits non-zero.
    """
    if '@' not in alert_email:
        raise ValueError(f"alert_email={alert_email!r} is not a valid email address.")

    resolved_data_drift_threshold = (
        data_drift_threshold if data_drift_threshold is not None else config.DATA_DRIFT_THRESHOLD
    )
    resolved_model_drift_threshold = (
        model_drift_threshold if model_drift_threshold is not None else config.MODEL_DRIFT_THRESHOLD
    )

    script_path = _PROJECT_ROOT / 'scripts' / 'deploy_lambda_container.sh'

    logger.info(f"Deploying drift-monitor Lambda container via {script_path}")
    logger.info(f"  Alert email: {alert_email}")
    logger.info(f"  Data drift threshold: {resolved_data_drift_threshold}")
    logger.info(f"  Model drift threshold: {resolved_model_drift_threshold}")

    try:
        subprocess.run(
            [
                'bash',
                str(script_path),
                alert_email,
                str(resolved_data_drift_threshold),
                str(resolved_model_drift_threshold),
            ],
            check=True,
            cwd=str(_PROJECT_ROOT),
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"deploy_lambda_container.sh exited with code {e.returncode}. "
            "See the streamed output above for details."
        ) from e

    logger.info("✓ Drift-monitor Lambda container deployed")
    return {'status': 'deployed', 'alert_email': alert_email}
