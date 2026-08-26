#!/usr/bin/env python3
"""
Deploy monitoring results writer Lambda function.

This Lambda consumes messages from SQS and writes monitoring results
to the monitoring_responses Iceberg table in Athena.
"""

import boto3
import json
import zipfile
import io
import time
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

def deploy_monitoring_writer(region='us-east-1'):
    """Deploy the monitoring writer Lambda."""

    # Load .env if available
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)

    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║  Deploying Monitoring Results Writer Lambda                       ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    print("")

    # Get AWS info
    sts = boto3.client('sts', region_name=region)
    account_id = sts.get_caller_identity()['Account']

    # Configuration (from .env with defaults)
    lambda_name = os.getenv('MONITORING_WRITER_LAMBDA_NAME', 'fraud-monitoring-results-writer')
    role_name = lambda_name + '-role'
    queue_name = os.getenv('MONITORING_SQS_QUEUE_NAME', 'fraud-monitoring-results')
    database = os.getenv('ATHENA_DATABASE', 'fraud_detection')
    table = os.getenv('MONITORING_TABLE_NAME', 'monitoring_responses')
    bucket = os.getenv('DATA_S3_BUCKET', f'fraud-detection-data-lake-skoppar-{account_id}')
    output_location = f's3://{bucket}/athena-query-results/'

    print(f"  Region: {region}")
    print(f"  Account: {account_id}")
    print(f"  Lambda: {lambda_name}")
    print(f"  Queue: {queue_name}")
    print("")

    # Clients
    iam = boto3.client('iam', region_name=region)
    sqs = boto3.client('sqs', region_name=region)
    lambda_client = boto3.client('lambda', region_name=region)

    # Step 1: Create SQS queue (+ dead-letter queue)
    print("[1/5] Creating SQS queue...")

    # Dead-letter queue first — poison messages that fail maxReceiveCount
    # deliveries land here instead of blocking the main queue forever.
    dlq_name = queue_name + '-dlq'
    try:
        dlq_url = sqs.create_queue(
            QueueName=dlq_name,
            Attributes={
                'MessageRetentionPeriod': '1209600',  # 14 days
            }
        )['QueueUrl']
        print(f"  ✓ DLQ created: {dlq_url}")
    except sqs.exceptions.QueueNameExists:
        dlq_url = sqs.get_queue_url(QueueName=dlq_name)['QueueUrl']
        print(f"  ✓ DLQ exists: {dlq_url}")

    dlq_arn = sqs.get_queue_attributes(
        QueueUrl=dlq_url,
        AttributeNames=['QueueArn']
    )['Attributes']['QueueArn']

    # Redrive policy: after maxReceiveCount failed receives a message is moved
    # to the DLQ. Applied on both create and reuse so existing queues get it.
    redrive_policy = json.dumps({
        'deadLetterTargetArn': dlq_arn,
        'maxReceiveCount': 5,
    })

    try:
        queue_url = sqs.create_queue(
            QueueName=queue_name,
            Attributes={
                'VisibilityTimeout': '300',  # 5 minutes
                'MessageRetentionPeriod': '1209600',  # 14 days
                'RedrivePolicy': redrive_policy,
            }
        )['QueueUrl']
        print(f"  ✓ Queue created: {queue_url}")
    except sqs.exceptions.QueueNameExists:
        queue_url = sqs.get_queue_url(QueueName=queue_name)['QueueUrl']
        # Ensure the redrive policy is present on a pre-existing queue.
        sqs.set_queue_attributes(
            QueueUrl=queue_url,
            Attributes={'RedrivePolicy': redrive_policy}
        )
        print(f"  ✓ Queue exists: {queue_url}")

    queue_arn = sqs.get_queue_attributes(
        QueueUrl=queue_url,
        AttributeNames=['QueueArn']
    )['Attributes']['QueueArn']

    # Step 2: Create IAM role
    print("")
    print("[2/5] Creating IAM role...")
    trust_policy = {
        'Version': '2012-10-17',
        'Statement': [{
            'Effect': 'Allow',
            'Principal': {'Service': 'lambda.amazonaws.com'},
            'Action': 'sts:AssumeRole'
        }]
    }

    try:
        iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description='Monitoring results writer Lambda role'
        )
        print("  ✓ Role created")
    except iam.exceptions.EntityAlreadyExistsException:
        print("  ✓ Role exists")

    role_arn = f'arn:aws:iam::{account_id}:role/{role_name}'

    # Attach managed policies
    for policy_arn in [
        'arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole',
        'arn:aws:iam::aws:policy/service-role/AWSLambdaSQSQueueExecutionRole',
        'arn:aws:iam::aws:policy/AmazonAthenaFullAccess',
    ]:
        iam.attach_role_policy(RoleName=role_name, PolicyArn=policy_arn)

    print("  ✓ Policies attached")
    print("  Waiting 10s for role propagation...")
    time.sleep(10)

    # Step 3: Create Lambda deployment package
    print("")
    print("[3/5] Creating Lambda package...")

    # Lambda code
    lambda_code = '''
import json
import boto3
import time
import os
from datetime import datetime

athena = boto3.client('athena')

ATHENA_DATABASE = os.environ['ATHENA_DATABASE']
ATHENA_TABLE = os.environ['ATHENA_TABLE']
ATHENA_OUTPUT = os.environ['ATHENA_OUTPUT']

def lambda_handler(event, context):
    """Process SQS messages and write to Athena.

    Uses partial batch response (ReportBatchItemFailures): only the message
    IDs that failed are returned in batchItemFailures, so successful records
    are deleted from the queue and NOT reprocessed, while a single poison
    record does not block the rest of the batch. After maxReceiveCount
    redeliveries a persistently failing record is moved to the DLQ.
    """
    records = event.get('Records', [])
    print(f"Processing {len(records)} messages")

    # Leave a safety margin before the Lambda timeout: if we're about to run
    # out of time, return the UNATTEMPTED records as batchItemFailures so SQS
    # redelivers them to a fresh invocation, rather than the runtime killing us
    # mid-write (which strands the whole in-flight batch and blocks the queue).
    SAFETY_MS = 5000
    batch_item_failures = []
    for idx, record in enumerate(records):
        message_id = record.get('messageId')
        remaining = context.get_remaining_time_in_millis() if context else None
        if remaining is not None and remaining < SAFETY_MS:
            deferred = records[idx:]
            print(f"Approaching timeout ({remaining} ms left); deferring "
                  f"{len(deferred)} unattempted record(s) for retry")
            for r in deferred:
                batch_item_failures.append({'itemIdentifier': r.get('messageId')})
            break
        try:
            body = json.loads(record['body'])
            write_to_athena(body, context)
        except Exception as e:
            print(f"Error processing message {message_id}: {e}")
            batch_item_failures.append({'itemIdentifier': message_id})

    return {'batchItemFailures': batch_item_failures}

def write_to_athena(data, context=None):
    """Write monitoring result to Athena Iceberg table (idempotently).

    Persistence is an Iceberg MERGE INTO keyed on monitoring_run_id: the row
    is inserted only WHEN NOT MATCHED, so SQS at-least-once redelivery of the
    same run is a no-op instead of a duplicate row. (Athena engine v3 supports
    MERGE INTO for Apache Iceberg tables.) The named column list is a plain
    projection here, so column ORDER is flexible.

    ⚠️ Source of truth: cloudformation/sagemaker-mlflow-setup.yaml AND
    src/setup/create_athena_tables.py monitoring_responses DDL block (search for
    "monitoring_run_id STRING, monitoring_timestamp TIMESTAMP"). Editing one
    without the others will cause silent INSERT failures or column misalignment.
    """
    columns = [
        'monitoring_run_id', 'monitoring_timestamp',
        'endpoint_name', 'model_version', 'model_package_arn',
        'evaluation_snapshot_id', 'training_snapshot_id',
        'data_drift_detected', 'drifted_columns_count', 'drifted_columns_share',
        'features_analyzed', 'data_sample_size', 'model_drift_detected',
        'baseline_roc_auc', 'current_roc_auc',
        'roc_auc_degradation', 'roc_auc_degradation_pct',
        'accuracy', 'precision', 'recall', 'f1_score',
        'model_sample_size', 'per_feature_drift_scores',
        'evidently_report_s3_path', 'mlflow_run_id',
        'alert_sent', 'detection_engine', 'created_at',
        # Problem-type-aware columns (nullable), appended after created_at to
        # match the monitoring_responses DDL. Numeric ones serialize via the
        # int/float branch, the two STRING ones via the string branch; none are
        # timestamps, so timestamp_cols is unchanged.
        'problem_type', 'primary_metric', 'baseline_primary', 'current_primary',
        'mae', 'rmse', 'r2', 'execution_status',
        'data_drift_status', 'model_drift_status',
    ]

    timestamp_cols = {'monitoring_timestamp', 'created_at'}

    values = []
    for col in columns:
        val = data.get(col)
        if val is None:
            values.append('NULL')
        elif isinstance(val, bool):
            values.append('TRUE' if val else 'FALSE')
        elif isinstance(val, (int, float)):
            values.append(str(val))
        elif col in timestamp_cols:
            # Iceberg TIMESTAMP columns need a TIMESTAMP literal, not a quoted string.
            val_str = str(val).replace("'", "''")
            values.append(f"TIMESTAMP '{val_str}'")
        else:
            val_str = str(val).replace("'", "''")
            values.append(f"'{val_str}'")

    col_list = ', '.join(columns)
    # Idempotent + MONOTONIC write: MERGE the single-row source (built from the
    # values above, each aliased to its column name) into the target on
    # monitoring_run_id. Insert when the run is new; when a row already exists,
    # UPDATE it ONLY to reconcile a prior best-effort ERROR row up to a real
    # verdict (ERROR → non-ERROR). This lets a retry that first wrote ERROR then
    # succeeded overwrite the ERROR row, WITHOUT letting a late/duplicate ERROR
    # delivery regress an already-recorded SUCCESS back to ERROR.
    select_cols = ', '.join(f"{v} AS {c}" for v, c in zip(values, columns))
    insert_vals = ', '.join(f"s.{c}" for c in columns)
    update_set = ', '.join(
        f"{c} = s.{c}" for c in columns if c != 'monitoring_run_id'
    )
    query = f"""
    MERGE INTO {ATHENA_DATABASE}.{ATHENA_TABLE} AS t
    USING (SELECT {select_cols}) AS s
    ON t.monitoring_run_id = s.monitoring_run_id
    WHEN MATCHED AND t.execution_status = 'ERROR' AND s.execution_status <> 'ERROR'
        THEN UPDATE SET {update_set}
    WHEN NOT MATCHED THEN INSERT ({col_list}) VALUES ({insert_vals})
    """

    # Execute query
    response = athena.start_query_execution(
        QueryString=query,
        QueryExecutionContext={'Database': ATHENA_DATABASE},
        ResultConfiguration={'OutputLocation': ATHENA_OUTPUT}
    )

    execution_id = response['QueryExecutionId']
    print(f"Athena query started: {execution_id}")

    # Wait for completion, but BOUND the poll by the invocation's remaining
    # time so a slow query can't run us into the hard Lambda timeout (which
    # would kill the process mid-poll and leave the record unacked at an
    # unpredictable point). If we're about to run out, raise so the caller
    # returns this record as a batchItemFailure for a clean SQS retry.
    SAFETY_MS = 3000
    while True:
        status = athena.get_query_execution(QueryExecutionId=execution_id)
        state = status['QueryExecution']['Status']['State']
        if state in ['SUCCEEDED', 'FAILED', 'CANCELLED']:
            break
        if context is not None and context.get_remaining_time_in_millis() < SAFETY_MS:
            raise Exception(
                f"Athena query {execution_id} still {state} with "
                f"< {SAFETY_MS} ms remaining; deferring for SQS retry"
            )
        time.sleep(0.5)

    if state == 'SUCCEEDED':
        print(f"✓ Written to {ATHENA_DATABASE}.{ATHENA_TABLE}")
    else:
        reason = status['QueryExecution']['Status'].get('StateChangeReason', 'Unknown')
        raise Exception(f"Athena query failed: {reason}")
'''

    # Create ZIP
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('lambda_function.py', lambda_code)
    zip_bytes = buf.getvalue()

    print(f"  ✓ Package created: {len(zip_bytes) / 1024:.1f} KB")

    # Step 4: Create/Update Lambda function
    print("")
    print("[4/5] Deploying Lambda function...")

    env_vars = {
        'ATHENA_DATABASE': database,
        'ATHENA_TABLE': table,
        'ATHENA_OUTPUT': output_location
    }

    try:
        lambda_client.get_function(FunctionName=lambda_name)
        # Update existing
        lambda_client.update_function_code(
            FunctionName=lambda_name,
            ZipFile=zip_bytes
        )
        lambda_client.get_waiter('function_updated_v2').wait(FunctionName=lambda_name)
        lambda_client.update_function_configuration(
            FunctionName=lambda_name,
            # Handler MUST be updated here too: an older deploy of this
            # function shipped the code as `index.py`
            # (Handler=index.lambda_handler). We now package it as
            # `lambda_function.py`, so without repinning the handler the
            # updated code loads under a module name the handler no longer
            # points at — every invocation fails with a handler-not-found
            # import error and results are silently never written to Athena.
            Handler='lambda_function.lambda_handler',
            Timeout=60,
            MemorySize=256,
            Environment={'Variables': env_vars}
        )
        print("  ✓ Lambda updated")
    except lambda_client.exceptions.ResourceNotFoundException:
        # Create new
        lambda_client.create_function(
            FunctionName=lambda_name,
            Runtime='python3.11',
            Role=role_arn,
            Handler='lambda_function.lambda_handler',
            Code={'ZipFile': zip_bytes},
            Timeout=60,
            MemorySize=256,
            Environment={'Variables': env_vars},
            Description='Write monitoring results to Athena'
        )
        print("  ✓ Lambda created")

    lambda_arn = f'arn:aws:lambda:{region}:{account_id}:function:{lambda_name}'

    # Step 5: Configure SQS trigger
    print("")
    print("[5/5] Configuring SQS trigger...")

    try:
        lambda_client.create_event_source_mapping(
            EventSourceArn=queue_arn,
            FunctionName=lambda_name,
            BatchSize=10,
            MaximumBatchingWindowInSeconds=5,
            # Honor the handler's batchItemFailures response so only failed
            # records are retried (and eventually sent to the DLQ).
            FunctionResponseTypes=['ReportBatchItemFailures']
        )
        print("  ✓ SQS trigger configured")
    except lambda_client.exceptions.ResourceConflictException:
        print("  ✓ SQS trigger exists")

    print("")
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║  ✅ MONITORING WRITER DEPLOYED                                     ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    print("")
    print(f"  Lambda: {lambda_arn}")
    print(f"  Queue: {queue_url}")
    print(f"  Target: {database}.{table}")
    print("")
    print("Test: Send message to SQS queue with monitoring data")
    print("")

    return queue_url

if __name__ == '__main__':
    region = sys.argv[1] if len(sys.argv) > 1 else 'us-east-1'
    deploy_monitoring_writer(region)
