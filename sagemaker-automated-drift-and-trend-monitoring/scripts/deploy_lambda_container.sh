#!/bin/bash
set -e

#############################################################################
# Deploy Drift Monitoring Lambda as Container Image
#############################################################################
# This script:
# 1. Creates SNS topic for alerts
# 2. Creates IAM role for Lambda
# 3. Builds and deploys Lambda as container image to ECR
# 4. Creates EventBridge schedule rule
# 5. Tests the Lambda function
#############################################################################

# All defaults are read from src/config/config.py via get_config — no
# shell-side `${VAR:-literal}` fallbacks anywhere. config.py is the single
# source of truth, so renaming any resource happens in exactly one place.
source "$(dirname "${BASH_SOURCE[0]}")/_read_config.sh"
REGION="$(get_config AWS_DEFAULT_REGION)"

ALERT_EMAIL="${1:-}"  # Pass email as first argument
DATA_DRIFT_THRESHOLD="${2:-0.2}"
MODEL_DRIFT_THRESHOLD="${3:-0.05}"

# Validate email
if [ -z "$ALERT_EMAIL" ]; then
    echo "Usage: $0 <email> [data_drift_threshold] [model_drift_threshold]"
    echo "Example: $0 your-email@example.com 0.2 0.05"
    exit 1
fi

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║  Deploying Drift Monitoring Infrastructure                         ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo "  Region: $REGION"
echo "  Alert Email: $ALERT_EMAIL"
echo "  Data Drift Threshold: $DATA_DRIFT_THRESHOLD"
echo "  Model Drift Threshold: $MODEL_DRIFT_THRESHOLD"
echo ""

# Load configuration from .env if available
if [ -f ../.env ]; then
    set -a
    source ../.env
    set +a
fi

# Get AWS account info
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# Every name below is sourced from src/config/config.py via get_config().
# Renaming any of these (Lambda, ECR repo, SNS topic, etc.) is a one-line
# edit in config.py — no shell-side fallbacks to keep in sync.
ATHENA_DB="$(get_config ATHENA_DATABASE)"
PROJECT_NAME_FOR_BUCKET="$(get_config PROJECT_NAME)"
ATHENA_OUTPUT="s3://${DATA_S3_BUCKET:-${PROJECT_NAME_FOR_BUCKET}-data-${ACCOUNT_ID}}/athena-query-results/"
SNS_TOPIC="$(get_config SNS_TOPIC_NAME)"
LAMBDA_NAME="$(get_config DRIFT_LAMBDA_NAME)"
# LAMBDA_EXEC_ROLE comes from CFN .env injection; if missing, derive the
# conventional role name from the Lambda name.
ROLE_NAME=$(echo "${LAMBDA_EXEC_ROLE:-${LAMBDA_NAME}-role}" | awk -F'/' '{print $NF}')
RULE_NAME="$(get_config EVENTBRIDGE_RULE_NAME)"
SCHEDULE="$(get_config DRIFT_MONITOR_SCHEDULE)"
REPO_NAME="$(get_config ECR_REPO_NAME)"
MODEL_PACKAGE_GROUP_NAME="$(get_config MLFLOW_MODEL_NAME)"

# Single source of truth for the CloudWatch namespace. Defaults to the CFN
# MetricNamespace default (FraudDetection/DriftMonitoring). This ONE value is
# used both in the IAM PutMetricData condition below AND passed to the Lambda
# as CLOUDWATCH_NAMESPACE, so a custom namespace can't drift between the two
# (which would otherwise cause AccessDenied or metrics-in-wrong-namespace).
CLOUDWATCH_NAMESPACE="${CLOUDWATCH_NAMESPACE:-FraudDetection/DriftMonitoring}"

# The monitoring results queue name is fixed; define it once so the IAM scope
# (Step 2) and the env-var wiring (Step 5) resolve the same queue.
SQS_QUEUE_NAME="fraud-monitoring-results"

# Step 1: SNS Topic
echo "[1/7] Creating SNS topic..."
TOPIC_ARN=$(aws sns create-topic --name $SNS_TOPIC --region $REGION --query 'TopicArn' --output text 2>/dev/null || \
            aws sns list-topics --region $REGION --query "Topics[?contains(TopicArn, '$SNS_TOPIC')].TopicArn" --output text)
echo "  ✓ Topic: $TOPIC_ARN"

if [ -n "$ALERT_EMAIL" ]; then
    aws sns subscribe --topic-arn $TOPIC_ARN --protocol email --notification-endpoint $ALERT_EMAIL --region $REGION 2>/dev/null || true
    echo "  ✓ Check $ALERT_EMAIL for confirmation link"
fi

# Step 2: IAM Role
echo ""
echo "[2/7] Creating IAM role..."
TRUST_POLICY=$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"Service": "lambda.amazonaws.com"},
    "Action": "sts:AssumeRole"
  }]
}
EOF
)

aws iam create-role \
    --role-name $ROLE_NAME \
    --assume-role-policy-document "$TRUST_POLICY" \
    --description "Drift monitoring Lambda role" \
    --region $REGION 2>/dev/null || echo "  (role already exists)"

ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${ROLE_NAME}"

# Attach managed policies
for policy in \
    "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole" \
    "arn:aws:iam::aws:policy/AmazonAthenaFullAccess" \
    "arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess"; do
    aws iam attach-role-policy --role-name $ROLE_NAME --policy-arn $policy --region $REGION 2>/dev/null || true
done

# Add SNS publish permission
SNS_POLICY=$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": ["sns:Publish"],
    "Resource": "$TOPIC_ARN"
  }]
}
EOF
)

aws iam put-role-policy \
    --role-name $ROLE_NAME \
    --policy-name SNSPublishPolicy \
    --policy-document "$SNS_POLICY" \
    --region $REGION 2>/dev/null || true

# Add CloudWatch PutMetricData permission. The scheduled Lambda calls
# cloudwatch.put_metric_data on every run (publish_cloudwatch_metrics in
# lambda_drift_monitor.py); without this it fails silently with AccessDenied.
# PutMetricData does not support resource-level ARNs, so Resource must be "*";
# we scope it with a condition on the CLOUDWATCH_NAMESPACE the Lambda actually
# publishes to (same $CLOUDWATCH_NAMESPACE passed to the function in Step 5).
CLOUDWATCH_POLICY=$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": ["cloudwatch:PutMetricData"],
    "Resource": "*",
    "Condition": {
      "StringEquals": {
        "cloudwatch:namespace": "$CLOUDWATCH_NAMESPACE"
      }
    }
  }]
}
EOF
)

aws iam put-role-policy \
    --role-name $ROLE_NAME \
    --policy-name CloudWatchPutMetricPolicy \
    --policy-document "$CLOUDWATCH_POLICY" \
    --region $REGION 2>/dev/null || true

# Add SQS send permission. The Lambda calls sqs.send_message against the
# monitoring queue (MONITORING_SQS_QUEUE_URL). No managed policy grants SQS,
# so add it inline. Scope it to the specific results queue ARN rather than "*":
# resolve the ARN from the (fixed) queue name, falling back to the conventional
# arn:aws:sqs:<region>:<account>:<name> pattern if the queue isn't created yet.
SQS_QUEUE_ARN=""
SQS_QUEUE_URL_FOR_ARN=$(aws sqs get-queue-url --queue-name "$SQS_QUEUE_NAME" --region $REGION --query 'QueueUrl' --output text 2>/dev/null || echo "")
if [ -n "$SQS_QUEUE_URL_FOR_ARN" ]; then
    SQS_QUEUE_ARN=$(aws sqs get-queue-attributes --queue-url "$SQS_QUEUE_URL_FOR_ARN" --attribute-names QueueArn --region $REGION --query 'Attributes.QueueArn' --output text 2>/dev/null || echo "")
fi
[ -z "$SQS_QUEUE_ARN" ] && SQS_QUEUE_ARN="arn:aws:sqs:${REGION}:${ACCOUNT_ID}:${SQS_QUEUE_NAME}"

# On-failure destination for the function's own ASYNCHRONOUS retries (see the
# EventInvokeConfig block after the function is created). Lambda writes the
# failed event to this queue using THIS execution role, so the role needs
# sqs:SendMessage on it. Conventional name/ARN (queue is created later).
ASYNC_DLQ_NAME="${LAMBDA_NAME}-async-dlq"
ASYNC_DLQ_ARN="arn:aws:sqs:${REGION}:${ACCOUNT_ID}:${ASYNC_DLQ_NAME}"

SQS_POLICY=$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": ["sqs:SendMessage"],
    "Resource": ["$SQS_QUEUE_ARN", "$ASYNC_DLQ_ARN"]
  }]
}
EOF
)

aws iam put-role-policy \
    --role-name $ROLE_NAME \
    --policy-name SQSSendPolicy \
    --policy-document "$SQS_POLICY" \
    --region $REGION 2>/dev/null || true

# Read the registered baseline.json from the Model Registry. The drift
# Lambda looks up the latest Approved ModelPackage to recover the
# baseline ROC-AUC and evaluation table — see load_baseline_from_registry
# in lambda_drift_monitor.py.
REGISTRY_POLICY=$(cat <<'EOF'
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": [
      "sagemaker:ListModelPackages",
      "sagemaker:DescribeModelPackage",
      "sagemaker:DescribeEndpoint",
      "sagemaker:DescribeEndpointConfig",
      "sagemaker:DescribeModel"
    ],
    "Resource": "*"
  }]
}
EOF
)

aws iam put-role-policy \
    --role-name $ROLE_NAME \
    --policy-name ModelRegistryReadPolicy \
    --policy-document "$REGISTRY_POLICY" \
    --region $REGION 2>/dev/null || true

echo "  ✓ Role: $ROLE_ARN"
echo "  ✓ Waiting 10s for role propagation..."
sleep 10

# Step 3: Create ECR repository if needed
echo ""
echo "[3/7] Setting up ECR repository..."
aws ecr describe-repositories --repository-names $REPO_NAME --region $REGION >/dev/null 2>&1 || \
aws ecr create-repository \
    --repository-name $REPO_NAME \
    --region $REGION \
    --image-scanning-configuration scanOnPush=false > /dev/null

# Attach repository policy so Lambda (this account) can pull the image. Without
# this, `aws lambda create-function --code ImageUri=...` fails with
# "Lambda does not have permission to access the ECR image" even when the
# Lambda execution role has ECR perms — container Lambdas authenticate via
# the ECR repo policy, not the execution role.
ECR_POLICY_FILE=$(mktemp -t ecr-policy.XXXXXX.json)
trap 'rm -f "$ENV_FILE" "$ECR_POLICY_FILE"' EXIT
cat > "$ECR_POLICY_FILE" <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "LambdaECRImageRetrievalPolicy",
      "Effect": "Allow",
      "Principal": {"Service": "lambda.amazonaws.com"},
      "Action": [
        "ecr:BatchGetImage",
        "ecr:GetDownloadUrlForLayer"
      ],
      "Condition": {
        "StringLike": {
          "aws:sourceArn": "arn:aws:lambda:${REGION}:${ACCOUNT_ID}:function:*"
        }
      }
    }
  ]
}
EOF
aws ecr set-repository-policy \
    --repository-name "$REPO_NAME" \
    --policy-text "file://$ECR_POLICY_FILE" \
    --region "$REGION" > /dev/null
echo "  ✓ Repository: $REPO_NAME (Lambda pull policy attached)"

# Step 4: Build and push Docker image
#
# Two paths, auto-detected:
#   - Local Docker daemon reachable → use `docker build` directly (fast, ~2 min)
#   - No local daemon (SageMaker Studio JupyterLab) → fall back to AWS
#     CodeBuild via src/setup/codebuild_image.py (~5-8 min)
#
# We use our own CodeBuild wrapper (codebuild_image.py) rather than the
# third-party sm-docker package because sm-docker calls v2-only SageMaker
# SDK APIs (sagemaker.get_execution_role, sagemaker.session) that were
# removed in v3 — and our project requires v3.
# Sanity-check the Lambda source before the (slow) container build. If the
# categorical-encoding fix for customer_gender is missing from the file, the
# built image will silently ship the pre-fix code and every drift run will
# fail with "empty column 'customer_gender'". Abort before CodeBuild.
echo ""
echo "[3.5/7] Verifying lambda_drift_monitor.py contents..."
TARGET_FILE="src/drift_monitoring/lambda_drift_monitor.py"
echo "  File: $TARGET_FILE"
echo "  Size: $(wc -c < "$TARGET_FILE") bytes, $(wc -l < "$TARGET_FILE") lines"
if grep -q "categorical_cols" "$TARGET_FILE"; then
    echo "  ✓ Found 'categorical_cols' — gender encoding fix is in place"
else
    echo "  ✗ 'categorical_cols' NOT found — fix was not saved. Aborting."
    exit 1
fi

echo ""
echo "[4/7] Building Docker image..."

IMAGE_URI="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO_NAME:latest"

if docker info >/dev/null 2>&1; then
    echo "  ✓ Local Docker daemon detected — using direct build"
    aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com
    docker build --platform linux/amd64 -t $REPO_NAME:latest -f src/drift_monitoring/Dockerfile.lambda . -q
    docker tag $REPO_NAME:latest $IMAGE_URI
    echo "  ✓ Image built"
    echo "  Pushing to ECR..."
    docker push $IMAGE_URI | tail -5
    echo "  ✓ Image pushed: $IMAGE_URI"
else
    echo "  ℹ No local Docker daemon — using AWS CodeBuild to build the image."
    echo "  This takes ~5-8 minutes; the build runs in AWS, output streams below."
    # Resolve the SageMaker execution role ARN (CodeBuild project needs one
    # role to assume — it needs ECR push perms + CloudWatch log perms).
    # Prefer SAGEMAKER_EXEC_ROLE from .env (set by CFN); fall back to the
    # current caller's role.
    if [ -n "${SAGEMAKER_EXEC_ROLE:-}" ]; then
        BUILD_ROLE_ARN="$SAGEMAKER_EXEC_ROLE"
    else
        CALLER_ARN=$(aws sts get-caller-identity --query Arn --output text)
        ROLE_NAME=$(echo "$CALLER_ARN" | sed -E 's|arn:aws:sts::[0-9]+:assumed-role/([^/]+)/.*|\1|')
        BUILD_ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${ROLE_NAME}"
    fi
    python3 -m src.setup.codebuild_image \
        --source-dir . \
        --dockerfile src/drift_monitoring/Dockerfile.lambda \
        --repository "$REPO_NAME" \
        --role-arn "$BUILD_ROLE_ARN" \
        --region "$REGION"
    echo "  ✓ Image built and pushed via CodeBuild: $IMAGE_URI"
fi

# Resolve the :latest tag to its immutable digest and deploy BY DIGEST.
#
# Why: `aws lambda update-function-code --image-uri ...:latest` pins whatever
# digest :latest resolves to AT THAT INSTANT and never re-resolves it on
# invocation. If a newer image is later pushed to :latest without re-running
# this deploy, the function silently keeps running the old digest. Deploying
# by the digest we JUST pushed makes "the function runs this exact image" an
# explicit, verifiable fact instead of a race against tag mutation.
IMAGE_DIGEST=$(aws ecr describe-images \
    --repository-name "$REPO_NAME" \
    --image-ids imageTag=latest \
    --region "$REGION" \
    --query 'imageDetails[0].imageDigest' --output text 2>/dev/null || echo "None")
if [ -z "$IMAGE_DIGEST" ] || [ "$IMAGE_DIGEST" = "None" ]; then
    echo "  ✗ ERROR: could not resolve a digest for $REPO_NAME:latest in ECR."
    echo "    The image build/push step above did not produce an image — aborting"
    echo "    rather than leaving the Lambda on a stale digest."
    exit 1
fi
IMAGE_URI="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO_NAME@$IMAGE_DIGEST"
echo "  ✓ Deploying pinned digest: $IMAGE_URI"

# Step 5: Create/Update Lambda function
echo ""
echo "[5/7] Deploying Lambda function..."

# Get MLflow tracking URI if available
MLFLOW_URI=$(aws sagemaker list-mlflow-tracking-servers --region $REGION --query 'TrackingServerSummaries[0].TrackingServerArn' --output text 2>/dev/null || echo "")
SQS_QUEUE_URL=$(aws sqs get-queue-url --queue-name "$SQS_QUEUE_NAME" --region $REGION --query 'QueueUrl' --output text 2>/dev/null || echo "")

# Endpoint name from config.py (single source). lambda_drift_monitor.py
# uses this for the row it writes to monitoring_responses.
ENDPOINT_NAME_FOR_LAMBDA="$(get_config ENDPOINT_NAME)"
# Resolve MODEL_VERSION from the latest approved package in the MPG; fall
# back to "latest" so the Lambda still runs against an unversioned setup.
MODEL_VERSION_FOR_LAMBDA="${MODEL_VERSION:-}"
if [ -z "$MODEL_VERSION_FOR_LAMBDA" ]; then
    MODEL_VERSION_FOR_LAMBDA=$(aws sagemaker list-model-packages \
        --model-package-group-name "$MODEL_PACKAGE_GROUP_NAME" \
        --model-approval-status Approved \
        --sort-by CreationTime --sort-order Descending \
        --max-results 1 --region "$REGION" \
        --query 'ModelPackageSummaryList[0].ModelPackageVersion' \
        --output text 2>/dev/null || echo "latest")
    [ "$MODEL_VERSION_FOR_LAMBDA" = "None" ] && MODEL_VERSION_FOR_LAMBDA="latest"
fi

# Write env config to a temp file and pass via file:// — using --environment
# with the shorthand "Variables={...}" or inline JSON trips the AWS CLI parser
# on the embedded braces/quotes. file:// sidesteps shell quoting entirely.
ENV_FILE=$(mktemp -t lambda-env.XXXXXX.json)
# Note: cleanup trap for $ENV_FILE was already registered alongside $ECR_POLICY_FILE
# in Step 3 — don't re-register here or it would shadow the earlier trap and
# leak the policy file.

cat > "$ENV_FILE" <<EOF
{
  "Variables": {
    "ATHENA_DATABASE": "$ATHENA_DB",
    "ATHENA_OUTPUT_S3": "$ATHENA_OUTPUT",
    "ATHENA_EVALUATION_TABLE": "evaluation_data",
    "MODEL_PACKAGE_GROUP": "$MODEL_PACKAGE_GROUP_NAME",
    "SNS_TOPIC_ARN": "$TOPIC_ARN",
    "MLFLOW_TRACKING_URI": "$MLFLOW_URI",
    "BASELINE_ROC_AUC": "0.92",
    "DATA_DRIFT_THRESHOLD": "$DATA_DRIFT_THRESHOLD",
    "KS_PVALUE_THRESHOLD": "0.05",
    "MODEL_DRIFT_THRESHOLD": "$MODEL_DRIFT_THRESHOLD",
    "MONITORING_SQS_QUEUE_URL": "$SQS_QUEUE_URL",
    "CLOUDWATCH_NAMESPACE": "$CLOUDWATCH_NAMESPACE",
    "DATA_DRIFT_LOOKBACK_DAYS": "1",
    "MODEL_DRIFT_LOOKBACK_DAYS": "1",
    "ENDPOINT_NAME": "$ENDPOINT_NAME_FOR_LAMBDA",
    "MODEL_VERSION": "$MODEL_VERSION_FOR_LAMBDA"
  }
}
EOF

# Check if function exists
# Suppress stdout too (>/dev/null): `aws lambda get-function` prints the full
# function JSON on success, which would otherwise be captured into
# FUNCTION_EXISTS along with the trailing "true" — making it match neither
# "true" nor "false" below, so BOTH the update and create branches were
# silently skipped and the function was never repointed to the new image.
FUNCTION_EXISTS=$(aws lambda get-function --function-name $LAMBDA_NAME --region $REGION >/dev/null 2>&1 && echo "true" || echo "false")

if [ "$FUNCTION_EXISTS" = "true" ]; then
    CURRENT_PKG_TYPE=$(aws lambda get-function-configuration --function-name $LAMBDA_NAME --region $REGION --query 'PackageType' --output text)

    if [ "$CURRENT_PKG_TYPE" = "Zip" ]; then
        echo "  Function exists with PackageType=Zip, recreating as Image..."
        # Remove permission first
        aws lambda remove-permission --function-name $LAMBDA_NAME --statement-id AllowEventBridgeInvoke --region $REGION 2>/dev/null || true
        # Delete function
        aws lambda delete-function --function-name $LAMBDA_NAME --region $REGION
        sleep 3
        FUNCTION_EXISTS="false"
    else
        echo "  Updating existing function..."
        aws lambda update-function-code --function-name $LAMBDA_NAME --image-uri $IMAGE_URI --region $REGION > /dev/null
        aws lambda wait function-updated-v2 --function-name $LAMBDA_NAME --region $REGION
        aws lambda update-function-configuration \
            --function-name $LAMBDA_NAME \
            --timeout 300 \
            --memory-size 512 \
            --environment "file://$ENV_FILE" \
            --region $REGION > /dev/null
        aws lambda wait function-updated-v2 --function-name $LAMBDA_NAME --region $REGION
    fi
fi

if [ "$FUNCTION_EXISTS" = "false" ]; then
    echo "  Creating new function..."
    aws lambda create-function \
        --function-name $LAMBDA_NAME \
        --package-type Image \
        --code ImageUri=$IMAGE_URI \
        --role $ROLE_ARN \
        --timeout 300 \
        --memory-size 512 \
        --description "Automated drift detection with Evidently + MLflow" \
        --environment "file://$ENV_FILE" \
        --region $REGION > /dev/null
    aws lambda wait function-active-v2 --function-name $LAMBDA_NAME --region $REGION
fi

FUNCTION_ARN="arn:aws:lambda:${REGION}:${ACCOUNT_ID}:function:${LAMBDA_NAME}"
echo "  ✓ Lambda: $FUNCTION_ARN"

# Verify the function actually resolved to the digest we just deployed.
# Guards against a silently-stale function (e.g. a skipped/failed push, or a
# tag that moved between resolve and deploy). Fail loudly if they diverge.
aws lambda wait function-updated-v2 --function-name $LAMBDA_NAME --region $REGION 2>/dev/null || true
DEPLOYED_IMAGE=$(aws lambda get-function --function-name $LAMBDA_NAME --region $REGION \
    --query 'Code.ResolvedImageUri' --output text 2>/dev/null || echo "")
DEPLOYED_DIGEST="${DEPLOYED_IMAGE##*@}"
if [ "$DEPLOYED_DIGEST" = "$IMAGE_DIGEST" ]; then
    echo "  ✓ Verified: Lambda is running the just-built image ($IMAGE_DIGEST)"
else
    echo "  ✗ ERROR: Lambda is running $DEPLOYED_DIGEST but expected $IMAGE_DIGEST."
    echo "    The function did NOT pick up the new image. Re-run this deploy."
    exit 1
fi

# Asynchronous-invoke retry policy + on-failure destination.
#
# EventBridge invokes this function ASYNCHRONOUSLY. The EventBridge target
# RetryPolicy/DeadLetterConfig configured in Step 6 only covers EventBridge
# FAILING TO DELIVER the event into Lambda's async queue (throttling,
# function-not-ready). Once Lambda accepts the invocation (HTTP 202), a handler
# exception — the ERROR path in lambda_handler — is governed by LAMBDA's own
# async retry policy, NOT EventBridge, and after those retries are exhausted the
# event is dropped SILENTLY unless an on-failure destination is set. So without
# this block a scoring bug that raises on every attempt would retry twice and
# then vanish with no DLQ record. Pin the retry count/age and route exhausted
# failures to a dedicated SQS DLQ so a persistently broken monitor is visible.
echo ""
echo "[6a/7] Configuring async retry + on-failure destination..."
ASYNC_DLQ_URL=$(aws sqs create-queue --queue-name "$ASYNC_DLQ_NAME" --region $REGION --query 'QueueUrl' --output text 2>/dev/null || \
                aws sqs get-queue-url --queue-name "$ASYNC_DLQ_NAME" --region $REGION --query 'QueueUrl' --output text)
ASYNC_DLQ_ARN=$(aws sqs get-queue-attributes --queue-url "$ASYNC_DLQ_URL" --attribute-names QueueArn --region $REGION --query 'Attributes.QueueArn' --output text)

aws lambda put-function-event-invoke-config \
    --function-name $LAMBDA_NAME \
    --maximum-retry-attempts 1 \
    --maximum-event-age-in-seconds 3600 \
    --destination-config "{\"OnFailure\": {\"Destination\": \"$ASYNC_DLQ_ARN\"}}" \
    --region $REGION > /dev/null
echo "  ✓ Async on-failure DLQ: $ASYNC_DLQ_ARN (maxRetry=1, maxAge=3600s)"

# Step 6: EventBridge Rule
echo ""
echo "[6/7] Creating EventBridge schedule..."
aws events put-rule \
    --name $RULE_NAME \
    --schedule-expression "$SCHEDULE" \
    --state ENABLED \
    --description "Trigger drift monitoring Lambda" \
    --region $REGION > /dev/null

RULE_ARN="arn:aws:events:${REGION}:${ACCOUNT_ID}:rule/${RULE_NAME}"

# Dead-letter queue for the EventBridge target. Without a DLQ + retry policy,
# a failed delivery to the Lambda (throttling, function-not-ready, etc.) is
# silently dropped and the scheduled run just never happens. Create an SQS DLQ,
# grant EventBridge permission to send to it (scoped to THIS rule's ARN), and
# attach a RetryPolicy + DeadLetterConfig to the target below.
DLQ_NAME="${RULE_NAME}-dlq"
DLQ_URL=$(aws sqs create-queue --queue-name "$DLQ_NAME" --region $REGION --query 'QueueUrl' --output text 2>/dev/null || \
          aws sqs get-queue-url --queue-name "$DLQ_NAME" --region $REGION --query 'QueueUrl' --output text)
DLQ_ARN=$(aws sqs get-queue-attributes --queue-url "$DLQ_URL" --attribute-names QueueArn --region $REGION --query 'Attributes.QueueArn' --output text)

# Allow events.amazonaws.com to SendMessage to the DLQ, but only for this rule.
DLQ_QUEUE_POLICY=$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [{
    "Sid": "AllowEventBridgeDLQ",
    "Effect": "Allow",
    "Principal": {"Service": "events.amazonaws.com"},
    "Action": "sqs:SendMessage",
    "Resource": "$DLQ_ARN",
    "Condition": {
      "ArnEquals": {"aws:SourceArn": "$RULE_ARN"}
    }
  }]
}
EOF
)
aws sqs set-queue-attributes \
    --queue-url "$DLQ_URL" \
    --attributes "$(jq -n --arg p "$DLQ_QUEUE_POLICY" '{Policy: $p}')" \
    --region $REGION
echo "  ✓ DLQ: $DLQ_ARN"

# Target JSON (list) with retry + dead-letter config. Shorthand can't express
# the nested RetryPolicy/DeadLetterConfig, so write JSON and pass via file://.
TARGETS_FILE=$(mktemp -t eb-targets.XXXXXX.json)
cat > "$TARGETS_FILE" <<EOF
[
  {
    "Id": "1",
    "Arn": "$FUNCTION_ARN",
    "RetryPolicy": {
      "MaximumRetryAttempts": 2,
      "MaximumEventAgeInSeconds": 3600
    },
    "DeadLetterConfig": {
      "Arn": "$DLQ_ARN"
    }
  }
]
EOF
aws events put-targets \
    --rule $RULE_NAME \
    --targets "file://$TARGETS_FILE" \
    --region $REGION > /dev/null
rm -f "$TARGETS_FILE"

aws lambda add-permission \
    --function-name $LAMBDA_NAME \
    --statement-id AllowEventBridgeInvoke \
    --action lambda:InvokeFunction \
    --principal events.amazonaws.com \
    --source-arn "arn:aws:events:${REGION}:${ACCOUNT_ID}:rule/${RULE_NAME}" \
    --region $REGION 2>/dev/null || true

echo "  ✓ Schedule: $SCHEDULE"

# Step 7: Test Lambda
echo ""
echo "[7/7] Testing Lambda function..."
aws lambda invoke \
    --function-name $LAMBDA_NAME \
    --region $REGION \
    /tmp/lambda_test_output.json > /tmp/lambda_test_response.json 2>&1

# A successful HTTP invoke (StatusCode 200) does NOT mean the function
# succeeded: Lambda still returns 200 when the handler raises, and flags it via
# the FunctionError field in the invoke response metadata. So we fail the deploy
# if ANY of these hold:
#   - the invoke transport itself didn't return 200,
#   - the response metadata carries a FunctionError (Handled/Unhandled),
#   - the output payload carries an errorMessage/errorType (raised exception),
#   - the payload's own statusCode is present and not 200.
STATUS_CODE=$(jq -r '.StatusCode // empty' /tmp/lambda_test_response.json 2>/dev/null || echo "")
FUNCTION_ERROR=$(jq -r '.FunctionError // empty' /tmp/lambda_test_response.json 2>/dev/null || echo "")
PAYLOAD_ERROR=$(jq -r '.errorMessage // .errorType // empty' /tmp/lambda_test_output.json 2>/dev/null || echo "")
PAYLOAD_STATUS=$(jq -r '.statusCode // empty' /tmp/lambda_test_output.json 2>/dev/null || echo "")

if [ "$STATUS_CODE" != "200" ]; then
    echo "  ❌ Lambda test failed (invoke StatusCode=$STATUS_CODE):"
    cat /tmp/lambda_test_response.json
    exit 1
fi

if [ -n "$FUNCTION_ERROR" ]; then
    echo "  ❌ Lambda executed but the function errored (FunctionError=$FUNCTION_ERROR):"
    jq '.' /tmp/lambda_test_output.json 2>/dev/null || cat /tmp/lambda_test_output.json
    exit 1
fi

if [ -n "$PAYLOAD_ERROR" ]; then
    echo "  ❌ Lambda returned an error payload: $PAYLOAD_ERROR"
    jq '.' /tmp/lambda_test_output.json 2>/dev/null || cat /tmp/lambda_test_output.json
    exit 1
fi

if [ -n "$PAYLOAD_STATUS" ] && [ "$PAYLOAD_STATUS" != "200" ]; then
    echo "  ❌ Lambda returned non-success statusCode=$PAYLOAD_STATUS:"
    jq '.' /tmp/lambda_test_output.json 2>/dev/null || cat /tmp/lambda_test_output.json
    exit 1
fi

echo "  ✓ Lambda test successful!"
jq '.' /tmp/lambda_test_output.json | head -20

# Save configuration
echo ""
echo "Saving deployment configuration..."
CONFIG_FILE="src/config/drift_monitoring_config.json"
cat > $CONFIG_FILE <<EOF
{
  "sns_topic_arn": "$TOPIC_ARN",
  "lambda_function_arn": "$FUNCTION_ARN",
  "eventbridge_rule_arn": "arn:aws:events:${REGION}:${ACCOUNT_ID}:rule/${RULE_NAME}",
  "ecr_image_uri": "$IMAGE_URI",
  "schedule": "$SCHEDULE",
  "data_drift_threshold": "$DATA_DRIFT_THRESHOLD",
  "model_drift_threshold": "$MODEL_DRIFT_THRESHOLD",
  "email": "$ALERT_EMAIL",
  "region": "$REGION"
}
EOF
echo "  ✓ Config saved: $CONFIG_FILE"

echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║  ✅ DEPLOYMENT COMPLETE                                            ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo "  SNS Topic:     $TOPIC_ARN"
echo "  Lambda:        $FUNCTION_ARN"
echo "  EventBridge:   $RULE_NAME ($SCHEDULE)"
echo "  Docker Image:  $IMAGE_URI"
echo ""
echo "Next steps:"
echo "  1. Check $ALERT_EMAIL for SNS subscription confirmation"
echo "  2. Monitor: aws logs tail /aws/lambda/$LAMBDA_NAME --follow"
echo "  3. Manual test: aws lambda invoke --function-name $LAMBDA_NAME output.json"
echo ""
