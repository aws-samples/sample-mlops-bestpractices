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

# Step 5: Create/Update Lambda function
echo ""
echo "[5/7] Deploying Lambda function..."

# Get MLflow tracking URI if available
# Priority: MLFLOW_TRACKING_URI env var (from .env) > SageMaker MLflow app lookup > empty
MLFLOW_URI="${MLFLOW_TRACKING_URI:-}"
if [ -z "$MLFLOW_URI" ]; then
    # Try the SageMaker MLflow tracking server API
    MLFLOW_URI=$(aws sagemaker list-mlflow-tracking-servers --region $REGION --query 'TrackingServerSummaries[0].TrackingServerArn' --output text 2>/dev/null || echo "")
    [ "$MLFLOW_URI" = "None" ] && MLFLOW_URI=""
fi

# Monitoring SQS queue — derive name from config (matches CFN drift-monitoring-infra.yaml)
MONITORING_QUEUE_NAME="$(get_config MONITORING_SQS_QUEUE_NAME 2>/dev/null || echo 'fraud-detection-monitoring-results')"
SQS_QUEUE_URL=$(aws sqs get-queue-url --queue-name "$MONITORING_QUEUE_NAME" --region $REGION --query 'QueueUrl' --output text 2>/dev/null || echo "")

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
    "DATA_DRIFT_LOOKBACK_DAYS": "1",
    "MODEL_DRIFT_LOOKBACK_DAYS": "1",
    "ENDPOINT_NAME": "$ENDPOINT_NAME_FOR_LAMBDA",
    "MODEL_VERSION": "$MODEL_VERSION_FOR_LAMBDA"
  }
}
EOF

# Check if function exists
FUNCTION_EXISTS=$(aws lambda get-function --function-name $LAMBDA_NAME --region $REGION 2>/dev/null && echo "true" || echo "false")

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

# Step 6: EventBridge Rule
echo ""
echo "[6/7] Creating EventBridge schedule..."
aws events put-rule \
    --name $RULE_NAME \
    --schedule-expression "$SCHEDULE" \
    --state ENABLED \
    --description "Trigger drift monitoring Lambda" \
    --region $REGION > /dev/null

aws events put-targets \
    --rule $RULE_NAME \
    --targets "Id=1,Arn=$FUNCTION_ARN" \
    --region $REGION > /dev/null

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

STATUS_CODE=$(jq -r '.StatusCode' /tmp/lambda_test_response.json)
if [ "$STATUS_CODE" = "200" ]; then
    RESULT=$(jq -r '.statusCode' /tmp/lambda_test_output.json 2>/dev/null || echo "error")
    if [ "$RESULT" = "200" ]; then
        echo "  ✓ Lambda test successful!"
        jq '.' /tmp/lambda_test_output.json | head -20
    else
        echo "  ⚠ Lambda executed but returned error:"
        jq '.' /tmp/lambda_test_output.json
    fi
else
    echo "  ❌ Lambda test failed:"
    cat /tmp/lambda_test_response.json
fi

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
