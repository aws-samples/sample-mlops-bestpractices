#!/bin/bash
#
# Deploy the drift-monitoring infrastructure stack.
#
# This creates the SNS topic, SQS queue, monitoring-results writer Lambda,
# drift-monitor IAM role, CloudWatch dashboard, and alarms. It does NOT
# create the scheduled drift-monitor Lambda (container image) — that's built
# separately with scripts/deploy_lambda_container.sh.
#
# Usage:
#   ./cloudformation/deploy-drift-monitoring.sh \
#       --data-bucket <bucket-name> \
#       --endpoint-name <endpoint-name> \
#       --alert-email <email>
#
#   Optional flags:
#     --stack-name <name>      (default: fraud-detection-drift-monitoring)
#     --region <region>        (default: from config.py AWS_DEFAULT_REGION)
#     --data-drift-threshold   (default: 0.2)
#     --model-drift-threshold  (default: 0.05)
#
# Example (from SageMaker notebook):
#   !bash ../cloudformation/deploy-drift-monitoring.sh \
#       --data-bucket my-fraud-detection-data \
#       --endpoint-name fraud-detector-endpoint \
#       --alert-email you@example.com
#
set -euo pipefail

# --- Defaults ---
STACK_NAME="fraud-detection-drift-monitoring"
DATA_BUCKET=""
ENDPOINT_NAME="fraud-detector-endpoint"
ALERT_EMAIL=""
DATA_DRIFT_THRESHOLD="0.2"
MODEL_DRIFT_THRESHOLD="0.05"
ATHENA_DATABASE="fraud_detection"

# --- Parse arguments ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stack-name)
      STACK_NAME="$2"
      shift 2
      ;;
    --data-bucket)
      DATA_BUCKET="$2"
      shift 2
      ;;
    --endpoint-name)
      ENDPOINT_NAME="$2"
      shift 2
      ;;
    --alert-email)
      ALERT_EMAIL="$2"
      shift 2
      ;;
    --data-drift-threshold)
      DATA_DRIFT_THRESHOLD="$2"
      shift 2
      ;;
    --model-drift-threshold)
      MODEL_DRIFT_THRESHOLD="$2"
      shift 2
      ;;
    --athena-database)
      ATHENA_DATABASE="$2"
      shift 2
      ;;
    --region)
      REGION="$2"
      shift 2
      ;;
    --help|-h)
      sed -n '1,/^set -euo pipefail$/p' "$0" | sed '$d'
      exit 0
      ;;
    *)
      echo "ERROR: Unknown option: $1"
      echo "Run with --help for usage."
      exit 1
      ;;
  esac
done

# --- Resolve region from config if not provided ---
if [ -z "${REGION:-}" ]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  if [ -f "$SCRIPT_DIR/../scripts/_read_config.sh" ]; then
    source "$SCRIPT_DIR/../scripts/_read_config.sh"
    REGION="$(get_config AWS_DEFAULT_REGION 2>/dev/null || echo 'us-east-1')"
  else
    REGION="${AWS_DEFAULT_REGION:-us-east-1}"
  fi
fi

# --- Validation ---
if [ -z "$DATA_BUCKET" ]; then
  echo "ERROR: --data-bucket is required"
  echo "Run with --help for usage."
  exit 1
fi

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  Deploying Drift-Monitoring Infrastructure Stack               ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "  Stack Name            : $STACK_NAME"
echo "  Region                : $REGION"
echo "  Data Bucket           : $DATA_BUCKET"
echo "  Endpoint Name         : $ENDPOINT_NAME"
echo "  Alert Email           : ${ALERT_EMAIL:-<none - no SNS subscription>}"
echo "  Data Drift Threshold  : $DATA_DRIFT_THRESHOLD"
echo "  Model Drift Threshold : $MODEL_DRIFT_THRESHOLD"
echo "  Athena Database       : $ATHENA_DATABASE"
echo ""

# --- Pre-flight checks ---
echo "[Pre-flight checks]"

# Check if the base stack exists. deploy-main-stack.sh names the base stack
# after PROJECT_NAME from config (e.g. "fraud-detection-monitoring"), so derive
# the same value here instead of hardcoding a stale name. Ensure get_config is
# available (the region block above only sources it when --region was omitted).
if ! command -v get_config >/dev/null 2>&1; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  [ -f "$SCRIPT_DIR/../scripts/_read_config.sh" ] && source "$SCRIPT_DIR/../scripts/_read_config.sh"
fi
BASE_STACK="$(get_config PROJECT_NAME 2>/dev/null || echo 'fraud-detection-monitoring')"
if ! aws cloudformation describe-stacks \
    --stack-name "$BASE_STACK" \
    --region "$REGION" &>/dev/null; then
  echo "⚠ WARNING: Base stack '$BASE_STACK' not found in $REGION"
  echo "  Deploy sagemaker-mlflow-setup.yaml first (it creates the data bucket + Athena tables)"
fi

# Check if bucket exists
if ! aws s3 ls "s3://$DATA_BUCKET" --region "$REGION" &>/dev/null; then
  echo "✗ ERROR: S3 bucket '$DATA_BUCKET' not found or not accessible"
  exit 1
fi
echo "✓ Data bucket '$DATA_BUCKET' exists"

# Check if endpoint exists (optional check - don't fail)
if aws sagemaker describe-endpoint \
    --endpoint-name "$ENDPOINT_NAME" \
    --region "$REGION" &>/dev/null 2>&1; then
  echo "✓ Endpoint '$ENDPOINT_NAME' exists"
else
  echo "⚠ Endpoint '$ENDPOINT_NAME' not found (deploy it first with notebook 2_deployment.ipynb)"
fi

echo ""
echo "[Deploying CloudFormation stack]"

# Build parameter overrides
PARAM_OVERRIDES="DataBucketName=$DATA_BUCKET"
PARAM_OVERRIDES="$PARAM_OVERRIDES EndpointName=$ENDPOINT_NAME"
PARAM_OVERRIDES="$PARAM_OVERRIDES AthenaDatabase=$ATHENA_DATABASE"
PARAM_OVERRIDES="$PARAM_OVERRIDES DataDriftThreshold=$DATA_DRIFT_THRESHOLD"
PARAM_OVERRIDES="$PARAM_OVERRIDES ModelDriftThreshold=$MODEL_DRIFT_THRESHOLD"

if [ -n "$ALERT_EMAIL" ]; then
  PARAM_OVERRIDES="$PARAM_OVERRIDES AlertEmail=$ALERT_EMAIL"
fi

TEMPLATE_FILE="$(dirname "$0")/drift-monitoring-infra.yaml"

if [ ! -f "$TEMPLATE_FILE" ]; then
  echo "✗ ERROR: Template file not found: $TEMPLATE_FILE"
  exit 1
fi

# Deploy the stack
aws cloudformation deploy \
  --template-file "$TEMPLATE_FILE" \
  --stack-name "$STACK_NAME" \
  --region "$REGION" \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameter-overrides $PARAM_OVERRIDES

if [ $? -eq 0 ]; then
  echo ""
  echo "✅ Stack deployed successfully!"
  echo ""

  # Show outputs
  echo "[Stack Outputs]"
  aws cloudformation describe-stacks \
    --stack-name "$STACK_NAME" \
    --region "$REGION" \
    --query 'Stacks[0].Outputs[*].[OutputKey,OutputValue]' \
    --output table

  echo ""
  echo "[Next Steps]"
  if [ -n "$ALERT_EMAIL" ]; then
    echo "  1. Check your email ($ALERT_EMAIL) and confirm the SNS subscription"
  fi
  echo "  2. (Optional) Deploy the scheduled drift-monitor Lambda:"
  echo "       ./scripts/deploy_lambda_container.sh"
  echo "  3. Run notebook 3_inference_monitoring.ipynb to test drift detection"
  echo ""
else
  echo ""
  echo "✗ Stack deployment failed!"
  echo "Check the error messages above for details."
  exit 1
fi
