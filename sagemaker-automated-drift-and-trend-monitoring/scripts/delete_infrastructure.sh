#!/bin/bash
set -e

#############################################################################
# Delete drift-monitor resources NOT managed by CloudFormation.
#
# The CFN stack (cloudformation/sagemaker-mlflow-setup.yaml) owns: SQS queues,
# Lambda IAM roles, the monitoring-writer Lambda, the SageMaker domain, S3
# buckets, Athena database. Those are torn down by:
#
#     ./cloudformation/delete-main-stack.sh
#
# This script tears down what `deploy_lambda_container.sh` and
# `create_cloudwatch_monitoring.py` create OUTSIDE CFN:
#
#   1. EventBridge rule + target (drift-monitor daily schedule)
#   2. Drift-monitor Lambda function (container-image based)
#   3. SNS topic (drift alerts) + its subscriptions
#   4. ECR repository for the drift-monitor image  (optional — pass --ecr)
#   5. CloudWatch dashboard + alarms (Section 6.6 of notebook 3)
#
# Run this BEFORE delete-main-stack.sh — the stack delete refuses to proceed if
# the drift Lambda still exists, because the Lambda's execution role and
# the SQS queue + SNS topic it references are all CFN-owned. Killing this
# Lambda first lets CFN tear those dependencies down cleanly.
#
# Usage:
#     scripts/delete_infrastructure.sh                # dry-run by default — shows what would happen, deletes nothing
#     scripts/delete_infrastructure.sh --execute      # actually delete
#     scripts/delete_infrastructure.sh --execute --ecr   # also delete ECR repo
#############################################################################

DRY_RUN=true
DELETE_ECR=false
for arg in "$@"; do
    case "$arg" in
        --execute|-x) DRY_RUN=false ;;
        --ecr|-e)     DELETE_ECR=true ;;
        --help|-h)
            sed -n '2,/^####/p' "$0" | sed 's/^#//' | head -n -1
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg" >&2
            echo "Run with --help to see usage." >&2
            exit 1
            ;;
    esac
done

# Every name below is sourced from src/config/config.py via get_config.
# Single source of truth — no shell-side defaults.
source "$(dirname "${BASH_SOURCE[0]}")/_read_config.sh"
REGION="$(get_config AWS_DEFAULT_REGION)"
LAMBDA_NAME="$(get_config DRIFT_LAMBDA_NAME)"
RULE_NAME="$(get_config EVENTBRIDGE_RULE_NAME)"
REPO_NAME="$(get_config ECR_REPO_NAME)"
SNS_TOPIC_NAME="$(get_config SNS_TOPIC_NAME)"
DASHBOARD_NAME="$(get_config CLOUDWATCH_DASHBOARD_NAME)"
ALARM_NAMES=(
    "FraudDetection-DataDrift-PSI"
    "FraudDetection-ModelDrift-ROCAUCDEGRADATION"
    "FraudDetection-ModelDrift-ACCURACY"
    "FraudDetection-ModelDrift-PRECISION"
    "FraudDetection-ModelDrift-RECALL"
)

if $DRY_RUN; then
    MODE="DRY-RUN (no changes — pass --execute to actually delete)"
else
    MODE="EXECUTE (resources WILL be deleted)"
fi

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║  Out-of-band drift-monitor resource cleanup                        ║"
echo "║  $MODE"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo "  Region:           $REGION"
echo "  EventBridge rule: $RULE_NAME"
echo "  Lambda function:  $LAMBDA_NAME"
echo "  SNS topic:        $SNS_TOPIC_NAME"
echo "  Dashboard:        $DASHBOARD_NAME"
echo "  CW alarms:        ${#ALARM_NAMES[@]} alarms"
echo "  ECR repo:         $REPO_NAME  (delete = $DELETE_ECR)"
echo ""

if ! $DRY_RUN; then
    read -p "Type 'yes' to confirm: " CONFIRM
    if [ "$CONFIRM" != "yes" ]; then
        echo "Cancelled."
        exit 0
    fi
    echo ""
fi

# Helper: prefix actions with [DRY-RUN] or run them depending on mode.
run() {
    if $DRY_RUN; then
        echo "  [DRY-RUN] would run: $*"
    else
        "$@"
    fi
}

# ────────────────────────────────────────────────────────────────────────────
# 1. EventBridge rule (remove targets first, then the rule)
# ────────────────────────────────────────────────────────────────────────────
echo "[1/5] EventBridge rule..."
if aws events describe-rule --name "$RULE_NAME" --region "$REGION" >/dev/null 2>&1; then
    TARGET_IDS=$(aws events list-targets-by-rule --rule "$RULE_NAME" --region "$REGION" \
        --query 'Targets[].Id' --output text 2>/dev/null || echo "")
    if [ -n "$TARGET_IDS" ]; then
        run aws events remove-targets --rule "$RULE_NAME" --ids $TARGET_IDS --region "$REGION"
    fi
    run aws events delete-rule --name "$RULE_NAME" --region "$REGION"
    echo "  ✓ Rule: $RULE_NAME"
else
    echo "  (rule not found, skipping)"
fi

# ────────────────────────────────────────────────────────────────────────────
# 2. Drift-monitor Lambda function
# ────────────────────────────────────────────────────────────────────────────
echo ""
echo "[2/5] Lambda function..."
if aws lambda get-function --function-name "$LAMBDA_NAME" --region "$REGION" >/dev/null 2>&1; then
    run aws lambda delete-function --function-name "$LAMBDA_NAME" --region "$REGION"
    echo "  ✓ Lambda: $LAMBDA_NAME"
else
    echo "  (function not found, skipping)"
fi

# ────────────────────────────────────────────────────────────────────────────
# 3. SNS topic + all confirmed subscriptions
# ────────────────────────────────────────────────────────────────────────────
echo ""
echo "[3/5] SNS topic..."
TOPIC_ARN=$(aws sns list-topics --region "$REGION" \
    --query "Topics[?ends_with(TopicArn, ':${SNS_TOPIC_NAME}')].TopicArn | [0]" \
    --output text 2>/dev/null || echo "")
if [ -n "$TOPIC_ARN" ] && [ "$TOPIC_ARN" != "None" ]; then
    SUBS=$(aws sns list-subscriptions-by-topic --topic-arn "$TOPIC_ARN" --region "$REGION" \
        --query "Subscriptions[?SubscriptionArn != 'PendingConfirmation'].SubscriptionArn" \
        --output text 2>/dev/null || echo "")
    for sub in $SUBS; do
        run aws sns unsubscribe --subscription-arn "$sub" --region "$REGION"
    done
    run aws sns delete-topic --topic-arn "$TOPIC_ARN" --region "$REGION"
    echo "  ✓ Topic: $SNS_TOPIC_NAME"
else
    echo "  (topic not found, skipping)"
fi

# ────────────────────────────────────────────────────────────────────────────
# 4. CloudWatch dashboard + alarms
# ────────────────────────────────────────────────────────────────────────────
echo ""
echo "[4/5] CloudWatch dashboard + alarms..."
if aws cloudwatch get-dashboard --dashboard-name "$DASHBOARD_NAME" --region "$REGION" >/dev/null 2>&1; then
    run aws cloudwatch delete-dashboards --dashboard-names "$DASHBOARD_NAME" --region "$REGION"
    echo "  ✓ Dashboard: $DASHBOARD_NAME"
else
    echo "  (dashboard not found, skipping)"
fi
# delete-alarms is a no-op for names that don't exist — safe to call once with the full list.
run aws cloudwatch delete-alarms --alarm-names "${ALARM_NAMES[@]}" --region "$REGION"
echo "  ✓ Alarms: ${#ALARM_NAMES[@]} (any non-existent ones silently skipped)"

# ────────────────────────────────────────────────────────────────────────────
# 5. ECR repository (only if --ecr — keeps the image around for fast redeploy)
# ────────────────────────────────────────────────────────────────────────────
echo ""
echo "[5/5] ECR repository..."
if $DELETE_ECR; then
    if aws ecr describe-repositories --repository-names "$REPO_NAME" --region "$REGION" >/dev/null 2>&1; then
        run aws ecr delete-repository --repository-name "$REPO_NAME" --force --region "$REGION"
        echo "  ✓ Repository: $REPO_NAME"
    else
        echo "  (repository not found, skipping)"
    fi
else
    echo "  Keeping ECR repository $REPO_NAME (re-run with --ecr to delete)"
fi

echo ""
if $DRY_RUN; then
    echo "╔════════════════════════════════════════════════════════════════════╗"
    echo "║  Dry-run complete. Re-run with --execute to actually delete.       ║"
    echo "╚════════════════════════════════════════════════════════════════════╝"
else
    echo "╔════════════════════════════════════════════════════════════════════╗"
    echo "║  ✅ Out-of-band resources deleted                                  ║"
    echo "╚════════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "Next: tear down everything else (SageMaker domain, Lambda IAM roles,"
    echo "SQS, S3, Athena DB) by running:"
    echo ""
    echo "    ./cloudformation/delete-main-stack.sh"
fi
