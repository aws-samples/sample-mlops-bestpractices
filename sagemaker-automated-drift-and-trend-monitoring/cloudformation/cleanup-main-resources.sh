#!/bin/bash
set -e

# Resolve stack name + region from config (matches deploy-main-stack.sh) so the
# defaults target the real base stack (PROJECT_NAME) instead of a stale name.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$SCRIPT_DIR/../scripts/_read_config.sh" ] && source "$SCRIPT_DIR/../scripts/_read_config.sh"
STACK_NAME="${1:-$(get_config PROJECT_NAME 2>/dev/null || echo 'fraud-detection-monitoring')}"
REGION="${AWS_DEFAULT_REGION:-$(get_config AWS_DEFAULT_REGION 2>/dev/null || echo 'us-east-1')}"

echo "=== SageMaker CloudFormation Stack Cleanup ==="
echo "Stack: $STACK_NAME"
echo "Region: $REGION"
echo ""

# Step 1: Get Domain ID from stack
echo "Step 1: Finding SageMaker Domain..."
DOMAIN_ID=$(aws cloudformation describe-stack-resources \
  --stack-name "$STACK_NAME" \
  --query "StackResources[?ResourceType=='AWS::SageMaker::Domain'].PhysicalResourceId" \
  --output text \
  --region "$REGION" 2>/dev/null || echo "")

if [ -n "$DOMAIN_ID" ]; then
  echo "Found Domain ID: $DOMAIN_ID"

  # Step 2: Delete all Spaces in the domain
  echo ""
  echo "Step 2: Deleting Spaces..."
  SPACES=$(aws sagemaker list-spaces \
    --domain-id "$DOMAIN_ID" \
    --query "Spaces[].SpaceName" \
    --output text \
    --region "$REGION" 2>/dev/null || echo "")

  if [ -n "$SPACES" ]; then
    for SPACE in $SPACES; do
      echo "  Deleting Space: $SPACE"
      aws sagemaker delete-space \
        --domain-id "$DOMAIN_ID" \
        --space-name "$SPACE" \
        --region "$REGION" 2>/dev/null || true
    done
    echo "  Waiting for Spaces to delete..."
    sleep 30
  else
    echo "  No Spaces found"
  fi

  # Step 3: Delete all Apps in the domain
  echo ""
  echo "Step 3: Deleting Apps..."
  USER_PROFILES=$(aws sagemaker list-user-profiles \
    --domain-id "$DOMAIN_ID" \
    --query "UserProfiles[].UserProfileName" \
    --output text \
    --region "$REGION" 2>/dev/null || echo "")

  if [ -n "$USER_PROFILES" ]; then
    for PROFILE in $USER_PROFILES; do
      echo "  Checking Apps for User Profile: $PROFILE"
      APPS=$(aws sagemaker list-apps \
        --domain-id "$DOMAIN_ID" \
        --user-profile-name "$PROFILE" \
        --query "Apps[?Status!='Deleted'].AppName" \
        --output text \
        --region "$REGION" 2>/dev/null || echo "")

      if [ -n "$APPS" ]; then
        for APP in $APPS; do
          APP_TYPE=$(aws sagemaker list-apps \
            --domain-id "$DOMAIN_ID" \
            --user-profile-name "$PROFILE" \
            --query "Apps[?AppName=='$APP'].AppType" \
            --output text \
            --region "$REGION" 2>/dev/null || echo "JupyterLab")

          echo "    Deleting App: $APP (Type: $APP_TYPE)"
          aws sagemaker delete-app \
            --domain-id "$DOMAIN_ID" \
            --user-profile-name "$PROFILE" \
            --app-type "$APP_TYPE" \
            --app-name "$APP" \
            --region "$REGION" 2>/dev/null || true
        done
      fi
    done
    echo "  Waiting for Apps to delete..."
    sleep 60
  else
    echo "  No User Profiles found"
  fi

  # Step 4: Find and delete orphaned ENIs
  echo ""
  echo "Step 4: Cleaning up orphaned ENIs..."
  VPC_ID=$(aws cloudformation describe-stack-resources \
    --stack-name "$STACK_NAME" \
    --query "StackResources[?ResourceType=='AWS::EC2::VPC'].PhysicalResourceId" \
    --output text \
    --region "$REGION" 2>/dev/null || echo "")

  if [ -n "$VPC_ID" ]; then
    echo "  VPC ID: $VPC_ID"
    ENIS=$(aws ec2 describe-network-interfaces \
      --filters "Name=vpc-id,Values=$VPC_ID" "Name=description,Values=*SageMaker*" \
      --query "NetworkInterfaces[].NetworkInterfaceId" \
      --output text \
      --region "$REGION" 2>/dev/null || echo "")

    if [ -n "$ENIS" ]; then
      for ENI in $ENIS; do
        echo "    Deleting ENI: $ENI"
        aws ec2 delete-network-interface \
          --network-interface-id "$ENI" \
          --region "$REGION" 2>/dev/null || echo "      (ENI may be in use, will retry)"
      done
      echo "  Waiting for ENIs to detach..."
      sleep 30
    else
      echo "  No orphaned ENIs found"
    fi
  fi
else
  echo "Domain not found or already deleted"
fi

# Step 5: Retry stack deletion
echo ""
echo "Step 5: Retrying stack deletion..."
aws cloudformation delete-stack \
  --stack-name "$STACK_NAME" \
  --region "$REGION"

echo ""
echo "✅ Stack deletion initiated. Monitor progress with:"
echo "  aws cloudformation describe-stacks --stack-name $STACK_NAME --region $REGION"
echo ""
echo "Or wait for completion:"
echo "  aws cloudformation wait stack-delete-complete --stack-name $STACK_NAME --region $REGION"
