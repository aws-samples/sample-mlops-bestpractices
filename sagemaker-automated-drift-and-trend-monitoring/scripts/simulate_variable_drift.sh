#!/bin/bash
#
# Simulate variable drift patterns across multiple monitoring runs
#
# This script:
# 1. Generates drifted datasets with time-varying patterns
# 2. Runs batch inference with each dataset
# 3. Triggers drift monitoring after each batch
# 4. Creates realistic timeline visualization in QuickSight
#
# Usage:
#   ./scripts/simulate_variable_drift.sh <endpoint-name>
#
# Example:
#   ./scripts/simulate_variable_drift.sh sg-xgboost-fraud-detector-endpoint
#

set -e  # Exit on error

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <endpoint-name>"
    echo ""
    echo "Example:"
    echo "  $0 sg-xgboost-fraud-detector-endpoint"
    exit 1
fi

ENDPOINT_NAME=$1
NUM_SAMPLES=${2:-2000}  # Default 2000 samples per run
WAIT_BETWEEN_RUNS=${3:-300}  # Default 5 minutes between runs

echo "=========================================="
echo "VARIABLE DRIFT SIMULATION"
echo "=========================================="
echo "Endpoint: $ENDPOINT_NAME"
echo "Samples per run: $NUM_SAMPLES"
echo "Wait between runs: ${WAIT_BETWEEN_RUNS}s"
echo ""

# Step 1: Generate drift datasets
echo "Step 1: Generating variable drift datasets..."
python src/drift_monitoring/generate_variable_drift_dataset.py

if [ $? -ne 0 ]; then
    echo "❌ Failed to generate drift datasets"
    exit 1
fi

echo "✅ Drift datasets generated"
echo ""

# Step 2: Run inference and monitoring for each dataset
RUNS=("run1" "run2" "run3" "run4" "run5" "run6")
DESCRIPTIONS=(
    "Baseline (minimal drift)"
    "Distance spike (travel/remote)"
    "Credit limit anomaly (system change)"
    "High velocity (increased activity)"
    "Recovery (returning to normal)"
    "Account age anomaly (new cohort)"
)

for i in "${!RUNS[@]}"; do
    RUN="${RUNS[$i]}"
    DESC="${DESCRIPTIONS[$i]}"

    echo "=========================================="
    echo "PROCESSING: $RUN"
    echo "=========================================="
    echo "Description: $DESC"
    echo "Dataset: data/drifted_data_${RUN}.csv"
    echo ""

    # Run batch inference
    echo "Running batch inference..."
    python main.py --mode test \
        --endpoint-name "$ENDPOINT_NAME" \
        --test-data "data/drifted_data_${RUN}.csv" \
        --num-samples "$NUM_SAMPLES"

    if [ $? -ne 0 ]; then
        echo "❌ Batch inference failed for $RUN"
        continue
    fi

    echo "✅ Batch inference completed for $RUN"

    # Wait for data to settle
    echo "Waiting 60s for data to settle..."
    sleep 60

    # Trigger drift monitoring
    echo "Triggering drift monitoring..."
    python src/drift_monitoring/run_drift_monitoring.py

    if [ $? -ne 0 ]; then
        echo "⚠️  Drift monitoring failed for $RUN (continuing...)"
    else
        echo "✅ Drift monitoring completed for $RUN"
    fi

    # Wait before next run (except last)
    if [ $i -lt $((${#RUNS[@]} - 1)) ]; then
        echo ""
        echo "Waiting ${WAIT_BETWEEN_RUNS}s before next run..."
        sleep "$WAIT_BETWEEN_RUNS"
        echo ""
    fi
done

echo ""
echo "=========================================="
echo "SIMULATION COMPLETE"
echo "=========================================="
echo ""
echo "✅ Processed ${#RUNS[@]} drift pattern runs"
echo ""
echo "📊 View Results in QuickSight:"
echo "  1. Open: https://quicksight.aws.amazon.com/"
echo "  2. Dashboard: fraud-monitoring-dashboard"
echo "  3. Sheet: Feature Drift Detail"
echo "  4. Visual: Feature Drift Score Timeline"
echo ""
echo "You should now see:"
echo "  • distance_from_home_km spiking in run 2"
echo "  • credit_limit spiking in run 3"
echo "  • velocity_score spiking in run 4"
echo "  • Varying patterns across all features over time"
echo ""
echo "=========================================="
