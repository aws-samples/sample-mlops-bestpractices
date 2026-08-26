#!/usr/bin/env python3
"""
Create CloudWatch Dashboard and Alarms for Drift Monitoring.

This script creates:
1. CloudWatch metrics for data drift (PSI) and model drift (ROC-AUC, etc.)
2. CloudWatch alarms that trigger on threshold violations
3. CloudWatch dashboard for visualizing drift trends
"""

import boto3
import json
import sys
import os
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv


def create_cloudwatch_monitoring(
    region='us-east-1',
    endpoint_name='fraud-detection-endpoint',
    drift_threshold=0.10,
    psi_threshold=0.2,
    evaluation_periods=1
):
    """
    Create CloudWatch dashboard and alarms for drift monitoring.

    Args:
        region: AWS region
        endpoint_name: SageMaker endpoint name
        drift_threshold: Threshold for model drift alarms (default: 10% degradation)
        psi_threshold: Threshold for PSI data drift alarm (default: 0.2)
        evaluation_periods: Number of evaluation periods for alarms
    """

    # Load .env if available
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)

    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║  Creating CloudWatch Drift Monitoring                             ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    print("")
    print(f"  Region: {region}")
    print(f"  Endpoint: {endpoint_name}")
    print(f"  Model Drift Threshold: {drift_threshold*100:.0f}%")
    print(f"  PSI Threshold: {psi_threshold}")
    print("")

    # AWS clients
    cw_client = boto3.client('cloudwatch', region_name=region)
    sts = boto3.client('sts', region_name=region)
    account_id = sts.get_caller_identity()['Account']

    # Configuration (from .env with defaults)
    NAMESPACE = os.getenv('CLOUDWATCH_NAMESPACE', 'FraudDetection/DriftMonitoring')
    DASHBOARD_NAME = os.getenv('CLOUDWATCH_DASHBOARD_NAME', 'FraudDetection-DriftMonitoring')
    ATHENA_DATABASE = os.getenv('ATHENA_DATABASE', 'fraud_detection')
    MONITORING_TABLE = os.getenv('MONITORING_TABLE_NAME', 'monitoring_responses')
    DATA_S3_BUCKET = os.getenv('DATA_S3_BUCKET', f'fraud-detection-data-lake-skoppar-{account_id}')

    # SNS topic ARN for alarm actions. Resolved the same way the rest of the
    # codebase does (manage_drift_lambda.py / config.py): accept an explicit
    # SNS_TOPIC_ARN if provided, otherwise construct it from SNS_TOPIC_NAME
    # (default matches the CFN SnsTopicName default 'fraud-detection-drift-alerts')
    # plus region + account. The CFN alarms wire AlarmActions to this topic, so
    # the imperative path must do the same to reach parity.
    SNS_TOPIC_NAME = os.getenv('SNS_TOPIC_NAME', 'fraud-detection-drift-alerts')
    SNS_TOPIC_ARN = os.getenv('SNS_TOPIC_ARN') or f'arn:aws:sns:{region}:{account_id}:{SNS_TOPIC_NAME}'
    print(f"  SNS Topic ARN (alarm actions): {SNS_TOPIC_ARN}")
    print("")

    # Step 1: Get latest drift metrics from monitoring_responses table
    print("[1/3] Fetching latest drift metrics from Athena...")

    try:
        athena = boto3.client('athena', region_name=region)
        # Column names must match the authoritative monitoring_responses DDL
        # (src/setup/create_athena_tables.py): monitoring_timestamp,
        # roc_auc_degradation, precision (not timestamp/degradation/
        # precision_score). Filter to THIS endpoint so we never publish another
        # endpoint's latest row under this endpoint's CloudWatch dimension. The
        # table is PARTITIONED BY day(monitoring_timestamp); order by it DESC.
        query = f"""
        SELECT
            monitoring_timestamp,
            drifted_columns_share,
            baseline_roc_auc,
            current_roc_auc,
            roc_auc_degradation,
            accuracy,
            precision,
            recall
        FROM {ATHENA_DATABASE}.{MONITORING_TABLE}
        WHERE endpoint_name = '{endpoint_name}'
        ORDER BY monitoring_timestamp DESC
        LIMIT 1
        """

        response = athena.start_query_execution(
            QueryString=query,
            QueryExecutionContext={'Database': ATHENA_DATABASE},
            ResultConfiguration={
                'OutputLocation': f's3://{DATA_S3_BUCKET}/athena-query-results/'
            }
        )

        execution_id = response['QueryExecutionId']

        # Wait for query to complete
        import time
        while True:
            status = athena.get_query_execution(QueryExecutionId=execution_id)
            state = status['QueryExecution']['Status']['State']
            if state in ['SUCCEEDED', 'FAILED', 'CANCELLED']:
                break
            time.sleep(1)

        if state == 'SUCCEEDED':
            # Get results
            result_s3_path = status['QueryExecution']['ResultConfiguration']['OutputLocation']
            s3 = boto3.client('s3', region_name=region)
            bucket, key = result_s3_path.replace('s3://', '').split('/', 1)
            obj = s3.get_object(Bucket=bucket, Key=key)

            import csv
            lines = obj['Body'].read().decode('utf-8').splitlines()
            reader = csv.DictReader(lines)
            results = list(reader)

            if results:
                latest = results[0]
                print(f"  ✓ Found latest metrics from {latest.get('monitoring_timestamp', 'N/A')}")

                # Publish metrics to CloudWatch
                metrics_to_publish = []

                # Data drift metric — share of columns that drifted this run
                # (0..1). Must match the metric name the drift Lambda publishes
                # in publish_cloudwatch_metrics() so the alarm/dashboard bind to
                # live data.
                if latest.get('drifted_columns_share'):
                    metrics_to_publish.append({
                        'MetricName': 'DriftedColumnsShare',
                        'Value': float(latest['drifted_columns_share']),
                        'Unit': 'None',
                        'Timestamp': datetime.now(timezone.utc)
                    })

                # Model performance metrics
                if latest.get('baseline_roc_auc') and latest.get('current_roc_auc'):
                    baseline = float(latest['baseline_roc_auc'])
                    current = float(latest['current_roc_auc'])
                    degradation = float(latest.get('roc_auc_degradation', 0))

                    metrics_to_publish.extend([
                        {'MetricName': 'BaselineROCAUC', 'Value': baseline, 'Unit': 'None'},
                        {'MetricName': 'CurrentROCAUC', 'Value': current, 'Unit': 'None'},
                        {'MetricName': 'ROCAUCDegradation', 'Value': degradation, 'Unit': 'None'},
                    ])

                if latest.get('accuracy'):
                    metrics_to_publish.append({
                        'MetricName': 'Accuracy',
                        'Value': float(latest['accuracy']),
                        'Unit': 'None'
                    })

                if latest.get('precision'):
                    metrics_to_publish.append({
                        'MetricName': 'Precision',
                        'Value': float(latest['precision']),
                        'Unit': 'None'
                    })

                if latest.get('recall'):
                    metrics_to_publish.append({
                        'MetricName': 'Recall',
                        'Value': float(latest['recall']),
                        'Unit': 'None'
                    })

                # Publish metrics in batches
                for i in range(0, len(metrics_to_publish), 20):
                    batch = metrics_to_publish[i:i+20]
                    cw_client.put_metric_data(
                        Namespace=NAMESPACE,
                        MetricData=[{
                            **m,
                            'Dimensions': [{'Name': 'Endpoint', 'Value': endpoint_name}]
                        } for m in batch]
                    )

                print(f"  ✓ Published {len(metrics_to_publish)} metrics to CloudWatch")
            else:
                print("  ⚠ No monitoring data found yet")
                print("    Run drift monitoring Lambda first to generate metrics")

    except Exception as e:
        print(f"  ⚠ Could not fetch metrics: {e}")
        print("    Dashboard and alarms will still be created, but may show no data")

    # Step 2: Create CloudWatch Alarms
    print("")
    print("[2/3] Creating CloudWatch alarms...")

    alarms_created = []

    # Data Drift Alarm: share of drifted columns > threshold.
    # `psi_threshold` is kept as the parameter name for CLI backward
    # compatibility (main.py passes it), but the metric is now the bounded
    # drifted-columns share (0..1) the drift Lambda publishes — not PSI.
    # Higher = more drift, so GreaterThanThreshold is the correct direction.
    #
    # EARLY-WARNING TIER (finding M1) — INTENTIONAL, DO NOT "fix" the gap:
    # This CloudWatch alarm trips at psi_threshold (default 0.20 = 20% of
    # columns drifted), which is deliberately BELOW the 0.50 (50%) share at
    # which the Evidently run's OVERALL drift VERDICT flips to drift-detected
    # (DRIFT_SHARE_THRESHOLD = 0.5 in evidently_reports.py) and fires the SNS
    # alert / DataDriftDetected. Consequence, by design: this CloudWatch alarm
    # can be in ALARM (RED) while the run's SNS alert / DataDriftDetected still
    # report no-drift, because 20% <= share < 50%. CloudWatch is the leading
    # early-warning signal; SNS/DataDriftDetected is the confirmed verdict.
    #
    # Alarm NAME is 'FraudDetection-DataDrift-PSI' to match the CloudFormation
    # DataDriftPsiAlarm (cloudformation/drift-monitoring-infra.yaml) so both the
    # imperative and CFN setup paths produce the SAME alarm. TreatMissingData is
    # notBreaching (a missing metric is not, by itself, data drift) and
    # AlarmActions notify the SNS topic, matching CFN.
    try:
        cw_client.put_metric_alarm(
            AlarmName='FraudDetection-DataDrift-PSI',
            AlarmDescription=f'Data drift early-warning: share of drifted columns exceeds {psi_threshold}.',
            MetricName='DriftedColumnsShare',
            Namespace=NAMESPACE,
            Statistic='Average',
            Period=300,
            EvaluationPeriods=evaluation_periods,
            Threshold=psi_threshold,
            ComparisonOperator='GreaterThanThreshold',
            TreatMissingData='notBreaching',
            Dimensions=[{'Name': 'Endpoint', 'Value': endpoint_name}],
            AlarmActions=[SNS_TOPIC_ARN],
        )
        alarms_created.append(f'FraudDetection-DataDrift-PSI (threshold: share > {psi_threshold})')
    except Exception as e:
        print(f"  ⚠ Failed to create data-drift alarm: {e}")

    # Model Drift Alarm: RELATIVE degradation of the primary metric > threshold.
    #
    # Alarms on PrimaryMetricDegradationRatio — the RELATIVE fraction the drift
    # Lambda publishes (value = degradation_pct/100, e.g. 0.10 == 10% relative
    # degradation from baseline). This matches drift_threshold's semantics (a
    # percentage), unlike the old ROCAUCDegradation metric which carried an
    # ABSOLUTE ROC-AUC delta and was inconsistent with a percentage threshold.
    # The primary metric is ROC-AUC for classification and RMSE for regression,
    # so the ratio is task-agnostic. Ratio is "bad = high", so
    # GreaterThanThreshold is the correct direction.
    #
    # The alarm NAME is 'FraudDetection-ModelDrift-ROCAUCDEGRADATION' (all-caps
    # DEGRADATION) to match the CloudFormation ModelDriftRocAucAlarm — the CFN
    # spelling is authoritative, so the imperative path adopts it (was
    # ...-ROCAUCDegradation here). The dashboard alarm-status widget below
    # references this exact name. Only the casing changed; MetricName is
    # PrimaryMetricDegradationRatio. TreatMissingData notBreaching + AlarmActions
    # to SNS mirror CFN.
    #
    # The former Accuracy/Precision/Recall alarms compared ABSOLUTE metric
    # values (~0.8–0.9) with GreaterThanThreshold(0.10), so they fired on every
    # publish — a false-positive by construction. Accuracy/Precision/Recall are
    # still plotted on the dashboard below as absolute-value time series.
    try:
        cw_client.put_metric_alarm(
            AlarmName='FraudDetection-ModelDrift-ROCAUCDEGRADATION',
            AlarmDescription=f'Model drift: relative degradation of the primary metric (ROC-AUC / accuracy / RMSE) from baseline exceeds {drift_threshold*100:.0f}%.',
            MetricName='PrimaryMetricDegradationRatio',
            Namespace=NAMESPACE,
            Statistic='Average',
            Period=300,
            EvaluationPeriods=evaluation_periods,
            Threshold=drift_threshold,
            ComparisonOperator='GreaterThanThreshold',
            TreatMissingData='notBreaching',
            Dimensions=[{'Name': 'Endpoint', 'Value': endpoint_name}],
            AlarmActions=[SNS_TOPIC_ARN],
        )
        alarms_created.append(f'FraudDetection-ModelDrift-ROCAUCDEGRADATION (threshold: > {drift_threshold})')
    except Exception as e:
        print(f"  ⚠ Failed to create ROC-AUC degradation alarm: {e}")

    # Heartbeat / no-data alarm: the drift Lambda emits DriftRunExecuted=1 on
    # EVERY scheduled run (including no-data runs). Its ABSENCE over the rolling
    # ~25h window (one daily run + slack) means the schedule stalled. Statistic
    # Sum + LessThanThreshold(1) + TreatMissingData=breaching fires when no
    # heartbeat arrived. Matches the CFN DriftMonitorHeartbeatAlarm exactly:
    # a rolling ~25h window built from Period 3600s × 25 consecutive missing
    # datapoints, so a missed daily run is detected within ~1h of the 25h mark
    # instead of up to ~50h with a single fixed 90000s period boundary. This is
    # the alarm the imperative path was missing entirely.
    try:
        cw_client.put_metric_alarm(
            AlarmName='FraudDetection-DriftMonitor-NoHeartbeat',
            AlarmDescription='The scheduled drift monitor has not run (no DriftRunExecuted heartbeat) in the expected window.',
            MetricName='DriftRunExecuted',
            Namespace=NAMESPACE,
            Statistic='Sum',
            Period=3600,             # rolling 25h window = 3600s × 25 datapoints
            EvaluationPeriods=25,
            DatapointsToAlarm=25,
            Threshold=1,
            ComparisonOperator='LessThanThreshold',
            TreatMissingData='breaching',
            Dimensions=[{'Name': 'Endpoint', 'Value': endpoint_name}],
            AlarmActions=[SNS_TOPIC_ARN],
        )
        alarms_created.append('FraudDetection-DriftMonitor-NoHeartbeat (fires if no run in ~25h)')
    except Exception as e:
        print(f"  ⚠ Failed to create heartbeat alarm: {e}")

    print(f"  ✓ Created {len(alarms_created)} alarms")

    # Step 3: Create CloudWatch Dashboard
    print("")
    print(f"[3/3] Creating CloudWatch dashboard: {DASHBOARD_NAME}...")

    # CloudWatch dashboard widget rules:
    #   - Header text uses type="text" with `markdown` property.
    #     (type="metric" with `markdown` is rejected as invalid.)
    #   - Every widget needs x/y/width/height. The grid is 24 columns wide.
    dashboard_body = {
        "widgets": [
            {
                "type": "text",
                "x": 0, "y": 0, "width": 24, "height": 2,
                "properties": {
                    "markdown": f"# Fraud Detection - Drift Monitoring Dashboard\n**Endpoint:** `{endpoint_name}` | **Threshold:** {drift_threshold*100:.0f}% variance | **Updated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
                }
            },
            {
                "type": "metric",
                "x": 0, "y": 2, "width": 12, "height": 6,
                "properties": {
                    "metrics": [
                        [NAMESPACE, "DriftedColumnsShare", "Endpoint", endpoint_name, {"stat": "Average"}]
                    ],
                    "view": "timeSeries",
                    "stacked": False,
                    "region": region,
                    "title": "Data Drift - Share of Drifted Columns",
                    "period": 300,
                    "yAxis": {"left": {"min": 0, "max": 1}},
                    "annotations": {
                        "horizontal": [
                            {"label": "Alarm Threshold", "value": psi_threshold, "fill": "above", "color": "#d62728"}
                        ]
                    }
                }
            },
            {
                "type": "metric",
                "x": 12, "y": 2, "width": 12, "height": 6,
                "properties": {
                    "metrics": [
                        [NAMESPACE, "PrimaryMetricDegradationRatio", "Endpoint", endpoint_name, {"stat": "Average", "label": "Primary Metric Degradation (relative)"}],
                        [NAMESPACE, "ROCAUCDegradation", "Endpoint", endpoint_name, {"stat": "Average", "label": "ROC-AUC Degradation (absolute, binary)"}],
                    ],
                    "view": "timeSeries",
                    "stacked": False,
                    "region": region,
                    "title": "Model Drift - Degradation from Baseline",
                    "period": 300,
                    "yAxis": {"left": {"min": 0, "max": 0.5}},
                    "annotations": {
                        "horizontal": [
                            {"label": f"{drift_threshold*100:.0f}% Alarm Threshold", "value": drift_threshold, "fill": "above", "color": "#d62728"}
                        ]
                    }
                }
            },
            {
                "type": "metric",
                "x": 0, "y": 8, "width": 12, "height": 6,
                "properties": {
                    "metrics": [
                        [NAMESPACE, "Accuracy", "Endpoint", endpoint_name, {"stat": "Average", "label": "Accuracy"}],
                        [NAMESPACE, "Precision", "Endpoint", endpoint_name, {"stat": "Average", "label": "Precision"}],
                        [NAMESPACE, "Recall", "Endpoint", endpoint_name, {"stat": "Average", "label": "Recall"}],
                        [NAMESPACE, "CurrentROCAUC", "Endpoint", endpoint_name, {"stat": "Average", "label": "Current ROC-AUC"}],
                        [NAMESPACE, "BaselineROCAUC", "Endpoint", endpoint_name, {"stat": "Average", "label": "Baseline ROC-AUC"}],
                    ],
                    "view": "timeSeries",
                    "stacked": False,
                    "region": region,
                    "title": "Model Quality - Current Metrics",
                    "period": 300,
                    "yAxis": {"left": {"min": 0, "max": 1}},
                }
            },
            {
                "type": "alarm",
                "x": 12, "y": 8, "width": 12, "height": 6,
                "properties": {
                    "title": "Drift Alarms Status",
                    "alarms": [
                        f"arn:aws:cloudwatch:{region}:{account_id}:alarm:FraudDetection-DataDrift-PSI",
                        f"arn:aws:cloudwatch:{region}:{account_id}:alarm:FraudDetection-ModelDrift-ROCAUCDEGRADATION",
                        f"arn:aws:cloudwatch:{region}:{account_id}:alarm:FraudDetection-DriftMonitor-NoHeartbeat",
                    ]
                }
            }
        ]
    }

    try:
        cw_client.put_dashboard(
            DashboardName=DASHBOARD_NAME,
            DashboardBody=json.dumps(dashboard_body),
        )
        print(f"  ✓ Dashboard created: {DASHBOARD_NAME}")
    except Exception as e:
        print(f"  ⚠ Failed to create dashboard: {e}")

    # Summary
    dashboard_url = f"https://console.aws.amazon.com/cloudwatch/home?region={region}#dashboards:name={DASHBOARD_NAME}"

    print("")
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║  ✅ CLOUDWATCH MONITORING CREATED                                  ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    print("")
    print(f"Dashboard URL:")
    print(f"  {dashboard_url}")
    print("")
    print(f"Alarms ({len(alarms_created)}):")
    for alarm in alarms_created:
        print(f"  - {alarm}")
    print("")
    print("Next steps:")
    print("  1. View dashboard in CloudWatch console")
    print(f"  2. Alarm actions are wired to SNS topic: {SNS_TOPIC_ARN}")
    print("     (subscribe an endpoint to this topic to receive notifications)")
    print("  3. Alarms will trigger when thresholds are exceeded")
    print("")

    return {
        'dashboard_url': dashboard_url,
        'dashboard_name': DASHBOARD_NAME,
        'alarms': alarms_created,
        'namespace': NAMESPACE
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Create CloudWatch drift monitoring')
    parser.add_argument('--region', default='us-east-1', help='AWS region')
    parser.add_argument('--endpoint', default='fraud-detection-endpoint', help='SageMaker endpoint name')
    parser.add_argument('--drift-threshold', type=float, default=0.10, help='Model drift threshold (default: 0.10 = 10%%)')
    parser.add_argument('--psi-threshold', type=float, default=0.2, help='PSI data drift threshold (default: 0.2)')
    parser.add_argument('--evaluation-periods', type=int, default=1, help='Alarm evaluation periods')

    args = parser.parse_args()

    result = create_cloudwatch_monitoring(
        region=args.region,
        endpoint_name=args.endpoint,
        drift_threshold=args.drift_threshold,
        psi_threshold=args.psi_threshold,
        evaluation_periods=args.evaluation_periods
    )

    sys.exit(0 if result else 1)
