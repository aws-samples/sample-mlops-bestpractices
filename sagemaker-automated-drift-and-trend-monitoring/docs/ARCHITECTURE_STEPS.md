# Architecture Flow: 11-Step Process

Based on `docs/MetaMonitoring.png` - Complete MLOps monitoring workflow

---

## 🔵 Training Pipeline (Steps 1-5)

### Step 1: Data Ingestion
**Component:** Amazon S3 → Training Data  
**Description:** Raw credit card transaction data is stored in Amazon S3. This serves as the source of truth for the training pipeline. The dataset contains 284K transactions with 33 features (transaction amount, customer age, merchant info, etc.) and fraud labels.

**Key Details:**
- Source: `s3://bucket/training_data/`
- Format: Parquet files
- Schema: 33 features + `is_fraud` label
- Volume: ~284K transactions

---

### Step 2: Feature Engineering
**Component:** Training Data → PySpark Processing  
**Description:** Raw data is transformed using PySpark on AWS Glue. This step handles feature engineering, data cleaning, missing value imputation, and creates derived features like velocity scores and transaction patterns.

**Key Details:**
- Platform: AWS Glue (PySpark)
- Transformations:
  - Calculate transaction velocity scores
  - Compute 24h/7-day transaction counts
  - Distance from home calculations
  - Time-based feature extraction
- Output: Processed training dataset ready for model training

---

### Step 3: Model Training
**Component:** PySpark Processing → XGBoost Training  
**Description:** Processed features are fed into XGBoost training on SageMaker. The model learns patterns to distinguish fraudulent from legitimate transactions. Training uses gradient boosting with hyperparameters optimized for fraud detection (high precision to minimize false positives).

**Key Details:**
- Algorithm: XGBoost
- Platform: Amazon SageMaker Training
- Hyperparameters: Configured in pipeline definition
- Training time: ~5-10 minutes
- Outputs: Model artifacts + evaluation metrics

---

### Step 4: Model Evaluation & Registration
**Component:** XGBoost Training → Evaluate → MLflow App  
**Description:** Trained model is evaluated against test data to calculate accuracy, precision, recall, F1-score, and AUC. If metrics pass quality gates (e.g., accuracy > 85%), the model is registered in MLflow with versioning and metadata (run ID, parameters, metrics).

**Key Details:**
- Quality gates: Accuracy threshold, precision threshold
- Metrics logged: Accuracy, precision, recall, F1, AUC, confusion matrix
- MLflow registry: Model versioned and tagged
- Artifacts: Model file, evaluation report, feature importance

---

### Step 5: Model Deployment
**Component:** Evaluate → SageMaker Endpoint  
**Description:** Approved model is deployed to a SageMaker real-time inference endpoint. The endpoint provides a REST API for prediction requests with configurable instance types (e.g., ml.m5.xlarge) and auto-scaling based on traffic.

**Key Details:**
- Endpoint name: `fraud-detector-endpoint-<timestamp>`
- Instance: ml.m5.xlarge (configurable)
- Auto-scaling: Min 1, Max 3 instances
- Latency: <100ms p99
- Output: Prediction (0/1) + fraud probability

---

## 🟠 Inference Monitoring Pipeline (Steps 6-11)

### Step 6: Real-Time Inference Logging
**Component:** SageMaker Endpoint → SQS → Lambda (Logger) → Athena (inference_responses)  
**Description:** Every prediction request is logged asynchronously via SQS to avoid blocking inference. Lambda consumes SQS messages and batch-writes predictions to Athena Iceberg table. This creates a complete audit trail of all model predictions with input features and outputs.

**Key Details:**
- Flow: Endpoint → SQS → Lambda → Athena
- Latency: Async (no impact on inference)
- Batch size: 100 records per Lambda invocation
- Table: `fraud_detection.inference_responses`
- Retention: Partitioned by day for efficient queries
- Fields logged:
  - `inference_id` (UUID)
  - `input_features` (JSON)
  - `prediction` (0/1)
  - `probability_fraud` (0.0-1.0)
  - `request_timestamp`
  - `endpoint_name`
  - `model_version`

---

### Step 7: Ground Truth Collection
**Component:** Ground Truth Simulator → Athena (ground_truth_updates)  
**Description:** Fraud confirmations arrive days/weeks after transactions (delayed labels). In production, this comes from fraud investigation teams. In dev/test, a simulator generates realistic delayed confirmations with correlation to predictions. These updates are written to `ground_truth_updates` table.

**Key Details:**
- Delay: 1-30 days after transaction
- Source: Manual fraud reviews, chargebacks, customer disputes
- Simulator: `simulate_ground_truth_from_athena.py` (dev/test only)
- Table: `fraud_detection.ground_truth_updates`
- Fields:
  - `inference_id` (links to prediction)
  - `actual_fraud` (TRUE/FALSE)
  - `confirmation_timestamp`
  - `days_since_prediction`

---

### Step 8: Ground Truth Backfill
**Component:** Ground Truth Updates → MERGE → Inference Responses  
**Description:** Batch job (Lambda or scheduled script) JOINs `ground_truth_updates` with `inference_responses` on `inference_id` and backfills the `ground_truth` column. This populates NULL values with confirmed fraud status, enabling performance monitoring.

**Key Details:**
- Frequency: Daily batch (3 AM UTC)
- Script: `update_ground_truth.py`
- SQL: MERGE operation on Iceberg table
- Updates: Sets `ground_truth`, `ground_truth_timestamp`, `days_to_ground_truth`
- Idempotent: Safe to re-run

```sql
MERGE INTO inference_responses AS target
USING ground_truth_updates AS source
ON target.inference_id = source.inference_id
WHEN MATCHED AND target.ground_truth IS NULL THEN
  UPDATE SET
    target.ground_truth = source.actual_fraud,
    target.ground_truth_timestamp = source.confirmation_timestamp
```

---

### Step 9: Scheduled Drift Detection
**Component:** EventBridge (2 AM) → Lambda (Drift Monitor) → Evidently AI  
**Description:** EventBridge triggers Lambda daily at 2 AM UTC. Lambda queries recent inference data (last 7 days) and baseline training data, then runs Evidently AI drift analysis. Detects distribution shifts in input features using statistical tests (Kolmogorov-Smirnov, PSI).

**Key Details:**
- Schedule: 2 AM UTC daily (configurable in config.yaml)
- Trigger: EventBridge rule
- Baseline: Random sample from `training_data` (5000 rows)
- Current: Recent predictions from `inference_responses` (last 7 days)
- Tests: Data drift (feature distributions) + Classification drift (model performance)
- Thresholds: Configured in `config.yaml`
  - `data_drift_share`: 0.3 (30% features drifted = alert)
  - `model_drift_threshold`: 0.05 (5% accuracy drop = alert)

---

### Step 10: Drift Report Generation & Storage
**Component:** Evidently AI → MLflow → Athena (monitoring_responses)  
**Description:** Evidently generates interactive HTML reports with drift visualizations (feature distributions, drift scores, statistical tests). Reports are logged to MLflow as artifacts. Summary metrics (drift detected, feature count, drift scores) are written to `monitoring_responses` Athena table for historical tracking.

**Key Details:**
- Report format: Interactive HTML (Evidently)
- Storage: MLflow artifacts + S3
- Metrics logged:
  - `drift_detected` (boolean)
  - `drifted_columns_count` (int)
  - `drifted_columns_share` (float, e.g., 0.35 = 35%)
  - Per-feature drift scores (PSI values)
- Table: `fraud_detection.monitoring_responses`
- Partitioned by: day(check_timestamp)

**Sample monitoring_responses record:**
```json
{
  "check_id": "drift_check_20260504_0200",
  "check_timestamp": "2026-05-04T02:00:00Z",
  "drift_detected": true,
  "drifted_columns_count": 12,
  "drifted_columns_share": 0.36,
  "feature_drift_scores": {
    "credit_limit": 74.47,
    "merchant_category_code": 28.25,
    "transaction_amount": 12.34
  },
  "mlflow_run_id": "abc123...",
  "alert_sent": true
}
```

---

### Step 11: Governance Dashboard Visualization
**Component:** QuickSight → Query → Athena (inference_responses + monitoring_responses)  
**Description:** QuickSight dashboard provides unified view of model health. Queries Athena tables directly (no SPICE caching) to show real-time metrics: inference volume, drift trends, model accuracy, feature-level drift scores. EventBridge triggers Lambda at 3 AM UTC to refresh calculated fields for performance optimization.

**Key Details:**
- Platform: Amazon QuickSight
- Data source: Athena direct query
- Refresh: EventBridge + Lambda (3 AM UTC) for calculated fields
- Dashboards:
  1. **Inference Overview**: Volume trends, latency, error rates
  2. **Drift Timeline**: Per-feature drift scores over time
  3. **Model Performance**: Accuracy, precision, recall (where ground_truth available)
  4. **Alert History**: Drift events and notifications

**Key Visualizations:**
- **Drift Score Trendlines**: Time-series of PSI scores per feature
  - Identify sudden spikes (e.g., credit_limit = 74.47)
  - Distinguish temporary anomalies from permanent shifts
- **Feature Ranking**: Top drifting features by score
- **Coverage Metrics**: % predictions with ground truth
- **Performance Degradation**: Accuracy over time

**Sample Queries:**
```sql
-- Drift timeline
SELECT 
  DATE_TRUNC('day', check_timestamp) as day,
  drifted_columns_count,
  drifted_columns_share
FROM monitoring_responses
WHERE check_timestamp >= CURRENT_DATE - INTERVAL '30' DAY
ORDER BY day;

-- Model accuracy (where ground truth available)
SELECT 
  DATE_TRUNC('day', request_timestamp) as day,
  COUNT(*) as predictions,
  AVG(CASE WHEN prediction = ground_truth THEN 1 ELSE 0 END) as accuracy
FROM inference_responses
WHERE ground_truth IS NOT NULL
  AND request_timestamp >= CURRENT_DATE - INTERVAL '30' DAY
GROUP BY DATE_TRUNC('day', request_timestamp);
```

---

## 🔄 Supporting Flows

### Alert Notifications (Step 9 → SNS)
**Component:** Lambda Drift Monitor → SNS → Email/SMS  
**Description:** When drift exceeds thresholds, Lambda publishes to SNS topic. Subscribers (email, SMS, Slack via webhook) receive alerts with summary: drift detected, feature count, top drifting features, MLflow report link.

**Alert Payload:**
```json
{
  "alert_type": "DATA_DRIFT_DETECTED",
  "severity": "HIGH",
  "check_timestamp": "2026-05-04T02:00:00Z",
  "drift_share": 0.36,
  "drifted_features": [
    {"feature": "credit_limit", "score": 74.47},
    {"feature": "merchant_category_code", "score": 28.25}
  ],
  "mlflow_report_url": "https://mlflow.../artifacts/drift_report.html",
  "recommended_action": "Investigate data pipeline or consider model retraining"
}
```

---

## 📊 Summary Table

| Step | Component | Purpose | Frequency | Output |
|------|-----------|---------|-----------|--------|
| 1 | S3 → Training Data | Data ingestion | One-time/batch | Raw dataset |
| 2 | PySpark Processing | Feature engineering | Per training run | Processed features |
| 3 | XGBoost Training | Model learning | Per training run | Model artifacts |
| 4 | Evaluate → MLflow | Quality gate | Per training run | Registered model |
| 5 | Deploy Endpoint | Production serving | Per deployment | Live API endpoint |
| 6 | Inference Logging | Audit trail | Real-time | inference_responses |
| 7 | Ground Truth Collection | Label gathering | Delayed (1-30d) | ground_truth_updates |
| 8 | Ground Truth Backfill | Merge labels | Daily batch (3 AM) | Updated inference_responses |
| 9 | Drift Detection | Monitor data shift | Daily (2 AM) | Drift metrics |
| 10 | Report Generation | Drift visualization | Daily (2 AM) | MLflow reports + monitoring_responses |
| 11 | Dashboard | Unified view | Real-time queries | QuickSight visuals |

---

## 🎯 Key Architecture Decisions

### Why This Design?

1. **Async Inference Logging (Step 6):**
   - SQS decouples logging from inference (no latency impact)
   - Lambda batch writes reduce Athena costs
   - Iceberg table enables efficient updates (Step 8)

2. **Delayed Ground Truth (Steps 7-8):**
   - Real-world fraud confirmations take days/weeks
   - Separate `ground_truth_updates` table tracks confirmations
   - MERGE operation backfills without duplicating data

3. **Scheduled Monitoring (Step 9):**
   - 2 AM UTC = low traffic time
   - Daily checks balance cost vs freshness
   - EventBridge eliminates need for cron servers

4. **MLflow Central Hub (Steps 4, 10):**
   - Single source of truth for experiments
   - Links training runs to monitoring runs
   - Stores drift reports as artifacts

5. **Direct Athena Queries (Step 11):**
   - No SPICE caching = always fresh data
   - Partitioned tables keep queries fast
   - Iceberg format enables efficient scans

---

## 🚀 Operational Metrics

**End-to-End Latency:**
- Inference: <100ms (Step 5)
- Logging: <5s (Step 6, async)
- Ground truth backfill: <2 min (Step 8, daily)
- Drift detection: ~5 min (Step 9, daily)
- Dashboard refresh: <30s (Step 11, on-demand)

**Cost (Monthly):**
- SageMaker Endpoint: ~$50-100 (depends on instance type)
- Lambda: ~$5 (millions of invocations free tier)
- Athena: ~$10 (query volume dependent)
- S3: ~$5 (storage + lifecycle policies)
- QuickSight: $0-24 (Reader/Author licenses)
- **Total: ~$30-150/month** (vs $200+ for managed alternatives)

**Scalability:**
- Inference: Auto-scales 1-3 instances (Step 5)
- Logging: SQS + Lambda scales to 1000s/sec (Step 6)
- Monitoring: Athena scales to petabytes (Steps 8-11)

---

## 📝 Configuration Files

All steps configurable via `config.yaml`:

```yaml
monitoring:
  # Step 9: Drift detection schedule
  drift_check_schedule: "cron(0 2 * * ? *)"  # 2 AM UTC daily
  
  # Step 9: Drift thresholds
  drift_thresholds:
    data_drift_share: 0.3  # 30% features drifted = alert
    model_drift_threshold: 0.05  # 5% accuracy drop = alert
  
  # Step 6: Logging batch size
  inference_logging:
    batch_size: 100
    max_wait_seconds: 60
  
  # Step 11: Dashboard refresh
  dashboard_refresh_schedule: "cron(0 3 * * ? *)"  # 3 AM UTC daily
```

---

## 🔗 Inter-Step Dependencies

```
Training (1-5) MUST complete before Inference (6-11)
Step 5 (Endpoint) enables Step 6 (Logging)
Step 6 (Logging) enables Step 7-8 (Ground Truth)
Step 8 (Backfill) enables Step 10 (Performance metrics)
Step 9 (Drift Check) reads from Step 1 (Training data) + Step 6 (Inference)
Step 11 (Dashboard) reads from Steps 6, 8, 10
```

**Critical Path:** Steps 1-5 must succeed for any monitoring to work. Steps 6-11 are production monitoring and can be deployed/updated independently.
