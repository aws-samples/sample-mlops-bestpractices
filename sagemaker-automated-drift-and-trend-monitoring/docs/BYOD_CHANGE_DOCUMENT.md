# "Bring Your Own Dataset" — Change Document

## Summary

This document catalogs every change required to switch the project from the
**Kaggle credit-card fraud detection** dataset to the **UCI Bank Marketing**
dataset (or any other binary-classification dataset). The intent is to identify
gaps in the `dataset_schema.yaml` abstraction layer — i.e., places that
**should** be schema-driven but today are hardcoded to fraud-specific values.

## Current Architecture

The project has a clean abstraction layer:

```
dataset_schema.yaml  →  schema.py  →  (downstream code)
```

`schema.py` provides: `feature_names()`, `target_column()`, `identifier_column()`,
`csv_column_order()`, `athena_feature_ddl()`, `cast_expr()`.

**The good news:** The schema layer itself is well-designed. Athena DDL
generation, the seed-table SQL, and feature-list propagation all use `schema.py`.

**The bad news:** 493 fraud-specific references exist across 40 Python files in
`src/`, plus 65 in CloudFormation templates, plus 477 in notebooks. Many of these
are **naming** (not schema) — resource identifiers like `fraud-detector-endpoint`,
`fraud-detection-drift-monitor`, `fraud_detection` database name — but some are
**semantic** (hardcoded column names, domain-specific logic).

---

## Change Classification

| Category | Effort | Description |
|---|---|---|
| ✅ Schema-driven (just works) | Zero | Code already reads from `schema.py` |
| 🔧 Config rename | Low | Change a string in `config.yaml` / `.env` |
| ⚠️ Hardcoded logic | Medium | Requires code changes |
| 🔴 Structural | High | Needs refactoring for dataset-agnostic design |

---

## Layer 1: Schema-Driven Changes (EASY — edit one file)

### File: `src/config/dataset_schema.yaml`

Replace the entire feature list, target, and identifier with the Bank Marketing
schema:

| Field | Fraud Detection (current) | Bank Marketing (new) |
|---|---|---|
| `identifier_column` | `transaction_id` | Synthetic ID (dataset has none) |
| `timestamp_column` | `transaction_timestamp` | N/A (no timestamp feature) |
| `target_column` | `is_fraud` (boolean) | `y` (boolean: subscribed yes/no) |
| `features` (count) | 30 numeric features | 20 features (mixed categorical → encoded) |
| `auxiliary_columns` | `fraud_prediction`, `fraud_probability` | None needed |

**What propagates automatically from this one change:**
- Athena table DDL (all 7 tables) — via `schema.athena_feature_ddl()`
- Training pipeline feature selection — via `schema.feature_names()`
- Drift Lambda feature list — via `TRAINING_FEATURES = schema.feature_names()`
- CSV column ordering — via `schema.csv_column_order()`

---

## Layer 2: Config Renames (LOW effort — edit `config.yaml` + `.env`)

### File: `src/config/config.yaml`

These are **naming** changes — the code reads these constants and passes them
through. Changing them here propagates everywhere:

| Config key | Current value | New value (example) |
|---|---|---|
| `project.name` | `fraud-detection-monitoring` | `bank-marketing-monitoring` |
| `mlflow.experiment_name` | `credit-card-fraud-detection-training` | `bank-marketing-training` |
| `mlflow.inference_experiment_name` | `credit-card-fraud-detection-inference` | `bank-marketing-inference` |
| `mlflow.batch_experiment_name` | `credit-card-fraud-detection-batch` | `bank-marketing-batch` |
| `mlflow.monitoring_experiment_name` | `credit-card-fraud-detection-monitoring` | `bank-marketing-monitoring` |
| `mlflow.model_name` | `fraud-detection` | `bank-marketing` |
| `athena.database` | `fraud_detection` | `bank_marketing` |
| `s3.prefix` | `fraud-detection/` | `bank-marketing/` |
| `endpoint.name` | `fraud-detector-endpoint` | `bank-marketing-endpoint` |
| `sqs.monitoring_results_queue` | `fraud-monitoring-results` | `bank-marketing-results` |

**Partial propagation issue:** These config values propagate to most code, BUT
several files have **hardcoded fallbacks** that won't update:

- `lambda_drift_monitor.py` line 50: `os.getenv('ATHENA_DATABASE', 'fraud_detection')`
- `lambda_drift_monitor.py` line 53: `os.getenv('MODEL_PACKAGE_GROUP', 'fraud-detection')`
- `lambda_drift_monitor.py` line 1067: `'fraud-detector-endpoint'`
- `config.py` line 225: fallback `"fraud-detection-monitoring"`
- `config.py` line 460: fallback `"fraud-detector-endpoint"`

These fallbacks exist because the Lambda runs in a container where `config.yaml`
isn't available — it relies on environment variables injected at deploy time. As
long as the deploy script passes the right env vars, these fallbacks never fire.
But they're misleading for a BYOD workshop.

---

## Layer 3: Hardcoded Logic That MUST Be Changed (MEDIUM effort)

### 3.1 Inference Handler (`src/train_pipeline/inference_code/inference.py`)

**Issues:**
- Line 181: `'transaction_id': str(original_data.iloc[idx].get('transaction_id', ''))`
  — hardcoded column lookup
- Line 182: `'transaction_amount': float(original_data.iloc[idx].get('transaction_amount', 0))`
  — hardcoded column lookup
- Lines 150-151: `"non_fraud"` / `"fraud"` response keys
- Lines 162-163: `fraud_prob` variable naming (cosmetic but confusing)
- Lines 173-174: `'probability_fraud'` / `'probability_non_fraud'` logged to SQS
- Line 41-49: `get_prediction_bucket()` with fraud-specific thresholds

**Fix:** Refactor to read identifier/amount columns from `schema.py` (which is
already bundled into the inference container). Rename response keys to generic
`"positive"` / `"negative"` or make them configurable.

### 3.2 Lambda Drift Monitor (`src/drift_monitoring/lambda_drift_monitor.py`)

**Issues:**
- Line 81: `PROBABILITY_COLUMN = os.getenv('PROBABILITY_COLUMN', 'probability_fraud')`
- Line 536: `{PROBABILITY_COLUMN} AS probability_fraud` — aliases to `probability_fraud`
  regardless of actual column name
- Line 556: `current_df['probability_fraud']` — hardcoded DataFrame column access
- Line 696: `subject = "🚨 ML Model Drift Alert - Fraud Detection"`
- Line 902: `mlflow.set_experiment("fraud-detection-drift_monitoring")`
- Line 1067: endpoint name fallback

**Fix:** The aliasing in the SQL query (line 536) forces `probability_fraud` as
the column name throughout the function. This could be changed to a generic alias
like `probability_positive` or made configurable. The alert subject and MLflow
experiment name should read from config/env.

### 3.3 Governance Dashboard (`src/governance/create_governance_dashboard.py`)

**Issues:**
- Line 956: `AVG(i.{PROBABILITY_COLUMN}) as avg_fraud_prob`
- Line 995: `{'Name': 'avg_fraud_prob', 'Type': 'DECIMAL'}`
- Lines 1350-1370: SQL with `predicted_fraud`, `actual_fraud`, `g.actual_fraud`
  column references
- Lines 1385-1386: QuickSight column declarations for `predicted_fraud`, `actual_fraud`

**Fix:** These SQL queries join `inference_responses` with `ground_truth_updates`
using fraud-specific column names. The ground-truth table itself stores
`actual_fraud` as a column name — this comes from `simulate_ground_truth_from_athena.py`.
Both must be updated together to use the generic target column name from schema.

### 3.4 Visualization Utils (`src/utils/visualization_utils.py`)

**Issues:**
- Lines 63-64: `probability_fraud FROM fraud_detection.inference_responses` (hardcoded SQL)
- Lines 157-158: Same pattern repeated
- Lines 285-288: `fraud_predictions`, `avg_fraud_prob` aliases
- Lines 402, 492: `FROM fraud_detection.inference_responses` (hardcoded DB name)

**Fix:** Replace hardcoded `fraud_detection` with `{ATHENA_DATABASE}` and
`probability_fraud` with `{PROBABILITY_COLUMN}` from config.

### 3.5 Ground Truth Simulator (`src/drift_monitoring/simulate_ground_truth_from_athena.py`)

**Issues:**
- Lines 59-60: `fraud_confirmation_days`, `non_fraud_confirmation_days` parameters
- Line 159: `df['actual_fraud'] = df['prediction'].astype(bool)` — hardcoded column name
- Lines 173-174, 181-182: `actual_fraud` column throughout
- Lines 185-188: Fraud-specific logging messages

**Fix:** Replace `actual_fraud` with a generic column name (e.g.,
`actual_{target_column}` or just `actual_target`). The confirmation-day concept
is fraud-specific (fraud cases confirmed faster than non-fraud); for bank
marketing, this would be "did they subscribe?" which might not need different
timelines.

### 3.6 Drift Dataset Generator (`src/drift_monitoring/generate_drift_dataset.py`)

**Issues:**
- Lines 66-70: Hardcoded `_FEATURE_DESCRIPTIONS` dict with fraud-specific features
  (`transaction_amount`, `distance_from_home_km`, `velocity_score`, etc.)
- Line 125: Non-negative constraint for fraud-specific features

**Fix:** The drift parameters should come from `config.yaml` (under a
`drift_generation` section) rather than being hardcoded for fraud features.
For bank marketing, you'd drift `age`, `duration`, `campaign`, etc.

### 3.7 Data Downloader (`src/setup/download_kaggle_dataset.py`)

**Issues:**
- Entire file is the Kaggle credit-card fraud fetcher with a
  `KAGGLE_COLUMN_MAP` that maps V1..V28 → fraud feature names

**Fix:** This file must be **replaced entirely** for a new dataset. The
`dataset_schema.yaml` header comment already says: "Point your own loader
(replacing `src/setup/download_kaggle_dataset.py`) at your source CSV/table."
For bank marketing, this would download from UCI and apply label encoding.

---

## Layer 4: CloudFormation & Infrastructure (LOW-MEDIUM effort)

### File: `cloudformation/drift-monitoring-infra.yaml`

24 fraud-specific strings, all as CloudFormation **parameter defaults**:

| Parameter | Current Default | Impact |
|---|---|---|
| `ProjectName` | `fraud-detection` | Derives resource names |
| `AthenaDatabase` | `fraud_detection` | Lambda env var |
| `EndpointName` | `fraud-detector-endpoint` | Lambda env var |
| `MonitoringSqsQueue` | `fraud-monitoring-results` | SQS queue name |
| `MonitoringWriterLambda` | `fraud-monitoring-results-writer` | Lambda name |
| `DriftAlertsSNS` | `fraud-detection-drift-alerts` | SNS topic |
| `DriftMonitorRole` | `fraud-detection-drift-monitor-role` | IAM role |
| `CloudWatchNamespace` | `FraudDetection/DriftMonitoring` | Metrics namespace |
| `CloudWatchDashboard` | `FraudDetection-DriftMonitoring` | Dashboard name |

**Plus hardcoded alarm names:** `FraudDetection-DataDrift-PSI`,
`FraudDetection-ModelDrift-ROCAUCDEGRADATION`, etc.

**Fix:** The parameter defaults can be overridden at deploy time via the
`--parameter-overrides` CLI flag. However, the **alarm names** and **dashboard
markdown text** on lines 354-517 are hardcoded resource names within the
template — these need `!Sub` with `${ProjectName}` interpolation.

### File: `cloudformation/sagemaker-mlflow-setup.yaml`

11 references — mostly in the same parameter-defaults pattern. Lower priority
since this stack is deployed once.

---

## Layer 5: Notebooks (MEDIUM effort for 3 & 4 specifically)

### `3_inference_monitoring.ipynb`

- Line 120: `ENDPOINT_NAME = 'fraud-detector-endpoint'`
- Lines 204-214: All resource names hardcoded with `fraud-*` fallbacks
- Line 221: `os.environ.setdefault('MODEL_PACKAGE_GROUP', 'fraud-detection')`
- Line 40-41: Markdown text referencing fraud detection

**Fix:** These should all read from `config.py` constants (which they partially
do via imports at the top). The hardcoded fallbacks in `os.environ.setdefault()`
calls should be removed or replaced with config references.

### `4_governance_dashboard.ipynb`

- This notebook calls `src/governance/create_governance_dashboard.py` which has
  the hardcoded SQL. The notebook itself is fairly clean — it reads from config.
- Fix the underlying `create_governance_dashboard.py` (Layer 3.3 above) and the
  notebook works.

---

## Summary: What Actually Needs to Change

### Tier 1 — Must change (blocks the workshop path)

| # | File | Change type | Effort |
|---|---|---|---|
| 1 | `dataset_schema.yaml` | Rewrite for Bank Marketing features | Low |
| 2 | `config.yaml` | Rename project/endpoint/DB/experiment names | Low |
| 3 | `download_kaggle_dataset.py` | Replace with UCI downloader | Medium |
| 4 | `inference.py` (inference handler) | Remove hardcoded `transaction_id`/`probability_fraud` | Medium |
| 5 | `generate_drift_dataset.py` | Replace fraud feature drift params with bank marketing ones | Medium |
| 6 | `simulate_ground_truth_from_athena.py` | Replace `actual_fraud` with schema-driven column | Medium |
| 7 | `drift-monitoring-infra.yaml` | Parameterize alarm names / dashboard text | Low |

### Tier 2 — Should change (cosmetic / confusing if left)

| # | File | Change type | Effort |
|---|---|---|---|
| 8 | `lambda_drift_monitor.py` | Change `probability_fraud` alias, alert subject, MLflow experiment | Medium |
| 9 | `create_governance_dashboard.py` | Update SQL column aliases (`predicted_fraud` → generic) | Medium |
| 10 | `visualization_utils.py` | Replace hardcoded SQL with config-driven queries | Medium |
| 11 | `pipeline.py` | Rename class `FraudDetectionPipeline`, job names | Low |
| 12 | Notebook 3 | Replace hardcoded fallback resource names | Low |

### Tier 3 — Nice to have (won't block functionality)

| # | File | Change type |
|---|---|---|
| 13 | All fallback strings in `config.py` | Rename from `fraud-*` to generic |
| 14 | Comments / docstrings / README | Update domain language |
| 15 | `batch_transform.py`, `test_endpoint.py` | Rename references |

---

## Recommendation for "Bring Your Own Dataset" Workshop

The current codebase is **~70% schema-driven** but the remaining 30% creates a
significant barrier. To make the BYOD path smooth, the codebase needs:

1. **Immediate fix (Tier 1):** ~7 files with targeted changes to make the system
   functionally work with a new dataset.

2. **Ideal refactoring:** Extract all domain-specific naming into `config.yaml`
   so that changing the project name + dataset schema is genuinely a
   two-file-edit workflow. Key gaps:
   - The inference handler should get column names from schema (already bundled)
   - The drift generator should read feature-specific drift params from config
   - Ground truth simulator should use `schema.target_column()` for its output
   - All SQL in `visualization_utils.py` and `create_governance_dashboard.py`
     should use config constants instead of string literals

3. **Estimated effort for a clean BYOD path:** 2-3 days of focused refactoring
   to make it truly a "change 2 files and go" experience. Without refactoring,
   switching datasets requires touching ~12 files with domain knowledge of what
   each hardcoded value means.
