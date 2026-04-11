# Data Directory

This directory holds the CSV datasets used by the fraud detection pipeline. The files are not checked into git — generate them locally using the included script.

## Generating the Data

```bash
# Generate all three CSVs
python data/generate_datasets.py

# Rename to drop the "generated_" prefix
cd data
mv generated_creditcard_predictions_final.csv creditcard_predictions_final.csv
mv generated_creditcard_drifted.csv creditcard_drifted.csv
mv generated_creditcard_ground_truth.csv creditcard_ground_truth.csv
```

You can also generate individual files:

```bash
python data/generate_datasets.py --predictions
python data/generate_datasets.py --drifted
python data/generate_datasets.py --ground-truth
```

## Dataset Descriptions

### creditcard_predictions_final.csv

The primary training and inference dataset.

- Rows: 284,807
- Columns: 35 (30 features + transaction_id, transaction_timestamp, transaction_amount, fraud_prediction, fraud_probability, customer_gender, is_fraud)
- Fraud rate: ~0.17% (highly imbalanced, realistic for credit card fraud)
- Features: PCA-normalised floats drawn from N(0,1), plus real-valued transaction_amount (lognormal, mean ~$88)

When to use:
- Training the XGBoost model via SageMaker Pipelines (`notebooks/1_training_pipeline.ipynb`)
- Uploading to S3 and migrating into Athena Iceberg tables (`main.py setup --migrate-data`)
- Baseline data for drift detection comparisons
- Testing inference endpoints with representative data

### creditcard_drifted.csv

A smaller dataset with intentional feature drift applied, used to validate that the monitoring system detects distribution shifts.

- Rows: 5,000 (configurable via `drift_generation.num_samples` in `src/config/config.yaml`)
- Columns: Same 35 as predictions_final
- Drift applied to 5 key features (default values shown, **all configurable in `src/config/config.yaml`**):
  - `transaction_amount`: +40% multiplicative increase (simulates inflation)
  - `transaction_timestamp`: +50,000 additive shift (simulates future time period)
  - `distance_from_home_km`: 2x multiplicative increase (simulates travel/remote transactions)
  - `velocity_score`: 1.5x multiplicative increase (simulates more active users)
  - `num_transactions_24h`: +3 additive shift (simulates higher transaction frequency)

**To adjust drift amounts:** Edit `src/config/config.yaml` under `drift_generation.default_drift`, then regenerate:
```bash
python src/drift_monitoring/generate_drift_dataset.py
```

When to use:
- Testing drift detection in `notebooks/2a_inference_monitoring.ipynb`
- Validating Evidently drift reports and PSI/KS thresholds
- Comparing drifted vs. baseline runs in MLflow
- Verifying SNS alerting triggers on drift

### creditcard_ground_truth.csv

Simulated ground truth confirmations with realistic delays and windowed structure.

- Rows: 50,000 (10 windows of 5,000 samples each)
- Columns: 36 (30 features + transaction_id, prediction_timestamp, window_id, transaction_timestamp, transaction_amount, ground_truth_fraud, observed_fraud, fraud_probability)
- `ground_truth_fraud`: the actual fraud label
- `observed_fraud`: ground truth with ~5% noise (simulates investigation errors)
- `window_id`: groups samples into time windows (1-10)

When to use:
- Testing ground truth integration and Athena MERGE updates
- Evaluating model performance metrics (accuracy, precision, recall, F1) with confirmed labels
- Simulating delayed fraud confirmations in `src/drift_monitoring/update_ground_truth.py`
- Validating the monitoring pipeline's ability to track performance over time

## Column Reference

The 30 feature columns map to fraud detection concepts:

| Feature Group | Columns |
|---|---|
| Transaction | transaction_hour, transaction_day_of_week, transaction_amount, transaction_type_code, transaction_timestamp |
| Customer | customer_age, customer_gender, customer_tenure_months, account_age_days |
| Geography | distance_from_home_km, distance_from_last_transaction_km, international_transaction, high_risk_country |
| Security | chip_transaction, pin_used, card_present, cvv_match, address_verification_match |
| Merchant | merchant_category_code, merchant_reputation_score |
| Behavior | num_transactions_24h, num_transactions_7days, avg_transaction_amount_30days, max_transaction_amount_30days, velocity_score, recurring_transaction, time_since_last_transaction_min, previous_fraud_incidents |
| Credit | credit_limit, available_credit_ratio |

## Reproducibility

The generator uses fixed random seeds (`RANDOM_STATE = 42` for predictions/ground truth, `123` for drift application) so repeated runs produce identical output.
