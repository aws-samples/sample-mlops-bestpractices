# QuickSight Setup Guide

This guide shows you how to set up Amazon QuickSight for the MLOps Governance Dashboard in a new AWS account.

---

## Prerequisites

- ✅ CloudFormation stack deployed
- ✅ Athena database with `fraud_detection` tables populated
- ✅ AWS account with admin access

---

## Setup Steps

### Step 1: Sign Up for QuickSight

1. Go to AWS Console → **Amazon QuickSight**
2. Click **"Sign up for QuickSight"**
3. Choose **Enterprise Edition** (required for programmatic dashboard creation)
4. Configure:
   - **Authentication**: IAM Identity Center or IAM only
   - **Account name**: Choose any name (e.g., `fraud-detection-monitoring`)
   - **Email**: Your admin email
5. Click **"Finish"** and wait 2-3 minutes

![QuickSight Sign Up](./Mlops-1.png)

---

### Step 2: Configure Regions in .env

QuickSight supports different regions for identity (where you signed up) and data (where your S3/Athena data lives).

Add to your `.env` file:

```bash
# The region where your data lives (S3, Athena)
AWS_DEFAULT_REGION=us-west-2

# The region where you signed up for QuickSight
QUICKSIGHT_IDENTITY_REGION=us-west-2
```

**Note**: If your data is in us-east-1, both values can be `us-east-1`. If you signed up for QuickSight in a different region, update `QUICKSIGHT_IDENTITY_REGION` accordingly.

These values are also configurable in `src/config/config.yaml`.

---

### Step 3: Run the Governance Dashboard Notebook

1. Open `notebooks/4_governance_dashboard.ipynb`
2. **Restart kernel** (to reload `.env` with the QuickSight region)
3. **Run all cells**

The notebook will create:
- Athena data source
- 5 datasets (inference, accuracy, drift, features, monitoring)
- Dashboard with visualizations

![Creating Data Source](./Mlops-2.png)

![Creating Datasets](./Mlops-3.png)

---

## Final Result

After running all notebooks (1-4), your QuickSight dashboard will look like this:

![QuickSight Dashboard](./Mlops-4.png)

The dashboard shows:
- **Model performance metrics**: Accuracy, precision, recall, F1, ROC-AUC
- **Drift detection trends**: Data drift and model drift over time
- **Inference metrics**: Volume and latency
- **Feature-level drift scores**: Per-feature drift analysis

---

## Reading the Dashboard: A Worked Drift Story

The dashboard has three sheets — **Model Drift**, **Data Drift**, and **Feature Drift** — laid out to be read top-down so you go from *symptom* → *cause* → *culprit* → *does it matter*. Full-page exports of each sheet are in this folder ([`Model_Drift_Trends.pdf`](Model_Drift_Trends.pdf), [`Data_Drift_Trends.pdf`](Data_Drift_Trends.pdf), [`Feature_Drift_Trends.pdf`](Feature_Drift_Trends.pdf)); the cropped panels below live in [`narrative/`](narrative/). All numbers are a real run of this repo against the demo data — the same Evidently output stored in `monitoring_responses`.

### Sheet 1 — Model Drift: is the model still healthy?

ROC-AUC dropped from the frozen baseline **~0.98** to **~0.48** (≈51% degradation); the model-drift verdict fired on every run. That's the alarm — the other two sheets explain it.

![ROC-AUC baseline vs current](narrative/model_rocauc_baseline_vs_current.png)

**Accuracy alone would have hidden this.** Sheet 1 also plots accuracy, precision, recall, and F1 as separate lines over time — and the pattern is the reason this dashboard tracks all four rather than accuracy alone:

![Accuracy stays high while precision, recall, F1 collapse](narrative/model_perf_accuracy_masks_zero_precision_recall.png)

Accuracy sits flat at **~0.85** across the window, while **precision, recall, and F1 all sit at 0** (they overlap into a single line along the x-axis). Under drifted inputs, the model has silently collapsed to always predicting the majority class (`non-fraud`): every prediction is `0`, so `TP = FP = 0` and precision / recall are structurally zero. Accuracy stays high because the ground truth is also mostly `non-fraud` — matching `0 → 0` counts as "correct" even though *zero actual fraud is being caught*. If the dashboard reported only accuracy, an on-call engineer would have concluded "0.85 — nothing wrong here" and missed the collapse entirely. This is why the drift-alert threshold in `config.yaml` is on `roc_auc_degradation`, not on accuracy.

> The collapse is a model-side effect, not a simulator artifact. The ground-truth simulator in `src/drift_monitoring/simulate_ground_truth_from_athena.py` (line 170) flips labels *symmetrically* — `~df.loc[error_indices, 'actual_fraud']` toggles both directions with equal probability. Precision and recall are zero because the *model* has stopped predicting `1` at all under drift (ROC-AUC ~0.48 confirms it's near-random on the raw scores), so there's nothing on the prediction side to flip *from* — every flip on the ground-truth side happens to a row already predicted `0`.

### Sheet 2 — Data Drift: did the inputs move?

**~93% of features drifted** (share ≈ 0.9–1.0, ~28 of 30 columns), with alerts firing. Broad input drift is the expected precursor to a performance collapse.

![Data drift share over time](narrative/data_drift_share_over_time.png)

### Sheet 3 — Feature Drift: which columns, ranked by `drift_magnitude`

Every column is ranked by **`drift_magnitude`** — a test-agnostic "× past threshold" number that stays comparable no matter which test Evidently picked per column. `num_transactions_24h` leads at **~11×**; the "Highest Drift Magnitude" KPI shows the worst single feature-run at **12.76×**.

![Top drifted features by magnitude](narrative/feat_top15_drifted.png)

The severity heatmap and distribution give the same ranking as bands (Low `<1.0` / Moderate `≥1.0` / Significant `≥3.0`):

![Feature drift severity distribution](narrative/feat_severity_dist.png)

> ⚠️ **These visuals use `drift_magnitude`, never raw `drift_score`.** Evidently picks a different test per column (KS p-value vs. Wasserstein distance), and those raw scores move in opposite directions on incomparable scales — so ranking or averaging raw `drift_score` across features is meaningless. Earlier versions of these screenshots (`DriftScore-Average.png`, `DriftScore-Sum.png`, `DriftScore-Variance-Population.png`, `Feature_drift_time.png`) plotted raw score and were dominated by a single mis-scaled feature reading ~1000×; they were **removed** because they told a false story.

### Sheet 3 + SHAP — does the drift actually matter?

Drift magnitude tells you *what changed*; SHAP (from `notebooks/7_optional_shap_explainability.ipynb`) tells you *what the model relies on*. Cross-referencing the two is what turns a list of anomalies into a prioritized action list. Two SHAP views help — the bar chart for **magnitude of impact**, and the beeswarm for **direction of impact**:

![SHAP feature importance — mean absolute magnitude](narrative/shap_feature_importance.png)

*Bar chart: mean absolute SHAP.* `account_age_days` (0.37) and `num_transactions_24h` (0.29) dominate — those are the two features the model relies on most on average. This chart is *unsigned*: it collapses positive and negative pushes into a single bar, so it says nothing about which direction a high or low value nudges predictions.

![SHAP beeswarm — signed SHAP by feature value](narrative/SHAP-beeswarm-plot.png)

*Beeswarm: signed SHAP per prediction, dots colored by feature value (blue = low, red = high).* For `account_age_days` (top row), red points cluster on the left and blue points cluster on the right — **older accounts push predictions away from fraud, newer accounts push toward fraud** (the inverse relationship you'd expect from domain intuition). `num_transactions_24h` (row 2) has color mixed at both extremes, indicating a non-monotonic relationship — consistent with it being a PCA-transformed feature (Kaggle's `V14`). The beeswarm answers "how will current input shifts translate into prediction shifts?", which the bar chart alone can't.

| Feature | Drift magnitude | SHAP importance (rank) | Verdict |
|---|---|---|---|
| `num_transactions_24h` | **×10.9** (worst) | **0.294 (#2)** | 🔴 High-risk — top-2 model driver *and* most-drifted → primary suspect for the AUC drop |
| `account_age_days` | ×2.1 | **0.365 (#1)** | 🔴 High-risk — the single most important feature is drifting |
| `customer_age`, `transaction_amount` | ×2.4, ×2.3 | 0.104, 0.083 (#7, #8) | 🟠 Important and drifting — monitor / retrain |
| `customer_gender` | ×4.9 (2nd-worst drift) | **0.001 (#30, last)** | 🟢 Low-risk noise — heavily drifted but the model ignores it |

**Takeaway:** ranked by drift alone you'd chase `customer_gender`; ranked by drift **×** importance the real culprit is `num_transactions_24h`. That intersection is the story the dashboard is built to tell.

> ⚠️ **QuickSight is not "discovering" what SHAP found — the alignment above is a property of the demo.**
> QuickSight only visualizes Evidently's distance / p-value statistics. It has no access to the model and no notion of feature importance. `num_transactions_24h` shows as the worst-drifted feature because the demo drift config (`src/config/config.yaml` → `drift_generation.default_drift`) applies an additive `shift: 1` to it, and that feature is a PCA component with near-unit standard deviation (Kaggle's `V14`), so a +1 shift becomes a ~10× Wasserstein magnitude. Change that config — bump another feature's factor, or lower this one's shift — and the QuickSight ranking will reshuffle on the very next drift run. **The SHAP chart won't move**, because SHAP is a property of the trained model and doesn't care about the drift-generation config. The demo aligns the two by construction; real production drift has no such guarantee. Use the *methodology* (cross-reference drift with SHAP before prioritizing), not the alignment shown here, as the takeaway.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| "QuickSight not subscribed" | Complete Step 1 (sign up) |
| "Access Denied" | Ensure your user has QuickSight Admin role |
| "Directory information not found" | Verify `QUICKSIGHT_IDENTITY_REGION` matches where you signed up |
| Dashboard shows "No data" | Verify Athena tables have data: `SELECT COUNT(*) FROM fraud_detection.inference_responses` |

---

## Cost

**QuickSight only:**

- **QuickSight Enterprise Author**: $24/user/month
- **Additional Readers** (read-only): $5/month (capped at $500/month per group)

**Full solution cost context:** QuickSight is only one line item in the total. The full stack (SageMaker AI **MLflow Apps** — usage-priced serverless, not the older EC2-backed MLflow Tracking Server, so no hourly `ml.*` instance charge for tracking — plus SageMaker Serverless Inference, Lambda, EventBridge, Athena, S3, SQS, SNS, and QuickSight) sits at **~$60/month directional** for a demo-scale deployment. See the [Cost section in the main README](../../../README.md#cost) and [`docs/ARCHITECTURE_STEPS.md`](../../ARCHITECTURE_STEPS.md#-configuration-files) for the per-service breakdown.

---

## Creating Custom Visualizations with Natural Language

QuickSight Q allows you to create custom charts by asking questions in plain English. This is especially powerful for ad-hoc drift analysis and trend exploration beyond the pre-built 32-visual dashboard.

![Build Visuals with Natural Language](./BuildVisualsWithNaturalLanguage.png)

### How to Use QuickSight Q

1. Open your QuickSight dashboard
2. Click the **Q search bar** at the top
3. Type your question in natural language
4. QuickSight automatically generates the appropriate visualization
5. Click **"Add to dashboard"** to save the visual

### Sample Natural Language Queries for Drift Analysis

#### **Drift Trend Analysis**

**Top drifting features over time:**
```
Show me a time series chart of the top 5 drifted features over the last 30 days 
with drift scores on the y-axis, grouped by day. Include a horizontal line at 
the 0.2 threshold. Add a second chart below showing daily prediction volume. 
Highlight days where more than 3 features crossed the drift threshold in red.
```

**What it generates:**
- Dual-axis chart with feature drift trends and volume correlation
- Threshold reference line to identify violations
- Color-coded alerts for high-drift days

---

**Feature drift severity distribution:**
```
Create a heatmap showing which features drifted each day over the last 14 days. 
Features on rows, dates on columns, color intensity by PSI score. Highlight 
any cell with PSI > 50 in dark red.
```

**What it generates:**
- Grid view of feature × date drift intensity
- Quick identification of chronic drifters vs. one-off spikes

---

**Drift score evolution by model version:**
```
Show drift score trends comparing model version 1 vs version 2 over the last 
7 days. Use separate lines for each version. Add a trend line to show if 
drift is improving or worsening after retraining.
```

**What it generates:**
- Multi-series comparison to validate retraining effectiveness
- Trend analysis showing drift trajectory

---

#### **Model Performance Analysis**

**ROC-AUC degradation timeline:**
```
Create a line chart showing ROC-AUC over the last 30 days with the baseline 
ROC-AUC as a horizontal reference line. Color the line green when above 
baseline, red when below. Add data labels on the 5 lowest points.
```

**What it generates:**
- Visual performance tracking with automatic baseline comparison
- Instant identification of worst-performing days

---

**Prediction distribution shift (Sankey diagram):**
```
Create a Sankey diagram showing how prediction bucket distributions shifted 
from the baseline week to last week. Show flows from training data buckets 
(very_low to very_high fraud probability) to current production buckets. 
Highlight any bucket that changed by more than 10 percentage points in red.
```

**What it generates:**
- Flow visualization of prediction score migration
- Identifies concept drift (score distribution shifts)
- Red highlighting for significant bucket changes (>10pp)

**Example use case:** If your "high confidence fraud" bucket (0.8-1.0 probability) was 5% of training predictions but is now 15% in production, the Sankey will show a thick red flow, indicating the model is predicting fraud more aggressively.

---

**Ground truth coverage and accuracy:**
```
Show a combo chart with ground truth coverage percentage (bars) and model 
accuracy (line) over the last 30 days. Add a warning annotation on days 
where coverage drops below 20%.
```

**What it generates:**
- Dual metric visualization showing data quality vs. performance
- Alerts when model drift metrics become unreliable (low coverage)

---

#### **Feature-Level Deep Dives**

**Credit limit drift investigation:**
```
Show me the credit_limit feature distribution from training data vs last 7 days 
of production data as overlapping histograms. Include mean, median, and standard 
deviation for both distributions. Highlight bins where production frequency is 
more than 2x training frequency in orange.
```

**What it generates:**
- Side-by-side distribution comparison
- Statistical summary to quantify shift magnitude
- Outlier bin identification

---

**Repeat offender features:**
```
Create a bar chart showing how many times each feature has been flagged for drift 
in the last 30 days. Sort descending. Color bars green for 0-2 flags, yellow for 
3-5 flags, red for 6+ flags. Add a table below listing retraining recommendations 
for features with 6+ flags.
```

**What it generates:**
- Chronic drift ranking (retraining priority list)
- Color-coded severity scoring
- Actionable retraining guidance

---

#### **Correlation and Anomaly Detection**

**Drift vs. volume correlation:**
```
Create a scatter plot with daily prediction volume on x-axis and drift percentage 
on y-axis for the last 60 days. Add a trend line. Highlight outliers where volume 
is high but drift is normal (green) or volume is low but drift is high (red).
```

**What it generates:**
- Identifies spurious drift caused by low sample sizes
- Finds genuine drift despite high volumes (true alerts)

---

**Cross-model drift comparison:**
```
Show a grouped bar chart comparing average drift scores across all deployed model 
versions over the last 14 days. Group by model version, color by severity 
(low/medium/high drift). Add a line showing inference volume per version.
```

**What it generates:**
- Multi-model drift landscape
- Version comparison for rollback decisions

---

### Tips for Writing Effective Queries

1. **Be specific about time ranges**: "last 30 days" vs. "last week" vs. "since Jan 1"
2. **Specify chart type**: line chart, bar chart, heatmap, Sankey, scatter plot
3. **Define thresholds explicitly**: "0.2 threshold", ">10 percentage points", ">50 PSI"
4. **Request color coding**: "highlight in red", "color green when above baseline"
5. **Ask for multiple visuals**: "Add a second chart below", "with a table underneath"
6. **Include statistical summaries**: "with mean and median", "add trend line"

### When to Use Natural Language Queries vs. Pre-Built Dashboard

| Use Natural Language Q | Use Pre-Built Dashboard |
|------------------------|-------------------------|
| Ad-hoc investigation of specific features | Daily/weekly monitoring routine |
| Custom time ranges (last 14 days, specific date range) | Standard 7/30-day lookback windows |
| Comparing multiple model versions side-by-side | Tracking single deployed model |
| Creating one-off reports for stakeholders | Ongoing governance and compliance |
| Exploring correlations not in the 32 visuals | Established drift/performance metrics |
| Testing "what-if" threshold changes | Production alerting at configured thresholds |

---

**Next Steps**: Once the dashboard is created, you can access it anytime at:
- QuickSight Console → **Dashboards** → **"Fraud Detection Governance Dashboard"**
- Use **QuickSight Q** (search bar) for custom natural language queries
