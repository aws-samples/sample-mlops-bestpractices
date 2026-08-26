# Schema Migration Runbook: Drift-Monitor Schema (Athena / Iceberg)

Operator runbook for upgrading **already-deployed** Athena/Iceberg tables to the
new drift-monitor schema. The code in `src/setup/create_athena_tables.py` (the
one authoritative source of table DDL) already reflects the target schema; this
runbook is how you apply that change to **LIVE** tables that were created
*before* the change landed.

> **Scope.** This affects five tables:
> - `monitoring_responses` — 10 new nullable columns appended after `created_at`.
> - `inference_responses` — `prediction` and `ground_truth` change from
>   `INT`/`INTEGER` to `DOUBLE`. Its `monitoring_run_id` column is now
>   **LEGACY/deprecated** — see [Section 1B](#1b-monitoring_run_inferences--create-new-bridge-table-safe-non-destructive).
> - `monitoring_run_inferences` — a **new** run→inference membership **bridge**
>   table, created via a safe `CREATE TABLE IF NOT EXISTS`. See
>   [Section 1B](#1b-monitoring_run_inferences--create-new-bridge-table-safe-non-destructive).
> - `monitoring_run_generations` — a **new** membership-completion **marker**
>   table (safe `CREATE TABLE IF NOT EXISTS`). See [Section 1C](#1c-monitoring_run_generations--create-membership-completion-marker-table-safe-non-destructive).
> - `monitoring_alerts` — a **new** notification **outbox** table (safe
>   `CREATE TABLE IF NOT EXISTS`). See [Section 1D](#1d-monitoring_alerts--create-notification-outbox-table-safe-non-destructive).
>
> **Greenfield / fresh deploys do NOT need this runbook.** See
> [Section 5](#5-greenfield--fresh-deploy-no-migration-needed).

---

## Conventions used below

- **Database name.** All SQL uses `fraud_detection`, which is the repo default
  for the `ATHENA_DATABASE` config value
  (`src/config/config.py`: `ATHENA_DATABASE ... "fraud_detection"`, also in
  `.env.example`). If your deployment overrides `ATHENA_DATABASE`, substitute
  your value everywhere `fraud_detection.` appears.
- **S3 data location.** Tables live under
  `s3://<DATA_S3_BUCKET>/<DATA_S3_PREFIX>/<table>/`, where `DATA_S3_PREFIX`
  defaults to `fraud-detection/`. Concrete SQL below uses
  `s3://<DATA_S3_BUCKET>/fraud-detection/...`; substitute your bucket
  (`DATA_S3_BUCKET`) and prefix if you overrode them.
- Run every query in the Athena console (or via `aws athena
  start-query-execution`) against the `fraud_detection` database.

---

## 0. Ordering constraint (READ FIRST — applies to BOTH migrations)

**Schema first, then code. Always.**

The drift Lambda and the monitoring-results writer Lambda
(`src/drift_monitoring/deploy_monitoring_writer.py`) use a **NAMED INSERT**
(`INSERT INTO ... (col_list) VALUES (...)`). A named INSERT that references a
column the table does not yet have **fails immediately**. Therefore:

1. **Apply the schema migrations (Sections 1 and 2) BEFORE deploying the new
   writer/drift Lambda code.** If you deploy the new Lambda first, its INSERT
   references `problem_type ... execution_status` (and expects `prediction` /
   `ground_truth` to be `DOUBLE`) and every write fails.
2. The additive `monitoring_responses` migration (Section 1) is
   backward-compatible: the OLD Lambda keeps working after you add the columns
   (its shorter INSERT column list is still a valid subset), and the NEW Lambda
   only works after. So there is a safe window — add columns any time, cut over
   the Lambda whenever ready.
3. `inference_responses` (Section 2) is the **hot ingest table** written by the
   inference-logger Lambda on every prediction. Its migration is a
   recreate-and-swap, so it has a **downtime / consistency window**. Plan it for
   a **low-traffic window** and quiesce ingest per **[Section Q](#q-ingest-quiesce-quiesce--drain--cutover--resume-run-before-any-migration)**
   before you touch the table.

   > **`ENABLE_ATHENA_LOGGING=false` is NOT enough on its own.** It stops only
   > the endpoint-side **producer**; the inference-logger **consumer** keeps
   > draining any already-queued messages into `inference_responses`, and the
   > monitoring plane (drift-monitor producer + monitoring-writer consumer) is
   > entirely unaffected by that flag. You must also stop the **consumers** and
   > the **monitoring producer** — that full QUIESCE → DRAIN → CUTOVER → RESUME
   > procedure is **Section Q**, and it applies to **both** migrations.

---

## Q. Ingest quiesce: QUIESCE → DRAIN → CUTOVER → RESUME (RUN BEFORE ANY MIGRATION)

> **Why this section exists.** Setting `ENABLE_ATHENA_LOGGING=false` only stops
> the **endpoint-side PRODUCER** (the custom inference handler skips its SQS
> send). It does **NOT** stop the **CONSUMER** — the writer Lambda that drains
> the SQS queue and `INSERT`s into Athena keeps running and will happily write
> to `inference_responses` / `monitoring_responses` **mid-migration** from
> messages already in the queue. The same is true on the monitoring side: the
> scheduled drift-monitor keeps producing, and the monitoring-writer keeps
> consuming, regardless of `ENABLE_ATHENA_LOGGING`. You must quiesce **both the
> producer and the consumer** on **both** ingest planes before you migrate.

There are **two independent ingest planes**, each a producer → SQS → consumer
chain, one per table this runbook migrates:

| Plane | Table (migration) | PRODUCER | SQS queue (+ DLQ) | CONSUMER (writer) + trigger |
|-------|-------------------|----------|-------------------|------------------------------|
| Monitoring | `monitoring_responses` (Section 1) | scheduled drift-monitor Lambda `fraud-detection-drift-monitor`, fired by EventBridge rule `fraud-detection-drift-check` | `fraud-monitoring-results` (DLQ `fraud-monitoring-results-dlq`) | `fraud-monitoring-results-writer` via SQS **event-source mapping** (CFN `MonitoringWriterEventSource`) |
| Inference (**HOT**) | `inference_responses` (Section 2) | endpoint inference handler, gated by `ENABLE_ATHENA_LOGGING` | `fraud-detection-inference-logging` (CFN `InferenceLoggingQueue`, `${ProjectName}-inference-logging`) | `fraud-detection-inference-logger` via SQS **event-source mapping** (CFN `InferenceLoggerEventSource`) |

> **Downtime / hot-ingest window.** The inference plane is **hot** (one message
> per prediction). Between **Q3 (consumers disabled)** and **Q7 (resume)** the
> endpoint keeps serving predictions, but their log rows sit **held in the
> inference-logging SQS queue** and are **NOT queryable in `inference_responses`**
> until you resume. That table is stale for the duration. **CRITICAL:** the
> inference-logging queue's `MessageRetentionPeriod` is only **86400s = 1 day**
> (CFN `InferenceLoggingQueue`), so any held inference message **expires and is
> permanently lost** if Q3→Q7 exceeds ~24h. The monitoring queue retains 14 days
> (`1209600s`). **Run this in a low-traffic window** and keep the Q3→Q7 window
> well under a day.

### Placeholders (match the CFN defaults / the rest of this runbook)

```bash
# --- General ---
export AWS_REGION=us-east-1
export DATABASE=fraud_detection            # ATHENA_DATABASE (repo default)

# --- Monitoring plane (writes monitoring_responses — Section 1) ---
export MON_QUEUE=fraud-monitoring-results
export MON_DLQ=fraud-monitoring-results-dlq
export MON_WRITER=fraud-monitoring-results-writer
export DRIFT_RULE=fraud-detection-drift-check       # EventBridge schedule

# --- Inference plane (writes inference_responses — Section 2, the HOT table) ---
export INF_QUEUE=fraud-detection-inference-logging
export INF_LOGGER=fraud-detection-inference-logger
```

> Substitute your values if you overrode `ProjectName`, `ATHENA_DATABASE`, or
> any of the queue/Lambda/rule names.

### Q1 — QUIESCE the producers (stop NEW messages entering the queues)

```bash
# Inference producer: turn off endpoint-side logging and CONFIRM it is live.
# ENABLE_ATHENA_LOGGING is a container env var, so this means pushing the
# endpoint-config change and waiting for the endpoint to reach InService with
# the new env. Confirm by observing the inference-logging queue stop growing
# (Q2 poll) — DescribeEndpoint should show Status=InService on the new config.
aws sagemaker describe-endpoint --endpoint-name fraud-detector-endpoint \
  --region "$AWS_REGION" --query '[EndpointStatus,EndpointConfigName]'

# Monitoring producer: disable the EventBridge schedule so no new run is emitted.
aws events disable-rule --name "$DRIFT_RULE" --region "$AWS_REGION"
aws events describe-rule --name "$DRIFT_RULE" --region "$AWS_REGION" \
  --query 'State'                      # MUST print "DISABLED"
```

> **STOP if the rule does not read `DISABLED`** or the endpoint has not picked
> up `ENABLE_ATHENA_LOGGING=false` — a live producer will keep feeding the
> queue and you can never reach a clean drain.

### Q2 — DRAIN the queues to empty (consumers still enabled)

Leave the writer Lambdas running so they flush everything already queued into
Athena (nothing is lost). Poll until **both** the visible and in-flight counts
are `0` for **each** queue, and confirm the DLQ is not accumulating:

```bash
for Q in "$MON_QUEUE" "$INF_QUEUE" "$MON_DLQ"; do
  URL=$(aws sqs get-queue-url --queue-name "$Q" --region "$AWS_REGION" \
        --query QueueUrl --output text)
  echo "== $Q =="
  aws sqs get-queue-attributes --queue-url "$URL" --region "$AWS_REGION" \
    --attribute-names ApproximateNumberOfMessages \
                      ApproximateNumberOfMessagesNotVisible \
    --query 'Attributes'
done
```

Re-run until, for `$MON_QUEUE` and `$INF_QUEUE`, both
`ApproximateNumberOfMessages` **and** `ApproximateNumberOfMessagesNotVisible`
are `0`.

> **STOP if `$MON_DLQ` (or the inference DLQ) is non-empty / growing:** messages
> there are **failed writes**. Investigate and redrive/hand-process them
> **before** migrating, or you will lose those rows. Do not proceed on a dirty
> DLQ.

### Q3 — STOP the consumers (disable the SQS event-source mappings)

Only after Q2 shows the queues empty. Disabling (not deleting) the mapping makes
RESUME a one-liner and leaves any late-arriving messages **held** safely in the
queue:

```bash
for FN in "$MON_WRITER" "$INF_LOGGER"; do
  UUID=$(aws lambda list-event-source-mappings --function-name "$FN" \
         --region "$AWS_REGION" --query 'EventSourceMappings[0].UUID' \
         --output text)
  echo "$FN event-source mapping: $UUID"
  aws lambda update-event-source-mapping --uuid "$UUID" --no-enabled \
    --region "$AWS_REGION"
  # Confirm — State passes through "Disabling" then settles on "Disabled":
  aws lambda get-event-source-mapping --uuid "$UUID" --region "$AWS_REGION" \
    --query 'State'
done
```

> Prefer `--no-enabled` (reversible) over deleting the mapping. If you must hard
> stop, `aws lambda delete-event-source-mapping --uuid "$UUID"` works but you
> then have to **recreate** it at RESUME (BatchSize 10,
> MaximumBatchingWindowInSeconds 5 — see the CFN mappings).

Then wait for any **in-flight Athena queries** against the two tables to finish
(the writer polls its `INSERT` to completion, but a manual/BI query could still
be running):

```bash
aws athena list-query-executions --region "$AWS_REGION" --work-group primary \
  --query 'QueryExecutionIds[:25]' --output text
# For any returned IDs, none touching inference_responses / monitoring_responses
# may be RUNNING or QUEUED:
aws athena batch-get-query-execution --region "$AWS_REGION" \
  --query-execution-ids <ids from above> \
  --query 'QueryExecutions[?Status.State==`RUNNING` || Status.State==`QUEUED`].[QueryExecutionId,Query]'
```

### Q4 — VERIFY no new writes before cutover (row counts stable)

```sql
-- Run, wait ~2–3 minutes with ingest quiesced, then run again.
-- Each pair MUST be identical across the two runs.
SELECT COUNT(*) AS inference_rows  FROM fraud_detection.inference_responses;
SELECT COUNT(*) AS monitoring_rows FROM fraud_detection.monitoring_responses;
```

> **STOP GATE — do NOT migrate if either count moved.** A moving count means a
> producer or consumer is still live. Re-check Q1 (rule `DISABLED`, endpoint on
> the no-logging config) and Q3 (both mappings `Disabled`) before continuing.

### Q5 — MIGRATE

Now run the table migrations against the **quiesced** tables:

- **Section 1** — `ALTER TABLE ... ADD COLUMNS` on `monitoring_responses`.
- **Section 2** — recreate + backfill + validate + swap `inference_responses`.

Deploy the new code (drift Lambda, monitoring-writer, inference-logger) per
**Section 3** — schema first, always.

### Q6 — RECONCILE messages held during the migration

Anything produced after Q1 but before the consumers were disabled was flushed in
Q2; anything that slipped in afterward is **held** in the now-disabled queue (or
sits in a DLQ). Before resuming, re-check depths (Q2 command) and the DLQ.

> **Safe by construction for the int→double change:** held inference messages
> still carry integer `prediction` / `ground_truth` in their JSON body. The
> **new** writer serializes those through the int/float branch and inserts them
> into the migrated `DOUBLE` column losslessly, so re-enabling the consumer
> (Q7) drains the held backlog straight into the migrated table. If you deleted
> (not disabled) the mappings, recreate them here.

### Q7 — RESUME (re-enable consumers, then producers)

```bash
# 1) Re-enable the consumers (flushes any held backlog into the migrated tables).
for FN in "$MON_WRITER" "$INF_LOGGER"; do
  UUID=$(aws lambda list-event-source-mappings --function-name "$FN" \
         --region "$AWS_REGION" --query 'EventSourceMappings[0].UUID' \
         --output text)
  aws lambda update-event-source-mapping --uuid "$UUID" --enabled \
    --region "$AWS_REGION"
done

# 2) Re-enable the monitoring producer schedule.
aws events enable-rule --name "$DRIFT_RULE" --region "$AWS_REGION"
aws events describe-rule --name "$DRIFT_RULE" --region "$AWS_REGION" \
  --query 'State'                      # MUST print "ENABLED"

# 3) Re-enable the inference producer: set ENABLE_ATHENA_LOGGING=true on the
#    endpoint container and push the endpoint-config change.
```

Confirm writes now flow to the **migrated** tables using the **Section 4**
verification checklist (new rows land; `execution_status` populates; `prediction`
/ `ground_truth` arrive as `DOUBLE`).

---

## 1. `monitoring_responses` — ADD COLUMNS (safe, non-breaking)

> **DANGER — do NOT run `src/drift_monitoring/create_monitoring_table.py` to
> migrate a live table.** That standalone helper is **destructive**: it issues
> `DROP TABLE IF EXISTS fraud_detection.monitoring_responses` and then re-creates
> the table from scratch, **destroying every historical monitoring row**. It is
> a **greenfield-only** tool. To add the 10 new columns to an **existing**,
> populated table, use the non-breaking `ALTER TABLE ... ADD COLUMNS` in this
> section — it preserves all data and requires no rewrite. Only use
> `create_monitoring_table.py` (or, if your copy exposes one, its opt-in
> destructive/recreate flag) for **greenfield recreation** where losing the
> existing table's contents is acceptable.

### What changes

10 new **nullable** columns are appended **after `created_at`**, in this exact
order (derived verbatim from the `monitoring_responses` DDL in
`src/setup/create_athena_tables.py`):

```
problem_type       STRING
primary_metric     STRING
baseline_primary   DOUBLE
current_primary    DOUBLE
mae                DOUBLE
rmse               DOUBLE
r2                 DOUBLE
execution_status   STRING
data_drift_status  STRING
model_drift_status STRING
```

For an Iceberg table this is a **SAFE, non-breaking** `ADD COLUMNS` operation:
existing rows read back `NULL` for the new columns, and there is **no data
rewrite**.

### Pre-flight checklist

- [ ] Confirm the table exists and is Iceberg:
  ```sql
  DESCRIBE fraud_detection.monitoring_responses;
  ```
- [ ] Confirm the 10 columns are **not already present** (skip this section if
      they are — the migration is idempotent in intent but `ADD COLUMNS` will
      error on a duplicate column name).
- [ ] Confirm you have not yet deployed the new writer Lambda (schema first).

### Migration SQL

```sql
ALTER TABLE fraud_detection.monitoring_responses
ADD COLUMNS (
    problem_type      STRING,
    primary_metric    STRING,
    baseline_primary  DOUBLE,
    current_primary   DOUBLE,
    mae                DOUBLE,
    rmse               DOUBLE,
    r2                 DOUBLE,
    execution_status   STRING,
    data_drift_status  STRING,
    model_drift_status STRING
);
```

### Validation

```sql
-- All 10 new columns must now appear, after created_at.
DESCRIBE fraud_detection.monitoring_responses;

-- Existing rows read back NULL for the new columns (no rewrite happened).
SELECT COUNT(*) AS total_rows,
       COUNT(execution_status) AS non_null_execution_status
FROM fraud_detection.monitoring_responses;
```

> **STOP if this validation fails:** if `DESCRIBE` does not show all 10 columns,
> do **not** deploy the new writer/drift Lambda — its named INSERT will fail on
> the missing columns. Re-run the `ALTER TABLE` and re-check before proceeding.

Only **after** the columns exist, proceed to deploy the new monitoring writer /
drift Lambda (Section 4 verification confirms it works).

---

## 1B. `monitoring_run_inferences` — CREATE new bridge table (safe, non-destructive)

> **What this is.** `monitoring_run_inferences` is a **new** run→inference
> membership **BRIDGE** table. It records the **exact** `inference_id`s each
> monitoring run scored, **per check**, replacing the old approach of back-filling
> `inference_responses.monitoring_run_id` with a timestamp window. Because
> `check_type ∈ {'data_drift','model_drift'}` and the two cohorts differ
> (data-drift is a **request-time** set; model-drift is a
> **ground-truth-arrival-time** set) yet **overlap**, a single scalar FK on
> `inference_responses` could not represent both — which is why the bridge exists.

### What changes

For an **existing / deployed** stack, this migration is simply to **CREATE the
new table**. It is a **safe, non-destructive** `CREATE TABLE IF NOT EXISTS`:

- **No data rewrite** and **no backfill of historical runs.** Runs that executed
  before the bridge landed will **not** have membership rows, and that is
  **expected and acceptable** — the bridge is populated **forward-only**, by new
  runs going forward.
- It is **independent of** and **separate from** the `monitoring_responses`
  `ADD COLUMNS` migration (Section 1) and the `inference_responses` int→double
  migration (Section 2). No quiesce window is required to create it.

How the drift Lambda populates it (for context — no operator action needed): the
Lambda writes membership **idempotently** — it `DELETE`s by
`(monitoring_run_id, check_type)` then does a batched `INSERT`. `monitoring_run_id`
is now **stable across retries**, so an SQS re-delivery re-materializes
**identical** membership rather than duplicating it.

> **`inference_responses.monitoring_run_id` is now LEGACY/deprecated.** It is
> retained for backward-compat with existing data but is **NO LONGER WRITTEN** by
> the Lambda. Consumers (e.g. the governance dashboard) now join **through the
> bridge** (`monitoring_run_inferences`) instead of reading that scalar column.
> **No `ALTER` is needed** — keep the legacy column as-is.

### Migration SQL

Substitute your bucket/prefix for `<bucket>/<prefix>` (repo default prefix is
`fraud-detection/`, so the concrete location is
`s3://<DATA_S3_BUCKET>/fraud-detection/monitoring_run_inferences/`). DDL is copied
verbatim from `src/setup/create_athena_tables.py`:

```sql
CREATE TABLE IF NOT EXISTS fraud_detection.monitoring_run_inferences (
    monitoring_run_id STRING, inference_id STRING, check_type STRING,
    endpoint_name STRING, created_at TIMESTAMP
)
PARTITIONED BY (day(created_at))
LOCATION 's3://<bucket>/<prefix>/monitoring_run_inferences/'
TBLPROPERTIES ('table_type' = 'ICEBERG', 'format' = 'parquet');
```

### Validation

```sql
-- The bridge table now exists and is Iceberg.
DESCRIBE fraud_detection.monitoring_run_inferences;

-- After the next scheduled drift run, membership rows appear for both checks.
SELECT check_type, COUNT(*) AS rows, COUNT(DISTINCT inference_id) AS inferences
FROM fraud_detection.monitoring_run_inferences
WHERE created_at > current_timestamp - interval '1' day
GROUP BY check_type;   -- expect check_type in {data_drift, model_drift}
```

> **Non-destructive, so no STOP gate on historical data:** an empty result before
> the first post-migration run is normal (forward-only, no historical backfill).
> Because the create is `IF NOT EXISTS`, re-running it is harmless.

---

## 1C. `monitoring_run_generations` — CREATE membership-completion marker table (safe, non-destructive)

> **What this is.** `monitoring_run_generations` is a **new** table holding one
> **completion marker** per `(monitoring_run_id, check_type)`. The drift Lambda
> writes the marker **only after** every `monitoring_run_inferences` row for that
> cohort was inserted **and count-verified** (`expected_count == actual_count`).
> It is the **commit point** for a cohort's bridge membership: a consumer that
> joins the bridge **through a `COMPLETE` marker** never reads a half-written
> cohort left behind by a crash mid-insert (partial bridge, **no** marker).

### What changes

For an **existing / deployed** stack, this migration is simply to **CREATE the
new table** — a **safe, non-destructive** `CREATE TABLE IF NOT EXISTS`:

- **No data rewrite** and **no backfill of historical runs.** Runs that executed
  before this table landed will have **no marker**; that is **expected and
  acceptable** — markers are populated **forward-only**, by new runs.
- **Independent of** all other migrations here; **no quiesce window** required.

How the drift Lambda populates it (context — no operator action): after the
count-verified bridge insert, the Lambda `MERGE`s a single marker row keyed on
`(monitoring_run_id, check_type)` — idempotent across retries of the same
(stable) `monitoring_run_id`. If the bridge insert fails its count check, the
marker is **withheld** and the whole run re-raises so EventBridge retries and
re-materializes the deterministic cohort.

### Migration SQL

Substitute your bucket/prefix for `<bucket>/<prefix>` (repo default prefix
`fraud-detection/`). DDL is copied verbatim from
`src/setup/create_athena_tables.py`:

```sql
CREATE TABLE IF NOT EXISTS fraud_detection.monitoring_run_generations (
    monitoring_run_id STRING, check_type STRING,
    expected_count INT, actual_count INT,
    status STRING, created_at TIMESTAMP
)
PARTITIONED BY (day(created_at))
LOCATION 's3://<bucket>/<prefix>/monitoring_run_generations/'
TBLPROPERTIES ('table_type' = 'ICEBERG', 'format' = 'parquet');
```

### Validation

```sql
DESCRIBE fraud_detection.monitoring_run_generations;

-- After the next scheduled run, a COMPLETE marker exists for each scored check
-- and its counts agree. Join the bridge THROUGH this table to read only whole
-- cohorts.
SELECT check_type, status, expected_count, actual_count
FROM fraud_detection.monitoring_run_generations
WHERE created_at > current_timestamp - interval '1' day
  AND status = 'COMPLETE';   -- expect expected_count = actual_count
```

> **Non-destructive, no STOP gate:** an empty result before the first
> post-migration run is normal (forward-only). `IF NOT EXISTS` makes re-running
> harmless.

---

## 1D. `monitoring_alerts` — CREATE notification outbox table (safe, non-destructive)

> **What this is.** `monitoring_alerts` is a **new** durable **outbox** with one
> row per `(monitoring_run_id, alert_type)`. Because EventBridge → Lambda is
> **asynchronous and retried** (and the handler now re-raises on a membership
> failure), the same run can be processed more than once; a naive `sns.publish`
> per pass would send **duplicate** drift alerts. The outbox makes alerting
> **exactly-once-ish**: a `PENDING` claim is inserted before publishing, then the
> row is flipped to `SENT` with the returned `MessageId`. A retry that sees
> `SENT` **skips** re-publishing; a retry that sees `PENDING` (a prior publish
> died mid-flight) **re-attempts**.

### What changes

For an **existing / deployed** stack, this migration is simply to **CREATE the
new table** — a **safe, non-destructive** `CREATE TABLE IF NOT EXISTS`:

- **No data rewrite** and **no backfill of historical runs.** Alerts already sent
  before this table landed have **no outbox row**; that is **expected** — the
  outbox is populated **forward-only**.
- **Independent of** all other migrations here; **no quiesce window** required.

> **Ordering:** the drift Lambda now alerts **after** the monitoring row + bridge
> membership are durably persisted, so a drift alert never fires for a run whose
> results didn't land. A publish failure re-raises → the run is retried.

### Migration SQL

Substitute your bucket/prefix for `<bucket>/<prefix>` (repo default prefix
`fraud-detection/`). DDL is copied verbatim from
`src/setup/create_athena_tables.py`:

```sql
CREATE TABLE IF NOT EXISTS fraud_detection.monitoring_alerts (
    monitoring_run_id STRING, alert_type STRING,
    subject STRING, message STRING,
    status STRING, sns_message_id STRING,
    created_at TIMESTAMP, sent_at TIMESTAMP
)
PARTITIONED BY (day(created_at))
LOCATION 's3://<bucket>/<prefix>/monitoring_alerts/'
TBLPROPERTIES ('table_type' = 'ICEBERG', 'format' = 'parquet');
```

### Validation

```sql
DESCRIBE fraud_detection.monitoring_alerts;

-- After a run that DETECTS drift, exactly one SENT row exists for that run's
-- 'drift' alert, carrying the SNS MessageId. Retries of the same run add no
-- second row.
SELECT monitoring_run_id, alert_type, status, sns_message_id
FROM fraud_detection.monitoring_alerts
WHERE created_at > current_timestamp - interval '1' day;
```

> **Non-destructive, no STOP gate:** an empty result before the first
> drift-detecting post-migration run is normal. `IF NOT EXISTS` makes re-running
> harmless.

---

## 2. `inference_responses` — RECREATE + BACKFILL (int→double)

### Why a plain ALTER will NOT work

Iceberg permits only a fixed set of safe type promotions: `int→long`,
`float→double`, and `decimal` precision widening. **`int→double` is NOT a
supported Iceberg promotion.** A plain `ALTER TABLE ... CHANGE`/`ALTER COLUMN`
to retype `prediction` or `ground_truth` from `INT`/`INTEGER` to `DOUBLE` will
fail or is unsupported. The safe procedure is **recreate a new table with the
corrected DDL, backfill it with an explicit CAST, validate, then swap.**

> Why `DOUBLE` at all: regression models emit continuous predictions/targets
> that `INT` would truncate, making regression drift meaningless. Binary /
> multiclass integer labels (`0`/`1`/`2`...) round-trip through `DOUBLE`
> **losslessly**, so no existing classification data is harmed.

### Pre-flight checklist

- [ ] You are in a **low-traffic window**.
- [ ] **Ingest is fully quiesced per [Section Q](#q-ingest-quiesce-quiesce--drain--cutover--resume-run-before-any-migration)**:
      the inference producer is off (`ENABLE_ATHENA_LOGGING=false`, endpoint on
      the new config), the queue is drained to 0, **and** the inference-logger
      **consumer's** event-source mapping is `Disabled`. Setting
      `ENABLE_ATHENA_LOGGING=false` alone does **not** stop the consumer, so
      rows could still land mid-migration — Q2/Q3 close that gap.
- [ ] Record the current row count for later parity checks:
  ```sql
  SELECT COUNT(*) AS baseline_row_count FROM fraud_detection.inference_responses;
  ```
- [ ] Confirm current types are the OLD int types:
  ```sql
  DESCRIBE fraud_detection.inference_responses;   -- prediction / ground_truth show int
  ```

### Step 2a — Create the corrected table `inference_responses_v2`

Same partitioning and location convention as the original, but a **distinct S3
location** (`inference_responses_v2/`) so Iceberg starts with clean metadata.
Column list is copied verbatim from the `inference_responses` DDL in
`src/setup/create_athena_tables.py`, with `prediction` and `ground_truth` now
`DOUBLE`. (`transaction_id` is the schema identifier column.)

```sql
CREATE TABLE IF NOT EXISTS fraud_detection.inference_responses_v2 (
    inference_id STRING, request_timestamp TIMESTAMP, endpoint_name STRING,
    model_version STRING, mlflow_run_id STRING,
    input_features STRING,
    prediction DOUBLE, probability_fraud DOUBLE, probability_non_fraud DOUBLE, confidence_score DOUBLE,
    ground_truth DOUBLE, ground_truth_timestamp TIMESTAMP, ground_truth_source STRING, days_to_ground_truth DOUBLE,
    inference_latency_ms DOUBLE, model_load_time_ms DOUBLE, preprocessing_time_ms DOUBLE,
    transaction_id STRING, transaction_amount DOUBLE, customer_id STRING,
    is_high_confidence BOOLEAN, is_low_confidence BOOLEAN, prediction_bucket STRING,
    request_id STRING, response_time TIMESTAMP, error_message STRING, inference_mode STRING,
    monitoring_run_id STRING
)
PARTITIONED BY (day(request_timestamp), endpoint_name)
LOCATION 's3://<DATA_S3_BUCKET>/fraud-detection/inference_responses_v2/'
TBLPROPERTIES ('table_type' = 'ICEBERG', 'format' = 'parquet');
```

### Step 2b — Backfill with explicit column list + CASTs

**Every column must be listed explicitly** in the SELECT so the two `CAST`s land
in the correct positions (`prediction` is column 7, `ground_truth` is column 11
in the DDL above). Do **not** use `SELECT *` — column order/position is
load-bearing.

```sql
INSERT INTO fraud_detection.inference_responses_v2
SELECT
    inference_id,
    request_timestamp,
    endpoint_name,
    model_version,
    mlflow_run_id,
    input_features,
    CAST(prediction AS DOUBLE)   AS prediction,      -- int -> double
    probability_fraud,
    probability_non_fraud,
    confidence_score,
    CAST(ground_truth AS DOUBLE) AS ground_truth,    -- int -> double
    ground_truth_timestamp,
    ground_truth_source,
    days_to_ground_truth,
    inference_latency_ms,
    model_load_time_ms,
    preprocessing_time_ms,
    transaction_id,
    transaction_amount,
    customer_id,
    is_high_confidence,
    is_low_confidence,
    prediction_bucket,
    request_id,
    response_time,
    error_message,
    inference_mode,
    monitoring_run_id
FROM fraud_detection.inference_responses;
```

Binary `0`/`1` labels round-trip losslessly through the cast; any future
regression values will no longer truncate.

### Step 2c — Validation (run ALL three; each is a STOP gate)

```sql
-- (1) Row-count parity: both numbers MUST be identical.
SELECT
  (SELECT COUNT(*) FROM fraud_detection.inference_responses)     AS old_count,
  (SELECT COUNT(*) FROM fraud_detection.inference_responses_v2)  AS new_count;
```

```sql
-- (2) Null-count parity on the two migrated columns: old vs new MUST match.
SELECT
  (SELECT COUNT(*) FROM fraud_detection.inference_responses
     WHERE prediction   IS NULL) AS old_pred_nulls,
  (SELECT COUNT(*) FROM fraud_detection.inference_responses_v2
     WHERE prediction   IS NULL) AS new_pred_nulls,
  (SELECT COUNT(*) FROM fraud_detection.inference_responses
     WHERE ground_truth IS NULL) AS old_gt_nulls,
  (SELECT COUNT(*) FROM fraud_detection.inference_responses_v2
     WHERE ground_truth IS NULL) AS new_gt_nulls;
```

```sql
-- (3) Spot-check that the CAST did not lose or alter rows. Both directions of
--     this bidirectional EXCEPT MUST return ZERO rows.
--
--     Do NOT use `NOT IN (subquery)`: if the subquery yields any NULL (e.g. a
--     NULL prediction), `x NOT IN (...)` evaluates to UNKNOWN for every x, the
--     WHERE filters it out, and the check returns 0 rows even when the sets
--     differ — silently HIDING a migration mismatch. `EXCEPT` uses set
--     semantics where NULL matches NULL, so it is null-safe.

-- (3a) Distinct cast values in OLD but absent from NEW — MUST return 0 rows.
SELECT CAST(prediction AS DOUBLE) AS v
FROM fraud_detection.inference_responses
EXCEPT
SELECT prediction AS v
FROM fraud_detection.inference_responses_v2;

-- (3b) Distinct values in NEW but absent from OLD — MUST return 0 rows.
SELECT prediction AS v
FROM fraud_detection.inference_responses_v2
EXCEPT
SELECT CAST(prediction AS DOUBLE) AS v
FROM fraud_detection.inference_responses;
```

> **Optional stronger check — row-level parity keyed on `transaction_id`** (the
> schema identifier column). A FULL OUTER JOIN with the null-safe
> `IS DISTINCT FROM` surfaces any key present on only one side **or** any row
> whose migrated values changed. MUST return 0 rows. (If `transaction_id` is not
> unique in your data, prefer the (3a)/(3b) set check above, since a non-unique
> join key can fan out.)
>
> ```sql
> SELECT o.transaction_id AS old_key, n.transaction_id AS new_key
> FROM fraud_detection.inference_responses      o
> FULL OUTER JOIN fraud_detection.inference_responses_v2 n
>   ON o.transaction_id = n.transaction_id
> WHERE o.transaction_id IS NULL                                       -- row only in NEW
>    OR n.transaction_id IS NULL                                       -- row only in OLD
>    OR CAST(o.prediction   AS DOUBLE) IS DISTINCT FROM n.prediction
>    OR CAST(o.ground_truth AS DOUBLE) IS DISTINCT FROM n.ground_truth;
> ```

> **STOP if any validation fails:** if `old_count != new_count`, the null counts
> differ, or **either** (3a) **or** (3b) (or the optional keyed check) returns
> any rows, **do NOT cut over**. Investigate, truncate/drop
> `inference_responses_v2`, and re-run the backfill. The original
> `inference_responses` is still intact and still the live table.

### Step 2d — Cutover (swap `_v2` into place)

Athena/Glue Iceberg does not support a single atomic table rename, so cut over
by dropping the old table and renaming the new one. `monitoring_run_id` (the
legacy monitoring-linkage column) is carried through the backfill unchanged,
so it is preserved.

> **`monitoring_run_id` is now LEGACY/deprecated.** It is kept in the `_v2` DDL
> and backfilled purely for backward-compat with existing data; the Lambda **no
> longer writes it**. Run→inference linkage now lives in the
> `monitoring_run_inferences` bridge ([Section 1B](#1b-monitoring_run_inferences--create-new-bridge-table-safe-non-destructive)),
> and consumers join through the bridge rather than this column.

Only after **all** of Step 2c passed:

```sql
-- Rename the validated original out of the way (keep it for rollback).
ALTER TABLE fraud_detection.inference_responses
  RENAME TO fraud_detection.inference_responses_old;

-- Promote the corrected table to the canonical name.
ALTER TABLE fraud_detection.inference_responses_v2
  RENAME TO fraud_detection.inference_responses;
```

> **If `ALTER TABLE ... RENAME TO` is not supported for your Iceberg/Glue
> setup**, perform the equivalent Glue-catalog swap instead: keep the old table
> as-is, and in the Glue Data Catalog rename/repoint the `inference_responses`
> catalog entry to the `_v2` table definition (e.g. via `aws glue update-table`
> / delete-and-recreate the catalog entry pointing at the `_v2` location). The
> underlying S3 data is untouched by a catalog swap.

Confirm the swap:

```sql
DESCRIBE fraud_detection.inference_responses;   -- prediction / ground_truth now DOUBLE
SELECT COUNT(*) FROM fraud_detection.inference_responses;  -- matches baseline_row_count
```

**Resume ingest per [Section Q, step Q7](#q-ingest-quiesce-quiesce--drain--cutover--resume-run-before-any-migration)**
once the swap is confirmed and the new code is deployed (Section 4): re-enable
the inference-logger **consumer's** event-source mapping, re-enable the
drift-monitor schedule, and set `ENABLE_ATHENA_LOGGING=true` on the endpoint.
Re-enabling the consumer flushes any held backlog into the migrated table.

### Step 2e — Rollback

- **Do not drop `inference_responses_old` (the renamed original) until the new
  schema has been validated in production** (Section 4 checklist all green).
- To revert before you have dropped the old table:
  ```sql
  ALTER TABLE fraud_detection.inference_responses
    RENAME TO fraud_detection.inference_responses_v2;   -- park the new one
  ALTER TABLE fraud_detection.inference_responses_old
    RENAME TO fraud_detection.inference_responses;      -- restore the original
  ```
  Then redeploy the **previous** Lambda code (int schema) if you had already
  cut over, and re-enable logging.
- Once you are confident, reclaim space:
  ```sql
  DROP TABLE fraud_detection.inference_responses_old;
  ```
  (Follow with `VACUUM` / `expire_snapshots` per
  `src/setup/create_athena_tables.py` maintenance helpers if desired.)

---

## 3. Order of operations across BOTH migrations + code deploy

1. **Section 1** — `ADD COLUMNS` to `monitoring_responses` (safe, no downtime).
   **Section 1B** — `CREATE TABLE IF NOT EXISTS monitoring_run_inferences`
   (safe, non-destructive, no downtime, no quiesce needed; can run any time).
2. **Section Q, steps Q1–Q4** — enter a low-traffic window and **quiesce both
   ingest planes**: stop the producers (`ENABLE_ATHENA_LOGGING=false` + disable
   the drift-monitor schedule), DRAIN the queues to 0, disable **both writer
   Lambdas' event-source mappings**, and verify row counts are stable. (Merely
   pausing the producer is **not** enough — the consumers must be stopped too.)
3. **Section 2a–2d** — recreate + backfill + validate + swap
   `inference_responses`.
4. **Deploy the new code** (schema is now ready):
   - new drift Lambda,
   - new monitoring-results writer Lambda
     (`deploy_monitoring_writer.py`),
   - new inference-logger Lambda (emits `prediction`/`ground_truth` as double).
5. **Section Q, steps Q6–Q7** — reconcile any held/DLQ messages, then RESUME:
   re-enable both consumers' event-source mappings, re-enable the drift-monitor
   schedule, and set `ENABLE_ATHENA_LOGGING=true`.
6. Run the **Section 4** verification checklist.
7. After production validation, **drop** `inference_responses_old`.

**Never** deploy new code before the schema migrations — the named INSERTs will
fail on unknown/mistyped columns.

---

## 4. Post-migration verification checklist

- [ ] `DESCRIBE fraud_detection.monitoring_responses` shows all 10 new columns
      (`problem_type` ... `model_drift_status`) after `created_at`.
- [ ] `DESCRIBE fraud_detection.inference_responses` shows `prediction` and
      `ground_truth` as `DOUBLE`; row count matches the pre-migration baseline.
- [ ] The scheduled **drift Lambda's named INSERT succeeds** (check Lambda logs
      for `✓ Written to fraud_detection.monitoring_responses`; no
      column-mismatch / Athena FAILED errors).
- [ ] **`execution_status` populates** on new rows:
  ```sql
  SELECT monitoring_run_id, execution_status, problem_type, primary_metric
  FROM fraud_detection.monitoring_responses
  WHERE created_at > current_timestamp - interval '1' day
  ORDER BY created_at DESC
  LIMIT 20;
  ```
- [ ] A **regression** deployment writes `mae` / `rmse` / `r2` as `DOUBLE`
      (non-null on regression runs):
  ```sql
  SELECT monitoring_run_id, problem_type, mae, rmse, r2, primary_metric,
         baseline_primary, current_primary
  FROM fraud_detection.monitoring_responses
  WHERE problem_type = 'regression'
  ORDER BY created_at DESC
  LIMIT 20;
  ```
- [ ] New inference rows land with `prediction`/`ground_truth` as double values
      (including any non-integer regression outputs, which no longer truncate).
- [ ] `DESCRIBE fraud_detection.monitoring_run_inferences` succeeds (bridge table
      exists and is Iceberg).
- [ ] After a scheduled run, the **bridge is populated** with membership rows for
      both checks:
  ```sql
  SELECT check_type, COUNT(*) AS rows, COUNT(DISTINCT inference_id) AS inferences
  FROM fraud_detection.monitoring_run_inferences
  WHERE created_at > current_timestamp - interval '1' day
  GROUP BY check_type;   -- expect rows for check_type in {data_drift, model_drift}
  ```
- [ ] Confirm re-delivery is **idempotent**: a re-run/retry of the same
      `monitoring_run_id` re-materializes identical membership (DELETE by
      `(monitoring_run_id, check_type)` then batched INSERT), not duplicates.
- [ ] `inference_responses.monitoring_run_id` is **no longer written** (it stays
      NULL on new inference rows); linkage now flows through the bridge and
      consumers join through `monitoring_run_inferences`.
- [ ] `inference_responses_old` retained until all of the above pass; then
      dropped.

---

## 5. Greenfield / fresh deploy (NO migration needed)

If you are standing up a **new** stack, you do **not** run this runbook. The
target schema is already baked into the code:

```bash
python -m src.setup.create_athena_tables
```

This (and the CloudFormation templates, which call the same script) create
`monitoring_responses` with the 10 columns and `inference_responses` with
`prediction`/`ground_truth` as `DOUBLE` directly. It also creates the
`monitoring_run_inferences` bridge table automatically — it is registered in
`ALL_TABLE_NAMES` / `ICEBERG_TABLES` / `PARTITIONED_TABLES` in
`src/setup/create_athena_tables.py`. New runs write membership rows to it for
`check_type` in `{data_drift, model_drift}`; the legacy
`inference_responses.monitoring_run_id` column is not written. This runbook
applies **only** to tables created *before* the drift-monitor schema change.
