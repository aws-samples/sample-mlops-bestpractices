"""
SageMaker Pipeline definition for fraud detection.

This pipeline implements a complete ML workflow:
1. ProcessingStep - Seed Athena training_data from predictions CSV (idempotent)
2. ProcessingStep - Data validation and preprocessing
3. TrainingStep - XGBoost model training with MLflow tracking
4. EvaluationStep - Model evaluation with quality gates
5. ConditionStep - Quality gate check (ROC-AUC >= threshold)
6. RegisterModelStep - Register model in SageMaker Model Registry
7. CreateModelStep - Create SageMaker model for deployment
8. LambdaStep - Deploy to serverless endpoint
9. LambdaStep - Test inference and log to MLflow

The pipeline supports:
- Parameterization for flexibility
- Conditional execution based on model quality
- MLflow integration for tracking and monitoring
- Athena integration for data and inference logging
- End-to-end automation from training to deployment
"""

import json
import logging
import os
import subprocess
from typing import Dict, Any, Optional, List

import boto3


def _resolve_code_commit_sha() -> str:
    """Capture the pipeline-code commit SHA on the orchestrator host.

    Runs once at pipeline-definition time (not in any container). CI/CD
    callers should override this via the CodeCommitSha pipeline parameter
    so the build server's SHA wins over a developer's local checkout. We
    fall back to 'unknown' if git isn't available — eval still completes,
    baseline.json just loses one lineage anchor.
    """
    sha = os.environ.get("CODE_COMMIT_SHA", "").strip()
    if sha:
        return sha
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"

# SageMaker v3 imports — Session and role helpers
from sagemaker.core.helper.session_helper import Session, get_execution_role
from sagemaker.core.workflow.pipeline_context import PipelineSession

# Processing
from sagemaker.core.processing import (
    ProcessingInput, ProcessingOutput, ScriptProcessor, FrameworkProcessor
)
from sagemaker.core.shapes.shapes import ProcessingS3Output, ProcessingS3Input
from sagemaker.core.spark.processing import PySparkProcessor

# Training (ModelTrainer replaces XGBoost estimator)
from sagemaker.train.model_trainer import ModelTrainer, SourceCode, Compute, InputData
from sagemaker.core.shapes.shapes import OutputDataConfig

# Inputs
from sagemaker.core.inputs import TrainingInput, CreateModelInput

# Model (ModelBuilder replaces the old sagemaker.model.Model in v3)
from sagemaker.serve.model_builder import ModelBuilder
from sagemaker.core.model_metrics import MetricsSource, ModelMetrics

# Workflow parameters
from sagemaker.core.workflow.parameters import (
    ParameterInteger, ParameterString, ParameterFloat, ParameterBoolean
)

# Pipeline and steps
from sagemaker.mlops.workflow.pipeline import Pipeline
from sagemaker.mlops.workflow.steps import ProcessingStep, TrainingStep, TransformStep
from sagemaker.mlops.workflow.model_step import ModelStep
from sagemaker.mlops.workflow.condition_step import ConditionStep
from sagemaker.mlops.workflow.lambda_step import LambdaStep, LambdaOutput, LambdaOutputTypeEnum
from sagemaker.mlops.workflow.fail_step import FailStep

# Workflow utilities
from sagemaker.core.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.core.workflow.functions import JsonGet, Join
from sagemaker.core.workflow.properties import PropertyFile
from sagemaker.core.workflow.execution_variables import ExecutionVariables

# Lambda helper
from sagemaker.core.lambda_helper import Lambda

# Infrastructure
from sagemaker.core.transformer import Transformer
from sagemaker.core.image_uris import retrieve as retrieve_image_uri

# Serverless
from sagemaker.serve.serverless import ServerlessInferenceConfig

# Local imports
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Load .env file BEFORE importing config
from dotenv import load_dotenv
env_path = project_root / '.env'
if env_path.exists():
    load_dotenv(env_path)
    logger_temp = logging.getLogger(__name__)
    logger_temp.info(f"Loaded environment from: {env_path}")
else:
    logger_temp = logging.getLogger(__name__)
    logger_temp.warning(f".env file not found at: {env_path}")

from src.config.config import (
    SAGEMAKER_EXEC_ROLE, LAMBDA_EXEC_ROLE, DATA_S3_BUCKET, MLFLOW_MODEL_NAME,
    S3_MODEL_ARTIFACTS, S3_TRAINING_DATA_EXPORT, ATHENA_TRAINING_TABLE,
    ATHENA_EVALUATION_TABLE, ATHENA_DATABASE, ATHENA_OUTPUT_S3,
    SERVERLESS_MEMORY_SIZE, SERVERLESS_MAX_CONCURRENCY, HIGH_CONFIDENCE_THRESHOLD,
    LOW_CONFIDENCE_LOWER, LOW_CONFIDENCE_UPPER, AWS_DEFAULT_REGION, SQS_QUEUE_URL,
    XGBOOST_PARAMS, XGBOOST_OBJECTIVE, XGBOOST_NUM_CLASS,
)
from src.config import schema

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get MLflow tracking URI from environment
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI')
if not MLFLOW_TRACKING_URI:
    logger.warning("MLFLOW_TRACKING_URI not set - MLflow logging will be disabled")
else:
    logger.info(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")


class FraudDetectionPipeline:
    """
    SageMaker Pipeline for fraud detection with end-to-end automation.
    
    Pipeline Flow:
    1. Preprocess data from Athena
    2. Train XGBoost model with MLflow tracking
    3. Evaluate model quality
    4. Quality gate check (ROC-AUC threshold)
    5. Register model in Model Registry
    6. Create SageMaker model
    7. Deploy to serverless endpoint
    8. Test inference and log metrics to MLflow
    """

    def __init__(
        self,
        pipeline_name: str = "fraud-detection-pipeline",
        role: Optional[str] = None,
        region: str = "us-east-1",
        **kwargs
    ):
        """
        Initialize pipeline.

        Args:
            pipeline_name: Name of the pipeline
            role: SageMaker execution role ARN
            region: AWS region
            **kwargs: Additional pipeline configuration
        """
        self.pipeline_name = pipeline_name
        self.region = region

        # Get execution role
        if role:
            self.role = role
        elif SAGEMAKER_EXEC_ROLE:
            self.role = SAGEMAKER_EXEC_ROLE
        else:
            try:
                self.role = get_execution_role()
            except:
                raise ValueError(
                    "Could not determine execution role. "
                    "Please provide role ARN or set SAGEMAKER_EXEC_ROLE"
                )

        # Initialize SageMaker PipelineSession — required for v3 so that
        # processor.run(), model_trainer.train(), model_builder.register/build()
        # return step_args instead of executing jobs immediately.
        self.session = PipelineSession(boto_session=boto3.Session(region_name=region))
        self.bucket = self.session.default_bucket()
        self.account_id = boto3.client('sts').get_caller_identity()['Account']

        # Pipeline configuration
        self.config = {
            'processing_instance_type': kwargs.get('processing_instance_type', 'ml.m5.xlarge'),
            'training_instance_type': kwargs.get('training_instance_type', 'ml.m5.xlarge'),
            'transform_instance_type': kwargs.get('transform_instance_type', 'ml.m5.xlarge'),
            'framework_version': kwargs.get('framework_version', '1.4-2'),
            'py_version': kwargs.get('py_version', 'py3'),
            'xgboost_version': kwargs.get('xgboost_version', '3.0-5'),
        }

        # Lambda configuration
        self.lambda_config = {
            'deploy_function_name': kwargs.get('deploy_lambda', 'fraud-detection-deploy-endpoint'),
            'test_function_name': kwargs.get('test_lambda', 'fraud-detection-test-inference'),
            'lambda_timeout': kwargs.get('lambda_timeout', 600),
            'lambda_memory': kwargs.get('lambda_memory', 1024),
        }

        logger.info(f"Initialized pipeline: {pipeline_name}")
        logger.info(f"  Role: {self.role}")
        logger.info(f"  Region: {region}")
        logger.info(f"  Bucket: {self.bucket}")
        logger.info(f"  Account: {self.account_id}")

    def _define_parameters(self) -> Dict[str, Any]:
        """
        Define pipeline parameters.

        Returns:
            Dictionary of pipeline parameters
        """
        logger.info("Defining pipeline parameters...")

        params = {
            # Data parameters
            'athena_table': ParameterString(
                name="AthenaTable",
                default_value=ATHENA_TRAINING_TABLE
            ),
            'athena_evaluation_table': ParameterString(
                name="AthenaEvaluationTable",
                default_value=ATHENA_EVALUATION_TABLE
            ),
            'athena_filter': ParameterString(
                name="AthenaFilter",
                default_value=""
            ),
            'target_column': ParameterString(
                name="TargetColumn",
                default_value=schema.target_column()
            ),

            # Training parameters
            'training_instance_type': ParameterString(
                name="TrainingInstanceType",
                default_value=self.config['training_instance_type']
            ),
            # XGBoost hyperparameters — single source of truth is
            # src/config/config.yaml → training.xgboost_params (loaded into
            # XGBOOST_PARAMS). To tune, edit YAML; defaults below pick up the
            # change automatically. Values are also exposed as pipeline
            # parameters so they can still be overridden per-execution.
            'max_depth': ParameterInteger(
                name="MaxDepth",
                default_value=int(XGBOOST_PARAMS["max_depth"])
            ),
            'learning_rate': ParameterFloat(
                name="LearningRate",
                default_value=float(XGBOOST_PARAMS["learning_rate"])
            ),
            'num_boost_round': ParameterInteger(
                name="NumBoostRound",
                default_value=int(XGBOOST_PARAMS["num_boost_round"])
            ),
            'min_child_weight': ParameterInteger(
                name="MinChildWeight",
                default_value=int(XGBOOST_PARAMS["min_child_weight"])
            ),
            'early_stopping_rounds': ParameterInteger(
                name="EarlyStoppingRounds",
                default_value=int(XGBOOST_PARAMS["early_stopping_rounds"])
            ),

            # Evaluation parameters
            'min_roc_auc': ParameterFloat(
                name="MinRocAuc",
                default_value=0.70  # Quality gate threshold
            ),
            'min_pr_auc': ParameterFloat(
                name="MinPrAuc",
                default_value=0.30  # Quality gate threshold
            ),

            # Deployment parameters
            'endpoint_name': ParameterString(
                name="EndpointName",
                default_value="fraud-detector"
            ),
            'endpoint_memory_size': ParameterInteger(
                name="EndpointMemorySize",
                default_value=SERVERLESS_MEMORY_SIZE
            ),
            'endpoint_max_concurrency': ParameterInteger(
                name="EndpointMaxConcurrency",
                default_value=SERVERLESS_MAX_CONCURRENCY
            ),
            'enable_athena_logging': ParameterString(
                name="EnableAthenaLogging",
                default_value="true"
            ),

            # Testing parameters
            'test_num_samples': ParameterInteger(
                name="TestNumSamples",
                default_value=50
            ),

            # Model registration
            'model_approval_status': ParameterString(
                name="ModelApprovalStatus",
                default_value="Approved"  # Auto-approve for pipeline
            ),
            'model_package_group': ParameterString(
                name="ModelPackageGroup",
                default_value=MLFLOW_MODEL_NAME
            ),

            # Lineage anchors written into baseline.json. CI/CD callers
            # override CodeCommitSha; FeatureSchemaVersion is bumped only
            # when the feature list itself changes (forces retrain so the
            # drift monitor doesn't compare incompatible schemas).
            'code_commit_sha': ParameterString(
                name="CodeCommitSha",
                default_value=_resolve_code_commit_sha(),
            ),
            'feature_schema_version': ParameterInteger(
                name="FeatureSchemaVersion",
                default_value=1,
            ),
        }

        logger.info(f"Defined {len(params)} parameters")
        return params

    def _create_seed_athena_step(self) -> ProcessingStep:
        """
        Create the seed step that idempotently loads the predictions CSV
        into the Athena ``training_data`` table.

        The CloudFormation lifecycle creates the empty Iceberg table; this
        step fills it. Runs as the first step of the pipeline so preprocessing
        can depend on the table being populated. Idempotent — if the
        integrity check already passes, the script is a few-second no-op.

        seed_athena_tables.py imports ``schema`` (i.e. src/config/schema.py)
        for the feature list, so its source_dir bundles that file AND its
        sibling dataset_schema.yaml alongside the entry script. This is the
        ONE place the dataset schema is packaged into a container — see
        schema.py's module docstring for why it can't just `import src...`.

        Returns:
            ProcessingStep
        """
        logger.info("Creating Athena seed step...")

        # The seed step only needs a generic Python runtime + boto3 (it does
        # NOT import sklearn). We pin the SageMaker scikit-learn image built on
        # Python 3.12 (ECR tag `<framework_version>-py312`) instead of the old
        # 1.2-1 image (Python 3.8). The version comes from config
        # (`framework_version`, default 1.4-2) — the ONLY sklearn version knob.
        #
        # The SDK's image_uris config has no version key that maps to the
        # py312 tag — retrieve() only ever emits `<version>-cpu-py3` (Python
        # 3.10 for 1.4-2) and rejects py_version="py312". So we let retrieve()
        # resolve the region-correct registry account + repository, then swap
        # the tag to `<version>-py312`. This keeps the account/region portable
        # (the account differs per region) without hardcoding it. Verified the
        # 1.4-2-py312 image is present with an identical digest across all
        # commercial regions checked (us/eu/ap/ca/sa).
        #
        # NOTE: the configured framework_version must have a published
        # `-py312` container build. Today only 1.4-2 does; if you change
        # framework_version, confirm the `<version>-py312` tag exists in ECR.
        _sklearn_version = self.config['framework_version']
        _sklearn_base_uri = retrieve_image_uri(
            framework="sklearn",
            region=self.region,
            version=_sklearn_version,
            image_scope="training",
        )
        # e.g. <acct>.dkr.ecr.<region>.amazonaws.com/sagemaker-scikit-learn:1.4-2-cpu-py3
        #   -> <acct>.dkr.ecr.<region>.amazonaws.com/sagemaker-scikit-learn:1.4-2-py312
        sklearn_image = f"{_sklearn_base_uri.rsplit(':', 1)[0]}:{_sklearn_version}-py312"

        # Bundle seed_athena_tables.py together with schema.py + its YAML
        # sidecar in a source_dir so the container-side `import schema`
        # resolves without needing the rest of the `src` package.
        # FrameworkProcessor (not plain ScriptProcessor) is required here —
        # ScriptProcessor.run(code=...) only ever uploads a single file;
        # FrameworkProcessor.run(code=..., source_dir=...) uploads the
        # whole source_dir as a tarball and unpacks it alongside the entry
        # script in the container.
        seed_source_dir = self._build_seed_source_dir()

        processor = FrameworkProcessor(
            image_uri=sklearn_image,
            role=self.role,
            instance_type=self.config['processing_instance_type'],
            instance_count=1,
            base_job_name="fraud-seed-athena",
            sagemaker_session=self.session,
            command=["python3"],
            env={
                # seed_athena_tables.py reads all config from env vars so it
                # has no dependency on src.config.config inside the
                # container (it DOES import the bundled schema.py sibling).
                'AWS_DEFAULT_REGION': AWS_DEFAULT_REGION,
                'ATHENA_DATABASE': ATHENA_DATABASE,
                'ATHENA_OUTPUT_S3': ATHENA_OUTPUT_S3,
                'ATHENA_TRAINING_TABLE': ATHENA_TRAINING_TABLE,
                'ATHENA_EVALUATION_TABLE': ATHENA_EVALUATION_TABLE,
                'DATA_S3_BUCKET': DATA_S3_BUCKET,
                'DATA_S3_PREFIX': os.environ.get('DATA_S3_PREFIX', 'fraud-detection/'),
            },
        )

        step = ProcessingStep(
            name="SeedAthenaTrainingData",
            step_args=processor.run(
                code="seed_athena_tables.py",
                source_dir=str(seed_source_dir),
                wait=False,
            ),
        )

        logger.info("✓ Athena seed step created")
        return step

    def _stage_schema_sibling(self, target_dir: Path) -> None:
        """Copy schema.py + dataset_schema.yaml into ``target_dir`` as
        sibling files.

        Shared by every bundling call site that needs the dataset schema
        available as a same-directory import inside a SageMaker container
        (the seed step today; the training and preprocessing steps use
        this same helper once their bundling is wired up) — one
        implementation instead of three divergent copies.
        """
        import shutil

        config_dir = Path(__file__).parent.parent / "config"

        shutil.copy(config_dir / "schema.py", target_dir)
        shutil.copy(config_dir / "dataset_schema.yaml", target_dir)

    def _build_seed_source_dir(self) -> Path:
        """Stage a temp directory containing seed_athena_tables.py plus its
        schema.py + dataset_schema.yaml dependencies, so
        FrameworkProcessor's source_dir packaging uploads all three files
        together as one tarball.

        Copying into a fresh temp dir (rather than symlinking into
        pipeline_steps/) keeps the repo's own pipeline_steps/ directory
        free of duplicated schema files — the copy is a build artifact,
        not a checked-in one.
        """
        import shutil
        import tempfile

        staging_dir = Path(tempfile.mkdtemp(prefix="seed-athena-source-"))
        pipeline_steps_dir = Path(__file__).parent / "pipeline_steps"

        shutil.copy(pipeline_steps_dir / "seed_athena_tables.py", staging_dir)
        self._stage_schema_sibling(staging_dir)

        # Write a requirements.txt so FrameworkProcessor's install_requirements.py
        # installs pyyaml before running seed_athena_tables.py. schema.py imports
        # yaml at module load; the base Processing Job image doesn't ship pyyaml,
        # so without this the step crashes on `import yaml` with
        # ModuleNotFoundError.
        (staging_dir / "requirements.txt").write_text("pyyaml\n")

        logger.info("Staged seed-step source dir: %s (seed_athena_tables.py + schema.py + dataset_schema.yaml + requirements.txt)",
                   staging_dir)
        return staging_dir

    def _create_preprocessing_step(
        self,
        params: Dict[str, Any],
        seed_step: ProcessingStep,
    ) -> ProcessingStep:
        """
        Create PySpark-based preprocessing step for distributed processing.

        Uses PySparkProcessor for scalable data processing that can handle
        millions of rows efficiently across a distributed cluster.

        Args:
            params: Pipeline parameters

        Returns:
            ProcessingStep
        """
        logger.info("Creating PySpark preprocessing step...")

        # Use PySparkProcessor for distributed Spark processing
        # The SageMaker Spark 3.3 container includes AWS Glue Data Catalog JARs
        # Glue config is set in SparkSession.builder in preprocessing_pyspark.py
        processor = PySparkProcessor(
            base_job_name="fraud-preprocessing-spark",
            framework_version="3.3",  # Spark version
            role=self.role,
            instance_type=self.config['processing_instance_type'],
            instance_count=1,  # Single instance to avoid YARN multi-node issues
            max_runtime_in_seconds=3600,
            sagemaker_session=self.session,
            env={
                'AWS_DEFAULT_REGION': AWS_DEFAULT_REGION,
                'ATHENA_DATABASE': ATHENA_DATABASE,
                'ATHENA_OUTPUT_S3': ATHENA_OUTPUT_S3,
                'DATA_S3_BUCKET': DATA_S3_BUCKET,
            }
        )

        # Build job arguments
        # Pass the same S3 URIs used by the ProcessingOutput entries below as
        # explicit output-dir args. This forces preprocessing_pyspark.py to run
        # its `if output_dir.startswith('s3://')` branch, which uses Spark to
        # write part-files then boto3 to consolidate them into train.csv/test.csv
        # at the train/test prefixes. Without these args the script falls back to
        # POSIX defaults and the local shutil consolidation branch, which fails to
        # produce the CSVs (only the JSON metadata reached S3), causing the
        # downstream training step to fail with "train.csv not found".
        job_arguments = [
            "--train-table", params['athena_table'],
            "--eval-table", params['athena_evaluation_table'],
            "--target-column", params['target_column'],
            "--train-output-dir", f"s3://{self.bucket}/fraud-detection/preprocessing/train",
            "--test-output-dir", f"s3://{self.bucket}/fraud-detection/preprocessing/test",
            "--stats-output-dir", f"s3://{self.bucket}/fraud-detection/preprocessing/stats",
        ]

        # Processing step with PySpark script — v3 uses step_args from processor.run().
        # depends_on=[seed_step] forces ordering since preprocessing reads from
        # Athena directly (no S3 data flow from the seed step).
        step = ProcessingStep(
            name="PreprocessData",
            depends_on=[seed_step],
            step_args=processor.run(
                submit_app=str(Path(__file__).parent / "pipeline_steps" / "preprocessing_pyspark.py"),
                arguments=job_arguments,
                outputs=[
                    ProcessingOutput(
                        output_name="train",
                        s3_output=ProcessingS3Output(
                            local_path="/opt/ml/processing/output/train",
                            s3_uri=f"s3://{self.bucket}/fraud-detection/preprocessing/train",
                            s3_upload_mode="EndOfJob",
                        ),
                    ),
                    ProcessingOutput(
                        output_name="test",
                        s3_output=ProcessingS3Output(
                            local_path="/opt/ml/processing/output/test",
                            s3_uri=f"s3://{self.bucket}/fraud-detection/preprocessing/test",
                            s3_upload_mode="EndOfJob",
                        ),
                    ),
                    ProcessingOutput(
                        output_name="stats",
                        s3_output=ProcessingS3Output(
                            local_path="/opt/ml/processing/output/stats",
                            s3_uri=f"s3://{self.bucket}/fraud-detection/preprocessing/stats",
                            s3_upload_mode="EndOfJob",
                        ),
                    ),
                ],
                wait=False,
            ),
        )

        logger.info("✓ PySpark preprocessing step created")
        logger.info("  Framework: Spark 3.3")
        logger.info("  Instances: 1x ml.m5.xlarge (single node Spark)")
        logger.info("  Data Source: Athena via Glue Data Catalog")
        return step

    def _create_training_step(
        self,
        params: Dict[str, Any],
        preprocessing_step: ProcessingStep
    ) -> TrainingStep:
        """
        Create training step with MLflow integration.

        Uses ModelTrainer (v3) instead of XGBoost estimator (v2).
        The ModelTrainer is configured with an explicit XGBoost training image,
        SourceCode for the training script, and Compute for instance config.

        Args:
            params: Pipeline parameters
            preprocessing_step: Preprocessing step for input dependencies

        Returns:
            TrainingStep
        """
        logger.info("Creating training step...")

        # Retrieve XGBoost training image URI for the target region
        xgboost_image = retrieve_image_uri(
            framework="xgboost",
            region=self.region,
            version=self.config['xgboost_version']
        )

        # Source code configuration for the training script
        source_code = SourceCode(
            source_dir=str(Path(__file__).parent / "pipeline_steps"),
            entry_script="train.py",
        )

        # Compute configuration for training instances
        compute = Compute(
            instance_type=params['training_instance_type'],
            instance_count=1,
        )

        # ModelTrainer replaces XGBoost estimator in v3
        # MLflow is auto-installed in train.py if not present
        model_trainer = ModelTrainer(
            training_image=xgboost_image,
            source_code=source_code,
            compute=compute,
            role=self.role,
            base_job_name="fraud-training",
            output_data_config=OutputDataConfig(
                s3_output_path=f"s3://{self.bucket}/fraud-detection/training/output"
            ),
            sagemaker_session=self.session,
            hyperparameters={
                # Training script hyperparameters — must be strings for SageMaker API.
                # Pipeline parameters (ParameterInteger/Float) are wrapped with Join()
                # to convert them to their string representation at execution time.
                'max-depth': Join(on="", values=[params['max_depth']]),
                'learning-rate': Join(on="", values=[params['learning_rate']]),
                'num-boost-round': Join(on="", values=[params['num_boost_round']]),
                'min-child-weight': Join(on="", values=[params['min_child_weight']]),
                'early-stopping-rounds': Join(on="", values=[params['early_stopping_rounds']]),
                'target-column': Join(on="", values=[params['target_column']]),
                # Target type + XGBoost objective override — resolved at pipeline
                # definition time from schema.target_type() and config constants.
                # These let the container-side train.py pick the right objective
                # (binary:logistic / reg:squarederror / multi:softprob) without
                # hardcoded assumptions about the target being 0/1.
                'target-type': schema.target_type(),
                'xgboost-objective': XGBOOST_OBJECTIVE,
                'xgboost-num-class': str(XGBOOST_NUM_CLASS),
            },
            environment={
                # MLflow configuration
                'MLFLOW_TRACKING_URI': MLFLOW_TRACKING_URI if MLFLOW_TRACKING_URI else '',
                'MLFLOW_EXPERIMENT_NAME': 'credit-card-fraud-detection-training',
                'MLFLOW_MODEL_NAME': MLFLOW_MODEL_NAME,
            },
        )

        # S3 URIs for training data from preprocessing step outputs
        train_s3_uri = preprocessing_step.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri
        validation_s3_uri = preprocessing_step.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri

        # Training step using step_args from model_trainer.train() (v3 pattern)
        # In pipeline context, .train(wait=False) returns step arguments
        # instead of executing the job directly
        step = TrainingStep(
            name="TrainModel",
            step_args=model_trainer.train(
                input_data_config=[
                    InputData(channel_name="train", data_source=train_s3_uri),
                    InputData(channel_name="validation", data_source=validation_s3_uri),
                ],
                wait=False,
            ),
        )

        logger.info("✓ Training step created")
        return step

    def _create_evaluation_step(
        self,
        params: Dict[str, Any],
        training_step: TrainingStep,
        preprocessing_step: ProcessingStep
    ) -> tuple:
        """
        Create evaluation step.

        Args:
            params: Pipeline parameters
            training_step: Training step for model input
            preprocessing_step: Preprocessing step for test data

        Returns:
            Tuple of (ProcessingStep, PropertyFile) for evaluation
        """
        logger.info("Creating evaluation step...")

        # Use ScriptProcessor with XGBoost image (has xgboost pre-installed)
        xgboost_image = retrieve_image_uri(
            framework="xgboost",
            region=self.region,
            version=self.config['xgboost_version']
        )
        
        processor = ScriptProcessor(
            image_uri=xgboost_image,
            role=self.role,
            instance_type=self.config['processing_instance_type'],
            instance_count=1,
            base_job_name="fraud-evaluation",
            sagemaker_session=self.session,
            command=["python3"],
            env={
                'MLFLOW_TRACKING_URI': MLFLOW_TRACKING_URI if MLFLOW_TRACKING_URI else '',
                'MLFLOW_EXPERIMENT_NAME': 'credit-card-fraud-detection-evaluation',
                'MLFLOW_MODEL_NAME': MLFLOW_MODEL_NAME,
                'AWS_DEFAULT_REGION': AWS_DEFAULT_REGION,
                'ATHENA_DATABASE': ATHENA_DATABASE,
                'ATHENA_OUTPUT_S3': ATHENA_OUTPUT_S3,
            }
        )

        # Define property file for evaluation metrics (used by ConditionStep)
        evaluation_report = PropertyFile(
            name="EvaluationReport",
            output_name="evaluation",
            path="evaluation.json"
        )

        step = ProcessingStep(
            name="EvaluateModel",
            step_args=processor.run(
                code=str(Path(__file__).parent / "pipeline_steps" / "evaluation.py"),
                arguments=[
                    "--target-column", params['target_column'],
                    "--min-roc-auc", Join(on="", values=[params['min_roc_auc']]),
                    "--min-pr-auc", Join(on="", values=[params['min_pr_auc']]),
                    "--evaluation-table", params['athena_evaluation_table'],
                    "--training-table", params['athena_table'],
                    "--model-package-group", params['model_package_group'],
                    "--code-commit-sha", params['code_commit_sha'],
                    "--feature-schema-version",
                    Join(on="", values=[params['feature_schema_version']]),
                    "--athena-database", ATHENA_DATABASE,
                    "--athena-output-s3", ATHENA_OUTPUT_S3,
                ],
                inputs=[
                    ProcessingInput(
                        input_name="model",
                        s3_input=ProcessingS3Input(
                            s3_uri=training_step.properties.ModelArtifacts.S3ModelArtifacts,
                            s3_data_type="S3Prefix",
                            local_path="/opt/ml/processing/model",
                        ),
                    ),
                    ProcessingInput(
                        input_name="test",
                        s3_input=ProcessingS3Input(
                            s3_uri=preprocessing_step.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
                            s3_data_type="S3Prefix",
                            local_path="/opt/ml/processing/test",
                        ),
                    ),
                ],
                outputs=[
                    ProcessingOutput(
                        output_name="evaluation",
                        s3_output=ProcessingS3Output(
                            local_path="/opt/ml/processing/evaluation",
                            s3_uri=f"s3://{self.bucket}/fraud-detection/evaluation",
                            s3_upload_mode="EndOfJob",
                        ),
                    ),
                ],
                wait=False,
            ),
            property_files=[evaluation_report],
        )

        logger.info("✓ Evaluation step created")
        return step, evaluation_report

    def _create_register_model_step(
        self,
        params: Dict[str, Any],
        training_step: TrainingStep,
        evaluation_step: ProcessingStep
    ) -> ModelStep:
        """
        Create model registration step.

        Uses ModelStep wrapping Model.register() (v3) instead of the removed
        RegisterModel step collection (v2). A Model object is created explicitly
        with the XGBoost image URI and training artifacts, then model.register()
        is called in pipeline context to produce step_args.

        Args:
            params: Pipeline parameters
            training_step: Training step for model artifacts
            evaluation_step: Evaluation step for metrics

        Returns:
            ModelStep for model registration
        """
        logger.info("Creating register model step...")

        # Register baseline.json (not evaluation.json) as the ModelPackage's
        # model_statistics. baseline.json contains the same metrics plus the
        # anchor fields the drift monitor needs (evaluation_table,
        # model_package_group), so the monitor can recover its reference by
        # looking up the registered ModelPackage instead of relying on a
        # hardcoded env var.
        model_metrics = ModelMetrics(
            model_statistics=MetricsSource(
                s3_uri=Join(
                    on="/",
                    values=[
                        evaluation_step.properties.ProcessingOutputConfig.Outputs["evaluation"].S3Output.S3Uri,
                        "baseline.json"
                    ]
                ),
                content_type="application/json"
            )
        )

        # Retrieve XGBoost image URI for the model
        xgboost_image = retrieve_image_uri(
            framework="xgboost",
            region=self.region,
            version=self.config['xgboost_version']
        )

        # Create ModelBuilder with training artifacts AND the custom inference
        # handler. Passing source_code here is critical: ModelBuilder.register()
        # repackages the training tarball to include code/inference.py and sets
        # SAGEMAKER_PROGRAM/SAGEMAKER_SUBMIT_DIRECTORY on the ModelPackage so
        # the container loads in script mode when a downstream
        # CreateModel/CreateEndpoint deploys from the registry. Omitting
        # source_code produces a bare xgboost tarball that loads in algorithm
        # mode — feature_names.json gets picked up as the model file and the
        # endpoint fails to start.
        model_builder = ModelBuilder(
            image_uri=xgboost_image,
            s3_model_data_url=training_step.properties.ModelArtifacts.S3ModelArtifacts,
            role_arn=self.role,
            source_code=SourceCode(
                source_dir=str(Path(__file__).parent / "pipeline_steps"),
                entry_script="inference.py",
            ),
            sagemaker_session=self.session,
            env_vars={
                'MLFLOW_TRACKING_URI': MLFLOW_TRACKING_URI if MLFLOW_TRACKING_URI else '',
                'MLFLOW_MODEL_NAME': MLFLOW_MODEL_NAME,
                'ENABLE_ATHENA_LOGGING': 'true',
                'ENDPOINT_NAME': params['endpoint_name'],
                'ATHENA_DATABASE': ATHENA_DATABASE,
                'ATHENA_OUTPUT_S3': ATHENA_OUTPUT_S3,
                'DATA_S3_BUCKET': DATA_S3_BUCKET,
                'SQS_QUEUE_URL': os.getenv('SQS_QUEUE_URL', SQS_QUEUE_URL),
                'MODEL_VERSION': 'pipeline',
                'MLFLOW_RUN_ID': 'pipeline',
            },
        )

        # ModelStep wraps model_builder.register() to produce a pipeline step
        step = ModelStep(
            name="RegisterModel",
            step_args=model_builder.register(
                content_types=["application/json", "text/csv"],
                response_types=["application/json"],
                inference_instances=["ml.m5.xlarge", "ml.m5.large"],
                transform_instances=["ml.m5.xlarge"],
                model_package_group_name=params['model_package_group'],
                approval_status=params['model_approval_status'],
                model_metrics=model_metrics,
            ),
        )

        logger.info("✓ Register model step created")
        return step

    def _create_model_step(
        self,
        params: Dict[str, Any],
        training_step: TrainingStep
    ) -> ModelStep:
        """
        Create SageMaker model step with custom inference handler for Athena logging.

        Uses ModelStep wrapping Model.create() (v3) instead of the removed
        CreateModelStep (v2). A generic Model object is created with an explicit
        XGBoost image URI, the custom inference script, and environment variables
        for Athena logging and MLflow tracking.

        Args:
            params: Pipeline parameters
            training_step: Training step for model artifacts

        Returns:
            ModelStep for model creation
        """
        logger.info("Creating model step with custom inference handler...")

        # Retrieve XGBoost image URI for the model
        xgboost_image = retrieve_image_uri(
            framework="xgboost",
            region=self.region,
            version=self.config['xgboost_version']
        )

        # Create ModelBuilder with explicit image URI (v3 pattern — replaces XGBoostModel)
        model_builder = ModelBuilder(
            image_uri=xgboost_image,
            s3_model_data_url=training_step.properties.ModelArtifacts.S3ModelArtifacts,
            role_arn=self.role,
            source_code=SourceCode(
                source_dir=str(Path(__file__).parent / "pipeline_steps"),
                entry_script="inference.py",
            ),
            sagemaker_session=self.session,
            env_vars={
                'MLFLOW_TRACKING_URI': MLFLOW_TRACKING_URI if MLFLOW_TRACKING_URI else '',
                'MLFLOW_MODEL_NAME': MLFLOW_MODEL_NAME,
                'ENABLE_ATHENA_LOGGING': 'true',
                'ENDPOINT_NAME': params['endpoint_name'],
                'ATHENA_DATABASE': ATHENA_DATABASE,
                'ATHENA_OUTPUT_S3': ATHENA_OUTPUT_S3,
                'DATA_S3_BUCKET': DATA_S3_BUCKET,
                'SQS_QUEUE_URL': os.getenv('SQS_QUEUE_URL', SQS_QUEUE_URL),
                'MODEL_VERSION': 'pipeline',
                'MLFLOW_RUN_ID': 'pipeline',
            },
        )

        # ModelStep wraps model_builder.build() to produce a pipeline step
        step = ModelStep(
            name="CreateModel",
            step_args=model_builder.build(
                sagemaker_session=self.session,
                role_arn=self.role,
            ),
        )

        logger.info("✓ Create model step with Athena logging created")
        return step

    def _create_deploy_lambda(self) -> Lambda:
        """
        Create or get Lambda function for endpoint deployment.

        Returns:
            Lambda helper object
        """
        logger.info("Creating deploy Lambda function...")

        # Use the Lambda script file
        lambda_script_path = str(Path(__file__).parent / "pipeline_steps" / "lambda_deploy_endpoint.py")

        # Use Lambda execution role if available, otherwise fall back to SageMaker role
        lambda_role = self.lambda_config.get('lambda_role') or LAMBDA_EXEC_ROLE or self.role
        
        if lambda_role == self.role:
            logger.warning("Using SageMaker role for Lambda - ensure it has lambda.amazonaws.com in trust policy")

        deploy_lambda = Lambda(
            function_name=self.lambda_config['deploy_function_name'],
            execution_role_arn=lambda_role,
            script=lambda_script_path,
            handler="lambda_deploy_endpoint.lambda_handler",
            timeout=self.lambda_config['lambda_timeout'],
            memory_size=self.lambda_config['lambda_memory'],
            session=self.session,
        )

        logger.info(f"✓ Deploy Lambda created: {self.lambda_config['deploy_function_name']}")
        return deploy_lambda

    def _create_test_lambda(self) -> Lambda:
        """
        Create or get Lambda function for inference testing.

        Returns:
            Lambda helper object
        """
        logger.info("Creating test Lambda function...")

        # Use the Lambda script file
        lambda_script_path = str(Path(__file__).parent / "pipeline_steps" / "lambda_test_inference.py")

        # Use Lambda execution role if available, otherwise fall back to SageMaker role
        lambda_role = self.lambda_config.get('lambda_role') or LAMBDA_EXEC_ROLE or self.role
        
        if lambda_role == self.role:
            logger.warning("Using SageMaker role for Lambda - ensure it has lambda.amazonaws.com in trust policy")

        test_lambda = Lambda(
            function_name=self.lambda_config['test_function_name'],
            execution_role_arn=lambda_role,
            script=lambda_script_path,
            handler="lambda_test_inference.lambda_handler",
            timeout=self.lambda_config['lambda_timeout'],
            memory_size=self.lambda_config['lambda_memory'],
            session=self.session,
        )

        logger.info(f"✓ Test Lambda created: {self.lambda_config['test_function_name']}")
        return test_lambda

    def _create_deploy_step(
        self,
        params: Dict[str, Any],
        create_model_step: ModelStep,
        register_step: ModelStep
    ) -> LambdaStep:
        """
        Create Lambda step for endpoint deployment.

        Args:
            params: Pipeline parameters
            create_model_step: ModelStep for model name
            register_step: ModelStep for model package ARN

        Returns:
            LambdaStep
        """
        logger.info("Creating deploy step...")

        deploy_lambda = self._create_deploy_lambda()

        step = LambdaStep(
            name="DeployEndpoint",
            lambda_func=deploy_lambda,
            inputs={
                "model_name": create_model_step.properties.ModelName,
                "endpoint_name": params['endpoint_name'],
                "memory_size_mb": params['endpoint_memory_size'],
                "max_concurrency": params['endpoint_max_concurrency'],
                "enable_athena_logging": params['enable_athena_logging'],
                "model_package_arn": register_step.properties.ModelPackageArn,
                "mlflow_run_id": "pipeline",
            },
            outputs=[
                LambdaOutput(output_name="endpoint_name", output_type=LambdaOutputTypeEnum.String),
                LambdaOutput(output_name="endpoint_arn", output_type=LambdaOutputTypeEnum.String),
                LambdaOutput(output_name="status", output_type=LambdaOutputTypeEnum.String),
            ],
        )

        logger.info("✓ Deploy step created")
        return step

    def _create_test_step(
        self,
        params: Dict[str, Any],
        deploy_step: LambdaStep
    ) -> LambdaStep:
        """
        Create Lambda step for inference testing.

        Args:
            params: Pipeline parameters
            deploy_step: Deploy step for endpoint name

        Returns:
            LambdaStep
        """
        logger.info("Creating test step...")

        test_lambda = self._create_test_lambda()

        step = LambdaStep(
            name="TestInference",
            lambda_func=test_lambda,
            inputs={
                "endpoint_name": params['endpoint_name'],
                "num_samples": params['test_num_samples'],
            },
            outputs=[
                LambdaOutput(output_name="total_invocations", output_type=LambdaOutputTypeEnum.Integer),
                LambdaOutput(output_name="successful_invocations", output_type=LambdaOutputTypeEnum.Integer),
                LambdaOutput(output_name="avg_latency_ms", output_type=LambdaOutputTypeEnum.Float),
            ],
        )

        logger.info("✓ Test step created")
        return step

    def _create_fail_step(self) -> FailStep:
        """
        Create fail step for quality gate failures.

        Returns:
            FailStep
        """
        logger.info("Creating fail step...")

        step = FailStep(
            name="ModelQualityFailed",
            error_message="Model quality check failed - ROC-AUC below threshold"
        )

        logger.info("✓ Fail step created")
        return step

    def _create_condition_step(
        self,
        params: Dict[str, Any],
        evaluation_step: ProcessingStep,
        evaluation_report: PropertyFile,
        success_steps: List,
        fail_step: FailStep
    ) -> ConditionStep:
        """
        Create condition step for quality gates.

        Args:
            params: Pipeline parameters
            evaluation_step: Evaluation step for metrics
            evaluation_report: PropertyFile from evaluation step
            success_steps: Steps to execute on success
            fail_step: FailStep to execute on failure

        Returns:
            ConditionStep
        """
        logger.info("Creating condition step...")

        # Define condition: ROC-AUC >= threshold
        condition = ConditionGreaterThanOrEqualTo(
            left=JsonGet(
                step_name=evaluation_step.name,
                property_file=evaluation_report,
                json_path="binary_classification_metrics.roc_auc.value"
            ),
            right=params['min_roc_auc']
        )

        step = ConditionStep(
            name="CheckModelQuality",
            conditions=[condition],
            if_steps=success_steps,
            else_steps=[fail_step]
        )

        logger.info("✓ Condition step created")
        return step


    def create_pipeline(self, include_deployment: bool = True) -> Pipeline:
        """
        Create the complete SageMaker Pipeline.

        Args:
            include_deployment: Include deployment and testing steps

        Returns:
            Pipeline instance
        """
        logger.info("=" * 80)
        logger.info(f"Creating pipeline: {self.pipeline_name}")
        logger.info(f"  Include deployment: {include_deployment}")
        logger.info("=" * 80)

        # Step 1: Define parameters
        params = self._define_parameters()

        # Step 2: Seed Athena training_data from the predictions CSV (idempotent)
        seed_step = self._create_seed_athena_step()

        # Step 3: Create preprocessing step (depends on seed_step)
        preprocessing_step = self._create_preprocessing_step(params, seed_step)

        # Step 4: Create training step
        training_step = self._create_training_step(params, preprocessing_step)

        # Step 4: Create evaluation step (returns step and property file)
        evaluation_step, evaluation_report = self._create_evaluation_step(
            params, training_step, preprocessing_step
        )

        # Step 5: Create fail step (for quality gate failures)
        fail_step = self._create_fail_step()

        # Step 6: Create register model step
        register_step = self._create_register_model_step(
            params, training_step, evaluation_step
        )

        # Build success steps based on configuration
        if include_deployment:
            # Step 7: Create model step
            create_model_step = self._create_model_step(params, training_step)

            # Step 8: Create deploy step
            deploy_step = self._create_deploy_step(params, create_model_step, register_step)

            # Step 9: Create test step
            test_step = self._create_test_step(params, deploy_step)

            # Success path: register → create model → deploy → test
            success_steps = [register_step, create_model_step, deploy_step, test_step]
        else:
            # Success path: register only
            success_steps = [register_step]

        # Step 10: Create condition step (quality gates)
        condition_step = self._create_condition_step(
            params, evaluation_step, evaluation_report, success_steps, fail_step
        )

        # Build pipeline steps
        pipeline_steps = [
            seed_step,
            preprocessing_step,
            training_step,
            evaluation_step,
            condition_step
        ]

        # Create pipeline
        pipeline = Pipeline(
            name=self.pipeline_name,
            parameters=list(params.values()),
            steps=pipeline_steps,
            sagemaker_session=self.session,
        )

        logger.info("=" * 80)
        logger.info("✓ Pipeline created successfully")
        logger.info(f"  Total steps: {len(pipeline_steps)}")
        logger.info(f"  Parameters: {len(params)}")
        if include_deployment:
            logger.info("  Flow: Seed → Preprocess → Train → Evaluate → Quality Gate → Register → Deploy → Test")
        else:
            logger.info("  Flow: Seed → Preprocess → Train → Evaluate → Quality Gate → Register")
        logger.info("=" * 80)

        return pipeline

    def upsert_pipeline(
        self,
        description: str = "Fraud detection pipeline with MLflow monitoring",
        include_deployment: bool = True,
        tags: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Create or update pipeline.

        Args:
            description: Pipeline description
            include_deployment: Include deployment and testing steps
            tags: Optional tags

        Returns:
            Dictionary with pipeline ARN and status
        """
        logger.info(f"Upserting pipeline: {self.pipeline_name}")

        # Create pipeline
        pipeline = self.create_pipeline(include_deployment=include_deployment)

        # Add default tags (don't override user-provided tags)
        if tags is None:
            tags = []

        # Define default tags
        default_tags = [
            {'Key': 'Project', 'Value': 'FraudDetection'},
            {'Key': 'ManagedBy', 'Value': 'SageMaker'},
            {'Key': 'MLflowIntegration', 'Value': 'true'},
            {'Key': 'IncludesDeployment', 'Value': str(include_deployment).lower()},
        ]

        # Create a dictionary of existing tag keys for quick lookup
        existing_keys = {tag['Key'] for tag in tags}

        # Only add default tags if their keys don't already exist
        for default_tag in default_tags:
            if default_tag['Key'] not in existing_keys:
                tags.append(default_tag)

        # Upsert pipeline
        response = pipeline.upsert(
            role_arn=self.role,
            description=description,
            tags=tags
        )

        logger.info(f"✓ Pipeline upserted: {response['PipelineArn']}")

        return {
            'pipeline_arn': response['PipelineArn'],
            'pipeline_name': self.pipeline_name,
            'status': 'created',
            'includes_deployment': include_deployment
        }

    def start_execution(
        self,
        execution_name: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        wait: bool = False
    ) -> Dict[str, Any]:
        """
        Start pipeline execution.

        Args:
            execution_name: Optional execution name
            parameters: Pipeline parameters to override
            wait: Wait for execution to complete

        Returns:
            Dictionary with execution details
        """
        from datetime import datetime

        if execution_name is None:
            timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
            execution_name = f"{self.pipeline_name}-{timestamp}"

        logger.info(f"Starting pipeline execution: {execution_name}")

        # Create pipeline
        pipeline = self.create_pipeline()

        # Start execution
        execution = pipeline.start(
            execution_display_name=execution_name,
            parameters=parameters or {}
        )

        result = {
            'execution_arn': execution.arn,
            'execution_name': execution_name,
            'pipeline_name': self.pipeline_name,
            'status': 'Executing'
        }

        if wait:
            logger.info("Waiting for execution to complete...")
            execution.wait()
            result['status'] = execution.describe()['PipelineExecutionStatus']

        logger.info(f"✓ Execution started: {execution.arn}")
        return result


def create_fraud_detection_pipeline(
    pipeline_name: str = "fraud-detection-pipeline",
    region: str = "us-east-1",
    role: Optional[str] = None,
    **kwargs
) -> FraudDetectionPipeline:
    """
    Factory function to create fraud detection pipeline.

    Args:
        pipeline_name: Pipeline name
        region: AWS region
        role: SageMaker execution role
        **kwargs: Additional configuration

    Returns:
        FraudDetectionPipeline instance
    """
    return FraudDetectionPipeline(
        pipeline_name=pipeline_name,
        region=region,
        role=role,
        **kwargs
    )


if __name__ == '__main__':
    """Test pipeline creation."""
    import argparse

    parser = argparse.ArgumentParser(description="Create SageMaker Pipeline")
    parser.add_argument('--pipeline-name', default='fraud-detection-pipeline',
                       help='Pipeline name')
    parser.add_argument('--region', default='us-east-1',
                       help='AWS region')
    parser.add_argument('--create', action='store_true',
                       help='Create/update pipeline in SageMaker')
    parser.add_argument('--no-deployment', action='store_true',
                       help='Exclude deployment steps')
    parser.add_argument('--start', action='store_true',
                       help='Start pipeline execution after creation')
    parser.add_argument('--wait', action='store_true',
                       help='Wait for execution to complete')

    args = parser.parse_args()

    # Create pipeline
    pipeline_builder = create_fraud_detection_pipeline(
        pipeline_name=args.pipeline_name,
        region=args.region
    )

    if args.create:
        # Upsert pipeline
        result = pipeline_builder.upsert_pipeline(
            include_deployment=not args.no_deployment
        )
        print(json.dumps(result, indent=2))

        if args.start:
            # Start execution
            exec_result = pipeline_builder.start_execution(wait=args.wait)
            print(json.dumps(exec_result, indent=2))
    else:
        # Just create definition (don't upsert)
        pipeline = pipeline_builder.create_pipeline(
            include_deployment=not args.no_deployment
        )
        print(f"Pipeline definition created: {pipeline.name}")
        print(f"Steps: {len(pipeline.steps)}")
