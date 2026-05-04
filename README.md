# Sample MLOps Best Practices

A collection of production-ready patterns and reference architectures for MLOps and LLMOps on AWS. Each project demonstrates end-to-end best practices for training, deploying, monitoring, and governing machine learning models in production.

## Projects

### [Automated Drift and Trend Monitoring](sagemaker-automated-drift-and-trend-monitoring/)

An end-to-end MLOps system built on Amazon SageMaker, MLflow, and Evidently AI that trains an XGBoost fraud detection model, logs every prediction to an Athena Iceberg data lake, and runs automated daily drift checks. Includes SNS alerting, ground truth integration, and a QuickSight governance dashboard.

Covers: SageMaker Pipelines, MLflow experiment tracking, Evidently drift detection, async inference logging (SQS + Lambda), EventBridge scheduling, QuickSight dashboards.

## Related Resources

- [sample-aiops-on-amazon-sagemakerai](https://github.com/aws-samples/sample-aiops-on-amazon-sagemakerai) — Additional AIOps patterns and operational best practices on Amazon SageMaker AI.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.
