"""Single source of truth for the dataset's feature schema.

Parses ``dataset_schema.yaml`` (a sibling file) and exposes typed accessors
for column names, types, and roles (identifier / timestamp / target /
feature / auxiliary). Every piece of code in this project that needs to
know a feature name, a column type, or the target/identifier column reads
it from here — nothing else in the codebase hardcodes a feature list.

Design constraint — this module MUST be importable with only the Python
standard library + PyYAML, and MUST NOT import anything from
``src.config.config`` or any other project module. It ships standalone
into SageMaker ScriptProcessor / PySparkProcessor containers and the
drift-monitor Lambda container image, none of which have the rest of the
``src`` package on PYTHONPATH. Locate ``dataset_schema.yaml`` via
``Path(__file__).parent`` so this works whether the file lives at its
normal repo path (``src/config/schema.py``) or is copied/mounted
standalone alongside its YAML sibling in a container.

Canonical column order (used everywhere — Athena DDL, the CSV loader, the
seed script's staging table): ``[identifier_column] + feature_names() +
[auxiliary column names] + [target_column]``. There is exactly one order;
nothing in this project should re-derive or hardcode a different one.

Usage::

    from src.config import schema

    schema.feature_names()          # -> ["transaction_timestamp", "transaction_hour", ...]
    schema.target_column()          # -> "is_fraud"
    schema.identifier_column()      # -> "transaction_id"
    schema.csv_column_order()       # -> full canonical column order
    schema.athena_feature_ddl()     # -> "transaction_timestamp DOUBLE, transaction_hour DOUBLE, ..."
    schema.cast_expr(feature)       # -> "CAST(transaction_hour AS DOUBLE) AS transaction_hour"
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml

_SCHEMA_YAML_PATH = Path(__file__).resolve().parent / "dataset_schema.yaml"

# Maps the friendly `type:` values used in dataset_schema.yaml to Athena/
# Iceberg SQL column types. Extend this if you need a type dataset_schema.yaml
# doesn't yet support (e.g. `bigint`, `float`).
ATHENA_TYPE_MAP: Dict[str, str] = {
    "double": "DOUBLE",
    "string": "STRING",
    "boolean": "BOOLEAN",
    "int": "INT",
    "timestamp": "TIMESTAMP",
}


@dataclass(frozen=True)
class Feature:
    """One column: its name and its declared friendly type."""

    name: str
    type: str  # one of ATHENA_TYPE_MAP's keys

    @property
    def athena_type(self) -> str:
        try:
            return ATHENA_TYPE_MAP[self.type]
        except KeyError:
            raise ValueError(
                f"Unknown type '{self.type}' for column '{self.name}' in "
                f"dataset_schema.yaml. Supported types: {sorted(ATHENA_TYPE_MAP)}"
            )


@functools.lru_cache(maxsize=1)
def _load() -> Dict[str, Any]:
    """Parse dataset_schema.yaml once per process and cache the result."""
    if not _SCHEMA_YAML_PATH.exists():
        raise FileNotFoundError(
            f"dataset_schema.yaml not found at {_SCHEMA_YAML_PATH}. "
            "This file must ship alongside schema.py — see "
            "src/config/dataset_schema.yaml in the repo."
        )
    with open(_SCHEMA_YAML_PATH, "r") as f:
        doc = yaml.safe_load(f) or {}
    dataset = doc.get("dataset")
    if not dataset:
        raise ValueError(
            f"dataset_schema.yaml at {_SCHEMA_YAML_PATH} is missing the top-level "
            "'dataset:' key."
        )
    return dataset


def identifier_column() -> str:
    """Unique row identifier column name (STRING in Athena)."""
    return _load()["identifier_column"]


def timestamp_column() -> str:
    """Primary event-timestamp column name (also present in features())."""
    return _load()["timestamp_column"]


def target_column() -> str:
    """Binary classification target column name."""
    return _load()["target_column"]


def target_type() -> str:
    """Friendly type of the target column (default: boolean)."""
    return _load().get("target_type", "boolean")


def problem_type() -> str:
    """ML problem type this model solves — drives which metrics and Evidently
    preset the model-quality monitor uses.

    Returns one of: ``"binary_classification"``, ``"multiclass_classification"``,
    ``"regression"``.

    Resolution order:
      1. ``PROBLEM_TYPE`` environment variable (lets the Lambda deploy step
         override without touching the schema file). Accepted values are the
         three above, plus the shorthands ``"binary"``, ``"multiclass"``,
         ``"regression"``.
      2. An explicit ``problem_type`` key in dataset_schema.yaml.
      3. Inferred from ``target_type()``:
           - boolean / bool                         -> binary_classification
           - float / double / numeric / real /
             regression / continuous               -> regression
           - categorical / string / multiclass      -> multiclass_classification
           - anything else (incl. int/integer)      -> binary_classification
             (the fraud reference target is an integer 0/1 label, so integer
             defaults to binary rather than regression).

    This is deliberately conservative: an unrecognized configuration falls
    back to binary classification, which is the historical behavior of the
    drift monitor — so adding this accessor never silently changes an
    existing binary deployment.
    """
    import os

    _CANON = {
        "binary": "binary_classification",
        "binary_classification": "binary_classification",
        "multiclass": "multiclass_classification",
        "multiclass_classification": "multiclass_classification",
        "regression": "regression",
    }

    env_val = os.environ.get("PROBLEM_TYPE")
    if env_val:
        canon = _CANON.get(env_val.strip().lower())
        if canon:
            return canon

    explicit = _load().get("problem_type")
    if explicit:
        canon = _CANON.get(str(explicit).strip().lower())
        if canon:
            return canon

    t = target_type().strip().lower()
    if t in ("boolean", "bool"):
        return "binary_classification"
    if t in ("float", "double", "numeric", "real", "regression", "continuous"):
        return "regression"
    if t in ("categorical", "string", "multiclass", "multiclass_classification"):
        return "multiclass_classification"
    return "binary_classification"


def features() -> List[Feature]:
    """All model feature columns, in canonical order."""
    return [Feature(f["name"], f["type"]) for f in _load()["features"]]


def feature_names() -> List[str]:
    """Feature column names only, in canonical order."""
    return [f.name for f in features()]


def auxiliary_columns() -> List[Feature]:
    """Columns present in source data but not used as model features
    (e.g. a stored prior prediction/probability for audit purposes)."""
    return [Feature(c["name"], c["type"]) for c in _load().get("auxiliary_columns", [])]


def split_config() -> Dict[str, Any]:
    """Train/eval split configuration (strategy, ratio, hash column)."""
    return _load().get("split", {
        "strategy": "deterministic_hash",
        "train_ratio": 0.8,
        "hash_column": identifier_column(),
    })


def csv_column_order() -> List[str]:
    """Canonical column order for the raw predictions CSV and the Athena
    staging table declared over it.

    Order: [identifier_column] + feature_names() + [auxiliary column
    names] + [target_column]. This is THE single order used everywhere —
    the CSV loader must write columns in this order, and the seed script's
    staging table declares columns in this same order.
    """
    return (
        [identifier_column()]
        + feature_names()
        + [c.name for c in auxiliary_columns()]
        + [target_column()]
    )


def athena_feature_ddl(extra_columns: List[Feature] = None) -> str:
    """Comma-joined ``name TYPE`` fragment for feature columns, suitable
    for splicing into a ``CREATE TABLE`` statement.

    Args:
        extra_columns: Optional additional columns to append after the
            features (e.g. auxiliary columns for training_data/
            evaluation_data/drifted_data DDL).
    """
    cols = list(features())
    if extra_columns:
        cols = cols + list(extra_columns)
    return ", ".join(f"{c.name} {c.athena_type}" for c in cols)


def cast_expr(feature: Feature) -> str:
    """SQL expression casting a staging (all-STRING) column to its
    declared Athena type, aliased back to its own name.

    STRING-typed columns pass through unchanged (already the staging
    type). BOOLEAN columns are lowercased before casting — pandas writes
    boolean columns as capitalized "True"/"False" strings, and Athena's
    ``CAST(x AS BOOLEAN)`` only recognizes lowercase "true"/"false".
    Every other type gets a direct CAST.
    """
    if feature.type == "string":
        return feature.name
    if feature.type == "boolean":
        return f"CAST(lower({feature.name}) AS BOOLEAN) AS {feature.name}"
    return f"CAST({feature.name} AS {feature.athena_type}) AS {feature.name}"


def all_feature_names_including_aux() -> List[str]:
    """Feature names plus auxiliary column names — the full non-target,
    non-identifier column set. Useful for iterating "everything except
    the target" without duplicating the concatenation logic."""
    return feature_names() + [c.name for c in auxiliary_columns()]


if __name__ == "__main__":
    # Quick manual sanity check: `python -m src.config.schema`
    print(f"Schema file:        {_SCHEMA_YAML_PATH}")
    print(f"Identifier column:  {identifier_column()}")
    print(f"Timestamp column:   {timestamp_column()}")
    print(f"Target column:      {target_column()} ({target_type()})")
    print(f"Feature count:      {len(features())}")
    print(f"Auxiliary columns:  {[c.name for c in auxiliary_columns()]}")
    print(f"CSV column order:   {csv_column_order()}")
    print(f"Athena feature DDL: {athena_feature_ddl()}")
