"""Unit tests for `src.config.schema`'s accessor functions.

Covers Requirement 2's accessor contracts (2.3-2.10, 2.20) against two
sources of truth:

- The real, checked-in `src/config/dataset_schema.yaml` -- exercises the
  "declared" path for every accessor (identifier/timestamp/target column,
  a declared `target_type`, `features()`/`feature_names()`,
  `auxiliary_columns()`, and a declared `split` section).
- A temporary, monkeypatched YAML file that omits `target_type` and/or
  `split` -- exercises the "omitted, falls back to default" path
  (`target_type()` defaulting to `"boolean"`, and `split_config()`
  defaulting `hash_column` to `identifier_column()`).

`schema.py` resolves its YAML path once at import time
(`_SCHEMA_YAML_PATH`) and caches the parsed document via
`functools.lru_cache` on `_load()`. To point the loader at a temporary
file within a test, we monkeypatch `schema._SCHEMA_YAML_PATH` and clear
`schema._load`'s cache so the next accessor call re-parses. An autouse
fixture clears the cache before and after every test so no test's parsed
document leaks into another test via the cache.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

from src.config import schema


@pytest.fixture(autouse=True)
def _clear_schema_cache():
    """Ensure every test starts and ends with a cold `_load()` cache.

    Without this, whichever test happens to run first would "win" the
    cache for the rest of the process, since `_load` is a
    process-lifetime `functools.lru_cache(maxsize=1)`.
    """
    schema._load.cache_clear()
    yield
    schema._load.cache_clear()


def _write_schema_yaml(tmp_path: Path, dataset: Dict[str, Any]) -> Path:
    """Write `{"dataset": dataset}` to a temp YAML file and return its path."""
    path = tmp_path / "dataset_schema.yaml"
    with open(path, "w") as f:
        yaml.safe_dump({"dataset": dataset}, f)
    return path


def _use_temp_schema(monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
    """Point `schema._load()` at `path` instead of the real checked-in YAML."""
    monkeypatch.setattr(schema, "_SCHEMA_YAML_PATH", path)
    schema._load.cache_clear()


# ---------------------------------------------------------------------------
# Against the real, checked-in dataset_schema.yaml
# ---------------------------------------------------------------------------


def test_identifier_column_returns_declared_value():
    assert schema.identifier_column() == "transaction_id"


def test_timestamp_column_returns_declared_value():
    assert schema.timestamp_column() == "transaction_timestamp"


def test_target_column_returns_declared_value():
    assert schema.target_column() == "is_fraud"


def test_target_type_returns_declared_value():
    # The checked-in YAML explicitly declares `target_type: boolean`.
    assert schema.target_type() == "boolean"


def test_features_returns_ordered_feature_objects():
    features = schema.features()

    assert len(features) > 0
    assert all(isinstance(f, schema.Feature) for f in features)
    # First and last entries match the checked-in YAML's declared order.
    assert features[0] == schema.Feature("transaction_hour", "double")
    assert features[-1] == schema.Feature("available_credit_ratio", "double")
    # A string-typed feature is preserved with its declared type.
    assert schema.Feature("customer_gender", "string") in features


def test_feature_names_matches_features_order():
    feature_names = schema.feature_names()
    features = schema.features()

    assert feature_names == [f.name for f in features]
    assert feature_names[0] == "transaction_hour"
    assert feature_names[-1] == "available_credit_ratio"


def test_auxiliary_columns_returns_declared_feature_objects():
    aux = schema.auxiliary_columns()

    assert aux == [
        schema.Feature("fraud_prediction", "boolean"),
        schema.Feature("fraud_probability", "double"),
    ]


def test_split_config_returns_declared_values():
    split = schema.split_config()

    assert split["strategy"] == "deterministic_hash"
    assert split["train_ratio"] == 0.8
    # The checked-in YAML declares hash_column explicitly, and it happens
    # to equal identifier_column() -- but this asserts the declared value
    # is returned as-is, not re-derived.
    assert split["hash_column"] == "transaction_id"
    assert split["hash_column"] == schema.identifier_column()


# ---------------------------------------------------------------------------
# Default behavior when optional keys are omitted from the YAML
# ---------------------------------------------------------------------------


def test_target_type_defaults_to_boolean_when_omitted(tmp_path, monkeypatch):
    dataset = {
        "identifier_column": "row_id",
        "timestamp_column": "event_time",
        "target_column": "label",
        # target_type intentionally omitted
        "features": [{"name": "f1", "type": "double"}],
    }
    _use_temp_schema(monkeypatch, _write_schema_yaml(tmp_path, dataset))

    assert schema.target_type() == "boolean"


def test_split_config_defaults_hash_column_to_identifier_column_when_split_omitted(
    tmp_path, monkeypatch
):
    dataset = {
        "identifier_column": "row_id",
        "timestamp_column": "event_time",
        "target_column": "label",
        "features": [{"name": "f1", "type": "double"}],
        # split section intentionally omitted entirely
    }
    _use_temp_schema(monkeypatch, _write_schema_yaml(tmp_path, dataset))

    split = schema.split_config()

    assert split["hash_column"] == schema.identifier_column() == "row_id"
    assert split["strategy"] == "deterministic_hash"
    assert split["train_ratio"] == 0.8


def test_auxiliary_columns_defaults_to_empty_list_when_omitted(tmp_path, monkeypatch):
    dataset = {
        "identifier_column": "row_id",
        "timestamp_column": "event_time",
        "target_column": "label",
        "features": [{"name": "f1", "type": "double"}],
        # auxiliary_columns intentionally omitted
    }
    _use_temp_schema(monkeypatch, _write_schema_yaml(tmp_path, dataset))

    assert schema.auxiliary_columns() == []


# ---------------------------------------------------------------------------
# problem_type() — drives which metrics/preset the model-quality monitor uses
# ---------------------------------------------------------------------------


def _base_dataset(**overrides: Any) -> Dict[str, Any]:
    dataset = {
        "identifier_column": "row_id",
        "timestamp_column": "event_time",
        "target_column": "label",
        "features": [{"name": "f1", "type": "double"}],
    }
    dataset.update(overrides)
    return dataset


def test_problem_type_defaults_to_binary_for_checked_in_boolean_target(monkeypatch):
    # The checked-in YAML declares a boolean target with no explicit
    # problem_type — a boolean target must resolve to binary_classification,
    # preserving the historical drift-monitor behavior.
    monkeypatch.delenv("PROBLEM_TYPE", raising=False)
    assert schema.problem_type() == "binary_classification"


def test_problem_type_env_var_takes_precedence(tmp_path, monkeypatch):
    # Even with a boolean target that would infer binary, the env override wins.
    _use_temp_schema(monkeypatch, _write_schema_yaml(tmp_path, _base_dataset(target_type="boolean")))
    monkeypatch.setenv("PROBLEM_TYPE", "regression")
    assert schema.problem_type() == "regression"


def test_problem_type_env_var_accepts_shorthand(tmp_path, monkeypatch):
    _use_temp_schema(monkeypatch, _write_schema_yaml(tmp_path, _base_dataset()))
    monkeypatch.setenv("PROBLEM_TYPE", "multiclass")
    assert schema.problem_type() == "multiclass_classification"


def test_problem_type_env_var_is_case_insensitive(tmp_path, monkeypatch):
    _use_temp_schema(monkeypatch, _write_schema_yaml(tmp_path, _base_dataset()))
    monkeypatch.setenv("PROBLEM_TYPE", "  Regression  ")
    assert schema.problem_type() == "regression"


def test_problem_type_invalid_env_var_falls_through_to_inference(tmp_path, monkeypatch):
    # An unrecognized env value is ignored (not honored, not crashing), so
    # resolution falls through to the schema / target_type inference below.
    _use_temp_schema(monkeypatch, _write_schema_yaml(tmp_path, _base_dataset(target_type="float")))
    monkeypatch.setenv("PROBLEM_TYPE", "not-a-real-type")
    assert schema.problem_type() == "regression"


def test_problem_type_explicit_yaml_key(tmp_path, monkeypatch):
    monkeypatch.delenv("PROBLEM_TYPE", raising=False)
    _use_temp_schema(
        monkeypatch,
        _write_schema_yaml(tmp_path, _base_dataset(problem_type="multiclass_classification")),
    )
    assert schema.problem_type() == "multiclass_classification"


def test_problem_type_explicit_yaml_key_beats_target_type_inference(tmp_path, monkeypatch):
    # A float target_type would infer regression, but an explicit problem_type
    # key overrides the inference.
    monkeypatch.delenv("PROBLEM_TYPE", raising=False)
    _use_temp_schema(
        monkeypatch,
        _write_schema_yaml(
            tmp_path, _base_dataset(target_type="float", problem_type="binary")
        ),
    )
    assert schema.problem_type() == "binary_classification"


@pytest.mark.parametrize(
    "target_type,expected",
    [
        ("boolean", "binary_classification"),
        ("bool", "binary_classification"),
        ("float", "regression"),
        ("double", "regression"),
        ("numeric", "regression"),
        ("real", "regression"),
        ("continuous", "regression"),
        ("categorical", "multiclass_classification"),
        ("string", "multiclass_classification"),
        ("multiclass", "multiclass_classification"),
        # int is the fraud reference's 0/1 label — must default to binary,
        # NOT regression, or an existing integer-labelled deployment flips
        # to the wrong metric path.
        ("int", "binary_classification"),
        ("integer", "binary_classification"),
        # Unrecognized types fall back conservatively to binary.
        ("something_weird", "binary_classification"),
    ],
)
def test_problem_type_inferred_from_target_type(
    tmp_path, monkeypatch, target_type, expected
):
    monkeypatch.delenv("PROBLEM_TYPE", raising=False)
    _use_temp_schema(
        monkeypatch, _write_schema_yaml(tmp_path, _base_dataset(target_type=target_type))
    )
    assert schema.problem_type() == expected
