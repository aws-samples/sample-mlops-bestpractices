"""Unit tests for `create_athena_tables.py`'s `create_all_tables()` column
ordering (Requirement 3, Task 4.1).

Exercises `create_all_tables()` against a fully mocked boto3 Athena/Glue
client (no real AWS calls) and inspects the exact `CREATE TABLE` DDL
strings that were submitted to `start_query_execution`. For each of the 7
tables, asserts that wherever the identifier column, `schema.
athena_feature_ddl()`'s feature-column block, the table's auxiliary
columns, and the typed target column are present, they appear in that
relative order -- identifier, then features, then auxiliary columns, then
target -- consistent with Requirement 3.2. Not every table uses every
piece (e.g. `monitoring_responses`' columns are drift-monitor internals
and are deliberately non-schema-driven, per design.md); the test only
asserts ordering among the pieces a given table actually contains, and
separately confirms `monitoring_responses` contains none of them.

Validates: Requirements 3.1, 3.2
"""

from __future__ import annotations

from typing import Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from src.config import schema
from src.setup import create_athena_tables as cat


# Tables whose columns are drift-monitor internals unrelated to the dataset
# schema (no identifier/feature/auxiliary/target blocks): monitoring_responses
# holds per-run verdicts, monitoring_run_inferences is the run->inference
# membership bridge, monitoring_run_generations holds membership-completion
# markers, and monitoring_alerts is the notification outbox. All deliberately
# non-schema-driven, per design.md.
NON_SCHEMA_DRIVEN_TABLES = {
    "monitoring_responses", "monitoring_run_inferences",
    "monitoring_run_generations", "monitoring_alerts",
}


class _EntityNotFoundException(Exception):
    """Stand-in for the real `glue.exceptions.EntityNotFoundException`,
    which boto3 normally generates dynamically per-client."""


def _make_mock_glue() -> MagicMock:
    glue = MagicMock()
    glue.exceptions.EntityNotFoundException = _EntityNotFoundException
    glue.get_table.side_effect = _EntityNotFoundException()
    return glue


def _make_mock_athena() -> MagicMock:
    athena = MagicMock()
    athena.start_query_execution.return_value = {"QueryExecutionId": "test-query-id"}
    athena.get_query_execution.return_value = {
        "QueryExecution": {"Status": {"State": "SUCCEEDED"}}
    }
    return athena


@pytest.fixture
def submitted_ddls(monkeypatch) -> Dict[str, str]:
    """Run `create_all_tables()` against mocked boto3 clients and return
    the CREATE TABLE SQL actually submitted to Athena, keyed by table name.

    Uses `skip_existing=False` so every table's DDL is submitted
    unconditionally (skip-existing goes through `table_exists()`/Glue,
    which is out of scope for this test -- covered by Task 4.2/4.3).
    """
    mock_athena = _make_mock_athena()
    mock_glue = _make_mock_glue()

    def _fake_client(service_name, **kwargs):
        if service_name == "athena":
            return mock_athena
        if service_name == "glue":
            return mock_glue
        raise AssertionError(f"Unexpected boto3 client requested: {service_name!r}")

    monkeypatch.setattr(cat.boto3, "client", _fake_client)

    results = cat.create_all_tables(skip_existing=False)
    assert results == {name: True for name in cat.ALL_TABLE_NAMES}
    assert mock_athena.start_query_execution.call_count == len(cat.ALL_TABLE_NAMES)

    submitted: Dict[str, str] = {}
    for call in mock_athena.start_query_execution.call_args_list:
        sql = call.kwargs["QueryString"]
        context = call.kwargs["QueryExecutionContext"]
        matches = [name for name in cat.ALL_TABLE_NAMES if f".{name} " in sql]
        assert len(matches) == 1, (
            f"Expected exactly one table name in submitted SQL, found {matches}: {sql}"
        )
        table_name = matches[0]
        assert table_name not in submitted, f"{table_name} DDL submitted more than once"
        assert context["Database"] == cat.ATHENA_DATABASE
        submitted[table_name] = sql

    return submitted


def test_all_tables_have_ddl_submitted(submitted_ddls):
    """Requirement 3.1: create_all_tables() creates every registered table
    (the schema/monitoring tables, the monitoring_run_inferences bridge, the
    monitoring_run_generations completion markers, and the monitoring_alerts
    outbox)."""
    assert set(submitted_ddls.keys()) == set(cat.ALL_TABLE_NAMES)
    assert len(cat.ALL_TABLE_NAMES) == 10


def _index_of(haystack: str, needle: str) -> Optional[int]:
    """Return the index of `needle` in `haystack`, or None if `needle` is
    empty or not present."""
    if not needle:
        return None
    idx = haystack.find(needle)
    return idx if idx != -1 else None


def _schema_driven_anchors(ddl: str) -> List[tuple]:
    """Locate the four Requirement-3.2 building blocks inside a table's raw
    DDL text, as exact substrings matching how `_table_ddls()` splices them
    in: the identifier column declaration, the whole
    `schema.athena_feature_ddl()` fragment, the whole joined auxiliary-column
    fragment, and the typed target-column declaration.

    Returns only the (label, index) pairs that are actually present, in the
    order they were checked (identifier, features, auxiliary, target) --
    NOT necessarily in the order they appear in the string. Callers compare
    indices to assert ordering.
    """
    id_decl = f"{schema.identifier_column()} STRING"
    feature_ddl = schema.athena_feature_ddl()
    aux_cols = schema.auxiliary_columns()
    aux_ddl = ", ".join(f"{c.name} {c.athena_type}" for c in aux_cols)
    target_athena_type = schema.ATHENA_TYPE_MAP[schema.target_type()]
    target_decl = f"{schema.target_column()} {target_athena_type}"

    anchors = [
        ("identifier", _index_of(ddl, id_decl)),
        ("features", _index_of(ddl, feature_ddl)),
        ("auxiliary", _index_of(ddl, aux_ddl)),
        ("target", _index_of(ddl, target_decl)),
    ]
    return [(label, idx) for label, idx in anchors if idx is not None]


@pytest.mark.parametrize("table_name", cat.ALL_TABLE_NAMES)
def test_column_order_for_table(submitted_ddls, table_name):
    """Requirement 3.2: wherever a table's DDL contains the identifier
    column, the feature-DDL block, the auxiliary-column block, and/or the
    typed target column, they appear in that relative order."""
    ddl = submitted_ddls[table_name]
    present_anchors = _schema_driven_anchors(ddl)

    # Every schema-driven table (all but the non-schema-driven monitoring
    # tables) must at least declare the identifier column.
    if table_name not in NON_SCHEMA_DRIVEN_TABLES:
        labels_present = {label for label, _ in present_anchors}
        assert "identifier" in labels_present, (
            f"{table_name}: expected the identifier column "
            f"({schema.identifier_column()!r}) to be declared"
        )

    # Anchors are appended in canonical order (identifier, features,
    # auxiliary, target) above, so a simple pairwise increasing-index check
    # over the *present* subset verifies Requirement 3.2's ordering.
    for (label_a, idx_a), (label_b, idx_b) in zip(present_anchors, present_anchors[1:]):
        assert idx_a < idx_b, (
            f"{table_name}: expected {label_a!r} (index {idx_a}) to appear "
            f"before {label_b!r} (index {idx_b}) in the generated DDL"
        )


def test_training_and_evaluation_data_use_full_pattern(submitted_ddls):
    """training_data/evaluation_data are the two tables that exercise the
    complete Requirement 3.2 pattern: identifier, features, auxiliary
    columns, typed target, then table-specific metadata columns."""
    aux_names = [c.name for c in schema.auxiliary_columns()]
    for table_name in ("training_data", "evaluation_data"):
        ddl = submitted_ddls[table_name]
        anchors = dict(_schema_driven_anchors(ddl))
        assert set(anchors.keys()) == {"identifier", "features", "auxiliary", "target"}, (
            f"{table_name}: expected all four building blocks to be present"
        )
        # Table-specific metadata columns (data_version, created_at,
        # updated_at) must follow the typed target column.
        metadata_idx = ddl.find("data_version STRING")
        assert metadata_idx > anchors["target"], (
            f"{table_name}: expected metadata columns after the target column"
        )
        if aux_names:
            assert anchors["identifier"] < anchors["features"] < anchors["auxiliary"] < anchors["target"]


def test_drifted_data_uses_identifier_features_and_target_only(submitted_ddls):
    """drifted_data omits auxiliary columns and trailing metadata columns."""
    ddl = submitted_ddls["drifted_data"]
    anchors = dict(_schema_driven_anchors(ddl))
    assert set(anchors.keys()) == {"identifier", "features", "target"}
    assert anchors["identifier"] < anchors["features"] < anchors["target"]


def test_monitoring_responses_is_not_schema_driven(submitted_ddls):
    """Per design.md: monitoring_responses' columns are drift-monitor
    internals unrelated to the dataset schema, kept as literal columns."""
    ddl = submitted_ddls["monitoring_responses"]
    anchors = _schema_driven_anchors(ddl)
    assert anchors == [], (
        "monitoring_responses is documented as non-schema-driven; expected "
        f"none of the identifier/feature/auxiliary/target anchors, found {anchors}"
    )
