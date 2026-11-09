"""Migrations build the same schema the ORM declares."""

from __future__ import annotations

from alembic import command
from sqlalchemy import create_engine, inspect, text

from app.entrypoint import _alembic_config
from tests.conftest import PG_DSN, sync_dsn

EXPECTED_TABLES = {"clients", "keys", "credentials", "models", "usage_log"}


def _upgrade_into(schema: str, revision: str = "head"):
    cfg = _alembic_config(sync_dsn(PG_DSN), schema)
    command.upgrade(cfg, revision)
    return cfg


def _engine(schema: str):
    return create_engine(
        sync_dsn(PG_DSN), connect_args={"options": f"-csearch_path={schema}"})


def _tables(schema: str) -> set[str]:
    engine = _engine(schema)
    try:
        return set(inspect(engine).get_table_names(schema=schema))
    finally:
        engine.dispose()


def _columns(schema: str, table: str) -> set[str]:
    engine = _engine(schema)
    try:
        return {c["name"] for c in inspect(engine).get_columns(table, schema=schema)}
    finally:
        engine.dispose()


def test_upgrade_creates_every_table(pg_schema):
    _upgrade_into(pg_schema)
    tables = _tables(pg_schema)
    assert EXPECTED_TABLES <= tables
    assert "users" not in tables
    assert "downloads" not in tables


def test_models_table_has_credential_id_and_no_api_key(pg_schema):
    _upgrade_into(pg_schema)
    cols = _columns(pg_schema, "models")
    assert "credential_id" in cols
    assert "auth_scheme" in cols
    assert "api_key" not in cols


def test_usage_log_has_cache_token_columns(pg_schema):
    _upgrade_into(pg_schema)
    cols = _columns(pg_schema, "usage_log")
    assert "cache_creation_input_tokens" in cols
    assert "cache_read_input_tokens" in cols


def test_models_has_cache_cost_columns(pg_schema):
    _upgrade_into(pg_schema)
    cols = _columns(pg_schema, "models")
    assert "cost_per_million_cache_write" in cols
    assert "cost_per_million_cache_read" in cols


def test_downgrade_drops_everything(pg_schema):
    cfg = _upgrade_into(pg_schema)
    command.downgrade(cfg, "base")
    assert not (EXPECTED_TABLES & _tables(pg_schema))


def test_a_custom_upstream_path_keeps_its_own_uri(pg_schema):
    """A route this migration cannot classify must go on working.

    Attaching such a row to a provider that serves no shape would resolve
    nothing, turning a deployment that worked before the upgrade into a 503
    that nothing in the migration output would have warned about.
    """
    cfg = _upgrade_into(pg_schema, "0003_model_cache_costs")

    engine = _engine(pg_schema)
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO models (deployment_name, target_uri, auth_scheme,"
                " enabled, admin_only) VALUES ('odd', 'https://vendor.example/infer',"
                " 'bearer', true, false)"
            )
        )
    engine.dispose()

    command.upgrade(cfg, "head")

    engine = _engine(pg_schema)
    try:
        with engine.begin() as conn:
            row = conn.execute(
                text("SELECT provider_id, target_uri FROM models WHERE deployment_name='odd'")
            ).one()
            providers = conn.execute(text("SELECT COUNT(*) FROM providers")).scalar_one()
    finally:
        engine.dispose()

    assert row.provider_id is None
    assert row.target_uri == "https://vendor.example/infer"
    assert providers == 0


def test_existing_models_are_backfilled_to_a_thirty_percent_rotation_threshold(pg_schema):
    """Every row that predates the column is listed with a threshold, so it
    must arrive with one rather than as null."""
    cfg = _upgrade_into(pg_schema, "0007_model_accepts_images")

    engine = _engine(pg_schema)
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO models (deployment_name, target_uri, auth_scheme,"
                " enabled, admin_only, context_window) VALUES"
                " ('old', 'https://vendor.example/v1/messages', 'bearer', true, false, 200000)"
            )
        )
    engine.dispose()

    command.upgrade(cfg, "head")

    engine = _engine(pg_schema)
    try:
        with engine.begin() as conn:
            value = conn.execute(
                text("SELECT rotation_threshold_percent FROM models WHERE deployment_name='old'")
            ).scalar_one()
    finally:
        engine.dispose()

    assert value == 30
