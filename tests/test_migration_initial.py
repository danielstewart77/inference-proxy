"""Migrations build the same schema the ORM declares."""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config as AlembicConfig
from sqlalchemy import create_engine, inspect

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPECTED_TABLES = {"clients", "keys", "credentials", "models", "usage_log"}


def _upgrade_into(db_path: Path) -> AlembicConfig:
    cfg = AlembicConfig(str(REPO_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(REPO_ROOT / "migrations"))
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
    command.upgrade(cfg, "head")
    return cfg


def test_upgrade_creates_every_table(tmp_path):
    db = tmp_path / "proxy.db"
    _upgrade_into(db)
    engine = create_engine(f"sqlite:///{db}")
    try:
        tables = set(inspect(engine).get_table_names())
    finally:
        engine.dispose()
    assert EXPECTED_TABLES <= tables
    assert "users" not in tables
    assert "downloads" not in tables


def test_models_table_has_credential_id_and_no_api_key(tmp_path):
    db = tmp_path / "proxy.db"
    _upgrade_into(db)
    engine = create_engine(f"sqlite:///{db}")
    try:
        cols = {c["name"] for c in inspect(engine).get_columns("models")}
    finally:
        engine.dispose()
    assert "credential_id" in cols
    assert "auth_scheme" in cols
    assert "api_key" not in cols


def test_usage_log_has_cache_token_columns(tmp_path):
    db = tmp_path / "proxy.db"
    _upgrade_into(db)
    engine = create_engine(f"sqlite:///{db}")
    try:
        cols = {c["name"] for c in inspect(engine).get_columns("usage_log")}
    finally:
        engine.dispose()
    assert "cache_creation_input_tokens" in cols
    assert "cache_read_input_tokens" in cols


def test_models_has_cache_cost_columns(tmp_path):
    db = tmp_path / "proxy.db"
    _upgrade_into(db)
    engine = create_engine(f"sqlite:///{db}")
    try:
        cols = {c["name"] for c in inspect(engine).get_columns("models")}
    finally:
        engine.dispose()
    assert "cost_per_million_cache_write" in cols
    assert "cost_per_million_cache_read" in cols


def test_downgrade_drops_everything(tmp_path):
    db = tmp_path / "proxy.db"
    cfg = _upgrade_into(db)
    command.downgrade(cfg, "base")
    engine = create_engine(f"sqlite:///{db}")
    try:
        tables = set(inspect(engine).get_table_names())
    finally:
        engine.dispose()
    assert not (EXPECTED_TABLES & tables)
