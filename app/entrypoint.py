"""Process entrypoint: bootstrap migrations, then run uvicorn.

Migration logic:
    1. If `alembic_version` exists in the target DB
       -> run `alembic upgrade head` (no-op if already at head).
    2. Else if any application table exists
       -> the schema was created out-of-band; `alembic stamp head` to mark
          the DB as already at head, then `alembic upgrade head` (no-op).
    3. Else (fresh DB)
       -> `alembic upgrade head` to create schema from migrations.

Then start uvicorn against `app.main:app`.
"""

from __future__ import annotations

import sys

import uvicorn
from alembic import command
from alembic.config import Config as AlembicConfig
from sqlalchemy import create_engine, inspect

from app.config import config, database_url

APP_TABLES = {"clients", "keys", "credentials", "models", "usage_log"}


def _alembic_config() -> AlembicConfig:
    cfg = AlembicConfig("alembic.ini")
    # set_main_option goes through configparser, which treats `%` as
    # interpolation; double any that appear in the path.
    cfg.set_main_option("sqlalchemy.url", database_url(driver="").replace("%", "%%"))
    return cfg


def _bootstrap_schema() -> None:
    """Inspect the target DB and run the right Alembic command for its state."""
    engine = create_engine(database_url(driver=""), pool_pre_ping=True)
    try:
        with engine.connect() as conn:
            tables = set(inspect(conn).get_table_names())
    finally:
        engine.dispose()

    cfg = _alembic_config()

    if "alembic_version" in tables:
        print("[entrypoint] alembic_version present; running upgrade head", flush=True)
        command.upgrade(cfg, "head")
        return

    if APP_TABLES & tables:
        print(
            "[entrypoint] application tables exist without alembic_version; "
            "stamping head before upgrade",
            flush=True,
        )
        command.stamp(cfg, "head")
        command.upgrade(cfg, "head")
        return

    print("[entrypoint] empty DB; running upgrade head to create schema", flush=True)
    command.upgrade(cfg, "head")


def main() -> int:
    print(
        f"\n{'='*60}\n  Inference proxy starting...\n{'='*60}\n"
        f"  Port:            {config.port}\n"
        f"  Deployments:     configured per-row in /admin/models\n"
        f"  Streaming:       {'Enabled' if config.enable_streaming else 'Disabled'}\n"
        f"  Logging:         {'Enabled' if config.enable_logging else 'Disabled'}\n"
        f"  Database:        {config.db_path}\n"
        f"{'='*60}\n",
        flush=True,
    )

    _bootstrap_schema()

    ssl_kwargs: dict = {}
    if config.use_https:
        ssl_kwargs["ssl_certfile"] = config.ssl_certfile
        ssl_kwargs["ssl_keyfile"] = config.ssl_keyfile

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=config.port,
        **ssl_kwargs,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
