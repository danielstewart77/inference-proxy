"""Process entrypoint: bring the schema to head, then run uvicorn.

The schema bootstrap is the riskiest thing this process does, because every
way it can be wrong looks like success. Three rules, each paid for:

1. It must know which schema it is talking to, and it must set that itself
   rather than trust the connection string to carry it. A Postgres DSN with
   no search path resolves to the default schema, where there are no tables
   -- so the old code declared the database empty and built a complete,
   correct, empty copy of everything beside the real data, reported healthy,
   and answered every request with a 401. The search path cannot live in the
   DSN either: `psycopg` accepts an `options` parameter and `asyncpg` has no
   such keyword at all, so one setting carrying it would satisfy the sync
   half and crash the async one. It comes from `DB_SCHEMA` on both sides.
2. It must stamp the revision the schema actually matches. Stamping at head
   over a schema that is a migration behind records that migration as applied
   and never runs it. The proxy starts clean and then refuses every request
   on a column that does not exist.
3. It must not print the password. The location is a DSN now, and this banner
   goes to a world-readable log.
"""

from __future__ import annotations

import sys
from typing import Optional

import uvicorn
from alembic import command
from alembic.config import Config as AlembicConfig
from sqlalchemy import create_engine, inspect, text

from app.config import config, database_url, redacted

APP_TABLES = {"clients", "keys", "credentials", "models", "providers", "usage_log"}

#: The column each revision introduces, newest first. A schema created out of
#: band carries no bookkeeping row, so the only way to know where it stands is
#: to look at what it has. The newest marker present is the revision it is at.
REVISION_MARKERS: tuple[tuple[str, str, str], ...] = (
    # 0010 adds no column, so it has no marker. A schema at 0010 reads as
    # 0009 and the upgrade re-runs 0010, which is why that migration looks
    # before it creates anything.
    ("0009_provider_systemone_path", "providers", "systemone_path"),
    ("0008_model_rotation_threshold", "models", "rotation_threshold_percent"),
    ("0007_model_accepts_images", "models", "accepts_images"),
    ("0006_model_effort_levels", "models", "effort_levels"),
    ("0005_model_context_window", "models", "context_window"),
    ("0004_providers", "models", "provider_id"),
    ("0003_model_cache_costs", "models", "cost_per_million_cache_write"),
    ("0002_usage_cache_tokens", "usage_log", "cache_creation_input_tokens"),
    ("0001_initial", "clients", "id"),
)


class BootstrapRefused(RuntimeError):
    """The target is not a database this process should migrate."""


def _engine(dsn: str, schema: Optional[str]):
    """A sync engine on `dsn`, with the search path and a connect deadline.

    The deadline matters as much as the path: libpq with no timeout against a
    paused container blocks forever, which presents as a process that is
    alive, has printed a healthy-looking banner, and is listening on nothing.
    """
    connect_args: dict = {}
    if schema:
        connect_args["options"] = f"-csearch_path={schema}"
        connect_args["connect_timeout"] = config.connect_timeout_seconds
    return create_engine(dsn, pool_pre_ping=True, connect_args=connect_args)


def _create_schema(dsn: str, schema: str) -> None:
    """Create the schema itself, for a genuine fresh install.

    `current_schema()` is NULL when the named schema does not exist, so
    without this an install against an empty database refuses rather than
    installing -- and nothing else in the proxy ever issues CREATE SCHEMA.
    """
    engine = create_engine(
        dsn, connect_args={"connect_timeout": config.connect_timeout_seconds})
    try:
        with engine.begin() as conn:
            conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{schema}"'))
    finally:
        engine.dispose()


def _alembic_config(dsn: str, schema: Optional[str]) -> AlembicConfig:
    cfg = AlembicConfig("alembic.ini")
    # set_main_option goes through configparser, which treats `%` as
    # interpolation; double any that appear in the URL.
    cfg.set_main_option("sqlalchemy.url", dsn.replace("%", "%%"))
    if schema:
        # Alembic builds its own engine from the URL, which would reach the
        # default schema. This is the search path it uses instead, and it is
        # also where the bookkeeping row goes -- stated rather than left to
        # follow whatever path the DSN happened to carry.
        cfg.attributes["search_path"] = schema
    return cfg


def _schema_state(dsn: str, schema: Optional[str]) -> tuple[set[str], dict[str, set[str]]]:
    """The tables in the target schema, and the columns of each."""
    engine = _engine(dsn, schema)
    try:
        with engine.connect() as conn:
            if schema:
                current = conn.execute(text("SELECT current_schema()")).scalar()
                if current != schema:
                    raise BootstrapRefused(
                        f"schema {schema!r} does not exist in this database "
                        f"(the connection resolved to {current!r}). Create it, "
                        "or correct DB_SCHEMA -- migrating the wrong schema is "
                        "how an empty copy of this proxy ends up beside the "
                        "real data.")
            insp = inspect(conn)
            tables = set(insp.get_table_names(schema=schema))
            columns = {
                t: {c["name"] for c in insp.get_columns(t, schema=schema)}
                for t in tables & (APP_TABLES | {"alembic_version"})
            }
    finally:
        engine.dispose()
    return tables, columns


def _revision_of(columns: dict[str, set[str]]) -> Optional[str]:
    """The newest revision whose column is present, or None for an empty schema."""
    for revision, table, column in REVISION_MARKERS:
        if column in columns.get(table, set()):
            return revision
    return None


def bootstrap_schema(dsn: Optional[str] = None,
                     schema: Optional[str] = None) -> None:
    """Bring the target schema to head, refusing anything ambiguous."""
    if dsn is None:
        dsn = database_url(driver="sync")
    if schema is None:
        schema = config.db_schema

    if schema and config.allow_schema_create:
        _create_schema(dsn, schema)

    tables, columns = _schema_state(dsn, schema)
    cfg = _alembic_config(dsn, schema)

    if "alembic_version" in tables:
        print("[entrypoint] alembic_version present; running upgrade head",
              flush=True)
        command.upgrade(cfg, "head")
        return

    present = APP_TABLES & tables
    if present:
        if present != APP_TABLES:
            raise BootstrapRefused(
                "the target holds some of the proxy's tables and not others "
                f"(missing {', '.join(sorted(APP_TABLES - present))}). A "
                "partial schema has no revision to stamp: guessing one "
                "records migrations as applied that then fail on a table "
                "that is not there, and the bookkeeping row left behind "
                "makes every later start fail the same way.")
        revision = _revision_of(columns)
        if revision is None:
            raise BootstrapRefused(
                "the target holds the proxy's tables but none of the columns "
                "any migration created. Refusing to guess where it stands.")
        print(
            f"[entrypoint] tables exist without alembic_version; the schema "
            f"matches {revision}, stamping there before upgrade",
            flush=True,
        )
        command.stamp(cfg, revision)
        command.upgrade(cfg, "head")
        return

    if not config.allow_schema_create:
        raise BootstrapRefused(
            f"the target holds none of the proxy's tables. Creating them is "
            "a fresh install, not a start -- set "
            "PROXY_ALLOW_SCHEMA_CREATE=true to say that is what this is. "
            "Unset, this refusal is what stops a wrong DB_SCHEMA from "
            "building an empty copy that answers every request with a 401.")

    print("[entrypoint] empty schema; running upgrade head to create it",
          flush=True)
    command.upgrade(cfg, "head")


def main() -> int:
    print(
        f"\n{'='*60}\n  Inference proxy starting...\n{'='*60}\n"
        f"  Port:            {config.port}\n"
        f"  Deployments:     configured per-row in /admin/models\n"
        f"  Streaming:       {'Enabled' if config.enable_streaming else 'Disabled'}\n"
        f"  Logging:         {'Enabled' if config.enable_logging else 'Disabled'}\n"
        f"  Database:        {redacted(database_url(driver='sync'))}\n"
        f"  Schema:          {config.db_schema}\n"
        f"{'='*60}\n",
        flush=True,
    )

    bootstrap_schema()

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
