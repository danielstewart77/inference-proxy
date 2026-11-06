"""Process entrypoint: bring the schema to head, then run uvicorn.

The schema bootstrap is the riskiest thing this process does, because every
way it can be wrong looks like success. Three rules, each paid for:

1. It must know which schema it is talking to. A Postgres DSN with no search
   path resolves to the default schema, where there are no tables -- so the
   old code declared the database empty and built a complete, correct, empty
   copy of everything beside the real data. The proxy then reported healthy
   and answered every request with a 401.
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

from app.config import config, database_url, is_postgres, redacted

APP_TABLES = {"clients", "keys", "credentials", "models", "providers", "usage_log"}

#: The column each revision introduces, newest first. A schema created out of
#: band carries no bookkeeping row, so the only way to know where it stands is
#: to look at what it has. The newest marker present is the revision it is at.
REVISION_MARKERS: tuple[tuple[str, str, str], ...] = (
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


def _alembic_config(dsn: str, schema: Optional[str]) -> AlembicConfig:
    cfg = AlembicConfig("alembic.ini")
    # set_main_option goes through configparser, which treats `%` as
    # interpolation; double any that appear in the URL.
    cfg.set_main_option("sqlalchemy.url", dsn.replace("%", "%%"))
    if schema:
        # Where the bookkeeping row goes. Without it the version table
        # follows the search path, which makes it an accident rather than a
        # decision -- and lands it in the shared schema the day a DSN omits
        # the search path.
        cfg.attributes["version_table_schema"] = schema
    return cfg


def _schema_state(dsn: str, schema: Optional[str]) -> tuple[set[str], dict[str, set[str]]]:
    """The tables in the target schema, and the columns of each."""
    engine = create_engine(dsn, pool_pre_ping=True)
    try:
        with engine.connect() as conn:
            if schema:
                current = conn.execute(text("SELECT current_schema()")).scalar()
                if current != schema:
                    raise BootstrapRefused(
                        f"this connection resolves to schema {current!r}, not "
                        f"{schema!r}. The DSN has to carry "
                        f"options=-csearch_path={schema}, or nothing here is "
                        "reading the proxy's own tables.")
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
        schema = config.db_schema if is_postgres() else None

    tables, columns = _schema_state(dsn, schema)
    cfg = _alembic_config(dsn, schema)

    if "alembic_version" in tables:
        print("[entrypoint] alembic_version present; running upgrade head",
              flush=True)
        command.upgrade(cfg, "head")
        return

    if APP_TABLES & tables:
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
        f"  Schema:          {config.db_schema if is_postgres() else 'n/a (sqlite)'}\n"
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
