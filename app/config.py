"""Configuration loaded from environment variables."""

from __future__ import annotations

import os
import re
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent


class Config:
    # Upstream deployments (target_uri / credential / api_version) live in the
    # `models` table, one row per deployment. There is no global upstream
    # endpoint configured here.

    # Server
    port: int = int(os.getenv("PORT", "8888"))
    enable_streaming: bool = os.getenv("ENABLE_STREAMING", "true").lower() == "true"
    enable_logging: bool = os.getenv("ENABLE_LOGGING", "true").lower() == "true"
    use_https: bool = os.getenv("USE_HTTPS", "false").lower() == "true"
    ssl_certfile: str = os.getenv("SSL_CERTFILE", "cert.pem")
    ssl_keyfile: str = os.getenv("SSL_KEYFILE", "key.pem")

    # Where the data lives. `DATABASE_URL` wins when set and is the whole
    # location -- host, database and credential. With it unset the proxy is on
    # its own SQLite file, which is what makes a revert one setting and a
    # restart. A DSN is never a filesystem path: it carries a password, and
    # the old code ran `mkdir` on whatever it was handed.
    database_url_override: str = os.getenv("DATABASE_URL", "")

    # Whether an empty target may be built from nothing. Off by default: the
    # branch that creates tables in a schema holding none of them is the one
    # that, pointed at the wrong schema, builds a complete empty copy beside
    # the real data and then answers every request with a 401. A fresh
    # install turns it on once; a cutover never does.
    allow_schema_create: bool = os.getenv(
        "PROXY_ALLOW_SCHEMA_CREATE", "false").lower() == "true"

    # The schema the proxy's tables live in, when the location is Postgres.
    # Load-bearing rather than cosmetic: `credentials` exists in three
    # schemas of the hive's one database meaning three different things, and
    # an unqualified name binds to whichever the search path reaches first.
    db_schema: str = os.getenv("DB_SCHEMA", "proxy")

    # SQLite database. Single file, no server. Override with an absolute path
    # to relocate it (e.g. onto a mounted volume).
    db_path: str = os.getenv("DB_PATH", str(REPO_ROOT / "data" / "proxy.db"))

    # Pool limits, which only matter on a server engine. A streamed turn holds
    # its connection for the life of the stream, so the ceiling here is the
    # number of concurrent turns the proxy can carry.
    pool_size: int = int(os.getenv("DB_POOL_SIZE", "10"))
    pool_max_overflow: int = int(os.getenv("DB_POOL_MAX_OVERFLOW", "20"))
    pool_timeout_seconds: int = int(os.getenv("DB_POOL_TIMEOUT_SECONDS", "10"))
    statement_timeout_seconds: int = int(
        os.getenv("DB_STATEMENT_TIMEOUT_SECONDS", "30"))
    connect_timeout_seconds: int = int(os.getenv("DB_CONNECT_TIMEOUT_SECONDS", "10"))

    # Admin console. The admin is a single operator identity held in env, not
    # a row in the database — there is no user table any more, only clients.
    session_secret: str = os.getenv("SESSION_SECRET", "change-me-in-production")
    admin_username: str = os.getenv("ADMIN_USERNAME", "")
    admin_password: str = os.getenv("ADMIN_PASSWORD", "")

    # Usage log retention
    usage_prune_interval_seconds: int = int(os.getenv("USAGE_PRUNE_INTERVAL_SECONDS", "86400"))
    usage_prune_days: int = int(os.getenv("USAGE_PRUNE_DAYS", "90"))
    usage_prune_disabled: bool = os.getenv("USAGE_PRUNE_DISABLE", "false").lower() == "true"


config = Config()


#: SQLAlchemy driver names, keyed by the face the caller needs. The app is
#: async; Alembic and the startup inspection are sync. One setting has to fan
#: out to both, which is why `driver` is a face rather than a driver name.
_DRIVERS = {
    ("postgresql", "async"): "postgresql+asyncpg",
    ("postgresql", "sync"): "postgresql+psycopg",
    ("sqlite", "async"): "sqlite+aiosqlite",
    ("sqlite", "sync"): "sqlite",
}

_POSTGRES_PREFIXES = (
    "postgresql+asyncpg://", "postgresql+psycopg://", "postgresql+psycopg2://",
    "postgresql://", "postgres://",
)


def is_postgres() -> bool:
    """Whether the configured location is a Postgres server."""
    return config.database_url_override.startswith(_POSTGRES_PREFIXES)


def database_url(*, driver: str = "async") -> str:
    """The SQLAlchemy URL for the configured location.

    `driver` is the face the caller needs -- 'async' for the app, 'sync' for
    Alembic and the startup inspection -- not a driver name, because one
    setting has to answer for both.
    """
    if driver not in ("async", "sync"):
        raise ValueError(f"driver must be 'async' or 'sync', not {driver!r}")

    override = config.database_url_override
    if override:
        if not is_postgres():
            raise ValueError(
                "DATABASE_URL names an engine this proxy does not speak: "
                f"{override.split('://')[0]}://. Postgres, or leave it unset "
                "for SQLite.")
        body = override.split("://", 1)[1]
        return f"{_DRIVERS[('postgresql', driver)]}://{body}"

    path = Path(config.db_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return f"{_DRIVERS[('sqlite', driver)]}:///{path}"


#: Query parameters that carry a credential. A socket DSN has no `@`, so this
#: is the only place a password can appear in one.
_SECRET_PARAMS = ("password", "sslpassword")


def redacted(url: str) -> str:
    """`url` with any password replaced, for printing.

    The location used to be a file path. It carries a credential now, and the
    startup banner goes to a world-readable log.
    """
    if "://" not in url:
        return url
    for param in _SECRET_PARAMS:
        url = re.sub(rf"(?i)([?&]{param}=)[^&]*", r"\1***", url)
    if "@" not in url:
        return url
    scheme, rest = url.split("://", 1)
    creds, host = rest.rsplit("@", 1)
    if ":" not in creds:
        # A DSN with a user and no password. Inventing one reads as a
        # credential that is set when it is not.
        return url
    user = creds.split(":", 1)[0]
    return f"{scheme}://{user}:***@{host}"
