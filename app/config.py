"""Configuration loaded from environment variables."""

from __future__ import annotations

import os
import re

from dotenv import load_dotenv

load_dotenv()


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

    # Where the data lives: a Postgres DSN, and the whole location -- host,
    # database and credential. Required; there is no other engine to fall
    # back to.
    database_url_override: str = os.getenv("DATABASE_URL", "")

    # Whether an empty target may be built from nothing. Off by default: the
    # branch that creates tables in a schema holding none of them is the one
    # that, pointed at the wrong schema, builds a complete empty copy beside
    # the real data and then answers every request with a 401. A fresh
    # install turns it on once; a cutover never does.
    allow_schema_create: bool = os.getenv(
        "PROXY_ALLOW_SCHEMA_CREATE", "false").lower() == "true"

    # The schema the proxy's tables live in.
    # Load-bearing rather than cosmetic: `credentials` exists in three
    # schemas of the hive's one database meaning three different things, and
    # an unqualified name binds to whichever the search path reaches first.
    db_schema: str = os.getenv("DB_SCHEMA", "proxy")

    # Pool limits. A streamed turn holds its connection for the life of the
    # stream, so the ceiling here is the number of concurrent turns the proxy
    # can carry.
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

    # How often a running proxy reloads its key cache from the database. A key
    # minted or revoked by another process -- `python -m app.cli`, a second
    # worker -- is otherwise invisible to this one until it restarts.
    key_cache_refresh_seconds: float = float(
        os.getenv("KEY_CACHE_REFRESH_SECONDS", "5"))

    # Usage log retention
    usage_prune_interval_seconds: int = int(os.getenv("USAGE_PRUNE_INTERVAL_SECONDS", "86400"))
    usage_prune_days: int = int(os.getenv("USAGE_PRUNE_DAYS", "90"))
    usage_prune_disabled: bool = os.getenv("USAGE_PRUNE_DISABLE", "false").lower() == "true"


config = Config()


#: SQLAlchemy driver names, keyed by the face the caller needs. The app is
#: async; Alembic and the startup inspection are sync. One setting has to fan
#: out to both, which is why `driver` is a face rather than a driver name.
_DRIVERS = {
    "async": "postgresql+asyncpg",
    "sync": "postgresql+psycopg",
}

_POSTGRES_PREFIXES = (
    "postgresql+asyncpg://", "postgresql+psycopg://", "postgresql+psycopg2://",
    "postgresql://", "postgres://",
)


def database_url(*, driver: str = "async") -> str:
    """The SQLAlchemy URL for the configured location.

    `driver` is the face the caller needs -- 'async' for the app, 'sync' for
    Alembic and the startup inspection -- not a driver name, because one
    setting has to answer for both.
    """
    if driver not in ("async", "sync"):
        raise ValueError(f"driver must be 'async' or 'sync', not {driver!r}")

    override = config.database_url_override
    if not override:
        raise ValueError(
            "DATABASE_URL is unset. The proxy runs only on Postgres; set it "
            "to the DSN of the database holding the proxy's schema.")
    if not override.startswith(_POSTGRES_PREFIXES):
        raise ValueError(
            "DATABASE_URL names an engine this proxy does not speak: "
            f"{override.split('://')[0]}://. It must be a Postgres DSN.")
    body = override.split("://", 1)[1]
    return f"{_DRIVERS[driver]}://{body}"


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
