"""Configuration loaded from environment variables."""

from __future__ import annotations

import os
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

    # SQLite database. Single file, no server. Override with an absolute path
    # to relocate it (e.g. onto a mounted volume).
    db_path: str = os.getenv("DB_PATH", str(REPO_ROOT / "data" / "proxy.db"))

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


def database_url(*, driver: str = "aiosqlite") -> str:
    """Build the SQLAlchemy connection URL for the SQLite database.

    `driver` selects the SQLAlchemy driver:
        - 'aiosqlite' for async (used by the app at runtime)
        - '' (empty) for sync (used by Alembic migrations)
    """
    path = Path(config.db_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    prefix = f"sqlite+{driver}" if driver else "sqlite"
    return f"{prefix}:///{path}"
