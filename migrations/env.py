"""Alembic migration environment."""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from app.config import database_url
from app.orm import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Use the sync sqlite driver for migrations (Alembic doesn't need async).
# Double `%` so configparser doesn't treat it as interpolation. A caller that
# already set a URL (the entrypoint, or a test pointing at a scratch file) wins.
if not config.get_main_option("sqlalchemy.url", ""):
    config.set_main_option("sqlalchemy.url", database_url(driver="").replace("%", "%%"))

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            # SQLite cannot ALTER most column properties; batch mode rebuilds
            # the table instead.
            render_as_batch=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
