"""Initial schema: clients, keys, credentials, models, usage_log.

Revision ID: 0001_initial
Revises:
Create Date: 2026-07-19
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "clients",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("name", sa.Unicode(64), nullable=False),
        sa.Column("kind", sa.Unicode(20), nullable=False, server_default="app"),
        sa.Column("description", sa.UnicodeText(), nullable=True),
        sa.Column("disabled", sa.Boolean(), nullable=False, server_default="0"),
        sa.Column("privileged", sa.Boolean(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("last_seen_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
        sa.CheckConstraint("kind IN ('app','agent','mind','human')", name="CK_clients_kind"),
    )

    op.create_table(
        "keys",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("client_id", sa.Integer(), nullable=False),
        sa.Column("label", sa.Unicode(64), nullable=False, server_default="default"),
        sa.Column("key_hash", sa.Unicode(64), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("revoked", sa.Boolean(), nullable=False, server_default="0"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("key_hash"),
        sa.ForeignKeyConstraint(["client_id"], ["clients.id"], ondelete="CASCADE"),
    )
    op.create_index("IX_keys_client", "keys", ["client_id"])

    op.create_table(
        "credentials",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("name", sa.Unicode(64), nullable=False),
        sa.Column("kind", sa.Unicode(32), nullable=False, server_default="static"),
        sa.Column("secret", sa.Unicode(2048), nullable=True),
        sa.Column("source_path", sa.Unicode(512), nullable=True),
        sa.Column("access_token", sa.Unicode(2048), nullable=True),
        sa.Column("expires_at", sa.DateTime(), nullable=True),
        sa.Column("refreshed_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
        sa.CheckConstraint(
            "kind IN ('static','anthropic_oauth','openai_oauth')", name="CK_credentials_kind"
        ),
    )

    op.create_table(
        "models",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("deployment_name", sa.Unicode(128), nullable=False),
        sa.Column("label", sa.Unicode(128), nullable=True),
        sa.Column("description", sa.UnicodeText(), nullable=True),
        sa.Column("target_uri", sa.Unicode(512), nullable=True),
        sa.Column("credential_id", sa.Integer(), nullable=True),
        sa.Column("api_version", sa.Unicode(64), nullable=True),
        sa.Column("auth_scheme", sa.Unicode(32), nullable=False, server_default="bearer"),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default="1"),
        sa.Column("admin_only", sa.Boolean(), nullable=False, server_default="0"),
        sa.Column("cost_per_million_input", sa.Numeric(10, 6), nullable=True),
        sa.Column("cost_per_million_output", sa.Numeric(10, 6), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("deployment_name"),
        sa.ForeignKeyConstraint(["credential_id"], ["credentials.id"], ondelete="SET NULL"),
    )

    op.create_table(
        "usage_log",
        # BigInteger degrades to Integer on SQLite; see the note in app/orm.py.
        sa.Column(
            "id",
            sa.BigInteger().with_variant(sa.Integer, "sqlite"),
            autoincrement=True,
            nullable=False,
        ),
        sa.Column("client_id", sa.Integer(), nullable=False),
        sa.Column("key_id", sa.Integer(), nullable=True),
        sa.Column("model_name", sa.Unicode(128), nullable=False),
        sa.Column("endpoint", sa.Unicode(64), nullable=False),
        sa.Column("input_tokens", sa.Integer(), nullable=True),
        sa.Column("output_tokens", sa.Integer(), nullable=True),
        sa.Column("total_tokens", sa.Integer(), nullable=True),
        sa.Column("status_code", sa.Integer(), nullable=False),
        sa.Column("error_type", sa.Unicode(64), nullable=True),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
        sa.Column("client_ip", sa.Unicode(45), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["client_id"], ["clients.id"]),
        sa.ForeignKeyConstraint(["key_id"], ["keys.id"]),
    )
    op.create_index("IX_usage_log_client_created", "usage_log", ["client_id", "created_at"])
    op.create_index("IX_usage_log_created", "usage_log", ["created_at"])
    op.create_index("IX_usage_log_model_created", "usage_log", ["model_name", "created_at"])


def downgrade() -> None:
    op.drop_table("usage_log")
    op.drop_table("models")
    op.drop_table("credentials")
    op.drop_table("keys")
    op.drop_table("clients")
