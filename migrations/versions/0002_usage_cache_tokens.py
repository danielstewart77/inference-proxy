"""Add cache token columns to usage_log.

Providers report cached prompt tokens separately from regular input, and bill
them differently (Anthropic: cache writes at 1.25x base input, reads at 0.1x;
OpenAI: cached reads at a discount). Two nullable columns record them exactly
as delivered so cost estimation can price each bucket at its own rate.

Revision ID: 0002_usage_cache_tokens
Revises: 0001_initial
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0002_usage_cache_tokens"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "usage_log",
        sa.Column("cache_creation_input_tokens", sa.Integer(), nullable=True),
    )
    op.add_column(
        "usage_log",
        sa.Column("cache_read_input_tokens", sa.Integer(), nullable=True),
    )


def downgrade() -> None:
    with op.batch_alter_table("usage_log") as batch:
        batch.drop_column("cache_read_input_tokens")
        batch.drop_column("cache_creation_input_tokens")
