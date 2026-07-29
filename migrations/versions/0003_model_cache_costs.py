"""Add cache-tier cost columns to models.

Cache tokens are billed at their own rates (Anthropic: writes at 1.25x base
input for the 5-minute tier, reads at 0.1x; OpenAI: discounted reads, no write
charge). Two nullable rate columns let the usage report price the cache token
counts metered in usage_log instead of ignoring them.

Revision ID: 0003_model_cache_costs
Revises: 0002_usage_cache_tokens
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0003_model_cache_costs"
down_revision = "0002_usage_cache_tokens"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "models",
        sa.Column("cost_per_million_cache_write", sa.Numeric(10, 6), nullable=True),
    )
    op.add_column(
        "models",
        sa.Column("cost_per_million_cache_read", sa.Numeric(10, 6), nullable=True),
    )


def downgrade() -> None:
    with op.batch_alter_table("models") as batch:
        batch.drop_column("cost_per_million_cache_read")
        batch.drop_column("cost_per_million_cache_write")
