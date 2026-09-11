"""Each deployment declares how much context it accepts.

The proxy owns the model map, so a reader asking "how full is this
conversation" has exactly one place to learn the denominator. Nothing else
in the hive held that number: the console had no table for it and the minds
only knew their own rotation threshold, which is a different figure.

Every existing row starts null on purpose. A default would be a guess
applied to twenty deployments across three providers, and a guessed
denominator is worse than an absent one — an absent one renders as unknown,
a wrong one renders as a confident percentage nobody can tell is false.

Revision ID: 0005_model_context_window
Revises: 0004_providers
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0005_model_context_window"
down_revision = "0004_providers"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("models", sa.Column("context_window", sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column("models", "context_window")
