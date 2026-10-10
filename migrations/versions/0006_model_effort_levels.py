"""Each deployment declares the reasoning-effort levels it accepts.

Which models take an effort setting, and which levels, is a fact about the
model — so it lives beside the model's other facts here, where every surface
asks, rather than in a table each bot would keep and let drift.

Every existing row starts null, meaning "takes no effort setting", until
someone declares otherwise.

Revision ID: 0006_model_effort_levels
Revises: 0005_model_context_window
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0006_model_effort_levels"
down_revision = "0005_model_context_window"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("models", sa.Column("effort_levels", sa.Unicode(128), nullable=True))


def downgrade() -> None:
    op.drop_column("models", "effort_levels")
