"""Each deployment declares the percentage of its window a conversation rotates at.

The threshold is a property of the model, not of a mind: the same mind on two
models rotates at whatever each model's row says. Every existing row is
backfilled to 30 and the column is not nullable — a listed model always
carries a threshold, so no caller has an "unknown" to invent a fallback for.

Revision ID: 0008_model_rotation_threshold
Revises: 0007_model_accepts_images
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0008_model_rotation_threshold"
down_revision = "0007_model_accepts_images"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("models") as batch:
        batch.add_column(
            sa.Column(
                "rotation_threshold_percent",
                sa.Integer(),
                nullable=False,
                server_default="30",
            )
        )


def downgrade() -> None:
    with op.batch_alter_table("models") as batch:
        batch.drop_column("rotation_threshold_percent")
