"""Each deployment declares whether it accepts images.

Null is "nobody has said", and stays distinct from "no": a conversation that
holds images is handed over rather than resumed on a model that cannot read
them, and only a declared answer decides that. Every existing row starts null.

Revision ID: 0007_model_accepts_images
Revises: 0006_model_effort_levels
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0007_model_accepts_images"
down_revision = "0006_model_effort_levels"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("models", sa.Column("accepts_images", sa.Boolean(), nullable=True))


def downgrade() -> None:
    op.drop_column("models", "accepts_images")
