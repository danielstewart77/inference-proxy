"""A provider declares the path its decision wire answers on.

A decision model is a fourth shape, not a chat model with a different body:
one state, named questions, one choice with a confidence per question, and no
output tokens at all. It therefore needs a path of its own beside the three
chat wires, for the same reason those are separate — the shape is whichever
endpoint the request came in on, and a provider that serves one wire and not
another has to be able to say so.

Backfilled for the `ollama` provider only, because Ollama is the only
upstream here that serves `/v1/systemone`. Every other provider keeps NULL,
which is a provider that does not serve the wire and refuses the request
rather than forwarding a decision call to an endpoint that would answer it
as chat.

Revision ID: 0009_provider_systemone_path
Revises: 0008_model_rotation_threshold
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0009_provider_systemone_path"
down_revision = "0008_model_rotation_threshold"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("providers") as batch:
        batch.add_column(sa.Column("systemone_path", sa.Unicode(256), nullable=True))
    op.execute(
        "UPDATE providers SET systemone_path = '/v1/systemone' WHERE name = 'ollama'"
    )


def downgrade() -> None:
    with op.batch_alter_table("providers") as batch:
        batch.drop_column("systemone_path")
