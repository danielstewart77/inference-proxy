"""A model declares its purpose, and a provider the path its embeddings answer on.

`models.purpose` is 'chat' or 'embedding'. Every existing row is a chat model,
so the column is backfilled to 'chat' and not nullable: a listed model always
says what it is for, and no caller has an "unknown" to guess about.

`providers.embeddings_path` is the embeddings wire, a shape of its own for the
same reason the decision wire has one (0009): the shape is whichever endpoint
the request came in on, and a provider that does not serve it must be able to
say so. Backfilled to '/v1/embeddings' on every provider whose chat path is
'/v1/chat/completions' -- Ollama and OpenAI both serve embeddings beside chat
at that prefix. Every other provider keeps NULL and refuses the wire.

Revision ID: 0011_model_purpose
Revises: 0010_model_provider_foreign_key
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0011_model_purpose"
down_revision = "0010_model_provider_foreign_key"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("providers") as batch:
        batch.add_column(sa.Column("embeddings_path", sa.Unicode(256), nullable=True))
    op.execute(
        "UPDATE providers SET embeddings_path = '/v1/embeddings' "
        "WHERE chat_completions_path = '/v1/chat/completions'"
    )
    with op.batch_alter_table("models") as batch:
        batch.add_column(
            sa.Column("purpose", sa.Unicode(16), nullable=False, server_default="chat")
        )


def downgrade() -> None:
    with op.batch_alter_table("models") as batch:
        batch.drop_column("purpose")
    with op.batch_alter_table("providers") as batch:
        batch.drop_column("embeddings_path")
