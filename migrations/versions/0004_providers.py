"""Providers own the upstream; models own only their name.

A model row used to carry a complete target URI, path included, which froze it
to one request shape: a locally-hosted model registered at `/v1/messages` could
answer a Claude harness and nothing else, even though the same Ollama serves
`/v1/responses` and `/v1/chat/completions` from the same host. Hoisting the
host, the credential and the per-shape paths onto a `providers` row lets the
path be chosen by the endpoint a request arrives on, so one local model serves
every harness. It is also what makes adding Google or Mistral a row rather than
a code change.

Existing rows are grouped into providers by (host, credential): every model
already sharing an upstream and a secret is, by definition, one provider. A
provider that is neither Anthropic nor the Codex backend gets the full set of
OpenAI-compatible paths, because that is what a local server offers and
withholding them is what this migration exists to undo.

Revision ID: 0004_providers
Revises: 0003_model_cache_costs
"""

from __future__ import annotations

from urllib.parse import urlsplit

import sqlalchemy as sa
from alembic import op

revision = "0004_providers"
down_revision = "0003_model_cache_costs"
branch_labels = None
depends_on = None

_ANTHROPIC_HOST = "api.anthropic.com"
_CODEX_HOST = "chatgpt.com"
# What an OpenAI-compatible local server (Ollama, vLLM, llama.cpp) exposes.
_LOCAL_PATHS = {
    "messages_path": "/v1/messages",
    "responses_path": "/v1/responses",
    "chat_completions_path": "/v1/chat/completions",
}


def _shape_column(path: str) -> str | None:
    tail = path.rstrip("/")
    if tail.endswith("/messages"):
        return "messages_path"
    if tail.endswith("/responses"):
        return "responses_path"
    if tail.endswith("/chat/completions"):
        return "chat_completions_path"
    return None


def _provider_name(host: str, taken: set[str]) -> str:
    if host == _ANTHROPIC_HOST:
        base = "anthropic"
    elif host == _CODEX_HOST:
        base = "openai"
    elif host.endswith(":11434"):
        base = "ollama"
    else:
        base = host.replace(".", "-").replace(":", "-")
    name, suffix = base, 2
    while name in taken:
        name, suffix = f"{base}-{suffix}", suffix + 1
    taken.add(name)
    return name


def upgrade() -> None:
    op.create_table(
        "providers",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("name", sa.Unicode(64), nullable=False, unique=True),
        sa.Column("label", sa.Unicode(128), nullable=True),
        sa.Column("base_url", sa.Unicode(512), nullable=False),
        sa.Column("messages_path", sa.Unicode(256), nullable=True),
        sa.Column("responses_path", sa.Unicode(256), nullable=True),
        sa.Column("chat_completions_path", sa.Unicode(256), nullable=True),
        sa.Column(
            "credential_id",
            sa.Integer(),
            sa.ForeignKey("credentials.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("auth_scheme", sa.Unicode(32), nullable=False, server_default="bearer"),
        sa.Column("api_version", sa.Unicode(64), nullable=True),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default="1"),
        sa.Column(
            "created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()
        ),
    )
    # Added without an inline foreign key: SQLite cannot attach one via ALTER,
    # and a batch rebuild trips over the unnamed constraints already on the
    # table. The relationship is declared in the ORM either way.
    op.add_column("models", sa.Column("provider_id", sa.Integer(), nullable=True))
    op.add_column("models", sa.Column("harnesses", sa.Unicode(128), nullable=True))

    _backfill()


def _backfill() -> None:
    bind = op.get_bind()
    rows = bind.execute(
        sa.text(
            "SELECT id, deployment_name, target_uri, credential_id, api_version,"
            " auth_scheme FROM models WHERE target_uri IS NOT NULL AND target_uri <> ''"
        )
    ).fetchall()

    groups: dict[tuple[str, int | None], dict] = {}
    for row in rows:
        parts = urlsplit(row.target_uri)
        if not parts.netloc:
            continue
        key = (f"{parts.scheme}://{parts.netloc}", row.credential_id)
        group = groups.setdefault(
            key,
            {
                "paths": {},
                "models": [],
                "api_version": row.api_version,
                "auth_scheme": row.auth_scheme,
                "host": parts.netloc,
            },
        )
        column = _shape_column(parts.path)
        if column:
            group["paths"].setdefault(column, parts.path)
        group["models"].append(row.id)

    taken: set[str] = set()
    for (base_url, credential_id), group in groups.items():
        host = group["host"]
        if not group["paths"]:
            # Every URI here had a path this migration cannot classify — a
            # custom upstream route. A provider serving no shape resolves
            # nothing, so these rows keep their own URIs and go on working
            # exactly as they did, rather than being attached to an upstream
            # that can never answer them. Guessing the standard paths onto a
            # host that has already shown it does not use them is how a working
            # deployment becomes a 503 nothing warned about.
            continue
        if host not in (_ANTHROPIC_HOST, _CODEX_HOST):
            # A local OpenAI-compatible server, recognised by already serving
            # one of the standard paths: give it the rest, which is the whole
            # point of the change.
            for column, path in _LOCAL_PATHS.items():
                group["paths"].setdefault(column, path)
        name = _provider_name(host, taken)
        result = bind.execute(
            sa.text(
                "INSERT INTO providers (name, label, base_url, messages_path,"
                " responses_path, chat_completions_path, credential_id, auth_scheme,"
                " api_version, enabled) VALUES (:name, :label, :base_url, :messages,"
                " :responses, :chat, :credential_id, :auth_scheme, :api_version, 1)"
            ),
            {
                "name": name,
                "label": name,
                "base_url": base_url,
                "messages": group["paths"].get("messages_path"),
                "responses": group["paths"].get("responses_path"),
                "chat": group["paths"].get("chat_completions_path"),
                "credential_id": credential_id,
                "auth_scheme": group["auth_scheme"] or "bearer",
                "api_version": group["api_version"],
            },
        )
        provider_id = result.lastrowid
        for model_id in group["models"]:
            bind.execute(
                sa.text("UPDATE models SET provider_id = :pid WHERE id = :mid"),
                {"pid": provider_id, "mid": model_id},
            )


def downgrade() -> None:
    op.drop_column("models", "harnesses")
    op.drop_column("models", "provider_id")
    op.drop_table("providers")
