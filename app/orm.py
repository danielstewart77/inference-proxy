"""SQLAlchemy ORM models for the inference proxy.

The proxy fronts one or more upstream inference endpoints and issues its own
keys to *clients* — apps and agents, not human users. There is no user table
and no self-service portal: the admin is a single operator identity held in
environment config, and every other principal is a `Client` row holding one
or more `ApiKey`s.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    Unicode,
    UnicodeText,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all ORM models. UTC datetimes throughout."""


class Client(Base):
    """An app or agent authorised to call the proxy.

    `name` is the stable identifier a caller is known by in logs and usage
    reporting (e.g. `skippy`, `frame-main`). `kind` records what sort of
    principal it is, purely for grouping in the admin console.
    """

    __tablename__ = "clients"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(Unicode(64), unique=True, nullable=False)
    kind: Mapped[str] = mapped_column(Unicode(20), nullable=False, server_default="app")
    description: Mapped[Optional[str]] = mapped_column(UnicodeText, nullable=True)
    disabled: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="0")
    # When true, this client may resolve admin-only models.
    privileged: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="0")
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    last_seen_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    keys: Mapped[list["ApiKey"]] = relationship(
        "ApiKey", back_populates="client", cascade="all, delete-orphan"
    )
    usage_logs: Mapped[list["UsageLog"]] = relationship("UsageLog", back_populates="client")

    __table_args__ = (
        CheckConstraint("kind IN ('app','agent','mind','human')", name="CK_clients_kind"),
    )


class ApiKey(Base):
    """A proxy key issued to a client.

    Renamed in Python to avoid shadowing the `keys` built-in name. A client may
    hold several keys at once so credentials can be rotated without downtime.
    """

    __tablename__ = "keys"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    client_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("clients.id", ondelete="CASCADE"), nullable=False
    )
    label: Mapped[str] = mapped_column(Unicode(64), nullable=False, server_default="default")
    key_hash: Mapped[str] = mapped_column(Unicode(64), nullable=False, unique=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    revoked: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="0")

    client: Mapped[Client] = relationship("Client", back_populates="keys")

    __table_args__ = (Index("IX_keys_client", "client_id"),)


class Credential(Base):
    """An upstream credential, shared by any number of `Model` rows.

    `kind` selects how the secret is obtained at request time:

    - `static` — `secret` holds the literal API key.
    - `anthropic_oauth` — a literal long-lived OAuth token in `secret`
      (returned as-is, no refresh), or — when `secret` is empty — the access
      token is read (and refreshed when stale) from the OAuth credential file
      at `source_path`.
    - `openai_oauth` — `secret` is ignored; the API key is read from a
      Codex-issued OAuth credential file at `source_path`.

    Refresh bookkeeping lives on the row so a restart doesn't lose track of an
    already-valid token.
    """

    __tablename__ = "credentials"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(Unicode(64), unique=True, nullable=False)
    kind: Mapped[str] = mapped_column(Unicode(32), nullable=False, server_default="static")
    secret: Mapped[Optional[str]] = mapped_column(Unicode(2048), nullable=True)
    source_path: Mapped[Optional[str]] = mapped_column(Unicode(512), nullable=True)
    # Cached access token + its expiry, for the OAuth kinds.
    access_token: Mapped[Optional[str]] = mapped_column(Unicode(2048), nullable=True)
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    refreshed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    __table_args__ = (
        CheckConstraint(
            "kind IN ('static','anthropic_oauth','openai_oauth')", name="CK_credentials_kind"
        ),
    )


class Provider(Base):
    """An upstream that hosts models — Anthropic, the Codex backend, a local
    Ollama, and whatever is added next.

    A provider owns the host and the credential; a `Model` row owns only the
    deployment name and its pricing. The three path columns say which request
    shapes this upstream serves and where each one lives, which is what lets
    one Ollama model answer a Claude harness and a Codex harness both: the
    path is chosen by the endpoint the request arrived on, not frozen into the
    model row. A null path means this upstream does not speak that shape, and
    a model on it is neither listed to nor resolvable by a harness that does.
    """

    __tablename__ = "providers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(Unicode(64), unique=True, nullable=False)
    label: Mapped[Optional[str]] = mapped_column(Unicode(128), nullable=True)
    # Scheme + host (+ any fixed prefix), without a trailing slash.
    base_url: Mapped[str] = mapped_column(Unicode(512), nullable=False)
    messages_path: Mapped[Optional[str]] = mapped_column(Unicode(256), nullable=True)
    responses_path: Mapped[Optional[str]] = mapped_column(Unicode(256), nullable=True)
    chat_completions_path: Mapped[Optional[str]] = mapped_column(
        Unicode(256), nullable=True
    )
    credential_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("credentials.id", ondelete="SET NULL"), nullable=True
    )
    auth_scheme: Mapped[str] = mapped_column(
        Unicode(32), nullable=False, server_default="bearer"
    )
    api_version: Mapped[Optional[str]] = mapped_column(Unicode(64), nullable=True)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="1")
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    credential: Mapped[Optional[Credential]] = relationship("Credential")

    def path_for(self, wire: str) -> Optional[str]:
        return {
            "anthropic_messages": self.messages_path,
            "openai_responses": self.responses_path,
            "chat_completions": self.chat_completions_path,
        }.get(wire)


class Model(Base):
    """Each row is a self-contained deployment: its own target URI, credential,
    and optional api_version. The path of `target_uri` determines the upstream
    protocol — `/anthropic/v1/messages` or a real `api.anthropic.com/v1/messages`
    (Anthropic Messages), `/openai/responses` or a real `api.openai.com/v1/responses`
    (Responses API), or `/chat/completions` (native). `auth_scheme` determines how
    the resolved secret is sent upstream — see `app.deployments.build_upstream_headers`.
    """

    __tablename__ = "models"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    deployment_name: Mapped[str] = mapped_column(Unicode(128), unique=True, nullable=False)
    label: Mapped[Optional[str]] = mapped_column(Unicode(128), nullable=True)
    description: Mapped[Optional[str]] = mapped_column(UnicodeText, nullable=True)
    # Upstream endpoint. Stored without the api-version query string; the
    # `api_version` column is appended at request time when set.
    target_uri: Mapped[Optional[str]] = mapped_column(Unicode(512), nullable=True)
    credential_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("credentials.id", ondelete="SET NULL"), nullable=True
    )
    # The upstream that hosts this model. When set it supplies the host, the
    # credential and the per-shape path, and `target_uri` is ignored.
    provider_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("providers.id", ondelete="SET NULL"), nullable=True
    )
    # Which harnesses may see and use this model, comma-separated ('claude',
    # 'codex'). Null — the default — means every harness the provider can
    # serve. Set it only to withhold a model from a harness it would otherwise
    # reach, as a model fine-tuned on one harness's turns is withheld from the
    # others.
    harnesses: Mapped[Optional[str]] = mapped_column(Unicode(128), nullable=True)
    api_version: Mapped[Optional[str]] = mapped_column(Unicode(64), nullable=True)
    # How the resolved secret is sent upstream: 'api_key_header' (Azure/Foundry
    # `api-key:`), 'bearer' (`Authorization: Bearer`, real OpenAI / Foundry's
    # Anthropic partner route), or 'x_api_key' (`x-api-key:`, real Anthropic).
    auth_scheme: Mapped[str] = mapped_column(
        Unicode(32), nullable=False, server_default="bearer"
    )
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="1")
    # When true, only privileged clients may resolve or see this model.
    admin_only: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="0")
    cost_per_million_input: Mapped[Optional[float]] = mapped_column(
        Numeric(10, 6), nullable=True
    )
    cost_per_million_output: Mapped[Optional[float]] = mapped_column(
        Numeric(10, 6), nullable=True
    )
    # Cache pricing. Anthropic bills cache writes at 1.25x base input (5-minute
    # TTL; the 1-hour tier is 2x — we store the 5-minute rate as the estimate)
    # and cache reads at 0.1x. OpenAI has no write charge and discounted reads.
    # Null means "price cache tokens at zero", matching providers without a
    # cache tier.
    cost_per_million_cache_write: Mapped[Optional[float]] = mapped_column(
        Numeric(10, 6), nullable=True
    )
    cost_per_million_cache_read: Mapped[Optional[float]] = mapped_column(
        Numeric(10, 6), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    credential: Mapped[Optional[Credential]] = relationship("Credential")
    provider: Mapped[Optional[Provider]] = relationship("Provider")

    def visible_to(self, harness: str) -> bool:
        """Whether `harness` may see and resolve this model.

        An unset list means every harness; the column exists to withhold, not
        to grant.
        """
        if not self.harnesses:
            return True
        allowed = {part.strip().lower() for part in self.harnesses.split(",")}
        allowed.discard("")
        return not allowed or harness.strip().lower() in allowed


class UsageLog(Base):
    __tablename__ = "usage_log"

    # SQLite only autoincrements a column declared exactly INTEGER PRIMARY KEY,
    # so BIGINT has to degrade to INTEGER there or inserts fail the NOT NULL
    # check on `id`.
    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True
    )
    client_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("clients.id"), nullable=False
    )
    key_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("keys.id"), nullable=True
    )
    model_name: Mapped[str] = mapped_column(Unicode(128), nullable=False)
    endpoint: Mapped[str] = mapped_column(Unicode(64), nullable=False)
    input_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    total_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    # Cache token counts, stored exactly as the provider delivered them.
    # Semantics differ: Anthropic's input_tokens EXCLUDES cached tokens
    # (cache_creation + cache_read are billed separately, at 1.25x and 0.1x
    # base input respectively); OpenAI's prompt_tokens INCLUDES cached_tokens
    # (mapped into cache_read here; OpenAI has no cache-creation charge).
    cache_creation_input_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    cache_read_input_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    status_code: Mapped[int] = mapped_column(Integer, nullable=False)
    error_type: Mapped[Optional[str]] = mapped_column(Unicode(64), nullable=True)
    duration_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    client_ip: Mapped[Optional[str]] = mapped_column(Unicode(45), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    client: Mapped[Client] = relationship("Client", back_populates="usage_logs")

    __table_args__ = (
        Index("IX_usage_log_client_created", "client_id", "created_at"),
        Index("IX_usage_log_created", "created_at"),
        Index("IX_usage_log_model_created", "model_name", "created_at"),
    )
