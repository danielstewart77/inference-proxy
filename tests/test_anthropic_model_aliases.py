"""Anthropic model alias resolution: dated ids rewrite, bracket variants don't."""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from app.orm import Credential, Model
from app.proxy.anthropic import (
    _base_model_from_bracket_variant,
    _resolve_anthropic_deployment,
)


async def _add_model(session, name: str, *, admin_only: bool = False) -> None:
    cred = Credential(name=f"cred-{name}", kind="static", secret="s")
    session.add(cred)
    await session.flush()
    session.add(
        Model(
            deployment_name=name,
            target_uri="https://api.anthropic.com/v1/messages",
            credential_id=cred.id,
            auth_scheme="bearer",
            enabled=True,
            admin_only=admin_only,
        )
    )
    await session.commit()


def test_bracket_variant_parsing():
    assert _base_model_from_bracket_variant("claude-fable-5[1m]") == "claude-fable-5"
    assert _base_model_from_bracket_variant("claude-fable-5") is None
    assert _base_model_from_bracket_variant(None) is None


async def test_bracket_variant_resolves_base_row_without_rewrite(session):
    await _add_model(session, "claude-fable-5")
    target, rewrite = await _resolve_anthropic_deployment(session, "claude-fable-5[1m]")
    assert target.name == "claude-fable-5"
    # The bracketed id carries the context-window choice — it must reach the
    # upstream unchanged.
    assert rewrite is False


async def test_dated_alias_still_rewrites(session):
    await _add_model(session, "claude-haiku-4-5")
    target, rewrite = await _resolve_anthropic_deployment(
        session, "claude-haiku-4-5-20251001"
    )
    assert target.name == "claude-haiku-4-5"
    assert rewrite is True


async def test_exact_match_never_rewrites(session):
    await _add_model(session, "claude-opus-5")
    target, rewrite = await _resolve_anthropic_deployment(session, "claude-opus-5")
    assert target.name == "claude-opus-5"
    assert rewrite is False


async def test_bracket_variant_of_admin_only_model_stays_gated(session):
    await _add_model(session, "claude-fable-5", admin_only=True)
    with pytest.raises(HTTPException) as exc:
        await _resolve_anthropic_deployment(session, "claude-fable-5[1m]", is_admin=False)
    assert exc.value.status_code == 404
    target, _ = await _resolve_anthropic_deployment(
        session, "claude-fable-5[1m]", is_admin=True
    )
    assert target.name == "claude-fable-5"


async def test_unknown_model_still_404s(session):
    with pytest.raises(HTTPException) as exc:
        await _resolve_anthropic_deployment(session, "claude-nope[1m]")
    assert exc.value.status_code == 404
