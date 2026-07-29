"""/admin/usage — aggregate report across all clients."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import HTMLResponse
from sqlalchemy import BigInteger, case, cast, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import require_html_admin
from app.db import get_session
from app.orm import Client, Model, UsageLog
from app.templating import templates

router = APIRouter()


def _since(days: int) -> datetime:
    return (datetime.now(timezone.utc) - timedelta(days=days)).replace(tzinfo=None)


def _sum_tokens(col):
    """SUM() that accumulates in BIGINT so large token totals don't overflow
    SQL Server's INT aggregate (pyodbc DataError 22003 / SQL Server error 8115).
    """
    return func.coalesce(func.sum(cast(col, BigInteger)), 0)


def _cost_expr():
    """Estimated USD cost: each token bucket priced at its model's rate.

    Anthropic reports cache tokens outside input_tokens, so the buckets add
    cleanly. OpenAI includes cached reads inside input_tokens; pricing those
    rows overstates slightly unless the model's cache_read rate is set to the
    (negative) discount — acceptable for an estimate.
    """

    def term(tokens_col, rate_col):
        return func.coalesce(
            func.sum(func.coalesce(tokens_col, 0) * rate_col) / 1_000_000, 0
        )

    return (
        term(UsageLog.input_tokens, Model.cost_per_million_input)
        + term(UsageLog.output_tokens, Model.cost_per_million_output)
        + term(UsageLog.cache_creation_input_tokens, Model.cost_per_million_cache_write)
        + term(UsageLog.cache_read_input_tokens, Model.cost_per_million_cache_read)
    )


@router.get("/admin/usage", response_class=HTMLResponse)
async def admin_usage(
    request: Request,
    days: int = Query(30, ge=1, le=365),
    session: AsyncSession = Depends(get_session),
):
    await require_html_admin(request)
    since = _since(days)

    by_model = (
        await session.execute(
            select(
                UsageLog.model_name.label("model_name"),
                func.count(UsageLog.id).label("requests"),
                _sum_tokens(UsageLog.input_tokens).label("input_tokens"),
                _sum_tokens(UsageLog.output_tokens).label("output_tokens"),
                _sum_tokens(UsageLog.cache_creation_input_tokens).label("cache_creation"),
                _sum_tokens(UsageLog.cache_read_input_tokens).label("cache_read"),
                _sum_tokens(UsageLog.total_tokens).label("total_tokens"),
                _cost_expr().label("cost"),
            )
            .outerjoin(Model, UsageLog.model_name == Model.deployment_name)
            .where(UsageLog.created_at >= since)
            .group_by(UsageLog.model_name)
            .order_by(UsageLog.model_name)
        )
    ).all()

    by_endpoint = (
        await session.execute(
            select(
                UsageLog.endpoint.label("endpoint"),
                func.count(UsageLog.id).label("requests"),
                _sum_tokens(UsageLog.total_tokens).label("total_tokens"),
                func.sum(case((UsageLog.error_type.is_(None), 0), else_=1)).label("errors"),
            )
            .where(UsageLog.created_at >= since)
            .group_by(UsageLog.endpoint)
            .order_by(UsageLog.endpoint)
        )
    ).all()

    by_client = (
        await session.execute(
            select(
                Client.name.label("client_name"),
                func.count(UsageLog.id).label("requests"),
                _sum_tokens(UsageLog.total_tokens).label("total_tokens"),
                _cost_expr().label("cost"),
            )
            .join(Client, UsageLog.client_id == Client.id)
            .outerjoin(Model, UsageLog.model_name == Model.deployment_name)
            .where(UsageLog.created_at >= since)
            .group_by(Client.name)
            .order_by(Client.name)
        )
    ).all()

    return templates.TemplateResponse(
        request,
        "admin/usage.html",
        {
            "days": days,
            "by_model": by_model,
            "by_endpoint": by_endpoint,
            "by_client": by_client,
        },
    )
