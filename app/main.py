"""FastAPI application factory: mounts the proxy routes and the admin console."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Request, WebSocket
from fastapi.responses import JSONResponse, RedirectResponse, Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.sessions import SessionMiddleware

from app.admin import clients as admin_clients
from app.admin import credentials as admin_credentials
from app.admin import login as admin_login
from app.admin import models as admin_models
from app.admin import usage as admin_usage
from app.auth import (
    is_admin_session,
    is_redirect_exception,
    refresh_key_cache,
    resolve_requester_role,
    validate_api_key,
)
from app.config import config
from app.db import SessionLocal, get_session
from app.orm import Model
from app.proxy import anthropic, chat_completions, compact, responses
from app.proxy import websocket as ws_router
from app.retention import usage_prune_loop
from app.status import connectivity_probe_loop
from app.status import router as status_router
from app.utils import ConnectionLoggerMiddleware, log

# Site favicon, loaded once at import. Served at /favicon.svg and /favicon.ico.
_FAVICON_SVG = (Path(__file__).resolve().parent / "static" / "favicon.svg").read_bytes()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Inference Proxy",
        version="3.0.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    # Order matters: ConnectionLogger first (sees raw scope), then SessionMiddleware.
    app.add_middleware(ConnectionLoggerMiddleware)
    app.add_middleware(
        SessionMiddleware,
        secret_key=config.session_secret,
        session_cookie="proxy_session",
        same_site="lax",
        https_only=False,  # A reverse proxy may terminate TLS upstream.
    )

    # ---- Redirect-exception handler -----------------------------------------
    @app.exception_handler(StarletteHTTPException)
    async def _handle_redirect_exc(request: Request, exc: StarletteHTTPException):
        loc = is_redirect_exception(exc) if isinstance(exc, HTTPException) else None
        if loc:
            return RedirectResponse(loc, status_code=303)
        # Default behavior: re-raise the standard handler shape.
        return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)

    # ---- Startup ------------------------------------------------------------
    @app.on_event("startup")
    async def _startup():
        async with SessionLocal() as session:
            n = await refresh_key_cache(session)
            print(f"[startup] proxy key cache loaded: {n} active key(s)", flush=True)
        # Background tasks
        asyncio.create_task(usage_prune_loop())
        asyncio.create_task(connectivity_probe_loop())

    @app.on_event("shutdown")
    async def _shutdown():
        from app.azure import close_shared_client
        await close_shared_client()

    # ---- Public / health endpoints ------------------------------------------
    @app.get("/favicon.ico", include_in_schema=False)
    @app.get("/favicon.svg", include_in_schema=False)
    async def favicon():
        # Same SVG for both paths; modern browsers accept image/svg+xml at .ico.
        return Response(
            content=_FAVICON_SVG,
            media_type="image/svg+xml",
            headers={"Cache-Control": "public, max-age=86400"},
        )

    @app.get("/health")
    async def health():
        return {
            "status": "healthy",
            "streaming_enabled": config.enable_streaming,
        }

    @app.get("/")
    async def root(request: Request):
        if not is_admin_session(request):
            return RedirectResponse("/login", status_code=303)
        return RedirectResponse("/admin/", status_code=303)

    @app.get("/admin")
    async def admin_no_slash():
        return RedirectResponse("/admin/", status_code=308)

    def _require_listing_role(
        authorization: str | None,
        api_key: str | None,
        x_api_key: str | None,
    ) -> bool:
        """Authenticate a model-listing request and return whether the caller
        is privileged.

        Listing requires a valid issued proxy key (these endpoints are not
        public). Unprivileged clients never see admin-only models. The
        credential may arrive as `Authorization: Bearer`, `api-key`, or the
        Anthropic `x-api-key` header.
        """
        api_key_header = api_key or x_api_key
        if not validate_api_key(authorization, api_key_header):
            raise HTTPException(status_code=401, detail="Invalid API key")
        return resolve_requester_role(authorization, api_key_header) == "admin"

    @app.get("/v1/models")
    async def list_models(
        authorization: str | None = Header(None, alias="Authorization"),
        api_key: str | None = Header(None, alias="api-key"),
        x_api_key: str | None = Header(None, alias="x-api-key"),
        session: AsyncSession = Depends(get_session),
    ):
        """OpenAI-style listing — Anthropic Messages deployments excluded."""
        is_admin = _require_listing_role(authorization, api_key, x_api_key)
        now = int(time.time())
        stmt = (
            select(Model.deployment_name)
            .where(Model.enabled.is_(True))
            .where(Model.target_uri.is_not(None))
            .where(~Model.target_uri.contains("/anthropic/"))
            .where(~Model.target_uri.contains("api.anthropic.com"))
            .order_by(Model.deployment_name)
        )
        if not is_admin:
            stmt = stmt.where(Model.admin_only.is_(False))
        rows = await session.execute(stmt)
        return {
            "object": "list",
            "data": [
                {"id": name, "object": "model", "created": now, "owned_by": "proxy"}
                for (name,) in rows.all()
            ],
        }

    @app.get("/v1/anthropic/models")
    async def list_anthropic_models(
        authorization: str | None = Header(None, alias="Authorization"),
        api_key: str | None = Header(None, alias="api-key"),
        x_api_key: str | None = Header(None, alias="x-api-key"),
        session: AsyncSession = Depends(get_session),
    ):
        """Anthropic-Messages-compatible deployments only."""
        is_admin = _require_listing_role(authorization, api_key, x_api_key)
        now = int(time.time())
        stmt = (
            select(Model.deployment_name, Model.label, Model.description)
            .where(Model.enabled.is_(True))
            .where(Model.target_uri.is_not(None))
            .where(
                Model.target_uri.contains("/anthropic/")
                | Model.target_uri.contains("api.anthropic.com")
            )
            .order_by(Model.deployment_name)
        )
        if not is_admin:
            stmt = stmt.where(Model.admin_only.is_(False))
        rows = await session.execute(stmt)
        return {
            "object": "list",
            "data": [
                {
                    "id": name,
                    "object": "model",
                    "created": now,
                    "owned_by": "anthropic",
                    "label": label,
                    "description": description,
                }
                for (name, label, description) in rows.all()
            ],
        }

    # ---- Proxy routers (auth via proxy API key, no session) -----------------
    app.include_router(chat_completions.router)
    app.include_router(responses.router)
    app.include_router(compact.router)
    app.include_router(anthropic.router)
    app.include_router(ws_router.router)

    # ---- Admin console (session-authed) ------------------------------------
    app.include_router(admin_login.router)
    app.include_router(admin_clients.router)
    app.include_router(admin_credentials.router)
    app.include_router(admin_models.router)
    app.include_router(admin_usage.router)

    # ---- Public status page (anonymous, no session) -------------------------
    app.include_router(status_router)

    # ---- Catch-all (must be last) -------------------------------------------
    @app.websocket("/{path:path}")
    async def websocket_catch_all(ws: WebSocket):
        log(
            f"[WS Catch-All] Unexpected WebSocket path: {ws.url.path} "
            f"headers: authorization={ws.headers.get('authorization', 'none')!r} "
            f"sec-websocket-protocol={ws.headers.get('sec-websocket-protocol', 'none')!r}"
        )
        await ws.accept()
        await ws.send_text(
            json.dumps({"error": f"Unknown WebSocket endpoint: {ws.url.path}. Use /v1/responses"})
        )
        await ws.close(code=4004, reason="Unknown endpoint")

    @app.api_route(
        "/{path:path}",
        methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    )
    async def catch_all(request: Request, path: str):
        print(f"\n[CATCH-ALL] {request.method} /{path}")
        print(f"  Query: {request.url.query}")
        print(f"  Headers: {dict(request.headers)}")
        try:
            body = await request.body()
            if body:
                print(f"  Body (first 500 chars): {body[:500].decode('utf-8', errors='replace')}")
        except Exception:
            pass
        return JSONResponse(
            status_code=404,
            content={"error": f"Unknown route: {request.method} /{path}"},
        )

    return app


app = create_app()
