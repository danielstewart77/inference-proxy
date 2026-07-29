"""Shared logging + ASGI middleware."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from app.config import config


_file_logger = logging.getLogger("proxy")
_file_logger.setLevel(logging.DEBUG)
_log_path = Path(__file__).resolve().parent.parent / "proxy.log"
_log_handler = logging.FileHandler(_log_path)
_log_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))
_file_logger.addHandler(_log_handler)


def log(msg: str) -> None:
    if config.enable_logging:
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] {msg}")
    _file_logger.info(msg)


class ConnectionLoggerMiddleware:
    """ASGI middleware that logs every incoming WebSocket connection before routing."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "websocket":
            path = scope.get("path", "?")
            headers = dict(scope.get("headers", []))
            subproto = headers.get(b"sec-websocket-protocol", b"none").decode()
            log(f"[ASGI-WS] Incoming WebSocket: path={path} subprotocols={subproto}")

            accepted = False

            async def _send_wrapper(message):
                nonlocal accepted
                msg_type = message.get("type", "")
                if msg_type == "websocket.accept":
                    accepted = True
                elif msg_type == "websocket.close" and not accepted:
                    code = message.get("code", "?")
                    log(f"[ASGI-WS] WS REJECTED (closed before accept): path={path} code={code}")
                elif msg_type == "websocket.http.response.start":
                    status = message.get("status", "?")
                    log(f"[ASGI-WS] WS rejected with HTTP {status}: path={path}")
                await send(message)

            try:
                await self.app(scope, receive, _send_wrapper)
            except Exception as e:
                log(f"[ASGI-WS] Exception during WS: path={path} {type(e).__name__}: {e}")
                raise
        else:
            await self.app(scope, receive, send)
