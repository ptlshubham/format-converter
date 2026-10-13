"""
Request guards for the image AI service: a per-IP rate limit and body-size caps.

Plain ASGI middleware, so an over-limit request is answered before FastAPI reads
its body or the route runs, and no model is ever touched for it. Mounted inside
CORSMiddleware so 429/413 replies still carry CORS headers and the browser can
read them.
"""

import math
import time
from collections import deque
from typing import Deque, Dict, Iterable, Optional

from fastapi import HTTPException
from starlette.responses import JSONResponse

from .settings import Settings

PREFIX = "/api/background-remover"
RATE_LIMITED_PATHS = (f"{PREFIX}/remove", f"{PREFIX}/export-hd")
JSON_PATHS = (f"{PREFIX}/export-hd", f"{PREFIX}/composite")
WINDOW_SECONDS = 60.0

TOO_MANY_REQUESTS = "Too many requests — please wait a minute and try again."


class SlidingWindowLimiter:
    """
    Counts accepted requests per key over the last `window` seconds.
    In-process only: the service runs as one worker, so one table is the truth.
    """

    def __init__(self, limit: int, window: float = WINDOW_SECONDS):
        self.limit = limit
        self.window = window
        self._hits: Dict[str, Deque[float]] = {}
        self._last_sweep = 0.0

    def hit(self, key: str, now: Optional[float] = None) -> int:
        """Records a request. Returns 0 when it is allowed, else the seconds to wait."""
        now = time.monotonic() if now is None else now
        self._sweep(now)
        hits = self._hits.setdefault(key, deque())
        while hits and hits[0] <= now - self.window:
            hits.popleft()
        if len(hits) >= self.limit:
            return max(1, math.ceil(hits[0] + self.window - now))
        hits.append(now)
        return 0

    def _sweep(self, now: float) -> None:
        # Once a window, forget clients that have gone quiet so the table stays small.
        if now - self._last_sweep < self.window:
            return
        self._last_sweep = now
        for key in [k for k, hits in self._hits.items() if not hits or hits[-1] <= now - self.window]:
            del self._hits[key]


def _header(scope, name: bytes) -> Optional[str]:
    values = [value.decode("latin-1") for key, value in scope.get("headers", []) if key == name]
    return ",".join(values) if values else None


def client_ip(scope, trusted_proxies: Iterable[str]) -> str:
    """
    The caller's address. X-Forwarded-For is believed only when the direct peer
    is a trusted proxy. Proxies append the address they saw, so the list is read
    from the right, skipping our own proxies: entries further left are whatever
    the client chose to send and could be forged.
    """
    trusted = set(trusted_proxies)
    peer = (scope.get("client") or ("unknown", 0))[0]
    forwarded = _header(scope, b"x-forwarded-for")
    if peer not in trusted or not forwarded:
        return peer
    for hop in reversed(forwarded.split(",")):
        hop = hop.strip()
        if hop and hop not in trusted:
            return hop
    return peer


def _too_large(limit: int) -> str:
    return f"This image is too large. The limit is {limit // (1024 * 1024)} MB."


class RequestGuards:
    def __init__(self, app, settings: Settings):
        self.app = app
        self.settings = settings
        self.limiter = (
            SlidingWindowLimiter(settings.rate_limit_per_minute) if settings.rate_limit_per_minute > 0 else None
        )

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST":
            await self.app(scope, receive, send)
            return

        path = scope["path"].rstrip("/")

        # 1. Rate limit, counted on arrival, before the body is read.
        if self.limiter and path in RATE_LIMITED_PATHS:
            wait = self.limiter.hit(client_ip(scope, self.settings.trusted_proxies))
            if wait:
                response = JSONResponse(
                    {"detail": TOO_MANY_REQUESTS}, status_code=429, headers={"Retry-After": str(wait)}
                )
                await response(scope, receive, send)
                return

        # 2. Body size: refuse a declared oversize body without reading it...
        limit = self.settings.max_json_bytes if path in JSON_PATHS else self.settings.max_upload_bytes
        declared = _header(scope, b"content-length")
        if declared is not None:
            try:
                too_big = int(declared) > limit
            except ValueError:
                await JSONResponse({"detail": "Invalid Content-Length header."}, status_code=400)(scope, receive, send)
                return
            if too_big:
                await JSONResponse({"detail": _too_large(limit)}, status_code=413)(scope, receive, send)
                return

        # ...and stop a body sent without a length once it passes the cap. FastAPI
        # re-raises an HTTPException met while reading the body as that response.
        received = 0

        async def capped_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    raise HTTPException(status_code=413, detail=_too_large(limit))
            return message

        await self.app(scope, capped_receive, send)
