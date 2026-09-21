"""
Basic per-client rate limiting.

DDWF has no rate limiting at all today: a single caller (misbehaving
client, retry storm, or just an uncapped loop on Vayu's side) can issue
unlimited /v1/forecast requests, each of which fans out to Open-Meteo/ESA/
MODIS and runs real model inference. That's both a self-inflicted-DoS risk
and a fast way to get this service's IP rate-limited or banned by the free
upstream APIs it depends on (see app/data/http_utils.py's docstring on
exactly that happening during manifest building).

This is an in-memory sliding-window limiter keyed by the caller's API key
(falling back to client IP for unauthenticated requests, so a 401 doesn't
skip limiting entirely). It is deliberately simple:

- Correct and sufficient for a single-process deployment (Dockerfile
  defaults to `--workers 1`; see its comment on why that's fine now that
  model inference no longer blocks the event loop).
- NOT correct across multiple processes/instances -- each process has its
  own counters, so real capacity under a multi-worker or multi-replica
  deploy is (limit * process_count), not `limit`. If DDWF ever scales
  beyond one process, swap this for a Redis-backed limiter (Redis is
  already a dependency here -- see app/services/cache.py) using
  INCR + EXPIRE instead of an in-memory dict.
"""
from __future__ import annotations

import time

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.logging import get_logger

log = get_logger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.limit = requests_per_minute
        self.window_s = 60.0
        self._hits: dict[str, list[float]] = {}

    def _client_key(self, request: Request) -> str:
        api_key = request.headers.get("x-api-key")
        if api_key:
            return f"key:{api_key}"
        client = request.client
        return f"ip:{client.host}" if client else "ip:unknown"

    async def dispatch(self, request: Request, call_next) -> Response:
        # Never rate-limit liveness/readiness probes or the metrics
        # scrape -- those are infrastructure, not caller traffic, and
        # throttling them can make an orchestrator think the service is
        # unhealthy under otherwise-legitimate load.
        if request.url.path in ("/health", "/metrics"):
            return await call_next(request)

        key = self._client_key(request)
        now = time.monotonic()
        hits = self._hits.setdefault(key, [])
        cutoff = now - self.window_s
        while hits and hits[0] < cutoff:
            hits.pop(0)

        if len(hits) >= self.limit:
            log.warning("rate_limit.exceeded", client=key, path=request.url.path)
            return Response(
                content='{"detail":"Rate limit exceeded. Try again shortly."}',
                status_code=429,
                media_type="application/json",
                headers={"Retry-After": str(int(self.window_s))},
            )

        hits.append(now)
        # Opportunistic cleanup so `_hits` doesn't grow forever with every
        # distinct API key / IP that has ever made one request -- same
        # unbounded-dict shape as the coarse-forecast cache bug, avoided
        # here from the start.
        if len(self._hits) > 10_000:
            stale = [k for k, v in self._hits.items() if not v or v[-1] < cutoff]
            for k in stale:
                del self._hits[k]

        return await call_next(request)
