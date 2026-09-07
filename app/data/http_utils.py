"""
Shared retry-with-backoff helper for the free external HTTP APIs this
project depends on (Open-Meteo, ORNL DAAC). None of them require a key,
but all of them rate-limit — and the fix for the earlier 414 bug (batching
large coordinate grids into many smaller requests) makes rate-limiting
more likely to surface, not less: a single AOI with a 32x32 fine grid now
issues ~11 chunked requests back-to-back, and five AOIs in a tight loop
send dozens of requests in quick succession with no spacing at all. That's
exactly what produced real 429s on a live run — expected behavior from a
free, fair-use API being hit faster than one-request-per-AOI, not a sign
Open-Meteo itself is unreliable.

This is deliberately generic (works with any httpx.AsyncClient) so
external_forecast.py, terrain_sources.py, and
scripts/build_aoi_pairs_manifest.py all share one retry policy instead of
three copies of similar-but-slightly-different logic.
"""
from __future__ import annotations

import asyncio
import random

from app.core.logging import get_logger

log = get_logger(__name__)


async def get_with_retry(client, url: str, params: dict | None = None,
                          max_retries: int = 5, base_delay_s: float = 1.0):
    """GET with exponential backoff + jitter on 429/5xx. Honors a
    Retry-After response header when present. Raises on the final attempt
    if still failing, and on any other 4xx immediately (no point retrying
    a malformed request)."""
    for attempt in range(max_retries + 1):
        resp = await client.get(url, params=params)

        if resp.status_code == 429 or resp.status_code >= 500:
            if attempt == max_retries:
                resp.raise_for_status()

            retry_after = resp.headers.get("Retry-After")
            if retry_after is not None:
                try:
                    delay = float(retry_after)
                except ValueError:
                    delay = base_delay_s * (2**attempt)
            else:
                delay = base_delay_s * (2**attempt)
            delay += random.uniform(0, delay * 0.25)  # jitter, avoid thundering-herd retries

            log.warning(
                "http.retrying_after_rate_limit_or_server_error",
                url=url, attempt=attempt, delay_s=round(delay, 1), status=resp.status_code,
            )
            await asyncio.sleep(delay)
            continue

        resp.raise_for_status()
        return resp

    raise RuntimeError("unreachable")  # loop always returns or raises above
