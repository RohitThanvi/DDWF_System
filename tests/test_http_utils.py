"""Tests for the shared retry-with-backoff helper -- the fix for real 429s
hit on a live manifest-build run once request chunking (the 414 fix) made
several rapid-fire requests per AOI normal."""
from __future__ import annotations

import pytest

from app.data.http_utils import get_with_retry


class FakeResponse:
    def __init__(self, status_code, json_data=None, headers=None):
        self.status_code = status_code
        self._json = json_data or {}
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._json


class FlakyClient:
    """Returns 429 for the first `fail_times` calls, then 200."""

    def __init__(self, fail_times, retry_after=None):
        self.fail_times = fail_times
        self.calls = 0
        self.retry_after = retry_after

    async def get(self, url, params=None):
        self.calls += 1
        if self.calls <= self.fail_times:
            headers = {"Retry-After": str(self.retry_after)} if self.retry_after else {}
            return FakeResponse(429, headers=headers)
        return FakeResponse(200, json_data={"ok": True})


@pytest.mark.anyio
async def test_retries_then_succeeds(monkeypatch):
    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr("app.data.http_utils.asyncio.sleep", fake_sleep)

    client = FlakyClient(fail_times=2)
    resp = await get_with_retry(client, "http://example.test", max_retries=5, base_delay_s=0.01)

    assert resp.json() == {"ok": True}
    assert client.calls == 3
    assert len(sleeps) == 2  # slept before each of the two retried attempts


@pytest.mark.anyio
async def test_gives_up_after_max_retries(monkeypatch):
    async def fake_sleep(seconds):
        pass

    monkeypatch.setattr("app.data.http_utils.asyncio.sleep", fake_sleep)

    client = FlakyClient(fail_times=100)  # always fails
    with pytest.raises(RuntimeError):
        await get_with_retry(client, "http://example.test", max_retries=2, base_delay_s=0.01)
    assert client.calls == 3  # initial attempt + 2 retries


@pytest.mark.anyio
async def test_honors_retry_after_header(monkeypatch):
    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr("app.data.http_utils.asyncio.sleep", fake_sleep)

    client = FlakyClient(fail_times=1, retry_after=5)
    await get_with_retry(client, "http://example.test", max_retries=3, base_delay_s=0.01)

    # delay should be based on the Retry-After value (5s), not the tiny base_delay_s
    assert sleeps[0] >= 5.0
