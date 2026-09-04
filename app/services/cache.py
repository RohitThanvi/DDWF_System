"""
Trajectory cache: the 30-day (120-step) global rollout is expensive
(a full autoregressive SFNO pass) and identical for every user within a
forecast cycle, so it is computed once per cycle and reused across all
AOI queries — this is the single biggest cost lever in the whole system
(see Module 5 of the TDD: "cached and refreshed once per forecast cycle,
not per-query").

Backing store: Redis for the hot path (small, fast, TTL'd), with an
optional on-disk parquet/zarr fallback for local dev without Redis.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.core.config import get_settings
from app.core.logging import get_logger

log = get_logger(__name__)


@dataclass
class TrajectoryCacheEntry:
    cycle_id: str
    computed_at: float
    trajectory: np.ndarray  # (steps, C, H, W) or a lazily-loaded zarr ref
    cond: np.ndarray


class TrajectoryCache:
    """Redis-backed if REDIS_URL is reachable, else falls back to a local
    on-disk cache under CACHE_DIR (fine for dev / single-instance deploys)."""

    def __init__(self):
        self.settings = get_settings()
        self._redis = None
        try:
            import redis

            self._redis = redis.from_url(self.settings.redis_url, socket_connect_timeout=1)
            self._redis.ping()
        except Exception:
            log.warning("cache.redis_unavailable", fallback="disk")
            self._redis = None

    def _disk_path(self, cycle_id: str) -> Path:
        return self.settings.cache_path / f"trajectory_{cycle_id}.npz"

    def get(self, cycle_id: str) -> TrajectoryCacheEntry | None:
        if self._redis is not None:
            raw = self._redis.get(f"traj:{cycle_id}:meta")
            if raw is None:
                return None
            meta = json.loads(raw)
            arr_bytes = self._redis.get(f"traj:{cycle_id}:arr")
            trajectory = np.frombuffer(arr_bytes, dtype=np.float32).reshape(meta["shape"])
            return TrajectoryCacheEntry(cycle_id, meta["computed_at"], trajectory, np.array(meta["cond"]))

        path = self._disk_path(cycle_id)
        if not path.exists():
            return None
        data = np.load(path, allow_pickle=True)
        return TrajectoryCacheEntry(cycle_id, float(data["computed_at"]), data["trajectory"], data["cond"])

    def set(self, entry: TrajectoryCacheEntry) -> None:
        ttl = self.settings.trajectory_cache_ttl_hours * 3600
        if self._redis is not None:
            meta = {"computed_at": entry.computed_at, "shape": list(entry.trajectory.shape), "cond": entry.cond.tolist()}
            self._redis.setex(f"traj:{entry.cycle_id}:meta", ttl, json.dumps(meta))
            self._redis.setex(f"traj:{entry.cycle_id}:arr", ttl, entry.trajectory.astype(np.float32).tobytes())
            return

        np.savez_compressed(
            self._disk_path(entry.cycle_id),
            computed_at=entry.computed_at,
            trajectory=entry.trajectory,
            cond=entry.cond,
        )

    def current_cycle_id(self) -> str:
        """Forecast cycles are anchored to 00z/06z/12z/18z, matching typical
        NWP initialization times; refresh cadence controlled by
        TRAJECTORY_CACHE_TTL_HOURS."""
        hop = self.settings.trajectory_cache_ttl_hours
        now = time.gmtime()
        cycle_hour = (now.tm_hour // hop) * hop
        return f"{now.tm_year:04d}{now.tm_mon:02d}{now.tm_mday:02d}T{cycle_hour:02d}Z"
