"""
Wraps the SFNO model for cached, cycle-level trajectory production.
This is Module 2 + the caching half of Module 5/6 in the TDD.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.sfno import SFNO
from app.services.cache import TrajectoryCache, TrajectoryCacheEntry

log = get_logger(__name__)


class GlobalEngineService:
    _instance: "GlobalEngineService | None" = None

    def __init__(self):
        self.settings = get_settings()
        self.device = torch.device(self.settings.device)
        self.model = SFNO(sht_backend="fft")  # served weights use the FFT fallback (see sfno.py)
        self._load_weights()
        self.model.to(self.device).eval()
        self.cache = TrajectoryCache()

    @classmethod
    def instance(cls) -> "GlobalEngineService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_weights(self) -> None:
        ckpt_path = self.settings.global_engine_checkpoint
        try:
            state = torch.load(ckpt_path, map_location="cpu")
            self.model.load_state_dict(state)
            log.info("global_engine.weights_loaded", path=ckpt_path)
        except FileNotFoundError:
            log.warning(
                "global_engine.weights_missing",
                path=ckpt_path,
                note="Serving randomly-initialized weights — inference output is NOT meteorologically valid until a trained checkpoint is placed here.",
            )

    def get_trajectory(self, cond: np.ndarray, force_refresh: bool = False) -> np.ndarray:
        """Returns the cached (or freshly computed) 120-step, 0.25 deg global
        trajectory for the current forecast cycle. Shape: (steps, C, H, W)."""
        cycle_id = self.cache.current_cycle_id()

        if not force_refresh:
            cached = self.cache.get(cycle_id)
            if cached is not None:
                log.info("global_engine.cache_hit", cycle_id=cycle_id)
                return cached.trajectory

        log.info("global_engine.cache_miss_computing", cycle_id=cycle_id)
        x0 = self._load_initial_condition()
        cond_t = torch.tensor(cond, dtype=torch.float32, device=self.device).unsqueeze(0)
        x0_t = torch.tensor(x0, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            traj = self.model.rollout(x0_t, cond_t, steps=self.settings.rollout_steps)
        traj_np = traj.squeeze(0).cpu().numpy()

        self.cache.set(TrajectoryCacheEntry(cycle_id, time.time(), traj_np, cond))
        return traj_np

    def _load_initial_condition(self) -> np.ndarray:
        """Pull the latest analysis state from the Zarr data lake
        (app/data/zarr_pipeline.py). Stubbed with zeros until wired to a
        real store — swap this out, everything downstream is unaffected."""
        return np.zeros((86, 721, 1440), dtype=np.float32)
