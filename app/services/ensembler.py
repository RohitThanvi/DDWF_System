"""
Two-tier ensemble + uncertainty quantification (Module 4).

Tier 1: initial-condition perturbation (N members) run through the SFNO
        global engine (bred-vector / EDA-style — owned by the training/IC
        generation pipeline, not this service).
Tier 2: model-form perturbation via different diffusion sampling seeds at
        the downscaling head, per IC member.

This service aggregates the resulting N x M member fields into p10/p50/p90
empirical quantiles per timestep, per variable.
"""
from __future__ import annotations

import numpy as np

from app.core.config import get_settings
from app.core.logging import get_logger

log = get_logger(__name__)


class EnsemblerService:
    def __init__(self):
        self.settings = get_settings()

    def aggregate(self, member_fields: np.ndarray) -> dict[str, np.ndarray]:
        """member_fields: (N_members, T, C, H, W) -> dict of p10/p50/p90,
        each (T, C, H, W)."""
        p10 = np.percentile(member_fields, 10, axis=0)
        p50 = np.percentile(member_fields, 50, axis=0)
        p90 = np.percentile(member_fields, 90, axis=0)
        return {"p10": p10, "p50": p50, "p90": p90}

    def confidence_flag(self, lead_time_days: float) -> str:
        """Day 15-30 outputs get an explicit low-skill flag rather than
        false precision — atmospheric predictability limits are structural,
        not a bug the ensemble can architect its way out of (see TDD
        'Known Limitations')."""
        if lead_time_days <= 10:
            return "high_skill"
        if lead_time_days <= 15:
            return "moderate_skill"
        return "low_skill"
