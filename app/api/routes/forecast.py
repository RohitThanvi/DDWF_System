"""
Core forecast endpoint — this is what Vayu calls.

POST /v1/forecast  ->  p10/p50/p90 timeseries for the requested AOI + horizon.

End-to-end flow mirrors Module 6 of the TDD:
  1. Global engine trajectory (cached per forecast cycle)
  2. Terrain fusion for the AOI
  3. Diffusion downscaling of the coarse patch
  4. Ensemble aggregation -> p10/p50/p90 + confidence flag
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import numpy as np
from fastapi import APIRouter, Depends

from app.api.schemas import ForecastRequest, ForecastResponse, TimestepValue
from app.core.logging import get_logger
from app.core.security import verify_api_key
from app.services.downscaler import DownscalerService
from app.services.ensembler import EnsemblerService
from app.services.global_engine import GlobalEngineService
from app.services.terrain_fusion import TerrainFusionService

router = APIRouter(prefix="/v1", tags=["forecast"], dependencies=[Depends(verify_api_key)])
log = get_logger(__name__)


@router.post("/forecast", response_model=ForecastResponse)
async def forecast(req: ForecastRequest) -> ForecastResponse:
    request_id = str(uuid.uuid4())
    log.info("forecast.requested", request_id=request_id, lat=req.lat, lon=req.lon, horizon_days=req.horizon_days)

    engine = GlobalEngineService.instance()
    terrain = TerrainFusionService.instance()
    downscaler = DownscalerService.instance()
    ensembler = EnsemblerService()

    # 1) cached global trajectory for this forecast cycle
    cond = np.zeros(8, dtype=np.float32)  # GHG anomaly + ENSO/MJO indices, see app/data/zarr_pipeline.py
    trajectory = engine.get_trajectory(cond)

    n_steps_requested = min(req.horizon_days * 4, trajectory.shape[0])  # 4 six-hourly steps/day

    # 2) terrain fusion for the AOI
    raster = terrain.fetch_raster_patch(tuple(req.bbox), resolution_m=req.resolution_m)

    # 3) downscale + ensemble per requested lead time (batched in production;
    #    looped here for clarity)
    timeseries: list[TimestepValue] = []
    for step_idx in range(0, n_steps_requested, 4):  # one point per day
        coarse_patch = trajectory[step_idx][:8]  # first 8 channels as a stand-in "target vars" slice
        coarse_tokens = coarse_patch.reshape(coarse_patch.shape[0], -1).T[:64]  # crude tokenization placeholder

        members = []
        for _m in range(3):  # small ensemble for a single request; full N/M grid runs offline
            member = downscaler.downscale(coarse_patch, raster, coarse_tokens)
            members.append(member)
        members_arr = np.stack(members)[None, ...]  # (1_ic_member, M_diffusion, C, H, W)

        agg = ensembler.aggregate(members_arr[0][None, ...])
        lead_days = step_idx / 4
        confidence = ensembler.confidence_flag(lead_days)

        def _summarize(field: np.ndarray) -> dict[str, float]:
            names = req.variables[: field.shape[0]]
            return {name: float(field[i].mean()) for i, name in enumerate(names)}

        timeseries.append(
            TimestepValue(
                valid_time=datetime.now(timezone.utc),
                lead_hours=step_idx * 6,
                p10=_summarize(agg["p10"][0]),
                p50=_summarize(agg["p50"][0]),
                p90=_summarize(agg["p90"][0]),
                confidence=confidence,
            )
        )

    warnings = []
    if req.horizon_days > 15:
        warnings.append(
            "Lead times beyond Day 15 are flagged low_skill: atmospheric predictability limits are structural, not a model deficiency."
        )

    return ForecastResponse(
        request_id=request_id,
        forecast_cycle=engine.cache.current_cycle_id(),
        aoi_center=[req.lat, req.lon],
        resolution_m=req.resolution_m,
        generated_at=datetime.now(timezone.utc),
        timeseries=timeseries,
        warnings=warnings,
    )
