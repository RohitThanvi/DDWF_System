"""
Core forecast endpoint — this is what Vayu calls.

POST /v1/forecast  ->  p10/p50/p90 timeseries for the requested AOI + horizon.

Option B end-to-end flow (see docs/ARCHITECTURE.md for the full rationale):
  1. Coarse global forecast pulled from Open-Meteo (free, no training/serving
     cost — replaces the SFNO global engine in the default request path)
  2. Terrain fusion for the AOI (DEM/LULC/LST)
  3. Diffusion downscaling of the coarse patch, conditioned on terrain
  4. Ensemble aggregation across diffusion-sampling seeds -> p10/p50/p90 + confidence flag

The SFNO global engine (app/services/global_engine.py) is a drop-in
alternative to step 1 if/when DDWF trains and owns that piece — swap the
`CoarseForecastService` call below for `GlobalEngineService.instance().get_trajectory(...)`.
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone

import httpx
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, status

from app.api.schemas import ForecastRequest, ForecastResponse, TimestepValue
from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.security import verify_api_key
from app.data.external_forecast import CoarseForecastService
from app.services.downscaler import DownscalerService
from app.services.ensembler import EnsemblerService
from app.services.terrain_fusion import TerrainFusionService

router = APIRouter(prefix="/v1", tags=["forecast"], dependencies=[Depends(verify_api_key)])
log = get_logger(__name__)


@router.post("/forecast", response_model=ForecastResponse)
async def forecast(req: ForecastRequest) -> ForecastResponse:
    request_id = str(uuid.uuid4())
    settings = get_settings()
    log.info("forecast.requested", request_id=request_id, lat=req.lat, lon=req.lon, horizon_days=req.horizon_days)

    coarse_source = CoarseForecastService.instance()
    terrain = TerrainFusionService.instance()
    downscaler = DownscalerService.instance()
    ensembler = EnsemblerService()

    # 1) coarse global forecast for the AOI (Open-Meteo grid, cached).
    # Unlike terrain fetches (which already degrade to zeros internally on
    # failure -- see TerrainFusionService), a coarse-forecast failure means
    # there is no signal to downscale at all, so surface it as a clean
    # 502/504 instead of letting httpx's exception propagate as a bare 500
    # with no indication of whose fault it is.
    try:
        coarse = await coarse_source.get_coarse_patch(
            tuple(req.bbox), grid_size=settings.coarse_grid_size, forecast_days=req.horizon_days
        )
    except httpx.TimeoutException as exc:
        log.warning("forecast.coarse_source_timeout", request_id=request_id, error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Coarse forecast source (Open-Meteo) timed out. Try again shortly.",
        ) from exc
    except httpx.HTTPError as exc:
        log.error("forecast.coarse_source_error", request_id=request_id, error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Coarse forecast source (Open-Meteo) is unavailable. Try again shortly.",
        ) from exc
    coarse_data = coarse["data"]  # (n_hours, n_vars, grid, grid)

    # 2) terrain fusion for the AOI
    raster = await terrain.fetch_raster_patch(tuple(req.bbox), resolution_m=req.resolution_m)

    # 3) downscale + ensemble, one point per day (take the 12:00 UTC hourly slice)
    timeseries: list[TimestepValue] = []
    hours_per_day = 24
    n_days = min(req.horizon_days, coarse_data.shape[0] // hours_per_day)

    for day in range(n_days):
        hour_idx = min(day * hours_per_day + 12, coarse_data.shape[0] - 1)
        coarse_patch = coarse_data[hour_idx]  # (n_vars, grid, grid)
        coarse_tokens = coarse_patch.reshape(coarse_patch.shape[0], -1).T  # (grid*grid, n_vars) tokens for cross-attn

        # downscaler.downscale() is synchronous, CPU-bound PyTorch inference
        # (an n_steps-iteration DDIM sampling loop) -- calling it directly
        # here would block the whole asyncio event loop for the entire
        # duration, including /health and every other concurrent request,
        # for as long as this one forecast takes to compute (multiplied by
        # horizon_days * 3 ensemble members). asyncio.to_thread runs it in
        # the default thread pool instead so the event loop stays free.
        members = await asyncio.gather(
            *(
                asyncio.to_thread(downscaler.downscale, coarse_patch, raster, coarse_tokens)
                for _seed in range(3)  # model-form perturbation via diffusion sampling seed
            )
        )
        members_arr = np.stack(members)  # (M, C, H, W)

        agg = ensembler.aggregate(members_arr)  # dict of p10/p50/p90, each (C, H, W)
        confidence = ensembler.confidence_flag(float(day))

        def _summarize(field: np.ndarray) -> dict[str, float]:
            names = req.variables[: field.shape[0]]
            return {name: float(field[i].mean()) for i, name in enumerate(names)}

        timeseries.append(
            TimestepValue(
                valid_time=datetime.now(timezone.utc),
                lead_hours=day * 24,
                p10=_summarize(agg["p10"]),
                p50=_summarize(agg["p50"]),
                p90=_summarize(agg["p90"]),
                confidence=confidence,
            )
        )

    warnings = []
    if not downscaler_has_weights():
        warnings.append(
            "Downscaler checkpoint not found — serving randomly-initialized weights. "
            "See docs/TRAINING.md to train real ones."
        )

    return ForecastResponse(
        request_id=request_id,
        forecast_cycle=coarse["hourly_time"][0] if coarse["hourly_time"] else "unknown",
        aoi_center=[req.lat, req.lon],
        resolution_m=req.resolution_m,
        generated_at=datetime.now(timezone.utc),
        timeseries=timeseries,
        warnings=warnings,
    )


def downscaler_has_weights() -> bool:
    from pathlib import Path

    return Path(get_settings().downscaler_checkpoint).exists()
