"""
Purpose-built shim for Project Vayu. `/v1/forecast` is already generic
enough for Vayu to call directly, but this route exists so:

  - Vayu never has to know DDWF's internal variable-naming/response shape
    changes across versions — this is the one contract Vayu's client code
    depends on, decoupled from DDWF's own evolving `/v1/forecast`.
  - The response is pre-flattened to whatever geospatial-layer shape
    Vayu's frontend map layer expects (lat/lon point + scalar fields per
    day), instead of the fuller p10/p50/p90-per-variable object.

If Vayu's needs converge with `/v1/forecast`, deprecate this in favor of
calling that directly with an API key.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.api.routes.forecast import forecast as run_forecast
from app.api.schemas import ForecastRequest
from app.core.security import verify_api_key

router = APIRouter(prefix="/v1/vayu", tags=["vayu-integration"], dependencies=[Depends(verify_api_key)])


class VayuWeatherLayerPoint(BaseModel):
    lat: float
    lon: float
    day: int
    temperature_c: float | None = None
    wind_ms: float | None = None
    precip_mm: float | None = None
    confidence: str


class VayuWeatherLayerResponse(BaseModel):
    aoi_center: list[float]
    forecast_cycle: str
    points: list[VayuWeatherLayerPoint]


@router.post("/weather-layer", response_model=VayuWeatherLayerResponse)
async def weather_layer(req: ForecastRequest) -> VayuWeatherLayerResponse:
    """Flattened, map-layer-friendly forecast — call this from Vayu's
    backend with the same body as /v1/forecast."""
    full = await run_forecast(req)

    points = []
    for i, ts in enumerate(full.timeseries):
        points.append(
            VayuWeatherLayerPoint(
                lat=full.aoi_center[0],
                lon=full.aoi_center[1],
                day=i,
                temperature_c=ts.p50.get("temperature_2m"),
                wind_ms=ts.p50.get("wind_10m"),
                precip_mm=ts.p50.get("precipitation"),
                confidence=ts.confidence,
            )
        )

    return VayuWeatherLayerResponse(
        aoi_center=full.aoi_center,
        forecast_cycle=full.forecast_cycle,
        points=points,
    )
