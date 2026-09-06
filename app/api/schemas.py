"""Pydantic request/response contracts — this is the API surface Vayu
integrates against, so keep it stable and additive (new optional fields,
never breaking renames) once v1 ships."""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, model_validator


class ForecastRequest(BaseModel):
    lat: float = Field(..., ge=-90, le=90, description="AOI center latitude")
    lon: float = Field(..., ge=-180, le=180, description="AOI center longitude")
    bbox: list[float] | None = Field(
        None, description="[min_lon, min_lat, max_lon, max_lat]; if omitted, a small default AOI around (lat, lon) is used"
    )
    horizon_days: int = Field(
        16, ge=1, le=16,
        description="Forecast horizon. Capped at 16 days — the coarse forecast source (Open-Meteo) doesn't offer further out; see docs/ARCHITECTURE.md.",
    )
    variables: list[str] = Field(
        default_factory=lambda: ["temperature_2m", "wind_10m", "precipitation", "surface_pressure"],
        description="Which variables to return",
    )
    resolution_m: int = Field(100, ge=100, le=1000, description="Target downscaled resolution in meters")

    @model_validator(mode="after")
    def _default_bbox(self) -> "ForecastRequest":
        if self.bbox is None:
            pad = 0.25
            self.bbox = [self.lon - pad, self.lat - pad, self.lon + pad, self.lat + pad]
        return self


class TimestepValue(BaseModel):
    valid_time: datetime
    lead_hours: int
    p10: dict[str, float]
    p50: dict[str, float]
    p90: dict[str, float]
    confidence: str  # high_skill | moderate_skill | low_skill


class ForecastResponse(BaseModel):
    request_id: str
    forecast_cycle: str
    aoi_center: list[float]
    resolution_m: int
    generated_at: datetime
    timeseries: list[TimestepValue]
    warnings: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    downscaler_loaded: bool
    coarse_forecast_provider: str
    sfno_checkpoint_present: bool  # optional upgrade path (Option A); unused by default request flow
    device: str
