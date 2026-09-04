# DDWF System — API Reference

Base URL: `http://<host>:8000` (local) or your deployed URL.

All non-health endpoints require `X-API-Key: <DDWF_API_KEY>`.

## `GET /health`

Liveness/readiness probe. No auth required.

```json
{
  "status": "ok",
  "global_engine_loaded": false,
  "downscaler_loaded": false,
  "device": "cpu"
}
```

`*_loaded: false` means a checkpoint file wasn't found at the configured
path — the service still runs (useful for smoke-testing the API surface)
but forecasts are not meteorologically meaningful until real weights are
placed at `GLOBAL_ENGINE_CHECKPOINT` / `DOWNSCALER_CHECKPOINT`.

## `POST /v1/forecast`

The primary contract. Full p10/p50/p90 timeseries.

**Request**
```json
{
  "lat": 26.9,
  "lon": 75.8,
  "bbox": [75.55, 26.65, 76.05, 27.15],
  "horizon_days": 30,
  "variables": ["temperature_2m", "wind_10m", "precipitation", "surface_pressure"],
  "resolution_m": 100
}
```
`bbox` is optional — a small default AOI is generated around `(lat, lon)` if omitted.

**Response**
```json
{
  "request_id": "…",
  "forecast_cycle": "20260904T06Z",
  "aoi_center": [26.9, 75.8],
  "resolution_m": 100,
  "generated_at": "2026-09-04T06:12:00Z",
  "timeseries": [
    {
      "valid_time": "2026-09-04T06:00:00Z",
      "lead_hours": 0,
      "p10": {"temperature_2m": 24.1, "wind_10m": 2.0, "precipitation": 0.0, "surface_pressure": 1004.2},
      "p50": {"temperature_2m": 27.3, "wind_10m": 3.4, "precipitation": 0.1, "surface_pressure": 1005.8},
      "p90": {"temperature_2m": 30.6, "wind_10m": 5.1, "precipitation": 2.4, "surface_pressure": 1007.1},
      "confidence": "high_skill"
    }
  ],
  "warnings": []
}
```
`confidence` is `high_skill` (Day 0-10), `moderate_skill` (Day 10-15), or
`low_skill` (Day 15-30) — see `EnsemblerService.confidence_flag`.

## `POST /v1/vayu/weather-layer`

Same request body as `/v1/forecast`. Returns a flattened, map-layer-ready
shape for Vayu's frontend instead of the fuller p10/p50/p90 object.

```json
{
  "aoi_center": [26.9, 75.8],
  "forecast_cycle": "20260904T06Z",
  "points": [
    {"lat": 26.9, "lon": 75.8, "day": 0, "temperature_c": 27.3, "wind_ms": 3.4, "precip_mm": 0.1, "confidence": "high_skill"}
  ]
}
```

## Calling it from Vayu

```python
import httpx

async def get_ddwf_layer(lat: float, lon: float, horizon_days: int = 14):
    async with httpx.AsyncClient(base_url="https://<ddwf-host>") as client:
        resp = await client.post(
            "/v1/vayu/weather-layer",
            headers={"X-API-Key": settings.ddwf_api_key},
            json={"lat": lat, "lon": lon, "horizon_days": horizon_days},
            timeout=30.0,
        )
        resp.raise_for_status()
        return resp.json()
```

Add `DDWF_BASE_URL` and `DDWF_API_KEY` to Vayu's own `.env` and it's a
two-line integration from there.
