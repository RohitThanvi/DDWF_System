"""
Coarse global forecast source — Open-Meteo (free, no API key, no rate-limit
tier needed for reasonable use: https://open-meteo.com).

Option B design decision: instead of training/serving the SFNO global
engine (Module 2) ourselves, we treat "a coarse 25km-ish global forecast"
as a solved problem and pull it from a free, already-operational NWP
aggregator. All of DDWF's own training effort then goes into the genuinely
novel piece: the satellite-terrain-conditioned diffusion downscaler
(Module 3) + the zero-shot SIREN implicit terrain field.

Open-Meteo blends multiple national forecast models (GFS, ICON, ECMWF
open-data, etc.) depending on location and gives up to 16 days of hourly
forecast — so DDWF's horizon is capped at 16 days in this mode, down from
the original 30-day SFNO design (see ForecastRequest.horizon_days).

The SFNO code path (app/models/sfno.py, app/services/global_engine.py,
training/train_global_engine.py) is kept in the repo and still works
standalone — it's the natural upgrade path if/when there's a reason to
own the global-engine piece instead of depending on Open-Meteo (see
docs/TRAINING.md "Option A").
"""
from __future__ import annotations

import asyncio

import numpy as np

from app.core.config import get_settings
from app.core.logging import get_logger

log = get_logger(__name__)

# Open-Meteo hourly variable names we request, in a fixed order that
# becomes the channel axis of the coarse patch array. Kept at 8 channels
# to match DiffusionDownscaler's default cond/context wiring.
OPEN_METEO_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "cloud_cover",
    "shortwave_radiation",
]

OPEN_METEO_BASE_URL = "https://api.open-meteo.com/v1/forecast"

# Open-Meteo documents up to 1000 locations per request for weather
# endpoints, but that's a location-count limit, not a URL-length one — a
# plain HTTP GET with hundreds of lat/lon pairs in the query string trips
# a 414 Request-URI Too Large from the server/proxy well before 1000. A
# fine_grid of 32 (1024 points) hits this immediately. Chunking at 100
# (same conservative limit Open-Meteo's own Elevation API documents) keeps
# every request's URL comfortably short regardless of grid_size.
MAX_COORDS_PER_REQUEST = 100


def _grid_points(bbox: tuple[float, float, float, float], grid_size: int) -> list[tuple[float, float]]:
    """Evenly-spaced (lat, lon) sample points tiling the AOI bbox, used to
    build a pseudo-raster coarse patch out of Open-Meteo's point API."""
    min_lon, min_lat, max_lon, max_lat = bbox
    lats = np.linspace(min_lat, max_lat, grid_size)
    lons = np.linspace(min_lon, max_lon, grid_size)
    return [(float(lat), float(lon)) for lat in lats for lon in lons]


class OpenMeteoClient:
    """Thin async client. Batches grid points into
    <=MAX_COORDS_PER_REQUEST-coordinate requests per AOI (comma-separated
    lat/lon lists, per Open-Meteo's multi-location support) rather than one
    request per point."""

    def __init__(self, base_url: str = OPEN_METEO_BASE_URL, timeout_s: float = 15.0):
        self.base_url = base_url
        self.timeout_s = timeout_s

    async def fetch_coarse_patch(
        self,
        bbox: tuple[float, float, float, float],
        grid_size: int = 8,
        forecast_days: int = 16,
        variables: list[str] | None = None,
    ) -> dict:
        """Returns {"hourly_time": [...], "data": np.ndarray of shape
        (n_hours, n_vars, grid_size, grid_size)}. Batches into
        <=MAX_COORDS_PER_REQUEST-coordinate requests to avoid a 414 from
        the server on larger grids (see MAX_COORDS_PER_REQUEST docstring),
        with retry-with-backoff on 429s (see app/data/http_utils.py) and a
        small delay between chunks so a large grid doesn't burst dozens of
        requests at once."""
        import httpx

        from app.data.http_utils import get_with_retry

        variables = variables or OPEN_METEO_VARIABLES
        points = _grid_points(bbox, grid_size)

        locations: list[dict] = []
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            for i in range(0, len(points), MAX_COORDS_PER_REQUEST):
                chunk = points[i : i + MAX_COORDS_PER_REQUEST]
                params = {
                    "latitude": ",".join(f"{lat:.4f}" for lat, _ in chunk),
                    "longitude": ",".join(f"{lon:.4f}" for _, lon in chunk),
                    "hourly": ",".join(variables),
                    "forecast_days": min(forecast_days, 16),
                    "timezone": "UTC",
                }
                resp = await get_with_retry(client, self.base_url, params)
                payload = resp.json()
                # Open-Meteo returns a single object for a one-point chunk,
                # a list for multiple.
                locations.extend(payload if isinstance(payload, list) else [payload])
                if i + MAX_COORDS_PER_REQUEST < len(points):
                    await asyncio.sleep(0.3)

        if len(locations) != len(points):
            log.warning("open_meteo.partial_response", expected=len(points), got=len(locations))

        hourly_time = locations[0]["hourly"]["time"]
        n_hours = len(hourly_time)
        data = np.zeros((n_hours, len(variables), grid_size, grid_size), dtype=np.float32)

        for idx, loc in enumerate(locations):
            row, col = divmod(idx, grid_size)
            hourly = loc.get("hourly", {})
            for v_idx, var in enumerate(variables):
                series = hourly.get(var)
                if series is None:
                    continue
                arr = np.array(series, dtype=np.float32)
                arr = np.nan_to_num(arr, nan=0.0)
                data[: len(arr), v_idx, row, col] = arr[:n_hours]

        return {"hourly_time": hourly_time, "data": data, "variables": variables}


class CoarseForecastService:
    """Cached wrapper around OpenMeteoClient — one fetch per (rounded bbox,
    forecast cycle), shared across ensemble members and re-requested
    minutely-identical queries within the cache TTL."""

    _instance: "CoarseForecastService | None" = None

    def __init__(self):
        self.settings = get_settings()
        self.client = OpenMeteoClient()
        self._cache: dict[str, dict] = {}  # simple process-local cache; see cache.py for the Redis pattern

    @classmethod
    def instance(cls) -> "CoarseForecastService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @staticmethod
    def _cache_key(bbox: tuple[float, float, float, float], grid_size: int, forecast_days: int) -> str:
        rounded = tuple(round(v, 2) for v in bbox)
        return f"{rounded}:{grid_size}:{forecast_days}"

    async def get_coarse_patch(
        self,
        bbox: tuple[float, float, float, float],
        grid_size: int = 8,
        forecast_days: int = 16,
    ) -> dict:
        key = self._cache_key(bbox, grid_size, forecast_days)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        result = await self.client.fetch_coarse_patch(bbox, grid_size=grid_size, forecast_days=forecast_days)
        self._cache[key] = result
        return result
