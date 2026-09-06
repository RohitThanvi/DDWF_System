"""
Real, free, key-less terrain data sources for the AOI downscaler.

DEM (elevation/slope/aspect): Open-Meteo's Elevation API
(https://open-meteo.com/en/docs/elevation-api), backed by Copernicus
DEM GLO-90 (90m, global, free license). Same provider as the coarse
forecast source (app/data/external_forecast.py) — no extra account, no key.

LULC (land use/land cover) and LST (land surface temperature) do NOT have
an equivalently trivial free/key-less source at the time of writing — ESA
WorldCover and MODIS LST both require registering with Copernicus/NASA
Earthdata and downloading raster tiles rather than a simple point API.
Those two channels stay as documented, explicit stubs
(`STUB_LULC_CHANNELS`, `STUB_LST_CHANNELS`) rather than fabricated numbers,
so it's always clear in the data exactly what's real terrain signal and
what isn't yet — see `TerrainFusionService.fetch_raster_patch`.
"""
from __future__ import annotations

import numpy as np

from app.core.logging import get_logger

log = get_logger(__name__)

OPEN_METEO_ELEVATION_URL = "https://api.open-meteo.com/v1/elevation"
MAX_COORDS_PER_REQUEST = 100  # Open-Meteo elevation API limit


class ElevationClient:
    """Fetches real elevation from Open-Meteo (Copernicus GLO-90), then
    derives slope/aspect via finite differences on the returned grid."""

    def __init__(self, base_url: str = OPEN_METEO_ELEVATION_URL, timeout_s: float = 15.0):
        self.base_url = base_url
        self.timeout_s = timeout_s

    async def fetch_elevation_grid(
        self, bbox: tuple[float, float, float, float], grid_size: int
    ) -> np.ndarray:
        """Returns a (grid_size, grid_size) elevation array in meters.
        Batches into <=100-coordinate requests per Open-Meteo's limit."""
        import httpx

        min_lon, min_lat, max_lon, max_lat = bbox
        lats = np.linspace(min_lat, max_lat, grid_size)
        lons = np.linspace(min_lon, max_lon, grid_size)
        points = [(lat, lon) for lat in lats for lon in lons]

        elevations: list[float] = []
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            for i in range(0, len(points), MAX_COORDS_PER_REQUEST):
                chunk = points[i : i + MAX_COORDS_PER_REQUEST]
                params = {
                    "latitude": ",".join(f"{lat:.5f}" for lat, _ in chunk),
                    "longitude": ",".join(f"{lon:.5f}" for _, lon in chunk),
                }
                resp = await client.get(self.base_url, params=params)
                resp.raise_for_status()
                elevations.extend(resp.json()["elevation"])

        return np.array(elevations, dtype=np.float32).reshape(grid_size, grid_size)


def slope_aspect_from_elevation(elevation: np.ndarray, cell_size_m: float = 1000.0) -> tuple[np.ndarray, np.ndarray]:
    """Standard finite-difference slope (degrees) + aspect (degrees from
    north) from a regularly-gridded elevation array. `cell_size_m` should
    be the approximate real-world spacing between grid points — pass the
    AOI's actual bbox width/grid_size in meters for a physically-correct
    slope; the default is a rough placeholder."""
    dz_dy, dz_dx = np.gradient(elevation, cell_size_m)
    slope = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))
    aspect = np.degrees(np.arctan2(dz_dy, -dz_dx))
    aspect = (aspect + 360) % 360
    return slope.astype(np.float32), aspect.astype(np.float32)


# Channel layout contract with TerrainFusionService.fetch_raster_patch:
# index 0: elevation (real, Open-Meteo/Copernicus GLO-90)
# index 1: slope (real, derived)
# index 2: aspect (real, derived)
# index 3-6: LULC one-hot (STUB — zeros until a real LULC source is wired)
# index 7: LST (STUB — zeros until a real LST source is wired)
REAL_DEM_CHANNELS = 3
STUB_LULC_CHANNELS = 4
STUB_LST_CHANNELS = 1
TOTAL_TERRAIN_CHANNELS = REAL_DEM_CHANNELS + STUB_LULC_CHANNELS + STUB_LST_CHANNELS
