"""
Real, free, key-less terrain data sources for the AOI downscaler.

DEM (elevation/slope/aspect): Open-Meteo's Elevation API
(https://open-meteo.com/en/docs/elevation-api), backed by Copernicus
DEM GLO-90 (90m, global, free license). Same provider as the coarse
forecast source (app/data/external_forecast.py) — no extra account, no key.

LULC (land use/land cover): ESA WorldCover 10m v200, read directly off its
public AWS COG bucket via HTTP range requests (no download, no key) — see
`LandCoverClient` below.

LST (land surface temperature) does NOT have an equivalently trivial free/
key-less source at the time of writing — MODIS LST requires a NASA
Earthdata login and a different access pattern (Earthdata Cloud / AppEEARS)
than the simple HTTP-range-read pattern the other two sources use. It
stays an explicit, documented stub (`STUB_LST_CHANNELS`) rather than a
fabricated number, so it's always clear in the data exactly what's real
signal and what isn't yet — see `TerrainFusionService.fetch_raster_patch`.
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
# index 3-6: LULC class-group fractions (real, ESA WorldCover 10m v200 —
#            see LandCoverClient below): vegetation, built-up, water, bare/snow
# index 7: LST (STUB — zeros until a real key-less LST source is wired;
#          MODIS LST requires a NASA Earthdata login, unlike the other
#          three sources, which is why it's the one channel left as zeros)
REAL_DEM_CHANNELS = 3
REAL_LULC_CHANNELS = 4
STUB_LST_CHANNELS = 1
TOTAL_TERRAIN_CHANNELS = REAL_DEM_CHANNELS + REAL_LULC_CHANNELS + STUB_LST_CHANNELS

# Backward-compatible alias (old name implied "stub"; LULC is real now)
STUB_LULC_CHANNELS = REAL_LULC_CHANNELS


WORLDCOVER_BASE_URL = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map"

# ESA WorldCover 10m v200 raw class codes, grouped into 4 semantic fractions.
# https://esa-worldcover.org (11 classes -> 4 groups for a compact, still
# meaningful conditioning signal for the downscaler).
_LULC_GROUPS: dict[str, tuple[int, ...]] = {
    "vegetation": (10, 20, 30, 40, 90, 95, 100),  # tree/shrub/grass/crop/wetland/mangrove/moss
    "built_up": (50,),
    "water": (80,),
    "bare_or_snow": (60, 70),                      # bare/sparse veg, snow/ice
}
LULC_GROUP_ORDER = ["vegetation", "built_up", "water", "bare_or_snow"]


def worldcover_tile_id(lat: float, lon: float) -> str:
    """ESA WorldCover tiles are 3x3 degree, named by their SW corner, e.g.
    'N24E075' or 'S48W012' — 2-digit zero-padded latitude, 3-digit
    zero-padded longitude (see https://esa-worldcover.org tile-naming docs)."""
    lat_sw = int(np.floor(lat / 3) * 3)
    lon_sw = int(np.floor(lon / 3) * 3)
    ns = "N" if lat_sw >= 0 else "S"
    ew = "E" if lon_sw >= 0 else "W"
    return f"{ns}{abs(lat_sw):02d}{ew}{abs(lon_sw):03d}"


class LandCoverClient:
    """Reads ESA WorldCover 10m v200 land-cover class fractions for an AOI
    directly off the public AWS COG via HTTP range requests (GDAL's
    /vsicurl/ driver) — no download of the ~50-200MB tile, no key.

    Known limitation: if the AOI bbox straddles more than one 3x3-degree
    WorldCover tile, only the tile containing the bbox center is read
    (edge pixels near a tile boundary would be wrong/missing rather than
    stitched from the neighbor tile). Fine for AOIs well inside a tile;
    revisit with real multi-tile mosaicking if AOIs routinely straddle
    boundaries.
    """

    def __init__(self, base_url: str = WORLDCOVER_BASE_URL):
        self.base_url = base_url

    def fetch_lulc_patch(self, bbox: tuple[float, float, float, float], target_size: int) -> np.ndarray:
        """Returns (4, target_size, target_size) class-group fraction
        arrays in `LULC_GROUP_ORDER`, each in [0, 1]. Fractions are
        approximated by linearly interpolating each class's binary mask
        down to `target_size` (a reasonable proxy for local class fraction,
        not a true block-average) — see module/class docstrings for why
        that trade-off is acceptable here."""
        import rasterio

        min_lon, min_lat, max_lon, max_lat = bbox
        center_lat, center_lon = (min_lat + max_lat) / 2, (min_lon + max_lon) / 2
        tile = worldcover_tile_id(center_lat, center_lon)
        url = f"/vsicurl/{self.base_url}/ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"

        with rasterio.open(url) as src:
            return read_lulc_fractions(src, bbox, target_size)


def read_lulc_fractions(src, bbox: tuple[float, float, float, float], target_size: int) -> np.ndarray:
    """Given an already-open rasterio dataset (real S3 COG, or an in-memory
    dataset in tests), read the AOI window and compute class-group
    fractions. Split out from `LandCoverClient.fetch_lulc_patch` so this
    logic is testable without network access — see tests/test_terrain_sources.py."""
    import rasterio
    from rasterio.windows import from_bounds
    from scipy.ndimage import zoom as _zoom

    min_lon, min_lat, max_lon, max_lat = bbox
    window = from_bounds(min_lon, min_lat, max_lon, max_lat, transform=src.transform)
    # Cap the native-resolution read so a large AOI doesn't pull an
    # unbounded number of 10m pixels over the network.
    max_native_px = 1024
    out_shape = (
        min(max_native_px, max(int(round(window.height)), 1)),
        min(max_native_px, max(int(round(window.width)), 1)),
    )
    raw = src.read(1, window=window, out_shape=out_shape, resampling=rasterio.enums.Resampling.nearest)

    fractions = np.zeros((len(LULC_GROUP_ORDER), target_size, target_size), dtype=np.float32)
    scale = (target_size / raw.shape[0], target_size / raw.shape[1]) if raw.size else (1.0, 1.0)
    for g_idx, group in enumerate(LULC_GROUP_ORDER):
        mask = np.isin(raw, _LULC_GROUPS[group]).astype(np.float32)
        resized = _zoom(mask, scale, order=1) if mask.size else np.zeros((target_size, target_size))
        fractions[g_idx] = resized[:target_size, :target_size]

    return fractions
