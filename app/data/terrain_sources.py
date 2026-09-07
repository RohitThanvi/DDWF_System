"""
Real, free, key-less terrain data sources for the AOI downscaler.

DEM (elevation/slope/aspect): Open-Meteo's Elevation API
(https://open-meteo.com/en/docs/elevation-api), backed by Copernicus
DEM GLO-90 (90m, global, free license). Same provider as the coarse
forecast source (app/data/external_forecast.py) — no extra account, no key.

LULC (land use/land cover): ESA WorldCover 10m v200, read directly off its
public AWS COG bucket via HTTP range requests (no download, no key) — see
`LandCoverClient` below.

LST (land surface temperature): MODIS MOD11A2, fetched via ORNL DAAC's free
REST subset service (https://modis.ornl.gov) — also no key, no account.
This one initially looked like it would need a NASA Earthdata login (the
usual path for MODIS data), but ORNL DAAC's subsetting service turned out
to be genuinely key-less; see `LSTClient` below. All three terrain sources
are real as a result — there is no remaining stub channel.
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
# index 7: LST (real, MODIS MOD11A2 via ORNL DAAC — see LSTClient below;
#          8-day composite, so treat as a slow surface-heating signal, not
#          live temperature)
REAL_DEM_CHANNELS = 3
REAL_LULC_CHANNELS = 4
REAL_LST_CHANNELS = 1
TOTAL_TERRAIN_CHANNELS = REAL_DEM_CHANNELS + REAL_LULC_CHANNELS + REAL_LST_CHANNELS

# Backward-compatible aliases (old names implied "stub"; both are real now)
STUB_LULC_CHANNELS = REAL_LULC_CHANNELS
STUB_LST_CHANNELS = REAL_LST_CHANNELS


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


# --- LST (land surface temperature): MODIS via ORNL DAAC's free,
# key-less REST subset service (https://modis.ornl.gov). Confirmed
# key-less and confirmed response schema (nrows/ncols/xllcorner/yllcorner
# + per-band 'data' arrays, reshape to (nrows, ncols)) against ORNL DAAC's
# own published documentation and tutorials before writing this client —
# this is a materially different, and pleasantly simpler, access pattern
# than NASA Earthdata/AppEEARS, which is why LST turned out not to need an
# account after all.
MODIS_BASE_URL = "https://modis.ornl.gov/rst/api/v1"
MODIS_LST_PRODUCT = "MOD11A2"      # Terra, 8-day composite, 1km
MODIS_LST_BAND = "LST_Day_1km"
MODIS_LST_SCALE = 0.02             # raw DN -> Kelvin
MODIS_LST_FILL = 0                 # raw DN used for invalid/no-data pixels
MODIS_MAX_HALF_WINDOW_KM = 100     # service-enforced cap on kmAboveBelow/kmLeftRight


class LSTClient:
    """Fetches the most recent available MOD11A2 daytime LST composite for
    an AOI from ORNL DAAC's free subset service — no key, no account.
    8-day compositing means this is never "right now" temperature; treat it
    as a slowly-varying land-surface-heating conditioning signal, not a
    live temperature reading (the coarse forecast's temperature_2m variable
    is the live signal; this is a complementary, surface-specific one)."""

    def __init__(self, base_url: str = MODIS_BASE_URL, timeout_s: float = 20.0):
        self.base_url = base_url
        self.timeout_s = timeout_s

    async def _latest_modis_date(self, lat: float, lon: float, client) -> str | None:
        resp = await client.get(
            f"{self.base_url}/{MODIS_LST_PRODUCT}/dates", params={"latitude": lat, "longitude": lon}
        )
        resp.raise_for_status()
        dates = resp.json().get("dates", [])
        if not dates:
            return None
        return dates[-1]["modis_date"]  # API returns dates in chronological order

    async def fetch_lst_patch(self, bbox: tuple[float, float, float, float], target_size: int) -> np.ndarray:
        """Returns (1, target_size, target_size) LST in Celsius, resampled
        from the native ~1km MODIS grid. Falls back to zeros (with a
        logged warning) if the service has no composite for this AOI, e.g.
        persistent cloud cover at the latest date."""
        import httpx
        from scipy.ndimage import zoom as _zoom

        min_lon, min_lat, max_lon, max_lat = bbox
        center_lat, center_lon = (min_lat + max_lat) / 2, (min_lon + max_lon) / 2
        width_km = abs(max_lon - min_lon) * 111 * np.cos(np.radians(center_lat))
        height_km = abs(max_lat - min_lat) * 111
        km_lr = int(np.clip(round(width_km / 2), 1, MODIS_MAX_HALF_WINDOW_KM))
        km_ab = int(np.clip(round(height_km / 2), 1, MODIS_MAX_HALF_WINDOW_KM))

        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            modis_date = await self._latest_modis_date(center_lat, center_lon, client)
            if modis_date is None:
                log.warning("terrain_fusion.lst_no_dates_available", lat=center_lat, lon=center_lon)
                return np.zeros((1, target_size, target_size), dtype=np.float32)

            resp = await client.get(
                f"{self.base_url}/{MODIS_LST_PRODUCT}/subset",
                params={
                    "latitude": center_lat, "longitude": center_lon, "band": MODIS_LST_BAND,
                    "startDate": modis_date, "endDate": modis_date,
                    "kmAboveBelow": km_ab, "kmLeftRight": km_lr,
                },
            )
            resp.raise_for_status()
            payload = resp.json()

        subset = payload.get("subset", [])
        nrows, ncols = payload.get("nrows"), payload.get("ncols")
        if not subset or not nrows or not ncols:
            log.warning("terrain_fusion.lst_empty_subset", lat=center_lat, lon=center_lon)
            return np.zeros((1, target_size, target_size), dtype=np.float32)

        raw = np.array(subset[0]["data"], dtype=np.float32).reshape(nrows, ncols)
        valid = raw != MODIS_LST_FILL
        lst_celsius = np.where(valid, raw * MODIS_LST_SCALE - 273.15, np.nan)
        fill_value = float(np.nanmean(lst_celsius)) if np.any(valid) else 0.0
        lst_celsius = np.nan_to_num(lst_celsius, nan=fill_value)

        scale = (target_size / nrows, target_size / ncols)
        resized = _zoom(lst_celsius, scale, order=1)
        return resized[:target_size, :target_size][np.newaxis, :, :].astype(np.float32)
