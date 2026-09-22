"""
Alternative coarse/target-grid source: ERA5 reanalysis streamed directly
from Google's public ARCO-ERA5 Zarr store, instead of Open-Meteo's
point-based historical archive API.

Why this exists: build_aoi_pairs_manifest.py's two weather-grid fetches
(coarse + target, both via archive-api.open-meteo.com) are by far the
highest-request-volume part of building a manifest -- a single AOI/day
is already a dozen-plus chunked point-API requests once you add both
grid densities together, and a --world, multi-year run multiplies that
by however many AOIs you're building. That volume, not the once-per-AOI
terrain fetches, is what actually trips Open-Meteo's rate limiting on a
large run.

ARCO-ERA5 (gs://gcp-public-data-arco-era5) is Google's free, public,
key-less mirror of ECMWF's ERA5 reanalysis -- the same underlying dataset
Open-Meteo's own historical archive is itself built from, just accessed
by opening a Zarr store directly and slicing it, streamed lazily over
HTTP rather than downloaded in bulk or fetched through a rate-limited
per-point REST API. Global, 0.25 degree (~28km) resolution, hourly,
1940-present, no login or API key.

--- Real caveats worth reading before trusting this at scale ---

1. Resolution: ERA5's native grid (~28km) is coarser than what
   OpenMeteoClient/_grid_points effectively samples for a small AOI's
   fine_grid (e.g. a 0.5-degree AOI at fine_grid=32 is ~0.016-degree
   spacing). Several adjacent fine-grid points will land on the same
   underlying ERA5 cell for a small AOI -- an intrinsic ERA5-resolution
   limit, not a bug here. Open-Meteo's own historical archive is itself
   built by interpolating this same ERA5 grid server-side, so it isn't
   actually higher-resolution "real" information either for this
   variable set -- this module just does the equivalent interpolation
   (nearest-neighbor) client-side instead of relying on Open-Meteo to
   do it server-side.

2. Accumulated-field unit conversion is a best-effort assumption I
   could not verify against the live store from this environment (no
   network egress to Google Cloud Storage from this project's sandbox --
   see the repo's network-allowlist notes). `total_precipitation` and
   `surface_solar_radiation_downwards` are ERA5 *accumulated* fields;
   this module assumes ARCO-ERA5's analysis-ready ("ar") tier delivers
   them as per-hour increments (the convention most ML-training
   pipelines built on this exact dataset, e.g. GraphCast's, rely on --
   ARCO-ERA5's own docs describe this tier as "oriented towards common
   research & ML workflows" and a superset of GraphCast's training
   data). If that assumption is wrong for this store, the derived
   precipitation/radiation values would be off by a constant factor
   (e.g. actually since-00Z-cumulative rather than hourly). **Please
   spot-check**: pull one grid point's value here and compare it
   against Open-Meteo's own historical archive for the same
   lat/lon/hour before trusting a large run's precipitation/radiation
   channels.

3. Real network dependency on Google Cloud Storage, separate from
   Open-Meteo -- run manifest building with `--source era5` from an
   environment with normal internet access (this repo's own CI/sandbox
   environments may not have GCS egress).

Not exercised against the live Zarr store anywhere in this repo's test
suite (no GCS network access in CI) -- tests/test_era5_zarr.py covers the
variable-derivation math (Magnus-formula RH, u/v -> speed+direction) and
grid-sampling logic against small synthetic in-memory xarray Datasets
instead, which is real coverage of the parts that don't require a live
Zarr store, just not of the store access itself.
"""
from __future__ import annotations

from datetime import date, datetime, timezone

import numpy as np

ARCO_ERA5_ZARR_PATH = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

# ARCO-ERA5's raw field names for everything OPEN_METEO_VARIABLES needs,
# either directly or as an input to a derived quantity below.
RAW_FIELDS = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "surface_pressure",
    "total_precipitation",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "total_cloud_cover",
    "surface_solar_radiation_downwards",
]

_dataset_cache: dict[str, object] = {}


def _open_dataset():
    """Opens (and caches for the life of the process -- repeated
    manifest-build calls should share one handle rather than re-opening
    the store's metadata per AOI) the ARCO-ERA5 Zarr store, anonymous
    read, no auth."""
    if "ds" not in _dataset_cache:
        import xarray as xr

        _dataset_cache["ds"] = xr.open_zarr(
            ARCO_ERA5_ZARR_PATH, chunks=None, storage_options={"token": "anon"}
        )
    return _dataset_cache["ds"]


def relative_humidity_from_dewpoint(t_celsius: np.ndarray, td_celsius: np.ndarray) -> np.ndarray:
    """Relative humidity (%) from 2m air temperature + dewpoint via the
    Alduchov & Eskridge (1996) Magnus-form approximation of saturation
    vapor pressure -- accurate to within ~0.4% over -40..50C, well within
    that range for essentially all real 2m air temperatures.
    RH = 100 * es(Td) / es(T), es(x) = exp(a*x / (b+x))."""
    a, b = 17.625, 243.04
    es_t = np.exp((a * t_celsius) / (b + t_celsius))
    es_td = np.exp((a * td_celsius) / (b + td_celsius))
    return np.clip(100.0 * es_td / es_t, 0.0, 100.0)


def wind_speed_direction(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(speed_km_h, direction_deg) from ERA5's u/v wind components (m/s,
    eastward/northward respectively). Direction uses the standard
    meteorological "from" convention (0=N, 90=E, 180=S, 270=W -- the
    compass direction the wind is blowing FROM, matching MetPy's
    wind_direction), not the mathematical vector angle. See
    tests/test_era5_zarr.py for worked sanity-check cases (a southward
    flow, i.e. wind FROM the north, must map to 0 degrees; a westward
    flow, wind FROM the east, must map to 90)."""
    speed_ms = np.sqrt(u**2 + v**2)
    speed_kmh = speed_ms * 3.6
    direction_deg = (90.0 - np.degrees(np.arctan2(-v, -u))) % 360.0
    return speed_kmh, direction_deg


def _derive_variables(raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """raw: {ARCO field name: array in ERA5's native units}. Returns
    {OPEN_METEO_VARIABLES name: array in Open-Meteo's units} -- see
    data/open_meteo_variable_stats.json's mean/std for the unit each
    Open-Meteo variable is implicitly assumed to be in elsewhere in this
    codebase (VariableNormalizer, the diffusion target scale, etc.)."""
    t_c = raw["2m_temperature"] - 273.15
    td_c = raw["2m_dewpoint_temperature"] - 273.15
    speed_kmh, direction_deg = wind_speed_direction(
        raw["10m_u_component_of_wind"], raw["10m_v_component_of_wind"]
    )
    return {
        "temperature_2m": t_c,
        "relative_humidity_2m": relative_humidity_from_dewpoint(t_c, td_c),
        "precipitation": raw["total_precipitation"] * 1000.0,  # m -> mm; see module docstring caveat 2
        "surface_pressure": raw["surface_pressure"] / 100.0,  # Pa -> hPa
        "wind_speed_10m": speed_kmh,
        "wind_direction_10m": direction_deg,
        "cloud_cover": raw["total_cloud_cover"] * 100.0,  # fraction -> percent
        "shortwave_radiation": raw["surface_solar_radiation_downwards"] / 3600.0,  # J/m^2 -> W/m^2; see caveat 2
    }


def sample_grid_from_dataset(ds, bbox: tuple[float, float, float, float], grid_size: int, when: datetime) -> dict[str, np.ndarray]:
    """Core sampling logic, factored out from fetch_grid so tests can pass
    a small synthetic in-memory Dataset instead of the real (2.05 PB) live
    store. Returns {OPEN_METEO_VARIABLES name: (grid_size, grid_size)
    array}, row=lat ascending / col=lon ascending -- the same convention
    _grid_points/_fetch_historical_grid use, so a manifest built this way
    is a drop-in replacement for one built from Open-Meteo."""
    min_lon, min_lat, max_lon, max_lat = bbox
    lats = np.linspace(min_lat, max_lat, grid_size)
    lons = np.linspace(min_lon, max_lon, grid_size)
    lons_0_360 = lons % 360.0  # ERA5's longitude coordinate runs 0..360, not -180..180

    ts = np.datetime64(when.replace(tzinfo=None))
    sel = ds[RAW_FIELDS].sel(time=ts, latitude=lats, longitude=lons_0_360, method="nearest")

    raw = {field: sel[field].values.astype(np.float32) for field in RAW_FIELDS}
    return _derive_variables(raw)


def fetch_grid(
    bbox: tuple[float, float, float, float],
    grid_size: int,
    day: date,
    variables: list[str],
    hour_utc: int = 12,
) -> np.ndarray:
    """Returns (n_vars, grid_size, grid_size) at `hour_utc` on `day`,
    sourced from the live ARCO-ERA5 Zarr store -- same output
    shape/semantics as
    scripts/build_aoi_pairs_manifest.py's _fetch_historical_grid, so it's
    a drop-in replacement for the Open-Meteo path there."""
    ds = _open_dataset()
    when = datetime(day.year, day.month, day.day, hour_utc, tzinfo=timezone.utc)
    by_name = sample_grid_from_dataset(ds, bbox, grid_size, when)
    out = np.zeros((len(variables), grid_size, grid_size), dtype=np.float32)
    for i, var in enumerate(variables):
        out[i] = by_name[var]
    return out
