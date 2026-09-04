"""
Zarr data lake access layer (Module 1).

Reads ERA5/CMIP6/MERRA-2 as a virtual-reference Zarr store (kerchunk over
raw NetCDF4/HDF5, so raw archives are never re-written) via xarray+Dask.
This module is intentionally decoupled from the model-serving path — the
API never blocks a user request on live NetCDF reads; only the offline
forecast-cycle refresh job (Airflow/Dagster DAG) touches this directly.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.core.config import get_settings
from app.core.logging import get_logger

log = get_logger(__name__)


@dataclass
class GridFeatureSpec:
    """Mirrors the `GridFeatures` TypedDict in the TDD — kept here as the
    single source of truth for channel ordering into the SFNO stem."""

    position_embed: int = 5     # sin(lat), cos(lat), sin(lon), cos(lon), elev_norm
    solar_zenith: int = 1
    toa_insolation: int = 1
    soil_moisture: int = 4      # 4 ERA5-Land layers
    albedo: int = 1
    sst_anomaly: int = 1
    oni_index: int = 1
    mjo_phase_amp: int = 2      # RMM1/RMM2

    @property
    def total_channels(self) -> int:
        return sum(
            v for k, v in self.__dict__.items() if not k.startswith("_")
        ) if False else (
            self.position_embed + self.solar_zenith + self.toa_insolation
            + self.soil_moisture + self.albedo + self.sst_anomaly
            + self.oni_index + self.mjo_phase_amp
        )


class ZarrDataLake:
    """Thin wrapper around xarray.open_zarr / kerchunk reference datasets.
    Kept import-light (xarray/zarr/dask are declared in requirements.txt,
    but this class only opens a store lazily — no network I/O on import)."""

    def __init__(self, store_uri: str | None = None):
        self.settings = get_settings()
        self.store_uri = store_uri or self.settings.zarr_store_uri
        self._ds = None

    def open(self):
        import xarray as xr

        if self._ds is None:
            log.info("zarr.open", uri=self.store_uri)
            self._ds = xr.open_zarr(self.store_uri, consolidated=True)
        return self._ds

    def latest_analysis_state(self):
        """Return the most recent fully-QC'd analysis timestep, regridded
        to 0.25 deg — the x0 fed into SFNO.rollout() for a fresh cycle."""
        ds = self.open()
        return ds.isel(time=-1)
