"""
Satellite terrain fusion (DEM + LULC + LST) for a given AOI, plus the
zero-shot implicit-field path for coordinates with no cached raster tile.

DEM (elevation/slope/aspect) is real, fetched live from Open-Meteo's free
Elevation API (Copernicus GLO-90) — see app/data/terrain_sources.py. LULC
and LST channels are explicit, documented stubs (zeros) until a real
free/key-less source is wired in; see that module's docstring for why.
"""
from __future__ import annotations

import asyncio

import numpy as np
import torch
from scipy.ndimage import zoom

from app.core.config import get_settings
from app.core.logging import get_logger
from app.data.terrain_sources import (
    STUB_LST_CHANNELS,
    ElevationClient,
    LandCoverClient,
    slope_aspect_from_elevation,
)
from app.models.siren import TerrainSIREN

log = get_logger(__name__)


class TerrainFusionService:
    _instance: "TerrainFusionService | None" = None

    def __init__(self):
        self.settings = get_settings()
        self.device = torch.device(self.settings.device)
        self.siren = TerrainSIREN().to(self.device).eval()
        self.elevation_client = ElevationClient()
        self.lulc_client = LandCoverClient()

    @classmethod
    def instance(cls) -> "TerrainFusionService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def fetch_raster_patch(
        self, bbox: tuple[float, float, float, float], resolution_m: int = 100, target_size: int = 256
    ) -> np.ndarray:
        """Returns (C=8, target_size, target_size): real elevation/slope/
        aspect from Open-Meteo/Copernicus GLO-90 and real land-cover class
        fractions (vegetation/built-up/water/bare-or-snow) from ESA
        WorldCover 10m, both upsampled/resampled to `target_size` (native
        DEM/LULC resolution runs out well before typical AOI target
        resolutions of 100m-1km, which is exactly why the diffusion head
        exists: it learns to hallucinate plausible sub-native-resolution
        structure rather than just upsampling terrain linearly). The LST
        channel is zeros (see app/data/terrain_sources.py for why)."""
        min_lon, min_lat, max_lon, max_lat = bbox
        approx_width_m = abs(max_lon - min_lon) * 111_000 * np.cos(np.radians((min_lat + max_lat) / 2))
        query_grid = min(32, max(4, int(approx_width_m / max(resolution_m, 90))))

        try:
            elevation = await self.elevation_client.fetch_elevation_grid(bbox, grid_size=query_grid)
        except Exception as exc:
            log.warning("terrain_fusion.elevation_fetch_failed", error=str(exc))
            elevation = np.zeros((query_grid, query_grid), dtype=np.float32)

        cell_size_m = approx_width_m / max(query_grid - 1, 1)
        slope, aspect = slope_aspect_from_elevation(elevation, cell_size_m=cell_size_m)

        scale = target_size / query_grid
        elevation_hi = zoom(elevation, scale, order=1)
        slope_hi = zoom(slope, scale, order=1)
        aspect_hi = zoom(aspect, scale, order=1)

        elevation_hi, slope_hi, aspect_hi = (
            a[:target_size, :target_size] for a in (elevation_hi, slope_hi, aspect_hi)
        )

        try:
            lulc_fractions = await asyncio.to_thread(self.lulc_client.fetch_lulc_patch, bbox, target_size)
        except Exception as exc:
            log.warning("terrain_fusion.lulc_fetch_failed", error=str(exc))
            lulc_fractions = np.zeros((4, target_size, target_size), dtype=np.float32)

        lst_stub = np.zeros((STUB_LST_CHANNELS, target_size, target_size), dtype=np.float32)

        dem_stack = np.stack([elevation_hi, slope_hi, aspect_hi]).astype(np.float32)
        return np.concatenate([dem_stack, lulc_fractions, lst_stub], axis=0)

    def implicit_terrain_embedding(self, lat: float, lon: float, raster_feats: np.ndarray) -> np.ndarray:
        """Zero-shot path (Module 3): query the SIREN field directly for
        coordinates with no cached tile — e.g. an unmapped island — using
        only the real elevation/slope/aspect patch fetched above."""
        lat_t = torch.tensor([lat], dtype=torch.float32, device=self.device)
        lon_t = torch.tensor([lon], dtype=torch.float32, device=self.device)
        feats_t = torch.tensor(raster_feats, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            emb = self.siren(lat_t, lon_t, feats_t)
        return emb.squeeze(0).cpu().numpy()
