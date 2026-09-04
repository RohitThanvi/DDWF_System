"""
Satellite terrain fusion (DEM + LULC + LST) for a given AOI, plus the
zero-shot implicit-field path for coordinates with no cached raster tile.
"""
from __future__ import annotations

import numpy as np
import torch

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.siren import TerrainSIREN

log = get_logger(__name__)


class TerrainFusionService:
    _instance: "TerrainFusionService | None" = None

    def __init__(self):
        self.settings = get_settings()
        self.device = torch.device(self.settings.device)
        self.siren = TerrainSIREN().to(self.device).eval()

    @classmethod
    def instance(cls) -> "TerrainFusionService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def fetch_raster_patch(self, bbox: tuple[float, float, float, float], resolution_m: int = 100) -> np.ndarray:
        """Fetch DEM/LULC/LST for the bbox from the terrain store
        (app/data zarr/COG readers). Returns (C=8, H, W): elevation, slope,
        aspect, LULC one-hot (reduced to a few dominant classes), LST.
        Stubbed until wired to Copernicus GLO-30 / a real LULC source."""
        h = w = 256
        return np.zeros((8, h, w), dtype=np.float32)

    def implicit_terrain_embedding(self, lat: float, lon: float, raster_feats: np.ndarray) -> np.ndarray:
        """Zero-shot path (Module 3): query the SIREN field directly for
        coordinates with no cached tile — e.g. an unmapped island — using
        only a coarse satellite-derived DEM patch (Copernicus GLO-30)."""
        lat_t = torch.tensor([lat], dtype=torch.float32, device=self.device)
        lon_t = torch.tensor([lon], dtype=torch.float32, device=self.device)
        feats_t = torch.tensor(raster_feats, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            emb = self.siren(lat_t, lon_t, feats_t)
        return emb.squeeze(0).cpu().numpy()
