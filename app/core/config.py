"""
Centralized settings, loaded from environment / .env.
Every other module reads config from here — never os.environ directly.
"""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Server ---
    env: str = "development"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"

    # --- Auth ---
    ddwf_api_key: str = "change-me-to-a-long-random-secret"

    # --- Model checkpoints ---
    # Downscaler is the model DDWF trains and serves (Option B design — see
    # docs/ARCHITECTURE.md). global_engine_checkpoint / rollout settings are
    # kept for the optional SFNO upgrade path (Option A) and unused by the
    # default request flow.
    global_engine_checkpoint: str = "./checkpoints/sfno_global_v1.pt"
    downscaler_checkpoint: str = "./checkpoints/diffusion_downscaler_v1.pt"
    device: str = "cpu"

    # --- Coarse forecast source (Option B: Open-Meteo, free/no API key) ---
    coarse_forecast_provider: str = "open-meteo"
    coarse_grid_size: int = 8          # NxN sample points tiling the AOI bbox
    max_horizon_days: int = 16         # Open-Meteo's free-tier forecast horizon

    # --- Data lake ---
    zarr_store_uri: str = "s3://ddwf-data-lake/era5.zarr"
    dem_store_uri: str = "s3://ddwf-data-lake/copernicus_glo30.zarr"
    cache_dir: str = "./data/cache"
    trajectory_cache_ttl_hours: int = 6

    # --- Redis ---
    redis_url: str = "redis://localhost:6379/0"

    # --- Rollout / ensemble (SFNO optional-upgrade-path settings; unused
    # by the default Open-Meteo-backed request flow) ---
    rollout_hop_hours: int = 6
    rollout_steps: int = 120         # 120 * 6h = 720h = 30 days
    ic_ensemble_members: int = 16
    diffusion_sampling_steps: int = 16

    # --- MLflow ---
    mlflow_tracking_uri: str = "./mlruns"

    @property
    def cache_path(self) -> Path:
        p = Path(self.cache_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p


@lru_cache
def get_settings() -> Settings:
    return Settings()
