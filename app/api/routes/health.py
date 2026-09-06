from pathlib import Path

from fastapi import APIRouter

from app.api.schemas import HealthResponse
from app.core.config import get_settings

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness/readiness probe. Kept dependency-free (no model load
    forced here) so orchestrators can hit it fast."""
    settings = get_settings()

    downscaler_loaded = False
    sfno_present = False
    try:
        downscaler_loaded = Path(settings.downscaler_checkpoint).exists()
        sfno_present = Path(settings.global_engine_checkpoint).exists()
    except Exception:
        pass

    return HealthResponse(
        status="ok",
        downscaler_loaded=downscaler_loaded,
        coarse_forecast_provider=settings.coarse_forecast_provider,
        sfno_checkpoint_present=sfno_present,
        device=settings.device,
    )
