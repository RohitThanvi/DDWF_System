from fastapi import APIRouter

from app.api.schemas import HealthResponse
from app.core.config import get_settings

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness/readiness probe. Kept dependency-free (no model load
    forced here) so orchestrators can hit it fast; use /health?deep=true
    semantics later if you want an actual forward-pass smoke test."""
    settings = get_settings()

    global_loaded = False
    downscaler_loaded = False
    try:
        from pathlib import Path
        global_loaded = Path(settings.global_engine_checkpoint).exists()
        downscaler_loaded = Path(settings.downscaler_checkpoint).exists()
    except Exception:
        pass

    return HealthResponse(
        status="ok",
        global_engine_loaded=global_loaded,
        downscaler_loaded=downscaler_loaded,
        device=settings.device,
    )
