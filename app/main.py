"""
DDWF System — FastAPI entrypoint.

Run locally:      uvicorn app.main:app --reload
Run in container:  see Dockerfile / docker-compose.yml

Vayu (or any other caller) integrates against:
  POST /v1/forecast          — full p10/p50/p90 timeseries
  POST /v1/vayu/weather-layer — flattened map-layer response for Vayu's frontend
  GET  /health                — liveness/readiness
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from app.api.routes import forecast, health, vayu_integration
from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger

configure_logging()
log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("ddwf.startup", env=get_settings().env)
    yield
    log.info("ddwf.shutdown")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="DDWF System",
        description="Global 30-Day Deterministic/Diffusion Weather Forecast + Satellite AOI Downscaling API",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.env != "production" else [],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(forecast.router)
    app.include_router(vayu_integration.router)

    Instrumentator().instrument(app).expose(app, endpoint="/metrics")

    return app


app = create_app()
