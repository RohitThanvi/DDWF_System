"""
DDWF System — FastAPI entrypoint.

Run locally:      uvicorn app.main:app --reload
Run in container:  see Dockerfile / docker-compose.yml

Vayu (or any other caller) integrates against:
  POST /v1/forecast          — full p10/p50/p90 timeseries
  POST /v1/vayu/weather-layer — flattened map-layer response for Vayu's frontend
  GET  /health                — liveness/readiness
"""
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator

from app.api.routes import forecast, health, vayu_integration
from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.core.rate_limit import RateLimitMiddleware
from app.core.security import verify_api_key

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
    app.add_middleware(RateLimitMiddleware, requests_per_minute=settings.rate_limit_per_minute)

    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        # Every log line in the forecast flow is already tagged with a
        # request_id generated inside the route handler -- this surfaces
        # the same idea at the HTTP layer so a caller (Vayu) can correlate
        # "my request X got error Y" with our server-side logs without us
        # having to thread the ID through every log call by hand. Stored on
        # request.state (not just added to the response after the fact) so
        # the exception handler below can attach the *same* ID to an error
        # response too -- BaseHTTPMiddleware re-raises rather than handing
        # back a response when the inner call raises, so this middleware
        # never gets a response object to attach a header to in that case.
        request.state.request_id = str(uuid.uuid4())
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        # Without this, an unexpected exception anywhere in the request
        # path (a bug in downscaler.downscale, a numpy shape error, an
        # unanticipated upstream response shape) propagates as FastAPI's
        # default plain-text 500 -- fine for a human staring at a terminal,
        # not fine for a caller (Vayu) trying to parse a JSON error body,
        # and prone to leaking implementation details (file paths, library
        # names) into the response by default. Log the full exception
        # server-side; return a generic, stable-shaped body to the caller.
        log.error("unhandled_exception", path=request.url.path, error=str(exc), exc_info=exc)
        request_id = getattr(request.state, "request_id", None)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error. Contact the DDWF maintainer with the X-Request-ID header if this persists."},
            headers={"X-Request-ID": request_id} if request_id else None,
        )

    app.include_router(health.router)
    app.include_router(forecast.router)
    app.include_router(vayu_integration.router)

    # /metrics exposes internal request-rate/latency detail per route --
    # not secret-sensitive, but not meant to be public either; require the
    # same API key as every other real endpoint rather than leaving it as
    # the one unauthenticated route on the service.
    Instrumentator().instrument(app).expose(
        app, endpoint="/metrics", dependencies=[Depends(verify_api_key)]
    )

    return app


app = create_app()
