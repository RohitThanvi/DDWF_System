# ---- Base ----
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgdal-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY checkpoints ./checkpoints
# NOTE: data/ (incl. open_meteo_variable_stats.json) MUST be copied — it's
# not optional. VariableNormalizer (app/data/normalization.py) reads it at
# DownscalerService construction time, i.e. on the first /v1/forecast
# request. Without this line the image builds and /health passes fine, but
# every real forecast request 500s with FileNotFoundError the moment it's
# hit — the kind of bug that never shows up until first production traffic.
COPY data ./data

RUN mkdir -p /srv/data/cache /srv/mlruns \
    && groupadd --system ddwf && useradd --system --gid ddwf --home-dir /srv --no-create-home ddwf \
    && chown -R ddwf:ddwf /srv
USER ddwf

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# WEB_CONCURRENCY overridable per deploy target; default to 1 since the
# forecast endpoint now offloads blocking model inference to a thread pool
# (see app/api/routes/forecast.py) rather than needing multiple worker
# processes just to stay responsive under one in-flight request.
ENV WEB_CONCURRENCY=1
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers ${WEB_CONCURRENCY}"]
