"""
Service-to-service auth for Vayu -> DDWF calls.
Simple static API key header for v1; swap for signed JWT/mTLS later
if DDWF ends up serving more than one internal caller.
"""
from fastapi import Header, HTTPException, status

from app.core.config import get_settings


async def verify_api_key(x_api_key: str = Header(default="")) -> None:
    settings = get_settings()
    if not x_api_key or x_api_key != settings.ddwf_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key header.",
        )
