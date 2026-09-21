"""
Service-to-service auth for Vayu -> DDWF calls.
Simple static API key header for v1; swap for signed JWT/mTLS later
if DDWF ends up serving more than one internal caller.
"""
import hmac

from fastapi import Header, HTTPException, status

from app.core.config import get_settings


async def verify_api_key(x_api_key: str = Header(default="")) -> None:
    settings = get_settings()
    # hmac.compare_digest instead of `!=`: a naive string comparison
    # short-circuits on the first mismatched byte, so response time leaks
    # how many leading characters of a guessed key were correct — a real,
    # if slow, timing side-channel against a static bearer credential.
    # compare_digest runs in constant time regardless of where the first
    # difference is.
    if not x_api_key or not hmac.compare_digest(x_api_key, settings.ddwf_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key header.",
        )
