"""
Authentication & rate-limiting dependencies for FastAPI.

Supports two identity modes:
  1. Clerk-issued JWT (Bearer token) → verified with the Clerk JWKS endpoint
  2. Guest ID header (X-Guest-Id: guest_<uuid>) → no cryptographic check, rate-limited

Environment variables required:
  CLERK_JWKS_URL  – https://<your-clerk-frontend-api>/.well-known/jwks.json
"""

import os
import logging

import jwt
from jwt import PyJWKClient
from fastapi import Header, HTTPException, Depends

from core.database.db_user_threads import check_and_increment_guest_usage, get_guest_usage_today

logger = logging.getLogger(__name__)

CLERK_JWKS_URL: str = os.getenv("CLERK_JWKS_URL", "")
GUEST_DAILY_LIMIT: int = int(os.getenv("GUEST_DAILY_LIMIT", "10"))

# Lazily initialised – avoids network calls at import time
_jwks_client: PyJWKClient | None = None


def _get_jwks_client() -> PyJWKClient:
    global _jwks_client
    if _jwks_client is None:
        if not CLERK_JWKS_URL:
            raise HTTPException(
                status_code=500,
                detail="CLERK_JWKS_URL is not configured on the server.",
            )
        _jwks_client = PyJWKClient(CLERK_JWKS_URL, cache_keys=True)
    return _jwks_client


def _verify_clerk_jwt(token: str) -> str:
    """Decode and verify a Clerk JWT. Returns the Clerk user-id (sub claim)."""
    try:
        client = _get_jwks_client()
        signing_key = client.get_signing_key_from_jwt(token)
        payload = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            options={"verify_aud": False},
        )
        return payload["sub"]
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning(f"[auth] JWT verification failed: {exc}")
        raise HTTPException(status_code=401, detail="Invalid or expired token.")


def get_current_user(
    authorization: str = Header(default=None),
    x_guest_id: str = Header(default=None),
) -> str:
    """
    FastAPI dependency that resolves the caller to either a Clerk user-id or a
    guest_<uuid> string.

    Priority:
      1. Authorization: Bearer <token>  →  verified Clerk user
      2. X-Guest-Id: guest_<uuid>       →  unverified guest (rate-limited elsewhere)
    """
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ", 1)[1]
        return _verify_clerk_jwt(token)

    if x_guest_id and x_guest_id.startswith("guest_"):
        return x_guest_id

    raise HTTPException(status_code=401, detail="Unauthorized.")


def get_optional_user(
    authorization: str = Header(default=None),
    x_guest_id: str = Header(default=None),
) -> str | None:
    """
    Like get_current_user but returns None instead of raising 401.
    Use on endpoints that work for anonymous callers but can also be identity-aware
    (e.g. /get_thread_id — no auth required, but we bind the thread if auth is present).
    """
    try:
        return get_current_user(authorization=authorization, x_guest_id=x_guest_id)
    except HTTPException:
        return None


def get_current_user_with_rate_limit(
    user_id: str = Depends(get_current_user),
) -> str:
    """
    Same as get_current_user but additionally enforces a daily cap for guests.
    Increments the counter. Use on endpoints that consume AI compute (e.g. /chat).
    """
    if user_id.startswith("guest_"):
        count = check_and_increment_guest_usage(user_id)
        if count > GUEST_DAILY_LIMIT:
            raise HTTPException(
                status_code=429,
                detail="Daily limit reached for guest users. Please sign in to continue.",
            )
    return user_id


def get_current_user_check_rate_limit(
    user_id: str = Depends(get_current_user),
) -> str:
    """
    Read-only quota check — does NOT increment the counter.
    Blocks with 429 if the guest has already exhausted today's quota,
    but does not consume a request itself.
    Use on pre-flight endpoints (e.g. /research_helper) so guests aren't
    allowed to keep prepping once their canvas quota is gone.
    """
    if user_id.startswith("guest_"):
        count = get_guest_usage_today(user_id)
        if count >= GUEST_DAILY_LIMIT:
            raise HTTPException(
                status_code=429,
                detail="Daily limit reached for guest users. Please sign in to continue.",
            )
    return user_id
