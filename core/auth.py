"""
Authentication & rate-limiting dependencies for FastAPI.

Supports two identity modes:
  1. Clerk-issued JWT (Bearer token) → verified with the Clerk JWKS endpoint
  2. Guest ID header (X-Guest-Id: guest_<uuid>) → no cryptographic check, rate-limited

Environment variables required:
  CLERK_JWKS_URL  – https://<your-clerk-frontend-api>/.well-known/jwks.json
"""

import hmac
import os
import re
import logging

import jwt
from jwt import PyJWKClient
from fastapi import Header, HTTPException, Depends

logger = logging.getLogger(__name__)

CLERK_JWKS_URL: str = os.getenv("CLERK_JWKS_URL", "")

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


def resolve_user(*, bearer_token: str | None, guest_id: str | None) -> str:
    """Shared identity resolution behind `get_current_user`.

    Pulled out so the voice WebSocket (core/routers/voice.py) can run the same
    Clerk-JWT-or-guest check on values it reads from query params — a browser
    WebSocket handshake can't carry a custom Authorization header the way a
    normal fetch() can, so token/guest id travel in the URL there instead, but
    the resolution logic itself must stay identical to /chat's.

    Priority:
      1. bearer_token   →  verified Clerk user
      2. guest_<uuid>   →  unverified guest (rate-limited elsewhere)
    """
    if bearer_token:
        return _verify_clerk_jwt(bearer_token)

    if guest_id and guest_id.startswith("guest_"):
        return guest_id

    raise HTTPException(status_code=401, detail="Unauthorized.")


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
    token = authorization.split(" ", 1)[1] if authorization and authorization.startswith("Bearer ") else None
    return resolve_user(bearer_token=token, guest_id=x_guest_id)


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


# ── training-data collector ────────────────────────────────────────────────
#
# The collector (core/routers/collector.py) is operated by a small number of
# trusted annotators through the password-protected /collect page of the frontend.
# That page's server holds COLLECTOR_API_KEY; browsers never see it. Everything
# the key unlocks lives under /api/collector, and it does NOT widen
# `resolve_user`: a Clerk-verified user or a guest can never reach those routes,
# and the key cannot be used as a user identity on /chat.
#
# Unset COLLECTOR_API_KEY disables the feature outright (503), so a deployment
# that never opted in exposes nothing.

COLLECTOR_USER_PREFIX = "collector_"
_COLLECTOR_ID_RE = re.compile(r"^[a-z0-9_-]{1,32}$")


def get_collector(
    x_collector_key: str = Header(default=None),
    x_collector_id: str = Header(default=None),
) -> str:
    """FastAPI dependency: verify the collector key, return `collector_<annotator>`.

    The annotator id only partitions rows by who collected them; it is not a
    credential. The returned id is what owns the collector's threads and
    `sft_examples.user_id`, so annotators cannot touch each other's threads.
    """
    expected = os.getenv("COLLECTOR_API_KEY", "")
    if not expected:
        raise HTTPException(status_code=503, detail="Collector is not enabled.")
    if not x_collector_key or not hmac.compare_digest(
        x_collector_key.encode(), expected.encode()
    ):
        raise HTTPException(status_code=401, detail="Unauthorized.")
    annotator = (x_collector_id or "").strip().lower()
    if not _COLLECTOR_ID_RE.match(annotator):
        raise HTTPException(status_code=400, detail="X-Collector-Id must match [a-z0-9_-]{1,32}.")
    return f"{COLLECTOR_USER_PREFIX}{annotator}"
