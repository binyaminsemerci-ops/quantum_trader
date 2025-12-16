"""Shared helpers for enforcing the admin token on mutable endpoints."""

from __future__ import annotations

import hmac
import os
from typing import Optional

from fastapi import Depends, Header, HTTPException, Request, status

from .admin_events import AdminEvent, record_admin_event


def get_configured_admin_token() -> Optional[str]:
    """Return the configured admin token (environment sourced)."""

    token = os.getenv("QT_ADMIN_TOKEN")
    if token and token.strip():
        return token
    return None


def enforce_admin_token(
    request: Request,
    provided_token: Optional[str],
    expected_token: Optional[str],
    *,
    missing_event: AdminEvent = AdminEvent.ADMIN_AUTH_MISSING,
    invalid_event: AdminEvent = AdminEvent.ADMIN_AUTH_INVALID,
) -> Optional[str]:
    """Validate the provided admin token against the expected value.

    Raises
    ------
    fastapi.HTTPException
        When authentication fails and an admin token is configured.
    """

    if not expected_token:
        # Guard disabled for this environment; allow the request through.
        return expected_token

    if provided_token is None:
        record_admin_event(
            missing_event,
            request=request,
            success=False,
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing admin token",
            headers={"WWW-Authenticate": "Token"},
        )

    if not hmac.compare_digest(provided_token, expected_token):
        record_admin_event(
            invalid_event,
            request=request,
            success=False,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid admin token",
        )

    return provided_token


async def require_admin_token(
    request: Request,
    provided_token: Optional[str] = Header(None, alias="X-Admin-Token"),
    expected_token: Optional[str] = Depends(get_configured_admin_token),
) -> Optional[str]:
    """FastAPI dependency that enforces the admin token header."""

    return enforce_admin_token(
        request,
        provided_token,
        expected_token,
    )


__all__ = [
    "enforce_admin_token",
    "get_configured_admin_token",
    "require_admin_token",
]
