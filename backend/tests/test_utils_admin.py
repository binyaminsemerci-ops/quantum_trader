import os

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from backend.utils import admin_auth
from backend.utils.admin_events import AdminEvent


def _make_request() -> Request:
    scope = {"type": "http", "method": "GET", "path": "/"}
    return Request(scope)


def test_enforce_admin_token_passes_when_guard_disabled(monkeypatch):
    monkeypatch.delenv("QT_ADMIN_TOKEN", raising=False)
    request = _make_request()
    assert admin_auth.enforce_admin_token(request, None, None) is None


def test_enforce_admin_token_missing(monkeypatch):
    monkeypatch.setenv("QT_ADMIN_TOKEN", "super-secret")
    request = _make_request()
    captured = []
    monkeypatch.setattr(admin_auth, "record_admin_event", lambda event, **kwargs: captured.append(event))

    with pytest.raises(HTTPException) as exc:
        admin_auth.enforce_admin_token(request, None, "super-secret")

    assert exc.value.status_code == 401
    assert captured == [AdminEvent.ADMIN_AUTH_MISSING]


def test_enforce_admin_token_invalid(monkeypatch):
    monkeypatch.setenv("QT_ADMIN_TOKEN", "super-secret")
    request = _make_request()
    captured = []
    monkeypatch.setattr(admin_auth, "record_admin_event", lambda event, **kwargs: captured.append(event))

    with pytest.raises(HTTPException) as exc:
        admin_auth.enforce_admin_token(request, "bad-token", "super-secret")

    assert exc.value.status_code == 403
    assert captured == [AdminEvent.ADMIN_AUTH_INVALID]


def test_enforce_admin_token_success(monkeypatch):
    monkeypatch.setenv("QT_ADMIN_TOKEN", "super-secret")
    request = _make_request()
    captured = []
    monkeypatch.setattr(admin_auth, "record_admin_event", lambda *args, **kwargs: captured.append(args[0]))

    token = admin_auth.enforce_admin_token(request, "super-secret", "super-secret")

    assert token == "super-secret"
    assert captured == []


@pytest.mark.asyncio
async def test_require_admin_token_accepts_valid_token(monkeypatch):
    monkeypatch.setenv("QT_ADMIN_TOKEN", "super-secret")
    request = _make_request()

    token = await admin_auth.require_admin_token(request, provided_token="super-secret", expected_token="super-secret")

    assert token == "super-secret"


@pytest.mark.asyncio
async def test_require_admin_token_raises_for_missing(monkeypatch):
    monkeypatch.setenv("QT_ADMIN_TOKEN", "super-secret")
    request = _make_request()
    captured = []
    monkeypatch.setattr(admin_auth, "record_admin_event", lambda event, **kwargs: captured.append(event))

    with pytest.raises(HTTPException) as exc:
        await admin_auth.require_admin_token(request, provided_token=None, expected_token="super-secret")

    assert exc.value.status_code == 401
    assert captured == [AdminEvent.ADMIN_AUTH_MISSING]
