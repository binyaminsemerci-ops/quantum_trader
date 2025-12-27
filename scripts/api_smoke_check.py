"""Simple smoke check for API exposure endpoints.

Run this script after deploying the backend to ensure the `/health`,
`/health/scheduler`, and `/risk` endpoints surface the expected fields.

Usage (PowerShell):

```
$env:QT_API_BASE="http://localhost:8000"
$env:QT_ADMIN_TOKEN="test-admin-token"  # optional if risk endpoint requires it
python scripts/api_smoke_check.py
```
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Any, Dict

import httpx


async def _fetch(client: httpx.AsyncClient, method: str, url: str, *, headers: Dict[str, str] | None = None) -> Dict[str, Any]:
    response = await client.request(method, url, headers=headers)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise SystemExit(f"Expected JSON object from {url}, got {type(payload)!r}")
    return payload


async def _run() -> None:
    base_url = os.getenv("QT_API_BASE", "http://localhost:8000").rstrip("/")
    admin_token = os.getenv("QT_ADMIN_TOKEN")

    headers = {"X-Admin-Token": admin_token} if admin_token else None

    timeout = httpx.Timeout(10.0, connect=5.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        health = await _fetch(client, "GET", f"{base_url}/health", headers=headers)
        scheduler = await _fetch(client, "GET", f"{base_url}/health/scheduler", headers=headers)
        risk = await _fetch(client, "GET", f"{base_url}/risk", headers=headers)

    _validate_health(health)
    _validate_scheduler(scheduler)
    _validate_risk(risk)
    print("API smoke check completed successfully.")


def _validate_health(payload: Dict[str, Any]) -> None:
    if payload.get("status") != "healthy":
        raise SystemExit(f"Unexpected health status: {payload.get('status')!r}")
    risk_snapshot = payload.get("risk")
    if not isinstance(risk_snapshot, dict):
        raise SystemExit("/health missing risk snapshot")
    _validate_positions(risk_snapshot.get("positions"))


def _validate_scheduler(payload: Dict[str, Any]) -> None:
    execution = payload.get("execution")
    if not isinstance(execution, dict):
        raise SystemExit("/health/scheduler missing execution block")
    if "gross_exposure" not in execution or "positions_synced" not in execution:
        raise SystemExit("/health/scheduler execution block missing exposure fields")


def _validate_risk(payload: Dict[str, Any]) -> None:
    status = payload.get("status")
    if status not in (None, "ok"):
        raise SystemExit(f"Unexpected risk status: {status!r}")
    _validate_positions(payload.get("positions"))


def _validate_positions(positions_block: Any) -> None:
    if not isinstance(positions_block, dict):
        raise SystemExit("Risk snapshot missing positions block")
    required_keys = {"positions", "total_notional", "as_of"}
    missing = required_keys - positions_block.keys()
    if missing:
        raise SystemExit(f"Positions snapshot missing keys: {sorted(missing)}")


if __name__ == "__main__":
    try:
        asyncio.run(_run())
    except httpx.HTTPError as exc:
        print(f"HTTP request failed: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as exc:  # pragma: no cover - unexpected failure
        print(f"Smoke check failed: {exc}", file=sys.stderr)
        sys.exit(1)
