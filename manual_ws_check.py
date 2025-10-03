"""Manual WebSocket connectivity check.

This script is intentionally named so pytest will NOT collect it as a test.
Run manually:
    python manual_ws_check.py
"""

import asyncio
import contextlib
import json

try:
    import websockets  # type: ignore
except ImportError as e:
    msg = "Install websockets package to run this manual check: pip install websockets"
    raise SystemExit(
        msg,
    ) from e


async def main() -> None:
    uri = "ws://127.0.0.1:8000/ws/dashboard"
    try:
        async with websockets.connect(uri) as ws:  # type: ignore[attr-defined]
            await ws.send(json.dumps({"type": "ping"}))
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(ws.recv(), timeout=5)
    except Exception as e:
        print(f"WebSocket connection failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
