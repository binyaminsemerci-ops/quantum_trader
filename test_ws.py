"""Manual websocket check script.

This file previously interfered with pytest because it defined an async test function
named test_ws(). We keep a manual runner instead. Pytest will not collect since
there is no top-level test_* function anymore.
"""

import asyncio
import contextlib
import json

try:
    import websockets  # type: ignore
except ImportError as e:  # pragma: no cover
    msg = "Install websockets to run this script: pip install websockets"
    raise SystemExit(msg) from e


async def _manual_ws_check() -> None:
    uri = "ws://127.0.0.1:8000/ws/dashboard"
    try:
        async with websockets.connect(uri) as ws:  # type: ignore[attr-defined]
            await ws.send(json.dumps({"type": "ping"}))
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(ws.recv(), timeout=5)
    except Exception as e:
        print(f"WebSocket test failed: {e}")


def main() -> None:  # pragma: no cover
    asyncio.run(_manual_ws_check())


if __name__ == "__main__":  # pragma: no cover
    main()
