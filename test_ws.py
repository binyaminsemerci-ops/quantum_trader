"""Manual websocket check script.

This file previously interfered with pytest because it defined an async test function
named test_ws(). We keep a manual runner instead. Pytest will not collect since
there is no top-level test_* function anymore.
"""
import asyncio
import json

try:
    import websockets  # type: ignore
except ImportError:  # pragma: no cover
    raise SystemExit("Install websockets to run this script: pip install websockets")


async def _manual_ws_check():
    uri = 'ws://127.0.0.1:8000/ws/dashboard'
    print(f'Connecting to {uri} ...')
    try:
        async with websockets.connect(uri) as ws:  # type: ignore[attr-defined]
            print('Connected. Sending ping...')
            await ws.send(json.dumps({'type': 'ping'}))
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=5)
                print('Received:', msg)
            except asyncio.TimeoutError:
                print('No message within 5s (ok if server silent).')
    except Exception as e:  # noqa: BLE001
        print('WebSocket check failed:', e)


def main():  # pragma: no cover
    asyncio.run(_manual_ws_check())


if __name__ == '__main__':  # pragma: no cover
    main()