"""Manual WebSocket connectivity check.

This script is intentionally named so pytest will NOT collect it as a test.
Run manually:
    python manual_ws_check.py
"""
import asyncio
import json

try:
    import websockets  # type: ignore
except ImportError:
    raise SystemExit("Install websockets package to run this manual check: pip install websockets")


async def main():
    uri = 'ws://127.0.0.1:8000/ws/dashboard'
    print(f'Connecting to {uri} ...')
    try:
        async with websockets.connect(uri) as ws:  # type: ignore[attr-defined]
            print('Connected. Sending ping message...')
            await ws.send(json.dumps({'type': 'ping'}))
            try:
                response = await asyncio.wait_for(ws.recv(), timeout=5)
                print('Received:', response)
            except asyncio.TimeoutError:
                print('No response within 5s (this may be normal if server does not echo).')
    except Exception as e:  # noqa: BLE001
        print(f'WebSocket test failed: {e}')


if __name__ == '__main__':
    asyncio.run(main())