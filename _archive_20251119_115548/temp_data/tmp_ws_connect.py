from fastapi.testclient import TestClient
from backend.main import app
from starlette.websockets import WebSocketDisconnect

def main() -> None:
    client = TestClient(app)
    headers = {"X-Admin-Token": "test-admin-token"}
    try:
        with client.websocket_connect("/ws/dashboard", headers=headers) as ws:
            print("connected")
            message = ws.receive_json()
            print("message", message)
    except WebSocketDisconnect as exc:
        reason = getattr(exc, "reason", None)
        if reason:
            print(f"closed {exc.code}: {reason}")
        else:
            print(f"closed {exc.code}")


if __name__ == "__main__":
    main()
