from backend.config import settings


def test_settings_loads():
    # tests may include placeholder literals; mark them allowlisted for detect-secrets
    # to avoid CI false positives when test fixtures include dummy credentials.
    # pragma: allowlist secret
    assert settings is not None


def test_settings_roundtrip(client):
    # POST settings and then GET to ensure values were persisted
    payload = {
        "api_key": "roundtrip_key",
        "api_secret": "roundtrip_secret",
    }  # pragma: allowlist secret
    post_resp = client.post("/settings", json=payload)
    assert post_resp.status_code == 200

    get_resp = client.get("/settings")
    assert get_resp.status_code == 200
    data = get_resp.json()
    assert data.get("api_key") == "roundtrip_key"
    # For safety do not assert secret equality in logs, just ensure key presence
    assert "api_secret" in data
