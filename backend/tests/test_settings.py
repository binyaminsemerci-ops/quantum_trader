

def test_settings_roundtrip(client):
    # POST settings and then GET to ensure values were persisted
    # The payload uses intentional test literals; allowlist for detect-secrets.
    payload = {"api_key": "roundtrip_key", "api_secret": "roundtrip_secret"}  # pragma: allowlist secret
    post_resp = client.post("/settings", json=payload)
    assert post_resp.status_code == 200

    get_resp = client.get("/settings")
    assert get_resp.status_code == 200
    data = get_resp.json()
    assert data.get("api_key") == "roundtrip_key"
    # For safety do not assert secret equality in logs, just ensure key presence
    assert "api_secret" in data
