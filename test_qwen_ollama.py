import urllib.request, json, time

payload = {
    "model": "qwen3:8b",
    "messages": [
        {"role": "system", "content": "Reply ONLY with JSON: {action, confidence, reason}. action must be one of: HOLD, PARTIAL_CLOSE_25, FULL_CLOSE, TIME_STOP_EXIT."},
        {"role": "user", "content": '{"symbol": "BTCUSDT", "formula_suggestion": {"action": "HOLD", "confidence": 0.4}}'}
    ],
    "stream": False,
    "format": "json"
}
t0 = time.monotonic()
req = urllib.request.Request("http://localhost:11434/api/chat", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
try:
    resp = urllib.request.urlopen(req, timeout=30)
    body = json.loads(resp.read())
    latency = (time.monotonic() - t0) * 1000
    content = body["message"]["content"]
    parsed = json.loads(content)
    print(f"LATENCY={latency:.0f}ms")
    print(f"ACTION={parsed.get('action')}")
    print(f"CONFIDENCE={parsed.get('confidence')}")
    print(f"REASON={str(parsed.get('reason'))[:80]}")
except Exception as e:
    print(f"ERROR: {e}")
