import requests, json, time

# Give service time to start
time.sleep(1)

try:
    r = requests.get("http://localhost:8001/health", timeout=5)
    print(json.dumps(r.json(), indent=2))
except Exception as e:
    print(f"Error: {e}")
