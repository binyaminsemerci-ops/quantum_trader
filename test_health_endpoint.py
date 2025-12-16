#!/usr/bin/env python3
import requests
import json
import sys

try:
    response = requests.get("http://localhost:8001/health", timeout=5)
    print(f"Status Code: {response.status_code}")
    print(f"\nResponse Body:")
    print(json.dumps(response.json(), indent=2))
    
    with open("/tmp/health_response.json", "w") as f:
        json.dump(response.json(), f, indent=2)
    print(f"\n✅ Response saved to /tmp/health_response.json")
    
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
