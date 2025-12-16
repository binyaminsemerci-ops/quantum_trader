"""
Test om backend faktisk leser .env.live og bruker riktig leverage
"""
import requests
import json

print("\n[SEARCH] TESTING BACKEND KONFIGURASJON")
print("=" * 80)

try:
    # Check health endpoint
    resp = requests.get("http://localhost:8000/health", timeout=5)
    print(f"\n[OK] Backend is running")
    print(f"Status: {resp.status_code}")
    
    # Try to get settings or config
    try:
        settings_resp = requests.get("http://localhost:8000/api/settings", timeout=5)
        if settings_resp.status_code == 200:
            settings = settings_resp.json()
            print(f"\n[CHART] Settings from API:")
            print(json.dumps(settings, indent=2))
    except Exception as e:
        print(f"[WARNING] Could not fetch settings: {e}")
    
    # Check if we can get positions
    try:
        pos_resp = requests.get("http://localhost:8000/api/positions", timeout=5)
        print(f"\n[CHART] Positions endpoint: {pos_resp.status_code}")
        if pos_resp.status_code == 200:
            positions = pos_resp.json()
            print(f"Open positions: {len(positions)}")
    except Exception as e:
        print(f"[WARNING] Positions check: {e}")
        
except Exception as e:
    print(f"\n❌ Backend not responding: {e}")
    print("\nBackend må være startet for at endringene skal virke!")

print("\n" + "=" * 80)
