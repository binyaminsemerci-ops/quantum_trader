"""Test what routes are actually registered in the live server."""
import requests
import json

print("=" * 80)
print("TESTING LIVE SERVER ROUTES")
print("=" * 80)

base_url = "http://localhost:8000"

# Test 1: Health endpoint (known to work)
print("\n1. Testing /health endpoint:")
try:
    response = requests.get(f"{base_url}/health", timeout=5)
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        print("   ✅ Health endpoint works")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 2: OpportunityRanker endpoint
print("\n2. Testing /opportunities/rankings endpoint:")
try:
    response = requests.get(f"{base_url}/opportunities/rankings", timeout=5)
    print(f"   Status: {response.status_code}")
    if response.status_code == 503:
        print("   ✅ CORRECT! Returns 503 (ranker not initialized)")
        print(f"   Response: {response.json()}")
    elif response.status_code == 404:
        print("   ❌ ERROR! Returns 404 (route not registered)")
        print(f"   Response: {response.json()}")
    elif response.status_code == 200:
        print("   ✅ WORKING! Returns 200")
        print(f"   Response: {json.dumps(response.json(), indent=2)}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 3: Try to get OpenAPI docs
print("\n3. Testing /docs endpoint:")
try:
    response = requests.get(f"{base_url}/docs", timeout=5)
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        print("   ✅ Docs endpoint accessible")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 4: Try alternative opportunity endpoint
print("\n4. Testing /opportunities/rankings/top endpoint:")
try:
    response = requests.get(f"{base_url}/opportunities/rankings/top", timeout=5)
    print(f"   Status: {response.status_code}")
    if response.status_code in [200, 503]:
        print(f"   ✅ Endpoint exists (status: {response.status_code})")
    elif response.status_code == 404:
        print("   ❌ 404 Not Found")
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n" + "=" * 80)
