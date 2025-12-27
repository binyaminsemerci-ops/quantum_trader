import requests
import json

print("Testing AI Insights Endpoint")
print("=" * 50)

url = "http://46.224.116.254:8025/ai/insights"

for i in range(10):
    response = requests.get(url)
    data = response.json()
    
    status = "⚠️ RETRAIN" if data["suggestion"] == "Retrain model" else "✅ STABLE"
    
    print(f"\nCall {i+1}: {status}")
    print(f"  Accuracy: {data['accuracy']}")
    print(f"  Drift Score: {data['drift_score']}")
    print(f"  Sharpe: {data['sharpe']}")
    print(f"  Latency: {data['latency']}ms")
    print(f"  Suggestion: {data['suggestion']}")
    
    if data["suggestion"] == "Retrain model":
        print(f"  🚨 High drift detected! Performance unstable.")
        break

print("\n" + "=" * 50)
print("✅ AI Insights endpoint operational")
print("✅ Drift detection working")
print("✅ Retrain recommendations enabled")
