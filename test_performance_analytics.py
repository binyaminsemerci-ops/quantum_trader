#!/usr/bin/env python3
"""
Performance Analytics Module Test Suite
Tests metrics computation, export functionality, and API endpoints
"""
import requests
import json
from datetime import datetime

API_BASE = "http://localhost:8000"

def test_performance_metrics():
    """Test /performance/metrics endpoint"""
    print("\n🧪 Testing GET /performance/metrics...")
    try:
        response = requests.get(f"{API_BASE}/performance/metrics", timeout=5)
        data = response.json()
        
        print(f"Status Code: {response.status_code}")
        print(f"Metrics: {json.dumps(data.get('metrics', {}), indent=2)}")
        print(f"Curve Points: {len(data.get('curve', []))}")
        
        assert response.status_code == 200, "Expected 200 OK"
        assert "metrics" in data, "Response must contain 'metrics'"
        assert "curve" in data, "Response must contain 'curve'"
        
        metrics = data["metrics"]
        required_fields = ["total_return", "winrate", "profit_factor", "sharpe", "sortino", "max_drawdown"]
        for field in required_fields:
            assert field in metrics, f"Missing required field: {field}"
        
        print("✅ Metrics endpoint working correctly")
        return True
    except Exception as e:
        print(f"❌ Metrics endpoint failed: {e}")
        return False

def test_export_json():
    """Test JSON export"""
    print("\n🧪 Testing GET /reports/export/json...")
    try:
        response = requests.get(f"{API_BASE}/reports/export/json", timeout=5)
        
        print(f"Status Code: {response.status_code}")
        print(f"Content-Type: {response.headers.get('Content-Type')}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Records exported: {len(data) if isinstance(data, list) else 'N/A'}")
            print("✅ JSON export working")
            return True
        else:
            print(f"⚠️ Status {response.status_code}: {response.text[:200]}")
            return False
    except Exception as e:
        print(f"❌ JSON export failed: {e}")
        return False

def test_export_csv():
    """Test CSV export"""
    print("\n🧪 Testing GET /reports/export/csv...")
    try:
        response = requests.get(f"{API_BASE}/reports/export/csv", timeout=5)
        
        print(f"Status Code: {response.status_code}")
        print(f"Content-Type: {response.headers.get('Content-Type')}")
        
        if response.status_code == 200:
            lines = response.text.split('\n')
            print(f"CSV Lines: {len(lines)}")
            print(f"Headers: {lines[0] if lines else 'None'}")
            print("✅ CSV export working")
            return True
        else:
            print(f"⚠️ Status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ CSV export failed: {e}")
        return False

def test_export_pdf():
    """Test PDF export"""
    print("\n🧪 Testing GET /reports/export/pdf...")
    try:
        response = requests.get(f"{API_BASE}/reports/export/pdf", timeout=10)
        
        print(f"Status Code: {response.status_code}")
        print(f"Content-Type: {response.headers.get('Content-Type')}")
        print(f"Content Size: {len(response.content)} bytes")
        
        if response.status_code == 200 and response.headers.get('Content-Type') == 'application/pdf':
            print("✅ PDF export working")
            return True
        else:
            print(f"⚠️ PDF export returned: {response.text[:200]}")
            return False
    except Exception as e:
        print(f"❌ PDF export failed: {e}")
        return False

def test_summary_endpoint():
    """Test legacy summary endpoint"""
    print("\n🧪 Testing GET /performance/summary...")
    try:
        response = requests.get(f"{API_BASE}/performance/summary", timeout=5)
        data = response.json()
        
        print(f"Status Code: {response.status_code}")
        print(f"Sharpe: {data.get('sharpe_ratio')}")
        print(f"Win Rate: {data.get('win_rate')}")
        print(f"Total Trades: {data.get('total_trades')}")
        
        assert response.status_code == 200
        print("✅ Summary endpoint working")
        return True
    except Exception as e:
        print(f"❌ Summary endpoint failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("  PHASE 21: PERFORMANCE ANALYTICS TEST SUITE")
    print("=" * 60)
    
    results = {
        "Metrics Endpoint": test_performance_metrics(),
        "Summary Endpoint": test_summary_endpoint(),
        "JSON Export": test_export_json(),
        "CSV Export": test_export_csv(),
        "PDF Export": test_export_pdf()
    }
    
    print("\n" + "=" * 60)
    print("  TEST RESULTS")
    print("=" * 60)
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:.<45} {status}")
    
    total = len(results)
    passed = sum(results.values())
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n>>> [Phase 21 Complete – Performance Analytics & Equity Visualization Operational]")
    else:
        print(f"\n⚠️ {total - passed} test(s) failed. Please check backend logs.")
