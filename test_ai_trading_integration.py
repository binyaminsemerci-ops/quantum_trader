#!/usr/bin/env python3
"""AI Auto Trading Integration Test.

This script tests the AI Auto Trading Service integration with the backend.
It verifies that all endpoints work correctly and that the AI service can
generate signals and execute trades.
"""

import os
import sys

import requests

# Backend URL
BASE_URL = "http://127.0.0.1:8000"


def test_endpoint(method, endpoint, data=None, description=""):
    """Test a single API endpoint."""
    url = f"{BASE_URL}{endpoint}"

    if description:
        pass

    try:
        if method.upper() == "GET":
            response = requests.get(url, timeout=10)
        elif method.upper() == "POST":
            response = requests.post(url, json=data, timeout=10)
        else:
            return False


        if response.status_code in [200, 201]:
            result = response.json()
            if isinstance(result, dict) and len(result) <= 5:
                # Print small responses inline
                pass
            return result
        else:
            if response.text:
                pass
            return None

    except requests.exceptions.RequestException:
        return None
    except Exception:
        return None


def main() -> int:

    # Test 1: Check if backend is running
    system_status = test_endpoint(
        "GET", "/api/v1/system/status", description="Check if backend is running",
    )
    if not system_status:
        import subprocess
        import time

        # Start backend in separate process
        backend_process = subprocess.Popen(
            [sys.executable, "backend\\simple_main.py"], cwd=os.getcwd(),
        )

        # Wait for backend to start
        time.sleep(5)

        # Try to connect again
        system_status = test_endpoint(
            "GET",
            "/api/v1/system/status",
            description="Check if backend is running after start",
        )
        if not system_status:
            backend_process.terminate()
            return 1

    # Test 2: AI Trading Status (initial)
    ai_status = test_endpoint(
        "GET", "/api/v1/ai-trading/status", description="Get initial AI trading status",
    )

    # Test 3: Update AI Configuration
    config = {
        "position_size": 500.0,
        "stop_loss_pct": 2.5,
        "take_profit_pct": 5.0,
        "min_confidence": 0.75,
        "max_positions": 3,
        "risk_limit": 5000.0,
    }
    test_endpoint(
        "POST", "/api/v1/ai-trading/config", config, "Update AI trading configuration",
    )

    # Test 4: Start AI Trading
    start_symbols = ["BTCUSDC", "ETHUSDC"]
    start_result = test_endpoint(
        "POST",
        "/api/v1/ai-trading/start",
        start_symbols,
        "Start AI trading with test symbols",
    )

    if start_result:

        # Wait a moment for the system to initialize
        time.sleep(3)

        # Test 5: Get AI Status (after start)
        test_endpoint(
            "GET",
            "/api/v1/ai-trading/status",
            description="Get AI status after starting",
        )

        # Test 6: Get AI Signals
        test_endpoint(
            "GET",
            "/api/v1/ai-trading/signals?limit=5",
            description="Get recent AI trading signals",
        )

        # Test 7: Get AI Executions
        test_endpoint(
            "GET",
            "/api/v1/ai-trading/executions?limit=5",
            description="Get recent AI trade executions",
        )

        # Let AI run for a few seconds to potentially generate signals
        time.sleep(5)

        # Test 8: Check for new signals
        test_endpoint(
            "GET",
            "/api/v1/ai-trading/signals?limit=10",
            description="Check for newly generated signals",
        )

        # Test 9: Stop AI Trading
        stop_result = test_endpoint(
            "POST", "/api/v1/ai-trading/stop", None, "Stop AI auto trading",
        )

        if stop_result:
            pass

    # Test 10: Final Status Check
    test_endpoint(
        "GET", "/api/v1/ai-trading/status", description="Get final AI trading status",
    )

    # Test 11: Basic Portfolio Check
    test_endpoint("GET", "/api/v1/portfolio", description="Check portfolio data access")

    # Summary

    if ai_status and start_result and stop_result:
        return 0
    else:
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
