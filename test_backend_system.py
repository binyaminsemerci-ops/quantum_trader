#!/usr/bin/env python3
"""
Backend System Tests

Comprehensive tests for the Quantum Trader backend including:
- API endpoint testing
- Database operations
- WebSocket connections
- Real-time data flow
- Error handling
- Performance benchmarks
"""

import json
import sys
import time
import requests
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import subprocess
import threading

try:
    import websocket
except ImportError:
    websocket = None

try:
    import psutil
except ImportError:
    psutil = None

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent / "backend"))


class BackendSystemTester:
    """Comprehensive backend system testing."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.ws_base_url = base_url.replace("http", "ws")
        self.backend_process: Optional[subprocess.Popen] = None
        self.test_results: List[Dict[str, Any]] = []

    def start_backend_server(self) -> bool:
        """Start the backend server for testing."""
        try:
            # Check if server is already running
            response = requests.get(f"{self.base_url}/health", timeout=2)
            if response.status_code == 200:
                print("✅ Backend server already running")
                return True
        except Exception:
            pass

        print("🚀 Starting backend server...")

        # Start the backend server
        self.backend_process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "backend.main:app",
                "--host",
                "0.0.0.0",
                "--port",
                "8000",
                "--reload",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=Path.cwd(),
        )

        # Wait for server to start
        for attempt in range(30):  # Wait up to 30 seconds
            try:
                time.sleep(1)
                response = requests.get(f"{self.base_url}/health", timeout=2)
                if response.status_code == 200:
                    print("✅ Backend server started successfully")
                    return True
            except Exception:
                continue

        print("❌ Failed to start backend server")
        return False

    def stop_backend_server(self):
        """Stop the backend server."""
        if self.backend_process:
            print("🛑 Stopping backend server...")
            self.backend_process.terminate()
            self.backend_process.wait(timeout=10)
            self.backend_process = None

    def test_health_endpoint(self) -> Dict[str, Any]:
        """Test the health endpoint."""
        start_time = time.time()

        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            response_time = (time.time() - start_time) * 1000  # ms

            result = {
                "test_name": "health_endpoint",
                "success": response.status_code == 200,
                "status_code": response.status_code,
                "response_time_ms": response_time,
                "response_data": (
                    response.json() if response.status_code == 200 else None
                ),
                "error": None,
            }

        except Exception as e:
            result = {
                "test_name": "health_endpoint",
                "success": False,
                "error": str(e),
                "response_time_ms": (time.time() - start_time) * 1000,
            }

        self.test_results.append(result)
        return result

    def test_api_endpoints(self) -> List[Dict[str, Any]]:
        """Test all major API endpoints."""
        endpoints = [
            ("/health", "GET", None),
            ("/api/trades", "GET", None),
            ("/api/stats", "GET", None),
            ("/api/chart/BTCUSDT", "GET", None),
            ("/api/settings", "GET", None),
            ("/api/prices", "GET", None),
            ("/api/candles/BTCUSDT", "GET", None),
            ("/api/signals", "GET", None),
            ("/api/portfolio", "GET", None),
            ("/api/watchlist", "GET", None),
        ]

        results = []

        for endpoint, method, payload in endpoints:
            start_time = time.time()

            try:
                if method == "GET":
                    response = requests.get(f"{self.base_url}{endpoint}", timeout=10)
                elif method == "POST":
                    response = requests.post(
                        f"{self.base_url}{endpoint}", json=payload, timeout=10
                    )

                response_time = (time.time() - start_time) * 1000  # ms

                # Determine if response is successful
                is_success = 200 <= response.status_code < 300

                result = {
                    "test_name": f"api_endpoint_{endpoint.replace('/', '_').replace(':', '_')}",
                    "endpoint": endpoint,
                    "method": method,
                    "success": is_success,
                    "status_code": response.status_code,
                    "response_time_ms": response_time,
                    "response_size": len(response.content),
                    "error": None if is_success else f"HTTP {response.status_code}",
                }

                # Check for JSON response
                try:
                    response_data = response.json()
                    result["has_json_response"] = True
                    result["response_keys"] = (
                        list(response_data.keys())
                        if isinstance(response_data, dict)
                        else None
                    )
                except Exception:
                    result["has_json_response"] = False

            except Exception as e:
                result = {
                    "test_name": f"api_endpoint_{endpoint.replace('/', '_').replace(':', '_')}",
                    "endpoint": endpoint,
                    "method": method,
                    "success": False,
                    "error": str(e),
                    "response_time_ms": (time.time() - start_time) * 1000,
                }

            results.append(result)
            self.test_results.append(result)

        return results

    def test_websocket_connection(self) -> Dict[str, Any]:
        """Test WebSocket connection and real-time data."""
        start_time = time.time()

        if websocket is None:
            return {
                "test_name": "websocket_connection",
                "success": False,
                "error": "websocket-client not installed",
                "test_duration_ms": (time.time() - start_time) * 1000,
                "skipped": True,
            }

        messages_received = []
        connection_successful = False

        def on_message(ws, message):
            try:
                data = json.loads(message)
                messages_received.append(data)
            except Exception:
                messages_received.append({"raw_message": message})

        def on_open(ws):
            nonlocal connection_successful
            connection_successful = True
            # Send a test message
            ws.send(
                json.dumps({"type": "ping", "timestamp": datetime.now().isoformat()})
            )

        def on_error(ws, error):
            print(f"WebSocket error: {error}")

        try:
            # Test dashboard WebSocket
            ws_url = f"{self.ws_base_url}/ws/dashboard"
            ws = websocket.WebSocketApp(
                ws_url, on_open=on_open, on_message=on_message, on_error=on_error
            )

            # Run WebSocket in thread for 5 seconds
            ws_thread = threading.Thread(target=ws.run_forever)
            ws_thread.daemon = True
            ws_thread.start()

            # Wait for connection and messages
            time.sleep(5)
            ws.close()

            result = {
                "test_name": "websocket_connection",
                "success": connection_successful,
                "messages_received": len(messages_received),
                "test_duration_ms": (time.time() - start_time) * 1000,
                "sample_messages": messages_received[:3] if messages_received else [],
                "error": (
                    None
                    if connection_successful
                    else "Failed to establish WebSocket connection"
                ),
            }

        except Exception as e:
            result = {
                "test_name": "websocket_connection",
                "success": False,
                "error": str(e),
                "test_duration_ms": (time.time() - start_time) * 1000,
            }

        self.test_results.append(result)
        return result

    def test_concurrent_requests(self, num_requests: int = 10) -> Dict[str, Any]:
        """Test concurrent API requests."""
        start_time = time.time()

        def make_request():
            try:
                response = requests.get(f"{self.base_url}/api/prices", timeout=10)
                return {
                    "success": response.status_code == 200,
                    "response_time": response.elapsed.total_seconds() * 1000,
                    "status_code": response.status_code,
                }
            except Exception as e:
                return {"success": False, "error": str(e)}

        # Execute concurrent requests
        with ThreadPoolExecutor(max_workers=num_requests) as executor:
            futures = [executor.submit(make_request) for _ in range(num_requests)]
            results = [future.result() for future in futures]

        # Analyze results
        successful_requests = sum(1 for r in results if r.get("success", False))
        response_times = [
            r.get("response_time", 0) for r in results if "response_time" in r
        ]

        result = {
            "test_name": "concurrent_requests",
            "total_requests": num_requests,
            "successful_requests": successful_requests,
            "success_rate": successful_requests / num_requests,
            "success": successful_requests
            >= num_requests * 0.8,  # 80% success rate required
            "avg_response_time_ms": (
                sum(response_times) / len(response_times) if response_times else 0
            ),
            "max_response_time_ms": max(response_times) if response_times else 0,
            "min_response_time_ms": min(response_times) if response_times else 0,
            "test_duration_ms": (time.time() - start_time) * 1000,
        }

        self.test_results.append(result)
        return result

    def test_ai_integration(self) -> Dict[str, Any]:
        """Test AI model integration endpoints."""
        start_time = time.time()

        try:
            # Test training endpoint (if available)

            # Note: This would typically be a POST to /api/ai/train
            # For now, we'll test the stats endpoint which shows AI status
            response = requests.get(f"{self.base_url}/api/stats", timeout=30)

            if response.status_code == 200:
                data = response.json()
                has_ai_data = any(
                    key in data
                    for key in ["model_accuracy", "backtest_results", "training_status"]
                )

                result = {
                    "test_name": "ai_integration",
                    "success": has_ai_data,
                    "response_time_ms": (time.time() - start_time) * 1000,
                    "has_ai_metrics": has_ai_data,
                    "response_keys": (
                        list(data.keys()) if isinstance(data, dict) else []
                    ),
                    "error": None,
                }
            else:
                result = {
                    "test_name": "ai_integration",
                    "success": False,
                    "error": f"HTTP {response.status_code}",
                    "response_time_ms": (time.time() - start_time) * 1000,
                }

        except Exception as e:
            result = {
                "test_name": "ai_integration",
                "success": False,
                "error": str(e),
                "response_time_ms": (time.time() - start_time) * 1000,
            }

        self.test_results.append(result)
        return result

    def test_database_operations(self) -> Dict[str, Any]:
        """Test database connectivity and operations."""
        start_time = time.time()

        try:
            # Test trades endpoint which requires database
            response = requests.get(f"{self.base_url}/api/trades?limit=10", timeout=10)

            if response.status_code == 200:
                data = response.json()
                is_array = isinstance(data, list)

                result = {
                    "test_name": "database_operations",
                    "success": True,
                    "response_time_ms": (time.time() - start_time) * 1000,
                    "trades_returned": len(data) if is_array else 0,
                    "response_format_valid": is_array,
                    "error": None,
                }
            else:
                result = {
                    "test_name": "database_operations",
                    "success": False,
                    "error": f"HTTP {response.status_code}",
                    "response_time_ms": (time.time() - start_time) * 1000,
                }

        except Exception as e:
            result = {
                "test_name": "database_operations",
                "success": False,
                "error": str(e),
                "response_time_ms": (time.time() - start_time) * 1000,
            }

        self.test_results.append(result)
        return result

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all backend system tests."""
        print("🔍 Running Backend System Tests...")
        print("=" * 50)

        # Start backend server
        if not self.start_backend_server():
            return {"error": "Failed to start backend server"}

        test_summary = {
            "start_time": datetime.now().isoformat(),
            "tests_run": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "total_duration_ms": 0,
        }

        start_time = time.time()

        try:
            # Run individual tests
            tests = [
                ("Health Check", self.test_health_endpoint),
                ("API Endpoints", self.test_api_endpoints),
                ("WebSocket Connection", self.test_websocket_connection),
                ("Concurrent Requests", lambda: self.test_concurrent_requests(10)),
                ("AI Integration", self.test_ai_integration),
                ("Database Operations", self.test_database_operations),
            ]

            for test_name, test_func in tests:
                print(f"\n📋 Running: {test_name}")

                try:
                    result = test_func()

                    if isinstance(
                        result, list
                    ):  # Multiple results (like API endpoints)
                        for r in result:
                            test_summary["tests_run"] += 1
                            if r.get("success", False):
                                test_summary["tests_passed"] += 1
                                print(f"  ✅ {r['test_name']}")
                            else:
                                test_summary["tests_failed"] += 1
                                print(
                                    f"  ❌ {r['test_name']}: {r.get('error', 'Failed')}"
                                )
                    else:  # Single result
                        test_summary["tests_run"] += 1
                        if result.get("success", False):
                            test_summary["tests_passed"] += 1
                            print(f"  ✅ {test_name}")
                        else:
                            test_summary["tests_failed"] += 1
                            print(f"  ❌ {test_name}: {result.get('error', 'Failed')}")

                except Exception as e:
                    test_summary["tests_run"] += 1
                    test_summary["tests_failed"] += 1
                    print(f"  ❌ {test_name}: {str(e)}")

            test_summary["total_duration_ms"] = (time.time() - start_time) * 1000
            test_summary["success_rate"] = (
                test_summary["tests_passed"] / test_summary["tests_run"]
                if test_summary["tests_run"] > 0
                else 0
            )
            test_summary["end_time"] = datetime.now().isoformat()
            test_summary["detailed_results"] = self.test_results

            # Print summary
            print("\n" + "=" * 50)
            print("📊 Backend System Test Summary:")
            print(f"  Tests Run: {test_summary['tests_run']}")
            print(f"  Tests Passed: {test_summary['tests_passed']}")
            print(f"  Tests Failed: {test_summary['tests_failed']}")
            print(f"  Success Rate: {test_summary['success_rate']:.1%}")
            print(f"  Total Duration: {test_summary['total_duration_ms']:.0f}ms")

            return test_summary

        finally:
            # Always stop the server
            self.stop_backend_server()


def main():
    """Main test execution."""
    import argparse

    parser = argparse.ArgumentParser(description="Backend System Tests")
    parser.add_argument("--url", default="http://localhost:8000", help="Backend URL")
    parser.add_argument("--output", help="Output file for test results")

    args = parser.parse_args()

    # Run tests
    tester = BackendSystemTester(args.url)
    results = tester.run_all_tests()

    # Save results if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"📄 Results saved to: {args.output}")

    # Exit with appropriate code
    success_rate = results.get("success_rate", 0)
    if success_rate >= 0.8:  # 80% success rate required
        print("✅ Backend system tests PASSED")
        return 0
    else:
        print("❌ Backend system tests FAILED")
        return 1


if __name__ == "__main__":
    exit(main())
