#!/usr/bin/env python3
"""Simple Backend System Test.

A simplified version of backend testing without external dependencies.
Tests basic API connectivity and core functionality.
"""

import json
import sys
import time
from datetime import datetime
from typing import Any, Dict, List

import requests


class SimpleBackendTester:
    """Simplified backend system tester."""

    def __init__(self) -> None:
        self.base_url = "http://localhost:8000"
        self.test_results: List[Dict[str, Any]] = []

    def test_health_endpoint(self) -> Dict[str, Any]:
        """Test health endpoint."""
        start_time = time.time()

        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)

            result = {
                "test_name": "health_endpoint",
                "success": response.status_code == 200,
                "status_code": response.status_code,
                "response_time_ms": response.elapsed.total_seconds() * 1000,
                "test_duration_ms": (time.time() - start_time) * 1000,
            }

        except Exception as e:
            result = {
                "test_name": "health_endpoint",
                "success": False,
                "error": str(e),
                "test_duration_ms": (time.time() - start_time) * 1000,
            }

        self.test_results.append(result)
        return result

    def test_api_endpoints(self) -> Dict[str, Any]:
        """Test main API endpoints."""
        start_time = time.time()

        endpoints = ["/api/prices", "/api/portfolio", "/api/trades", "/api/stats"]

        endpoint_results = []

        for endpoint in endpoints:
            try:
                response = requests.get(f"{self.base_url}{endpoint}", timeout=5)
                endpoint_results.append(
                    {
                        "endpoint": endpoint,
                        "success": response.status_code == 200,
                        "status_code": response.status_code,
                        "response_time_ms": response.elapsed.total_seconds() * 1000,
                    },
                )
            except Exception as e:
                endpoint_results.append(
                    {"endpoint": endpoint, "success": False, "error": str(e)},
                )

        successful_endpoints = sum(
            1 for ep in endpoint_results if ep.get("success", False)
        )
        success_rate = successful_endpoints / len(endpoints)

        result = {
            "test_name": "api_endpoints",
            "success": success_rate >= 0.5,  # At least half should work
            "success_rate": success_rate,
            "endpoints_tested": len(endpoints),
            "endpoints_successful": successful_endpoints,
            "endpoint_results": endpoint_results,
            "test_duration_ms": (time.time() - start_time) * 1000,
        }

        self.test_results.append(result)
        return result

    def run_simple_tests(self) -> Dict[str, Any]:
        """Run simple backend tests."""
        start_time = time.time()

        # Check if backend is running
        health_result = self.test_health_endpoint()

        if health_result["success"]:

            api_result = self.test_api_endpoints()

            if api_result["success"]:
                pass
            else:
                pass
        else:
            # Still try API tests to see what fails
            api_result = self.test_api_endpoints()

        # Calculate overall results
        total_tests = len(self.test_results)
        successful_tests = sum(
            1 for result in self.test_results if result.get("success", False)
        )

        return {
            "tests_run": total_tests,
            "tests_passed": successful_tests,
            "tests_failed": total_tests - successful_tests,
            "success_rate": successful_tests / total_tests if total_tests > 0 else 0,
            "total_duration_ms": (time.time() - start_time) * 1000,
            "detailed_results": self.test_results,
            "timestamp": datetime.now().isoformat(),
        }



def main() -> int:
    """Main test execution."""
    import argparse

    parser = argparse.ArgumentParser(description="Simple Backend System Test")
    parser.add_argument("--output", help="Output file for test results")

    args = parser.parse_args()

    # Run tests
    tester = SimpleBackendTester()
    results = tester.run_simple_tests()

    # Print summary

    # Save results if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)

    # Exit with appropriate code
    if results["success_rate"] >= 0.5:
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
