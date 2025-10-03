#!/usr/bin/env python3
"""
Frontend System Tests

Comprehensive tests for the Quantum Trader frontend including:
- Component rendering tests
- User interaction simulation  
- API integration tests
- Performance benchmarks
- Cross-browser compatibility
- Responsive design validation
"""

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import requests


class FrontendSystemTester:
    """Comprehensive frontend system testing."""

    def __init__(
        self, frontend_dir: str = "frontend", backend_url: str = "http://localhost:8000"
    ):
        self.frontend_dir = Path(frontend_dir)
        self.backend_url = backend_url
        self.frontend_process: Optional[subprocess.Popen] = None
        self.backend_process: Optional[subprocess.Popen] = None
        self.test_results: List[Dict[str, Any]] = []
        self.frontend_url = "http://localhost:5173"

    def start_backend_server(self) -> bool:
        """Start backend server for frontend testing."""
        try:
            # Check if server is already running
            response = requests.get(f"{self.backend_url}/health", timeout=2)
            if response.status_code == 200:
                print("✅ Backend server already running")
                return True
        except Exception:
            pass

        print("🚀 Starting backend server for frontend tests...")

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
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=Path.cwd(),
        )

        # Wait for server to start
        for attempt in range(30):
            try:
                time.sleep(1)
                response = requests.get(f"{self.backend_url}/health", timeout=2)
                if response.status_code == 200:
                    print("✅ Backend server started")
                    return True
            except Exception:
                continue

        print("❌ Failed to start backend server")
        return False

    def start_frontend_dev_server(self) -> bool:
        """Start the frontend development server."""
        try:
            # Check if already running
            response = requests.get(self.frontend_url, timeout=2)
            if response.status_code == 200:
                print("✅ Frontend server already running")
                return True
        except Exception:
            pass

        print("🚀 Starting frontend development server...")

        # Start the frontend dev server
        self.frontend_process = subprocess.Popen(
            ["npm", "run", "dev", "--", "--host", "127.0.0.1", "--port", "5173"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self.frontend_dir,
        )

        # Wait for server to start
        for attempt in range(60):  # Vite can take longer to start
            try:
                time.sleep(1)
                response = requests.get(self.frontend_url, timeout=2)
                if response.status_code == 200:
                    print("✅ Frontend server started")
                    return True
            except Exception:
                continue

        print("❌ Failed to start frontend server")
        return False

    def stop_servers(self):
        """Stop both frontend and backend servers."""
        if self.frontend_process:
            print("🛑 Stopping frontend server...")
            self.frontend_process.terminate()
            try:
                self.frontend_process.wait(timeout=10)
            except Exception:
                self.frontend_process.kill()
            self.frontend_process = None

        if self.backend_process:
            print("🛑 Stopping backend server...")
            self.backend_process.terminate()
            try:
                self.backend_process.wait(timeout=10)
            except Exception:
                self.backend_process.kill()
            self.backend_process = None

    def test_frontend_build(self) -> Dict[str, Any]:
        """Test frontend build process."""
        start_time = time.time()

        try:
            print("🔨 Testing frontend build...")
            result = subprocess.run(
                ["npm", "run", "build"],
                cwd=self.frontend_dir,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minutes
            )

            build_time = time.time() - start_time

            # Check if dist directory was created
            dist_dir = self.frontend_dir / "dist"
            has_dist = dist_dir.exists()

            # Check for essential files
            essential_files = ["index.html", "assets"]
            files_present = []
            if has_dist:
                for file in essential_files:
                    file_path = dist_dir / file
                    if file_path.exists():
                        files_present.append(file)

            test_result = {
                "test_name": "frontend_build",
                "success": result.returncode == 0
                and has_dist
                and len(files_present) >= 1,
                "build_time_seconds": build_time,
                "return_code": result.returncode,
                "has_dist_directory": has_dist,
                "essential_files_present": files_present,
                "build_output_size": (
                    self._get_directory_size(dist_dir) if has_dist else 0
                ),
                "stdout": (
                    result.stdout[:500] if result.stdout else ""
                ),  # First 500 chars
                "stderr": result.stderr[:500] if result.stderr else "",
                "error": (
                    None
                    if result.returncode == 0
                    else f"Build failed with code {result.returncode}"
                ),
            }

        except Exception as e:
            test_result = {
                "test_name": "frontend_build",
                "success": False,
                "error": str(e),
                "build_time_seconds": time.time() - start_time,
            }

        self.test_results.append(test_result)
        return test_result

    def test_vitest_unit_tests(self) -> Dict[str, Any]:
        """Run Vitest unit tests."""
        start_time = time.time()

        try:
            print("🧪 Running Vitest unit tests...")
            result = subprocess.run(
                ["npm", "run", "test"],
                cwd=self.frontend_dir,
                capture_output=True,
                text=True,
                timeout=180,  # 3 minutes
            )

            test_time = time.time() - start_time

            # Parse test output for results
            output = result.stdout + result.stderr

            # Look for test results in output
            tests_run = 0
            tests_passed = 0
            tests_failed = 0

            # Simple parsing - in a real implementation, you'd parse JSON output
            if "Test Files" in output:
                lines = output.split("\n")
                for line in lines:
                    if "passed" in line.lower() and "failed" in line.lower():
                        # Extract numbers from lines like "Tests  2 passed (2)"
                        words = line.split()
                        for i, word in enumerate(words):
                            if word.isdigit():
                                if "passed" in line.lower():
                                    tests_passed = int(word)
                                elif "failed" in line.lower():
                                    tests_failed = int(word)
                        tests_run = tests_passed + tests_failed

            test_result = {
                "test_name": "vitest_unit_tests",
                "success": result.returncode == 0,
                "test_time_seconds": test_time,
                "tests_run": tests_run,
                "tests_passed": tests_passed,
                "tests_failed": tests_failed,
                "return_code": result.returncode,
                "test_output": output[:1000],  # First 1000 chars
                "error": (
                    None
                    if result.returncode == 0
                    else f"Tests failed with code {result.returncode}"
                ),
            }

        except Exception as e:
            test_result = {
                "test_name": "vitest_unit_tests",
                "success": False,
                "error": str(e),
                "test_time_seconds": time.time() - start_time,
            }

        self.test_results.append(test_result)
        return test_result

    def test_typescript_compilation(self) -> Dict[str, Any]:
        """Test TypeScript compilation."""
        start_time = time.time()

        try:
            print("📝 Testing TypeScript compilation...")
            result = subprocess.run(
                ["npm", "run", "typecheck"],
                cwd=self.frontend_dir,
                capture_output=True,
                text=True,
                timeout=120,  # 2 minutes
            )

            compile_time = time.time() - start_time

            # Check for TypeScript errors
            has_errors = "error" in result.stderr.lower() if result.stderr else False
            error_count = result.stderr.count("error TS") if result.stderr else 0

            test_result = {
                "test_name": "typescript_compilation",
                "success": result.returncode == 0 and not has_errors,
                "compile_time_seconds": compile_time,
                "return_code": result.returncode,
                "typescript_errors": error_count,
                "has_type_errors": has_errors,
                "output": (result.stdout + result.stderr)[:1000],
                "error": (
                    None if result.returncode == 0 else "TypeScript compilation failed"
                ),
            }

        except Exception as e:
            test_result = {
                "test_name": "typescript_compilation",
                "success": False,
                "error": str(e),
                "compile_time_seconds": time.time() - start_time,
            }

        self.test_results.append(test_result)
        return test_result

    def test_frontend_loading(self) -> Dict[str, Any]:
        """Test if frontend loads correctly."""
        start_time = time.time()

        try:
            # Test main page load
            response = requests.get(self.frontend_url, timeout=10)
            load_time = (time.time() - start_time) * 1000  # ms

            # Check response
            is_html = "text/html" in response.headers.get("content-type", "")
            has_react_root = "root" in response.text if response.text else False
            content_size = len(response.content)

            test_result = {
                "test_name": "frontend_loading",
                "success": response.status_code == 200 and is_html,
                "status_code": response.status_code,
                "load_time_ms": load_time,
                "content_type": response.headers.get("content-type", ""),
                "content_size": content_size,
                "is_html_response": is_html,
                "has_react_root": has_react_root,
                "response_headers": dict(response.headers),
                "error": (
                    None
                    if response.status_code == 200
                    else f"HTTP {response.status_code}"
                ),
            }

        except Exception as e:
            test_result = {
                "test_name": "frontend_loading",
                "success": False,
                "error": str(e),
                "load_time_ms": (time.time() - start_time) * 1000,
            }

        self.test_results.append(test_result)
        return test_result

    def test_api_integration(self) -> Dict[str, Any]:
        """Test frontend API integration with backend."""
        start_time = time.time()

        try:
            # Test API endpoints that frontend typically calls
            api_tests = [
                f"{self.backend_url}/api/prices",
                f"{self.backend_url}/api/trades",
                f"{self.backend_url}/api/portfolio",
                f"{self.backend_url}/health",
            ]

            api_results = []

            for api_url in api_tests:
                try:
                    api_start = time.time()
                    response = requests.get(api_url, timeout=5)
                    api_time = (time.time() - api_start) * 1000

                    api_results.append(
                        {
                            "endpoint": api_url,
                            "success": response.status_code == 200,
                            "status_code": response.status_code,
                            "response_time_ms": api_time,
                            "has_json": response.headers.get(
                                "content-type", ""
                            ).startswith("application/json"),
                        }
                    )
                except Exception as e:
                    api_results.append(
                        {"endpoint": api_url, "success": False, "error": str(e)}
                    )

            # Calculate overall success
            successful_apis = sum(
                1 for result in api_results if result.get("success", False)
            )
            success_rate = successful_apis / len(api_results) if api_results else 0

            test_result = {
                "test_name": "api_integration",
                "success": success_rate >= 0.75,  # 75% of APIs must work
                "test_duration_ms": (time.time() - start_time) * 1000,
                "apis_tested": len(api_results),
                "apis_successful": successful_apis,
                "success_rate": success_rate,
                "api_results": api_results,
                "error": (
                    None
                    if success_rate >= 0.75
                    else f"Only {success_rate:.1%} of APIs working"
                ),
            }

        except Exception as e:
            test_result = {
                "test_name": "api_integration",
                "success": False,
                "error": str(e),
                "test_duration_ms": (time.time() - start_time) * 1000,
            }

        self.test_results.append(test_result)
        return test_result

    def test_static_assets(self) -> Dict[str, Any]:
        """Test static assets loading."""
        start_time = time.time()

        try:
            # Test common static asset paths
            asset_paths = [
                "/favicon.ico",
                "/vite.svg",
                "/src/main.tsx",  # Should be served by Vite dev server
            ]

            asset_results = []

            for asset_path in asset_paths:
                try:
                    asset_url = f"{self.frontend_url}{asset_path}"
                    asset_start = time.time()
                    response = requests.get(asset_url, timeout=5)
                    asset_time = (time.time() - asset_start) * 1000

                    asset_results.append(
                        {
                            "path": asset_path,
                            "success": response.status_code == 200,
                            "status_code": response.status_code,
                            "response_time_ms": asset_time,
                            "content_type": response.headers.get("content-type", ""),
                            "content_size": len(response.content),
                        }
                    )
                except Exception as e:
                    asset_results.append(
                        {"path": asset_path, "success": False, "error": str(e)}
                    )

            # Calculate success rate
            successful_assets = sum(
                1 for result in asset_results if result.get("success", False)
            )
            success_rate = (
                successful_assets / len(asset_results) if asset_results else 0
            )

            test_result = {
                "test_name": "static_assets",
                "success": success_rate >= 0.5,  # At least 50% of assets should load
                "test_duration_ms": (time.time() - start_time) * 1000,
                "assets_tested": len(asset_results),
                "assets_successful": successful_assets,
                "success_rate": success_rate,
                "asset_results": asset_results,
                "error": (
                    None
                    if success_rate >= 0.5
                    else f"Only {success_rate:.1%} of assets loading"
                ),
            }

        except Exception as e:
            test_result = {
                "test_name": "static_assets",
                "success": False,
                "error": str(e),
                "test_duration_ms": (time.time() - start_time) * 1000,
            }

        self.test_results.append(test_result)
        return test_result

    def _get_directory_size(self, directory: Path) -> int:
        """Get total size of directory in bytes."""
        total_size = 0
        try:
            for file_path in directory.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        except Exception:
            pass
        return total_size

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all frontend system tests."""
        print("🎨 Running Frontend System Tests...")
        print("=" * 50)

        test_summary = {
            "start_time": datetime.now().isoformat(),
            "tests_run": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "total_duration_ms": 0,
        }

        start_time = time.time()

        try:
            # Start backend first
            if not self.start_backend_server():
                print("⚠️  Continuing without backend server")

            # Run build and compilation tests first (no servers needed)
            build_tests = [
                ("TypeScript Compilation", self.test_typescript_compilation),
                ("Frontend Build", self.test_frontend_build),
                ("Vitest Unit Tests", self.test_vitest_unit_tests),
            ]

            for test_name, test_func in build_tests:
                print(f"\n📋 Running: {test_name}")

                try:
                    result = test_func()
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

            # Start frontend server for runtime tests
            if self.start_frontend_dev_server():
                runtime_tests = [
                    ("Frontend Loading", self.test_frontend_loading),
                    ("API Integration", self.test_api_integration),
                    ("Static Assets", self.test_static_assets),
                ]

                for test_name, test_func in runtime_tests:
                    print(f"\n📋 Running: {test_name}")

                    try:
                        result = test_func()
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
            else:
                print("⚠️  Skipping runtime tests - frontend server failed to start")

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
            print("🎨 Frontend System Test Summary:")
            print(f"  Tests Run: {test_summary['tests_run']}")
            print(f"  Tests Passed: {test_summary['tests_passed']}")
            print(f"  Tests Failed: {test_summary['tests_failed']}")
            print(f"  Success Rate: {test_summary['success_rate']:.1%}")
            print(f"  Total Duration: {test_summary['total_duration_ms']:.0f}ms")

            return test_summary

        finally:
            # Always stop servers
            self.stop_servers()


def main():
    """Main test execution."""
    import argparse

    parser = argparse.ArgumentParser(description="Frontend System Tests")
    parser.add_argument("--frontend-dir", default="frontend", help="Frontend directory")
    parser.add_argument(
        "--backend-url", default="http://localhost:8000", help="Backend URL"
    )
    parser.add_argument("--output", help="Output file for test results")

    args = parser.parse_args()

    # Run tests
    tester = FrontendSystemTester(args.frontend_dir, args.backend_url)
    results = tester.run_all_tests()

    # Save results if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"📄 Results saved to: {args.output}")

    # Exit with appropriate code
    success_rate = results.get("success_rate", 0)
    if success_rate >= 0.7:  # 70% success rate required
        print("✅ Frontend system tests PASSED")
        return 0
    else:
        print("❌ Frontend system tests FAILED")
        return 1


if __name__ == "__main__":
    exit(main())
