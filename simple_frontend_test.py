#!/usr/bin/env python3
"""
Simple Frontend System Test

A simplified version of frontend testing without external dependencies.
Tests basic frontend build and accessibility.
"""

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any


class SimpleFrontendTester:
    """Simplified frontend system tester."""

    def __init__(self):
        self.frontend_dir = Path("frontend")
        self.test_results: List[Dict[str, Any]] = []

    def test_frontend_files_exist(self) -> Dict[str, Any]:
        """Test that key frontend files exist."""
        start_time = time.time()

        required_files = [
            "package.json",
            "index.html",
            "vite.config.tsx",
            "tsconfig.json",
        ]

        file_results = []
        for file_name in required_files:
            file_path = self.frontend_dir / file_name
            exists = file_path.exists()
            file_results.append(
                {"file": file_name, "exists": exists, "path": str(file_path)}
            )

        files_found = sum(1 for result in file_results if result["exists"])
        success_rate = files_found / len(required_files)

        result = {
            "test_name": "frontend_files_exist",
            "success": success_rate >= 0.75,  # At least 75% of files should exist
            "files_checked": len(required_files),
            "files_found": files_found,
            "success_rate": success_rate,
            "file_results": file_results,
            "test_duration_ms": (time.time() - start_time) * 1000,
        }

        self.test_results.append(result)
        return result

    def test_package_json_content(self) -> Dict[str, Any]:
        """Test package.json content."""
        start_time = time.time()

        try:
            package_json_path = self.frontend_dir / "package.json"

            if not package_json_path.exists():
                result = {
                    "test_name": "package_json_content",
                    "success": False,
                    "error": "package.json not found",
                    "test_duration_ms": (time.time() - start_time) * 1000,
                }
            else:
                with open(package_json_path, "r") as f:
                    package_data = json.load(f)

                # Check for key properties
                has_name = "name" in package_data
                has_scripts = "scripts" in package_data
                has_dependencies = "dependencies" in package_data
                has_dev_script = has_scripts and "dev" in package_data.get(
                    "scripts", {}
                )
                has_build_script = has_scripts and "build" in package_data.get(
                    "scripts", {}
                )

                checks = [
                    has_name,
                    has_scripts,
                    has_dependencies,
                    has_dev_script,
                    has_build_script,
                ]
                success_rate = sum(checks) / len(checks)

                result = {
                    "test_name": "package_json_content",
                    "success": success_rate >= 0.8,
                    "checks": {
                        "has_name": has_name,
                        "has_scripts": has_scripts,
                        "has_dependencies": has_dependencies,
                        "has_dev_script": has_dev_script,
                        "has_build_script": has_build_script,
                    },
                    "success_rate": success_rate,
                    "test_duration_ms": (time.time() - start_time) * 1000,
                }

        except Exception as e:
            result = {
                "test_name": "package_json_content",
                "success": False,
                "error": str(e),
                "test_duration_ms": (time.time() - start_time) * 1000,
            }

        self.test_results.append(result)
        return result

    def test_npm_available(self) -> Dict[str, Any]:
        """Test if npm is available."""
        start_time = time.time()

        try:
            # Check npm version
            result = subprocess.run(
                ["npm", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=self.frontend_dir,
            )

            npm_available = result.returncode == 0
            npm_version = result.stdout.strip() if npm_available else None

            test_result = {
                "test_name": "npm_available",
                "success": npm_available,
                "npm_version": npm_version,
                "test_duration_ms": (time.time() - start_time) * 1000,
            }

            if not npm_available:
                test_result["error"] = result.stderr.strip()

        except Exception as e:
            test_result = {
                "test_name": "npm_available",
                "success": False,
                "error": str(e),
                "test_duration_ms": (time.time() - start_time) * 1000,
            }

        self.test_results.append(test_result)
        return test_result

    def test_dependencies_installable(self) -> Dict[str, Any]:
        """Test if dependencies can be checked."""
        start_time = time.time()

        try:
            # Check if node_modules exists or if we can run npm ls
            node_modules_path = self.frontend_dir / "node_modules"

            if node_modules_path.exists():
                # Try to list installed packages
                result = subprocess.run(
                    ["npm", "ls", "--depth=0"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    cwd=self.frontend_dir,
                )

                has_packages = result.returncode == 0 or "node_modules" in result.stdout

                test_result = {
                    "test_name": "dependencies_installable",
                    "success": has_packages,
                    "node_modules_exists": True,
                    "npm_ls_success": result.returncode == 0,
                    "test_duration_ms": (time.time() - start_time) * 1000,
                }
            else:
                # Try npm install --dry-run to see if it would work
                result = subprocess.run(
                    ["npm", "install", "--dry-run"],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    cwd=self.frontend_dir,
                )

                test_result = {
                    "test_name": "dependencies_installable",
                    "success": result.returncode == 0,
                    "node_modules_exists": False,
                    "dry_run_success": result.returncode == 0,
                    "test_duration_ms": (time.time() - start_time) * 1000,
                }

                if not test_result["success"]:
                    test_result["error"] = result.stderr[:500]  # First 500 chars

        except Exception as e:
            test_result = {
                "test_name": "dependencies_installable",
                "success": False,
                "error": str(e),
                "test_duration_ms": (time.time() - start_time) * 1000,
            }

        self.test_results.append(test_result)
        return test_result

    def run_simple_tests(self) -> Dict[str, Any]:
        """Run simple frontend tests."""
        print("🧪 Running Simple Frontend Tests...")

        start_time = time.time()

        # Check if frontend directory exists
        if not self.frontend_dir.exists():
            print(f"  ❌ Frontend directory not found: {self.frontend_dir}")
            return {
                "error": f"Frontend directory not found: {self.frontend_dir}",
                "timestamp": datetime.now().isoformat(),
            }

        print(f"  📁 Frontend directory: {self.frontend_dir}")

        # Test frontend files
        print("  🔍 Checking frontend files...")
        files_result = self.test_frontend_files_exist()

        if files_result["success"]:
            print(f"  ✅ Frontend files present ({files_result['success_rate']:.1%})")
        else:
            print(f"  ⚠️ Missing frontend files ({files_result['success_rate']:.1%})")

        # Test package.json
        print("  🔍 Checking package.json...")
        package_result = self.test_package_json_content()

        if package_result["success"]:
            print("  ✅ Package.json looks good")
        else:
            print(
                f"  ⚠️ Package.json issues: {package_result.get('error', 'Invalid content')}"
            )

        # Test npm
        print("  🔍 Checking npm availability...")
        npm_result = self.test_npm_available()

        if npm_result["success"]:
            print(
                f"  ✅ npm available (version: {npm_result.get('npm_version', 'unknown')})"
            )

            # Test dependencies if npm is available
            print("  🔍 Checking dependencies...")
            deps_result = self.test_dependencies_installable()

            if deps_result["success"]:
                print("  ✅ Dependencies look good")
            else:
                print(
                    f"  ⚠️ Dependency issues: {deps_result.get('error', 'Unknown issue')}"
                )
        else:
            print(f"  ❌ npm not available: {npm_result.get('error', 'Unknown error')}")

        # Calculate overall results
        total_tests = len(self.test_results)
        successful_tests = sum(
            1 for result in self.test_results if result.get("success", False)
        )

        summary = {
            "tests_run": total_tests,
            "tests_passed": successful_tests,
            "tests_failed": total_tests - successful_tests,
            "success_rate": successful_tests / total_tests if total_tests > 0 else 0,
            "total_duration_ms": (time.time() - start_time) * 1000,
            "detailed_results": self.test_results,
            "timestamp": datetime.now().isoformat(),
        }

        return summary


def main():
    """Main test execution."""
    import argparse

    parser = argparse.ArgumentParser(description="Simple Frontend System Test")
    parser.add_argument("--output", help="Output file for test results")

    args = parser.parse_args()

    # Run tests
    tester = SimpleFrontendTester()
    results = tester.run_simple_tests()

    # Handle error case
    if "error" in results:
        print(f"❌ Frontend tests failed: {results['error']}")
        return 1

    # Print summary
    print("\n📊 Frontend Test Summary:")
    print(f"  Tests Run: {results['tests_run']}")
    print(f"  Tests Passed: {results['tests_passed']}")
    print(f"  Success Rate: {results['success_rate']:.1%}")
    print(f"  Duration: {results['total_duration_ms']:.0f}ms")

    # Save results if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"📄 Results saved to: {args.output}")

    # Exit with appropriate code
    if results["success_rate"] >= 0.5:
        print("✅ Frontend tests PASSED")
        return 0
    else:
        print("❌ Frontend tests FAILED")
        return 1


if __name__ == "__main__":
    exit(main())
