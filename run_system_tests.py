#!/usr/bin/env python3
"""
Complete System Test Runner

Master test runner that executes all system tests:
- Backend system tests
- Frontend system tests  
- End-to-end integration tests
- Performance validation
- Comprehensive reporting

This provides a single command to validate the entire Quantum Trader system.
"""

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any


class SystemTestRunner:
    """Master system test runner."""

    def __init__(self):
        self.test_results: Dict[str, Any] = {}
        self.overall_summary = {
            "start_time": None,
            "end_time": None,
            "total_duration_ms": 0,
            "test_categories": {},
            "overall_success": False,
            "overall_success_rate": 0.0,
        }

        # Test scripts
        self.test_scripts = [
            {
                "name": "Backend System Tests",
                "script": "test_backend_system.py",
                "description": "API endpoints, WebSocket, database, AI integration",
            },
            {
                "name": "Frontend System Tests",
                "script": "test_frontend_system.py",
                "description": "Build process, TypeScript, components, API integration",
            },
            {
                "name": "End-to-End Integration Tests",
                "script": "test_e2e_integration.py",
                "description": "Full stack integration, user workflows, load testing",
            },
        ]

    def check_prerequisites(self) -> Dict[str, Any]:
        """Check system prerequisites for testing."""
        print("🔍 Checking system prerequisites...")

        prereqs = {
            "python_version": sys.version,
            "python_executable": sys.executable,
            "working_directory": str(Path.cwd()),
            "test_scripts_present": {},
            "dependencies_available": {},
            "errors": [],
        }

        # Check test scripts exist
        for test in self.test_scripts:
            script_path = Path(test["script"])
            prereqs["test_scripts_present"][test["script"]] = script_path.exists()
            if not script_path.exists():
                prereqs["errors"].append(f"Test script not found: {test['script']}")

        # Check key dependencies
        dependencies = [
            ("requests", "requests"),
            ("websocket-client", "websocket"),
            ("fastapi", "fastapi"),
            ("uvicorn", "uvicorn"),
        ]
        for dep_name, import_name in dependencies:
            try:
                __import__(import_name)
                prereqs["dependencies_available"][dep_name] = True
            except ImportError:
                prereqs["dependencies_available"][dep_name] = False
                prereqs["errors"].append(f"Missing dependency: {dep_name}")

        # Check if requirements files exist
        req_files = ["requirements.txt", "backend/requirements.txt"]
        for req_file in req_files:
            if Path(req_file).exists():
                print(f"  ✅ Found {req_file}")

        # Summary
        all(prereqs["test_scripts_present"].values())
        all(prereqs["dependencies_available"].values())

        prereqs["all_prerequisites_met"] = len(prereqs["errors"]) == 0

        if prereqs["all_prerequisites_met"]:
            print("  ✅ All prerequisites met")
        else:
            print("  ❌ Prerequisites missing:")
            for error in prereqs["errors"]:
                print(f"    - {error}")

        return prereqs

    def run_test_script(self, test_info: Dict[str, str]) -> Dict[str, Any]:
        """Run a single test script and capture results."""
        script_name = test_info["script"]
        test_name = test_info["name"]

        print(f"\n🧪 Running {test_name}...")
        print(f"   Script: {script_name}")
        print(f"   Testing: {test_info['description']}")

        start_time = time.time()

        try:
            # Run the test script
            result = subprocess.run(
                [
                    sys.executable,
                    script_name,
                    "--output",
                    f"results_{script_name}.json",
                ],
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            duration_ms = (time.time() - start_time) * 1000

            # Try to load detailed results from output file
            results_file = f"results_{script_name}.json"
            detailed_results = None
            try:
                if Path(results_file).exists():
                    with open(results_file, "r") as f:
                        detailed_results = json.load(f)
            except Exception as e:
                print(f"   ⚠️ Could not load detailed results: {e}")

            test_result = {
                "test_name": test_name,
                "script": script_name,
                "success": result.returncode == 0,
                "return_code": result.returncode,
                "duration_ms": duration_ms,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "detailed_results": detailed_results,
                "timestamp": datetime.now().isoformat(),
            }

            # Extract key metrics from detailed results
            if detailed_results:
                # Common metrics extraction
                if "success_rate" in detailed_results:
                    test_result["success_rate"] = detailed_results["success_rate"]
                if (
                    "tests_passed" in detailed_results
                    and "tests_run" in detailed_results
                ):
                    test_result["tests_passed"] = detailed_results["tests_passed"]
                    test_result["tests_run"] = detailed_results["tests_run"]
                    if detailed_results["tests_run"] > 0:
                        test_result["success_rate"] = (
                            detailed_results["tests_passed"]
                            / detailed_results["tests_run"]
                        )

            if test_result["success"]:
                success_rate = test_result.get("success_rate", 1.0)
                print(f"   ✅ {test_name} completed successfully")
                if "success_rate" in test_result:
                    print(f"      Success Rate: {success_rate:.1%}")
                if "tests_passed" in test_result and "tests_run" in test_result:
                    print(
                        f"      Tests: {test_result['tests_passed']}/{test_result['tests_run']} passed"
                    )
                print(f"      Duration: {duration_ms:.0f}ms")
            else:
                print(f"   ❌ {test_name} failed (exit code: {result.returncode})")
                if result.stderr:
                    print(f"      Error: {result.stderr[:200]}...")

            return test_result

        except subprocess.TimeoutExpired:
            return {
                "test_name": test_name,
                "script": script_name,
                "success": False,
                "error": "Test timed out after 5 minutes",
                "duration_ms": (time.time() - start_time) * 1000,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            return {
                "test_name": test_name,
                "script": script_name,
                "success": False,
                "error": str(e),
                "duration_ms": (time.time() - start_time) * 1000,
                "timestamp": datetime.now().isoformat(),
            }

    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        print("\n📊 Generating comprehensive test report...")

        # Calculate overall metrics
        total_categories = len(self.test_results)
        successful_categories = sum(
            1 for result in self.test_results.values() if result.get("success", False)
        )

        # Aggregate test counts
        total_tests_run = 0
        total_tests_passed = 0

        for category_result in self.test_results.values():
            if "tests_run" in category_result:
                total_tests_run += category_result["tests_run"]
            if "tests_passed" in category_result:
                total_tests_passed += category_result["tests_passed"]

        # Calculate success rates
        category_success_rate = (
            successful_categories / total_categories if total_categories > 0 else 0
        )
        test_success_rate = (
            total_tests_passed / total_tests_run if total_tests_run > 0 else 0
        )

        # Determine overall success
        overall_success = category_success_rate >= 0.75 and test_success_rate >= 0.75

        self.overall_summary.update(
            {
                "total_categories": total_categories,
                "successful_categories": successful_categories,
                "category_success_rate": category_success_rate,
                "total_tests_run": total_tests_run,
                "total_tests_passed": total_tests_passed,
                "test_success_rate": test_success_rate,
                "overall_success": overall_success,
                "overall_success_rate": min(category_success_rate, test_success_rate),
            }
        )

        # Performance summary
        durations = [
            result.get("duration_ms", 0) for result in self.test_results.values()
        ]
        performance_summary = {
            "total_test_duration_ms": sum(durations),
            "average_category_duration_ms": (
                sum(durations) / len(durations) if durations else 0
            ),
            "fastest_category_ms": min(durations) if durations else 0,
            "slowest_category_ms": max(durations) if durations else 0,
        }

        # Error summary
        errors = []
        for category, result in self.test_results.items():
            if not result.get("success", False):
                errors.append(
                    {
                        "category": category,
                        "error": result.get("error", "Unknown error"),
                        "return_code": result.get("return_code"),
                    }
                )

        # Coverage analysis
        coverage_analysis = {
            "backend_coverage": "backend"
            in [r.get("script", "").lower() for r in self.test_results.values()],
            "frontend_coverage": "frontend"
            in [r.get("script", "").lower() for r in self.test_results.values()],
            "integration_coverage": "e2e"
            in [r.get("script", "").lower() for r in self.test_results.values()],
            "api_testing": any(
                "api" in str(r.get("detailed_results", {}))
                for r in self.test_results.values()
            ),
            "websocket_testing": any(
                "websocket" in str(r.get("detailed_results", {}))
                for r in self.test_results.values()
            ),
            "load_testing": any(
                "load" in str(r.get("detailed_results", {}))
                for r in self.test_results.values()
            ),
        }

        report = {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "system": "Quantum Trader",
                "test_environment": "Development",
                "python_version": sys.version,
            },
            "executive_summary": self.overall_summary,
            "performance_summary": performance_summary,
            "error_summary": errors,
            "coverage_analysis": coverage_analysis,
            "detailed_results": self.test_results,
            "recommendations": self.generate_recommendations(),
        }

        return report

    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []

        # Check individual test categories
        for category, result in self.test_results.items():
            if not result.get("success", False):
                recommendations.append(
                    f"🔧 Fix issues in {category}: {result.get('error', 'Unknown error')}"
                )
            elif result.get("success_rate", 1.0) < 0.9:
                recommendations.append(
                    f"⚠️ Improve reliability in {category} (current: {result.get('success_rate', 0):.1%})"
                )

        # Performance recommendations
        durations = [
            result.get("duration_ms", 0) for result in self.test_results.values()
        ]
        avg_duration = sum(durations) / len(durations) if durations else 0

        if avg_duration > 60000:  # > 1 minute average
            recommendations.append(
                "⚡ Consider optimizing test performance - average duration is > 1 minute"
            )

        # Coverage recommendations
        coverage_gaps = []
        if not any(
            "backend" in r.get("script", "").lower() for r in self.test_results.values()
        ):
            coverage_gaps.append("backend API testing")
        if not any(
            "frontend" in r.get("script", "").lower()
            for r in self.test_results.values()
        ):
            coverage_gaps.append("frontend component testing")
        if not any(
            "e2e" in r.get("script", "").lower() for r in self.test_results.values()
        ):
            coverage_gaps.append("end-to-end integration testing")

        if coverage_gaps:
            recommendations.append(
                f"📊 Add missing test coverage: {', '.join(coverage_gaps)}"
            )

        # Success recommendations
        overall_success_rate = self.overall_summary.get("overall_success_rate", 0)
        if overall_success_rate >= 0.95:
            recommendations.append(
                "🎉 Excellent test coverage and reliability! System is production-ready."
            )
        elif overall_success_rate >= 0.85:
            recommendations.append(
                "✅ Good test coverage. Address failing tests for production readiness."
            )
        elif overall_success_rate >= 0.75:
            recommendations.append(
                "⚠️ Moderate test coverage. Significant improvements needed before production."
            )
        else:
            recommendations.append(
                "❌ Poor test coverage. Major issues must be resolved before deployment."
            )

        return recommendations

    def run_all_system_tests(self) -> Dict[str, Any]:
        """Run complete system test suite."""
        print("🚀 Quantum Trader - Complete System Test Suite")
        print("=" * 70)

        self.overall_summary["start_time"] = datetime.now().isoformat()
        start_time = time.time()

        # Check prerequisites
        prereq_check = self.check_prerequisites()
        if not prereq_check["all_prerequisites_met"]:
            print("❌ Prerequisites not met. Cannot run tests.")
            return {
                "error": "Prerequisites not met",
                "prerequisite_check": prereq_check,
                "timestamp": datetime.now().isoformat(),
            }

        # Run each test category
        for test_info in self.test_scripts:
            try:
                result = self.run_test_script(test_info)
                self.test_results[test_info["name"]] = result
            except Exception as e:
                print(f"❌ Failed to run {test_info['name']}: {e}")
                self.test_results[test_info["name"]] = {
                    "test_name": test_info["name"],
                    "script": test_info["script"],
                    "success": False,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                }

        # Complete summary
        self.overall_summary["end_time"] = datetime.now().isoformat()
        self.overall_summary["total_duration_ms"] = (time.time() - start_time) * 1000

        # Generate comprehensive report
        report = self.generate_comprehensive_report()

        # Print summary
        self.print_test_summary(report)

        return report

    def print_test_summary(self, report: Dict[str, Any]):
        """Print formatted test summary."""
        print("\n" + "=" * 70)
        print("📊 SYSTEM TEST SUMMARY")
        print("=" * 70)

        summary = report["executive_summary"]

        # Overall results
        print(
            f"🎯 Overall Success: {'✅ PASS' if summary['overall_success'] else '❌ FAIL'}"
        )
        print(f"📈 Success Rate: {summary['overall_success_rate']:.1%}")
        print(f"⏱️ Total Duration: {summary['total_duration_ms']:.0f}ms")

        # Category breakdown
        print("\n📋 Test Categories:")
        print(f"   Categories Run: {summary['total_categories']}")
        print(f"   Categories Passed: {summary['successful_categories']}")
        print(f"   Category Success Rate: {summary['category_success_rate']:.1%}")

        # Individual test breakdown
        if "total_tests_run" in summary:
            print("\n🧪 Individual Tests:")
            print(f"   Tests Run: {summary['total_tests_run']}")
            print(f"   Tests Passed: {summary['total_tests_passed']}")
            print(f"   Test Success Rate: {summary['test_success_rate']:.1%}")

        # Performance metrics
        perf = report["performance_summary"]
        print("\n⚡ Performance:")
        print(
            f"   Average Category Duration: {perf['average_category_duration_ms']:.0f}ms"
        )
        print(f"   Fastest Category: {perf['fastest_category_ms']:.0f}ms")
        print(f"   Slowest Category: {perf['slowest_category_ms']:.0f}ms")

        # Errors (if any)
        if report["error_summary"]:
            print("\n❌ Errors Encountered:")
            for error in report["error_summary"]:
                print(f"   - {error['category']}: {error['error']}")

        # Coverage analysis
        coverage = report["coverage_analysis"]
        print("\n📊 Test Coverage:")
        coverage_items = [
            ("Backend API", coverage["backend_coverage"]),
            ("Frontend Components", coverage["frontend_coverage"]),
            ("E2E Integration", coverage["integration_coverage"]),
            ("WebSocket Testing", coverage["websocket_testing"]),
            ("Load Testing", coverage["load_testing"]),
        ]

        for item, covered in coverage_items:
            status = "✅" if covered else "❌"
            print(f"   {status} {item}")

        # Recommendations
        print("\n💡 Recommendations:")
        for rec in report["recommendations"]:
            print(f"   {rec}")

        print("\n" + "=" * 70)


def main():
    """Main system test execution."""
    import argparse

    parser = argparse.ArgumentParser(description="Complete System Test Runner")
    parser.add_argument("--output", help="Output file for comprehensive test report")
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")

    args = parser.parse_args()

    # Set verbosity
    if args.quiet:
        print("Running in quiet mode...")

    # Run complete system tests
    runner = SystemTestRunner()
    report = runner.run_all_system_tests()

    # Save report if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"📄 Comprehensive test report saved to: {args.output}")

    # Exit with appropriate code
    if report.get("error"):
        return 1

    success = report.get("executive_summary", {}).get("overall_success", False)
    if success:
        print("🎉 All system tests PASSED - System is ready for production!")
        return 0
    else:
        print("❌ System tests FAILED - Issues must be resolved before production")
        return 1


if __name__ == "__main__":
    exit(main())
