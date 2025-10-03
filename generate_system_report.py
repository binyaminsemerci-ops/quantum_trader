#!/usr/bin/env python3
"""
System Test Summary Report

Generates a comprehensive summary of backend and frontend system tests.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List


class SystemTestSummaryGenerator:
    """Generate comprehensive system test summary."""

    def __init__(self):
        self.results: Dict[str, Any] = {}

    def load_test_results(self):
        """Load available test results."""
        result_files = [
            ("backend", "backend_test_results.json"),
            ("frontend", "frontend_test_results.json"),
            ("e2e", "e2e_test_results.json"),
            ("system", "system_test_report.json"),
        ]

        for category, filename in result_files:
            file_path = Path(filename)
            if file_path.exists():
                try:
                    with open(file_path, "r") as f:
                        self.results[category] = json.load(f)
                    print(f"✅ Loaded {category} test results from {filename}")
                except Exception as e:
                    print(f"⚠️ Could not load {filename}: {e}")
                    self.results[category] = {"error": f"Failed to load: {e}"}
            else:
                print(f"❌ Test results not found: {filename}")
                self.results[category] = {"error": "Results file not found"}

    def check_system_components(self) -> Dict[str, Any]:
        """Check system components availability."""
        components = {
            "backend_directory": Path("backend").exists(),
            "frontend_directory": Path("frontend").exists(),
            "backend_main": Path("backend/main.py").exists(),
            "backend_simple": Path("backend/simple_main.py").exists(),
            "frontend_package": Path("frontend/package.json").exists(),
            "frontend_vite": Path("frontend/vite.config.tsx").exists(),
            "database": Path("backend/quantum_trader.db").exists(),
            "requirements_backend": Path("backend/requirements.txt").exists(),
            "requirements_root": Path("requirements.txt").exists(),
        }

        return {
            "components": components,
            "backend_ready": components["backend_directory"]
            and (components["backend_main"] or components["backend_simple"]),
            "frontend_ready": components["frontend_directory"]
            and components["frontend_package"],
            "database_ready": components["database"],
            "overall_structure": sum(components.values()) / len(components),
        }

    def analyze_test_coverage(self) -> Dict[str, Any]:
        """Analyze test coverage across system."""
        coverage = {
            "backend_api_tested": False,
            "backend_health_tested": False,
            "frontend_structure_tested": False,
            "frontend_build_tested": False,
            "e2e_integration_tested": False,
            "websocket_tested": False,
            "load_testing_performed": False,
            "error_handling_tested": False,
        }

        # Check backend coverage
        if "backend" in self.results and self.results["backend"].get(
            "detailed_results"
        ):
            for result in self.results["backend"]["detailed_results"]:
                if result.get("test_name") == "health_endpoint":
                    coverage["backend_health_tested"] = True
                if result.get("test_name") == "api_endpoints":
                    coverage["backend_api_tested"] = True

        # Check frontend coverage
        if "frontend" in self.results and self.results["frontend"].get(
            "detailed_results"
        ):
            for result in self.results["frontend"]["detailed_results"]:
                if "files" in result.get("test_name", ""):
                    coverage["frontend_structure_tested"] = True
                if "package" in result.get("test_name", ""):
                    coverage["frontend_build_tested"] = True

        # Check E2E coverage
        if "e2e" in self.results:
            coverage["e2e_integration_tested"] = True
            # Look for specific E2E test types
            e2e_data = self.results["e2e"]
            if isinstance(e2e_data, dict) and "detailed_results" in e2e_data:
                for result in e2e_data["detailed_results"]:
                    test_name = result.get("test_name", "").lower()
                    if "websocket" in test_name:
                        coverage["websocket_tested"] = True
                    if "load" in test_name:
                        coverage["load_testing_performed"] = True
                    if "error" in test_name:
                        coverage["error_handling_tested"] = True

        coverage_score = sum(coverage.values()) / len(coverage)

        return {
            "coverage_details": coverage,
            "coverage_score": coverage_score,
            "coverage_percentage": coverage_score * 100,
        }

    def calculate_system_health(self) -> Dict[str, Any]:
        """Calculate overall system health score."""
        scores = []

        # Component structure score
        components = self.check_system_components()
        structure_score = components["overall_structure"]
        scores.append(("structure", structure_score, 0.3))  # 30% weight

        # Test success scores
        test_categories = ["backend", "frontend", "e2e"]
        test_scores = []

        for category in test_categories:
            if category in self.results:
                result = self.results[category]
                if "success_rate" in result:
                    test_scores.append(result["success_rate"])
                elif result.get("error"):
                    test_scores.append(0.0)  # Failed to run
                else:
                    test_scores.append(0.5)  # Partial/unknown
            else:
                test_scores.append(0.0)  # Not tested

        avg_test_score = sum(test_scores) / len(test_scores) if test_scores else 0
        scores.append(("testing", avg_test_score, 0.4))  # 40% weight

        # Coverage score
        coverage = self.analyze_test_coverage()
        coverage_score = coverage["coverage_score"]
        scores.append(("coverage", coverage_score, 0.3))  # 30% weight

        # Calculate weighted average
        weighted_score = sum(score * weight for _, score, weight in scores)

        # Determine health category
        if weighted_score >= 0.8:
            health_category = "Excellent"
            health_emoji = "🟢"
        elif weighted_score >= 0.6:
            health_category = "Good"
            health_emoji = "🟡"
        elif weighted_score >= 0.4:
            health_category = "Fair"
            health_emoji = "🟠"
        else:
            health_category = "Poor"
            health_emoji = "🔴"

        return {
            "overall_score": weighted_score,
            "score_percentage": weighted_score * 100,
            "health_category": health_category,
            "health_emoji": health_emoji,
            "component_scores": dict((name, score) for name, score, _ in scores),
            "recommendations": self.generate_health_recommendations(
                weighted_score, components, coverage
            ),
        }

    def generate_health_recommendations(
        self, overall_score: float, components: Dict[str, Any], coverage: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on system health."""
        recommendations = []

        # Structure recommendations
        if not components["backend_ready"]:
            recommendations.append(
                "🔧 Fix backend structure: Ensure backend directory has working main.py or simple_main.py"
            )

        if not components["frontend_ready"]:
            recommendations.append(
                "🔧 Fix frontend structure: Ensure frontend directory has valid package.json"
            )

        if not components["database_ready"]:
            recommendations.append("🔧 Initialize database: Run database setup scripts")

        # Testing recommendations
        test_gaps = []
        coverage_details = coverage.get("coverage_details", {})

        if not coverage_details.get("backend_api_tested"):
            test_gaps.append("backend API testing")
        if not coverage_details.get("frontend_build_tested"):
            test_gaps.append("frontend build testing")
        if not coverage_details.get("e2e_integration_tested"):
            test_gaps.append("end-to-end integration testing")
        if not coverage_details.get("websocket_tested"):
            test_gaps.append("WebSocket functionality testing")
        if not coverage_details.get("load_testing_performed"):
            test_gaps.append("load/performance testing")

        if test_gaps:
            recommendations.append(
                f"📊 Add missing test coverage: {', '.join(test_gaps)}"
            )

        # Score-based recommendations
        if overall_score >= 0.8:
            recommendations.append(
                "🎉 Excellent system health! Ready for production deployment."
            )
        elif overall_score >= 0.6:
            recommendations.append(
                "✅ Good system health. Address minor issues for production readiness."
            )
        elif overall_score >= 0.4:
            recommendations.append(
                "⚠️ Fair system health. Significant improvements needed before production."
            )
        else:
            recommendations.append(
                "❌ Poor system health. Major issues must be resolved before deployment."
            )

        # Specific technical recommendations
        if "backend" in self.results:
            backend_result = self.results["backend"]
            if backend_result.get("success_rate", 0) < 0.5:
                recommendations.append(
                    "🔧 Fix backend connectivity issues - server may not be starting properly"
                )

        if "frontend" in self.results:
            frontend_result = self.results["frontend"]
            # Check for npm issues
            if "detailed_results" in frontend_result:
                for result in frontend_result["detailed_results"]:
                    if result.get("test_name") == "npm_available" and not result.get(
                        "success"
                    ):
                        recommendations.append(
                            "🛠️ Install Node.js and npm for frontend development"
                        )

        return recommendations

    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive system test report."""
        print("📊 Generating System Test Summary Report...")

        # Load all available test results
        self.load_test_results()

        # Analyze system components
        components = self.check_system_components()

        # Analyze test coverage
        coverage = self.analyze_test_coverage()

        # Calculate system health
        health = self.calculate_system_health()

        # Compile individual test summaries
        test_summaries = {}
        for category, results in self.results.items():
            if "error" not in results:
                test_summaries[category] = {
                    "tests_run": results.get("tests_run", 0),
                    "tests_passed": results.get("tests_passed", 0),
                    "success_rate": results.get("success_rate", 0),
                    "duration_ms": results.get(
                        "total_duration_ms", results.get("test_duration_ms", 0)
                    ),
                    "status": (
                        "✅ PASS"
                        if results.get("success_rate", 0) >= 0.5
                        else "❌ FAIL"
                    ),
                }
            else:
                test_summaries[category] = {
                    "status": "❌ ERROR",
                    "error": results["error"],
                }

        report = {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "system": "Quantum Trader",
                "report_type": "System Health & Test Summary",
            },
            "system_health": health,
            "component_analysis": components,
            "test_coverage": coverage,
            "test_summaries": test_summaries,
            "detailed_results": self.results,
        }

        return report

    def print_report(self, report: Dict[str, Any]):
        """Print formatted system test report."""
        print("\n" + "=" * 70)
        print("📊 QUANTUM TRADER SYSTEM TEST SUMMARY")
        print("=" * 70)

        # System health overview
        health = report["system_health"]
        print(
            f"\n{health['health_emoji']} OVERALL SYSTEM HEALTH: {health['health_category'].upper()}"
        )
        print(f"📈 Health Score: {health['score_percentage']:.1f}%")

        # Component status
        components = report["component_analysis"]
        print("\n🏗️ SYSTEM COMPONENTS:")
        print(f"   Backend Ready: {'✅' if components['backend_ready'] else '❌'}")
        print(f"   Frontend Ready: {'✅' if components['frontend_ready'] else '❌'}")
        print(f"   Database Ready: {'✅' if components['database_ready'] else '❌'}")
        print(f"   Structure Score: {components['overall_structure']:.1%}")

        # Test coverage
        coverage = report["test_coverage"]
        print("\n📊 TEST COVERAGE:")
        print(f"   Coverage Score: {coverage['coverage_percentage']:.1f}%")

        coverage_items = [
            ("Backend API", coverage["coverage_details"]["backend_api_tested"]),
            (
                "Frontend Structure",
                coverage["coverage_details"]["frontend_structure_tested"],
            ),
            ("E2E Integration", coverage["coverage_details"]["e2e_integration_tested"]),
            ("WebSocket", coverage["coverage_details"]["websocket_tested"]),
            ("Load Testing", coverage["coverage_details"]["load_testing_performed"]),
        ]

        for item, tested in coverage_items:
            status = "✅" if tested else "❌"
            print(f"   {status} {item}")

        # Test results summary
        test_summaries = report["test_summaries"]
        print("\n🧪 TEST RESULTS:")

        for category, summary in test_summaries.items():
            print(f"   {summary['status']} {category.title()}:")
            if "error" in summary:
                print(f"       Error: {summary['error']}")
            else:
                print(
                    f"       Tests: {summary['tests_passed']}/{summary['tests_run']} passed ({summary['success_rate']:.1%})"
                )
                if summary.get("duration_ms"):
                    print(f"       Duration: {summary['duration_ms']:.0f}ms")

        # Recommendations
        print("\n💡 RECOMMENDATIONS:")
        for rec in health["recommendations"]:
            print(f"   {rec}")

        print("\n" + "=" * 70)


def main():
    """Main report generation."""
    import argparse

    parser = argparse.ArgumentParser(description="System Test Summary Report Generator")
    parser.add_argument("--output", help="Output file for comprehensive report")

    args = parser.parse_args()

    # Generate report
    generator = SystemTestSummaryGenerator()
    report = generator.generate_comprehensive_report()

    # Print report
    generator.print_report(report)

    # Save report if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"📄 Comprehensive report saved to: {args.output}")

    # Exit with appropriate code based on system health
    health_score = report["system_health"]["overall_score"]
    if health_score >= 0.6:
        print("✅ System health acceptable for development/testing")
        return 0
    else:
        print("❌ System health needs improvement")
        return 1


if __name__ == "__main__":
    exit(main())
