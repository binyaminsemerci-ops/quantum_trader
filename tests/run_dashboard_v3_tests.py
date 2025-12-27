"""
Dashboard V3.0 - Comprehensive Test Suite Runner
DASHBOARD-V3-QA: Complete Test & Validation Suite

Runs all Dashboard V3.0 tests:
1. Backend API Contract Tests (7 files, 168 tests)
2. Integration Tests (2 files, 38 tests)
3. Frontend Component Tests (4 files, 69 tests)
4. E2E Tests (6 tests)

Usage:
    python run_dashboard_v3_tests.py              # Run all tests
    python run_dashboard_v3_tests.py --quick      # Quick smoke test
    python run_dashboard_v3_tests.py --api        # API tests only
    python run_dashboard_v3_tests.py --integration # Integration tests only
    python run_dashboard_v3_tests.py --frontend   # Frontend tests only
    python run_dashboard_v3_tests.py --e2e        # E2E tests only
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    RESET = '\033[0m'


def print_header(title: str):
    """Print section header"""
    print(f"\n{Colors.BLUE}{'='*70}{Colors.RESET}")
    print(f"{Colors.BLUE}{title}{Colors.RESET}")
    print(f"{Colors.BLUE}{'='*70}{Colors.RESET}\n")


def run_command(cmd: list, description: str) -> tuple[bool, int, int]:
    """Run a command and return (success, passed_count, failed_count)"""
    print(f"{Colors.CYAN}Running: {description}...{Colors.RESET}")
    print(f"{Colors.YELLOW}Command: {' '.join(cmd)}{Colors.RESET}\n")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )
        
        # Parse pytest output for test counts
        passed = 0
        failed = 0
        output = result.stdout + result.stderr
        
        # Look for pytest summary line (e.g., "10 passed, 2 failed")
        for line in output.split('\n'):
            if 'passed' in line.lower():
                import re
                match = re.search(r'(\d+)\s+passed', line)
                if match:
                    passed = int(match.group(1))
            if 'failed' in line.lower():
                import re
                match = re.search(r'(\d+)\s+failed', line)
                if match:
                    failed = int(match.group(1))
        
        if result.returncode == 0:
            print(f"\n{Colors.GREEN}✓ {description} PASSED ({passed} tests){Colors.RESET}\n")
            return True, passed, failed
        else:
            print(f"\n{Colors.RED}✗ {description} FAILED (exit code: {result.returncode}){Colors.RESET}\n")
            if failed > 0:
                print(f"{Colors.RED}Failed tests: {failed}{Colors.RESET}\n")
            return False, passed, failed
    
    except FileNotFoundError:
        print(f"\n{Colors.RED}✗ Command not found: {cmd[0]}{Colors.RESET}\n")
        return False, 0, 0
    except Exception as e:
        print(f"\n{Colors.RED}✗ Error running {description}: {str(e)}{Colors.RESET}\n")
        return False, 0, 0


def run_api_tests() -> Dict[str, tuple[bool, int, int]]:
    """Run all backend API contract tests"""
    print_header("Backend API Contract Tests (168 tests)")
    
    tests_dir = Path(__file__).parent / "api"
    results = {}
    
    api_test_files = [
        "test_dashboard_overview.py",
        "test_dashboard_trading.py",
        "test_dashboard_risk.py",
        "test_dashboard_system.py",
        "test_dashboard_stream.py",
        "test_dashboard_numeric_safety.py",
        "test_dashboard_logging.py"
    ]
    
    for test_file in api_test_files:
        test_path = tests_dir / test_file
        if test_path.exists():
            results[test_file] = run_command(
                ["pytest", str(test_path), "-v", "--tb=short"],
                f"API Test: {test_file}"
            )
        else:
            print(f"{Colors.YELLOW}⚠ Test file not found: {test_path}{Colors.RESET}\n")
            results[test_file] = (False, 0, 0)
    
    return results


def run_integration_tests() -> Dict[str, tuple[bool, int, int]]:
    """Run integration tests (Portfolio, Risk, ESS)"""
    print_header("Integration Tests (38 tests)")
    
    tests_dir = Path(__file__).parent / "integrations" / "dashboard"
    results = {}
    
    integration_test_files = [
        "test_portfolio_dashboard_integration.py",
        "test_risk_dashboard_integration.py"
    ]
    
    for test_file in integration_test_files:
        test_path = tests_dir / test_file
        if test_path.exists():
            results[test_file] = run_command(
                ["pytest", str(test_path), "-v", "--tb=short"],
                f"Integration Test: {test_file}"
            )
        else:
            print(f"{Colors.YELLOW}⚠ Test file not found: {test_path}{Colors.RESET}\n")
            results[test_file] = (False, 0, 0)
    
    return results


def run_frontend_tests() -> Dict[str, tuple[bool, int, int]]:
    """Run frontend component tests"""
    print_header("Frontend Component Tests (69 tests)")
    
    frontend_dir = Path(__file__).parent.parent / "frontend"
    results = {}
    
    # Check if Jest is configured
    jest_config = frontend_dir / "jest.config.js"
    if not jest_config.exists():
        print(f"{Colors.YELLOW}⚠ Jest not configured. See frontend/__tests__/README_TEST_SETUP.md{Colors.RESET}\n")
        results["frontend_setup"] = (False, 0, 0)
        return results
    
    # Run npm test
    success, passed, failed = run_command(
        ["npm", "test", "--", "--watchAll=false"],
        "Frontend Component Tests (Jest + RTL)"
    )
    results["frontend_components"] = (success, passed, failed)
    
    return results


def run_e2e_tests() -> Dict[str, tuple[bool, int, int]]:
    """Run E2E tests (Playwright)"""
    print_header("E2E Tests (6 tests)")
    
    tests_dir = Path(__file__).parent / "e2e"
    results = {}
    
    e2e_test_file = tests_dir / "test_dashboard_v3_e2e.py"
    
    if e2e_test_file.exists():
        print(f"{Colors.YELLOW}Note: E2E tests require frontend running on localhost:3000{Colors.RESET}\n")
        results["e2e"] = run_command(
            ["pytest", str(e2e_test_file), "-v", "-m", "e2e", "--tb=short"],
            "E2E Tests (Playwright)"
        )
    else:
        print(f"{Colors.YELLOW}⚠ Test file not found: {e2e_test_file}{Colors.RESET}\n")
        results["e2e"] = (False, 0, 0)
    
    return results


def run_quick_smoke_tests() -> Dict[str, tuple[bool, int, int]]:
    """Run quick smoke tests (1-2 tests per category)"""
    print_header("Quick Smoke Tests (~2 minutes)")
    
    tests_dir = Path(__file__).parent
    results = {}
    
    # API smoke: Just overview endpoint
    api_test = tests_dir / "api" / "test_dashboard_overview.py"
    if api_test.exists():
        results["api_smoke"] = run_command(
            ["pytest", str(api_test), "-k", "test_overview_schema_valid", "-v"],
            "API Smoke Test"
        )
    
    # Integration smoke: Just portfolio
    integration_test = tests_dir / "integrations" / "dashboard" / "test_portfolio_dashboard_integration.py"
    if integration_test.exists():
        results["integration_smoke"] = run_command(
            ["pytest", str(integration_test), "-k", "test_portfolio_data_flows", "-v"],
            "Integration Smoke Test"
        )
    
    return results


def main():
    """Run Dashboard V3.0 test suite"""
    parser = argparse.ArgumentParser(
        description="Dashboard V3.0 Comprehensive Test Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_dashboard_v3_tests.py              # Run all tests (~20 min)
    python run_dashboard_v3_tests.py --quick      # Quick smoke test (~2 min)
    python run_dashboard_v3_tests.py --api        # API tests only (~5 min)
    python run_dashboard_v3_tests.py --integration # Integration tests (~3 min)
    python run_dashboard_v3_tests.py --frontend   # Frontend tests (~4 min)
    python run_dashboard_v3_tests.py --e2e        # E2E tests (~5 min)
        """
    )
    
    parser.add_argument('--quick', action='store_true', help='Run quick smoke tests only')
    parser.add_argument('--api', action='store_true', help='Run API contract tests only')
    parser.add_argument('--integration', action='store_true', help='Run integration tests only')
    parser.add_argument('--frontend', action='store_true', help='Run frontend tests only')
    parser.add_argument('--e2e', action='store_true', help='Run E2E tests only')
    
    args = parser.parse_args()
    
    print_header("Dashboard V3.0 - Comprehensive Test Suite")
    
    all_results = {}
    
    # Determine which tests to run
    if args.quick:
        all_results.update(run_quick_smoke_tests())
    elif args.api:
        all_results.update(run_api_tests())
    elif args.integration:
        all_results.update(run_integration_tests())
    elif args.frontend:
        all_results.update(run_frontend_tests())
    elif args.e2e:
        all_results.update(run_e2e_tests())
    else:
        # Run all tests
        all_results.update(run_api_tests())
        all_results.update(run_integration_tests())
        all_results.update(run_frontend_tests())
        all_results.update(run_e2e_tests())
    
    # Print Final Summary
    print_header("Final Test Summary")
    
    total_tests = sum(passed + failed for _, passed, failed in all_results.values())
    total_passed = sum(passed for _, passed, _ in all_results.values())
    total_failed = sum(failed for _, _, failed in all_results.values())
    
    suites_passed = sum(1 for success, _, _ in all_results.values() if success)
    suites_total = len(all_results)
    suites_failed = suites_total - suites_passed
    
    print(f"{Colors.CYAN}Test Suite Results:{Colors.RESET}\n")
    
    for name, (success, passed, failed) in all_results.items():
        status = f"{Colors.GREEN}✓ PASS{Colors.RESET}" if success else f"{Colors.RED}✗ FAIL{Colors.RESET}"
        test_summary = f"({passed} passed, {failed} failed)" if (passed + failed) > 0 else ""
        print(f"  {status}  {name:<45} {test_summary}")
    
    print(f"\n{Colors.CYAN}Overall Test Results:{Colors.RESET}")
    print(f"  {Colors.GREEN}Tests Passed: {total_passed}/{total_tests}{Colors.RESET}")
    if total_failed > 0:
        print(f"  {Colors.RED}Tests Failed: {total_failed}/{total_tests}{Colors.RESET}")
    
    print(f"\n{Colors.CYAN}Test Suite Summary:{Colors.RESET}")
    print(f"  {Colors.GREEN}Suites Passed: {suites_passed}/{suites_total}{Colors.RESET}")
    if suites_failed > 0:
        print(f"  {Colors.RED}Suites Failed: {suites_failed}/{suites_total}{Colors.RESET}")
    
    print()
    
    if suites_failed == 0 and total_failed == 0:
        print(f"{Colors.GREEN}{'='*70}{Colors.RESET}")
        print(f"{Colors.GREEN}✓ ALL TESTS PASSED ({total_passed}/{total_tests}){Colors.RESET}")
        print(f"{Colors.GREEN}✓ Dashboard V3.0 is production ready!{Colors.RESET}")
        print(f"{Colors.GREEN}{'='*70}{Colors.RESET}\n")
        
        print(f"{Colors.CYAN}Next Steps:{Colors.RESET}")
        print("  1. Review manual checklist: tests/MANUAL_TEST_CHECKLIST.md")
        print("  2. Deploy to staging environment")
        print("  3. Run E2E tests against staging")
        print("  4. Get QA sign-off from manual checklist")
        print()
        
        sys.exit(0)
    else:
        print(f"{Colors.RED}{'='*70}{Colors.RESET}")
        print(f"{Colors.RED}✗ SOME TESTS FAILED{Colors.RESET}")
        print(f"{Colors.RED}✗ {total_failed} test(s) failed across {suites_failed} suite(s){Colors.RESET}")
        print(f"{Colors.RED}{'='*70}{Colors.RESET}\n")
        
        # Provide helpful hints
        print(f"{Colors.YELLOW}Troubleshooting:{Colors.RESET}")
        print("  • Review test output above for specific failures")
        print("  • Check that all services are running:")
        print("    - Backend API: http://localhost:8000/health")
        print("    - Frontend: http://localhost:3000")
        print("    - Portfolio Service: http://localhost:8004/health")
        print("  • Ensure test dependencies installed:")
        print("    - Backend: pip install pytest pytest-asyncio")
        print("    - Frontend: npm install (see frontend/__tests__/README_TEST_SETUP.md)")
        print("    - E2E: pip install pytest-playwright && playwright install")
        print()
        
        sys.exit(1)


if __name__ == "__main__":
    main()
