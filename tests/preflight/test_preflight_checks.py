"""
Pre-Flight Check Tests

EPIC-PREFLIGHT-001: Validate pre-flight check system functionality.

Tests verify registry, execution, and error handling without requiring
actual service dependencies.
"""

import pytest
from backend.preflight.types import PreflightResult
from backend.preflight.checks import register_check, run_all_preflight_checks, CHECKS


def test_preflight_result_creation():
    """Test PreflightResult dataclass creation."""
    result = PreflightResult(
        name="test_check",
        success=True,
        reason="ok",
        details={"key": "value"},
    )
    
    assert result.name == "test_check"
    assert result.success is True
    assert result.reason == "ok"
    assert result.details == {"key": "value"}


def test_preflight_result_str():
    """Test PreflightResult string formatting."""
    pass_result = PreflightResult(name="check1", success=True, reason="ok")
    fail_result = PreflightResult(name="check2", success=False, reason="failed")
    
    assert "✅" in str(pass_result)
    assert "❌" in str(fail_result)
    assert "check1" in str(pass_result)
    assert "check2" in str(fail_result)


def test_checks_registry_not_empty():
    """Test that CHECKS registry has registered checks."""
    # CHECKS should have at least 6 core checks registered in checks.py
    assert len(CHECKS) >= 6, "Expected at least 6 pre-flight checks registered"


def test_register_check_decorator():
    """Test that @register_check decorator adds function to registry."""
    initial_count = len(CHECKS)
    
    @register_check
    async def dummy_check() -> PreflightResult:
        return PreflightResult(name="dummy", success=True, reason="test")
    
    assert len(CHECKS) == initial_count + 1
    assert dummy_check in CHECKS


@pytest.mark.asyncio
async def test_run_all_preflight_checks_returns_list():
    """Test that run_all_preflight_checks returns a list of results."""
    results = await run_all_preflight_checks()
    
    assert isinstance(results, list)
    assert len(results) > 0
    
    for result in results:
        assert isinstance(result, PreflightResult)
        assert hasattr(result, "name")
        assert hasattr(result, "success")
        assert hasattr(result, "reason")


@pytest.mark.asyncio
async def test_exception_handling_in_check():
    """Test that exceptions in checks are caught and converted to failed results."""
    # Create a check that raises an exception
    @register_check
    async def failing_check() -> PreflightResult:
        raise ValueError("Test exception")
    
    results = await run_all_preflight_checks()
    
    # Find the failing_check result
    failing_result = next((r for r in results if r.name == "failing_check"), None)
    
    assert failing_result is not None
    assert failing_result.success is False
    assert "exception" in failing_result.reason.lower()
    assert "ValueError" in failing_result.reason
    assert failing_result.details is not None
    assert "error" in failing_result.details


@pytest.mark.asyncio
async def test_successful_check_result():
    """Test that successful checks return success=True."""
    @register_check
    async def passing_check() -> PreflightResult:
        return PreflightResult(
            name="passing_check",
            success=True,
            reason="all_good",
        )
    
    results = await run_all_preflight_checks()
    
    passing_result = next((r for r in results if r.name == "passing_check"), None)
    
    assert passing_result is not None
    assert passing_result.success is True
    assert passing_result.reason == "all_good"


@pytest.mark.asyncio
async def test_failed_check_result():
    """Test that failed checks return success=False."""
    @register_check
    async def blocking_check() -> PreflightResult:
        return PreflightResult(
            name="blocking_check",
            success=False,
            reason="something_wrong",
            details={"issue": "test failure"},
        )
    
    results = await run_all_preflight_checks()
    
    blocking_result = next((r for r in results if r.name == "blocking_check"), None)
    
    assert blocking_result is not None
    assert blocking_result.success is False
    assert blocking_result.reason == "something_wrong"
    assert blocking_result.details.get("issue") == "test failure"


@pytest.mark.asyncio
async def test_all_core_checks_execute():
    """Test that all core checks execute without crashing."""
    results = await run_all_preflight_checks()
    
    # Core check names (from checks.py)
    core_checks = [
        "check_health_endpoints",
        "check_risk_system",
        "check_exchange_connectivity",
        "check_database_redis",
        "check_observability",
        "check_stress_scenarios",
    ]
    
    result_names = [r.name for r in results]
    
    for check_name in core_checks:
        assert check_name in result_names, f"Core check {check_name} not executed"


def test_preflight_result_default_details():
    """Test that PreflightResult has empty dict as default details."""
    result = PreflightResult(name="test", success=True, reason="ok")
    
    assert result.details == {}
    assert isinstance(result.details, dict)
