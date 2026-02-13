"""
🧪 FAILURE SCENARIO TEST CASES (INSTITUSJONELT NIVÅ)
=====================================================
Standard format for hver test:
- ID
- Trigger
- Forventet respons
- Akseptkriterier
- Fail hvis...

Disse test-cases kan kjøres før live for å verifisere at
alle failure scenarios håndteres korrekt.
"""

import pytest
import asyncio
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# TEST CASE DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

class TestSeverity(str, Enum):
    """Test severity levels matching failure classes."""
    CRITICAL = "🔴"      # Class A - Kill-switch
    SERIOUS = "🟠"       # Class B - Pause
    MODERATE = "🟡"      # Class C - Exit-only
    EXTERNAL = "🔵"      # Class E - Infrastructure


@dataclass
class FailureTestCase:
    """Definition of a failure scenario test case."""
    id: str
    name: str
    severity: TestSeverity
    trigger_description: str
    expected_response: str
    acceptance_criteria: List[str]
    fail_if: str
    setup_fn: Optional[Callable] = None
    trigger_fn: Optional[Callable] = None
    verify_fn: Optional[Callable] = None


# ═══════════════════════════════════════════════════════════════════════════════
# ALL TEST CASES
# ═══════════════════════════════════════════════════════════════════════════════

TEST_CASES: Dict[str, FailureTestCase] = {
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 🔴 CRITICAL (Class A) - Kill-Switch Level
    # ═══════════════════════════════════════════════════════════════════════════
    
    "TC-A1": FailureTestCase(
        id="TC-A1",
        name="Menneskelig override",
        severity=TestSeverity.CRITICAL,
        trigger_description="Forsøk på manuell endring av size/stop i live",
        expected_response="Umiddelbar freeze + cooldown",
        acceptance_criteria=[
            "Ingen nye orders aksepteres",
            "Full audit-logg genereres",
            "Cooldown timer starter",
            "System state = FROZEN",
        ],
        fail_if="Endring slipper gjennom",
    ),
    
    "TC-A2": FailureTestCase(
        id="TC-A2",
        name="Risiko per trade overskrides",
        severity=TestSeverity.CRITICAL,
        trigger_description="Size × SL > policy-grense (2%)",
        expected_response="Trade avvises",
        acceptance_criteria=[
            "Ingen ordre sendt til exchange",
            "Violation logges",
            "Risk kernel blokkerer",
        ],
        fail_if="Ordre når execution",
    ),
    
    "TC-A3": FailureTestCase(
        id="TC-A3",
        name="Daglig tapsgrense",
        severity=TestSeverity.CRITICAL,
        trigger_description="PnL ≤ daglig maks tap (-5%)",
        expected_response="Kill-switch aktiveres",
        acceptance_criteria=[
            "Kun reduce-only modus",
            "Nye entries blokkert",
            "Critical incident logges",
            "Cooldown før restart",
        ],
        fail_if="Ny entry tillates",
    ),
    
    "TC-A4": FailureTestCase(
        id="TC-A4",
        name="Data-inkonsistens",
        severity=TestSeverity.CRITICAL,
        trigger_description="Manglende ticks / klokke-drift > 1s",
        expected_response="SAFE MODE",
        acceptance_criteria=[
            "Entry blokkert",
            "Exits tillatt",
            "Data re-synk initieres",
            "System state = SAFE_MODE",
        ],
        fail_if="Entry gjennomføres",
    ),
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 🟠 SERIOUS (Class B) - Strategic Pause
    # ═══════════════════════════════════════════════════════════════════════════
    
    "TC-B1": FailureTestCase(
        id="TC-B1",
        name="Regime-usikkerhet",
        severity=TestSeverity.SERIOUS,
        trigger_description="Motstridende regimesignaler",
        expected_response="Pause trading",
        acceptance_criteria=[
            "Ingen nye entries",
            "Observer-only modus",
            "Regime detector logger usikkerhet",
        ],
        fail_if="System trader videre",
    ),
    
    "TC-B2": FailureTestCase(
        id="TC-B2",
        name="Edge-forfall",
        severity=TestSeverity.SERIOUS,
        trigger_description="Negativ expectancy over N trades",
        expected_response="Strategi fryses",
        acceptance_criteria=[
            "Shadow-krav før live restart",
            "Strategi flagges for review",
            "Performance metrics logges",
        ],
        fail_if="Live trading fortsetter",
    ),
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 🟡 MODERATE (Class C) - Exit-Only
    # ═══════════════════════════════════════════════════════════════════════════
    
    "TC-C1": FailureTestCase(
        id="TC-C1",
        name="Tid i trade for lang",
        severity=TestSeverity.MODERATE,
        trigger_description="Holdetid > policy (24h)",
        expected_response="Tvungen exit",
        acceptance_criteria=[
            "Posisjon reduseres/lukkes",
            "Time-stop logges",
            "Funding-kostnad beregnes",
        ],
        fail_if="Trade holdes videre",
    ),
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 🔵 EXTERNAL (Class E) - Infrastructure
    # ═══════════════════════════════════════════════════════════════════════════
    
    "TC-E1": FailureTestCase(
        id="TC-E1",
        name="Exchange-anomali",
        severity=TestSeverity.EXTERNAL,
        trigger_description="API-feil / balance mismatch",
        expected_response="Isoler exchange",
        acceptance_criteria=[
            "Flat-mandat",
            "Ingen nye ordrer forsøkes",
            "Midler verifiseres",
            "Exchange health logged",
        ],
        fail_if="Ordre forsøkes mot ustabil exchange",
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# TEST RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TestResult:
    """Result of a single test case execution."""
    test_id: str
    passed: bool
    message: str
    duration_ms: float
    acceptance_results: Dict[str, bool] = field(default_factory=dict)
    error: Optional[str] = None


class FailureTestRunner:
    """
    Runs failure scenario tests against the governance system.
    
    Usage:
        runner = FailureTestRunner()
        results = await runner.run_all_tests()
        runner.print_report(results)
    """
    
    def __init__(self):
        self._results: List[TestResult] = []
        self._governance_initialized = False
        
    async def setup_governance(self):
        """Initialize governance components for testing."""
        try:
            from backend.domains.governance import (
                get_failure_monitor,
                get_grunnlov_registry,
                get_restart_manager,
            )
            
            self._failure_monitor = get_failure_monitor()
            self._grunnlov_registry = get_grunnlov_registry()
            self._restart_manager = get_restart_manager()
            self._governance_initialized = True
            
            logger.info("✅ Governance components initialized for testing")
            
        except ImportError as e:
            logger.warning(f"⚠️ Could not import governance: {e}")
            self._governance_initialized = False
    
    async def run_test(self, test_case: FailureTestCase) -> TestResult:
        """Run a single test case."""
        start_time = datetime.utcnow()
        
        try:
            # Run test-specific logic
            if test_case.id == "TC-A1":
                passed, acceptance = await self._test_human_override()
            elif test_case.id == "TC-A2":
                passed, acceptance = await self._test_risk_per_trade()
            elif test_case.id == "TC-A3":
                passed, acceptance = await self._test_daily_loss_limit()
            elif test_case.id == "TC-A4":
                passed, acceptance = await self._test_data_inconsistency()
            elif test_case.id == "TC-B1":
                passed, acceptance = await self._test_regime_uncertainty()
            elif test_case.id == "TC-B2":
                passed, acceptance = await self._test_edge_decay()
            elif test_case.id == "TC-C1":
                passed, acceptance = await self._test_time_in_trade()
            elif test_case.id == "TC-E1":
                passed, acceptance = await self._test_exchange_anomaly()
            else:
                passed = False
                acceptance = {"unknown_test": False}
            
            duration = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return TestResult(
                test_id=test_case.id,
                passed=passed,
                message="PASS" if passed else f"FAIL: {test_case.fail_if}",
                duration_ms=duration,
                acceptance_results=acceptance,
            )
            
        except Exception as e:
            duration = (datetime.utcnow() - start_time).total_seconds() * 1000
            return TestResult(
                test_id=test_case.id,
                passed=False,
                message=f"ERROR: {str(e)}",
                duration_ms=duration,
                error=str(e),
            )
    
    async def run_all_tests(self) -> List[TestResult]:
        """Run all test cases."""
        await self.setup_governance()
        
        results = []
        for test_id, test_case in TEST_CASES.items():
            logger.info(f"Running {test_case.severity.value} {test_id}: {test_case.name}")
            result = await self.run_test(test_case)
            results.append(result)
            
            status = "✅ PASS" if result.passed else "❌ FAIL"
            logger.info(f"  {status} ({result.duration_ms:.1f}ms)")
        
        self._results = results
        return results
    
    # ═══════════════════════════════════════════════════════════════════════════
    # INDIVIDUAL TEST IMPLEMENTATIONS
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def _test_human_override(self) -> tuple[bool, Dict[str, bool]]:
        """TC-A1: Test human override detection and freeze."""
        acceptance = {}
        
        if not self._governance_initialized:
            return False, {"governance_not_initialized": False}
        
        try:
            # Simulate human override attempt
            from backend.domains.governance import (
                FailureScenario,
                SystemState,
            )
            
            # Trigger the failure
            success, _ = self._failure_monitor.trigger_failure(
                FailureScenario.D1_MANUAL_OVERRIDE_ATTEMPT,
                {"action": "manual_stop_loss_change", "user": "test"}
            )
            acceptance["failure_triggered"] = success
            
            # Check system state is FROZEN
            acceptance["state_is_frozen"] = (
                self._failure_monitor.current_state == SystemState.FROZEN
            )
            
            # Check trading is blocked
            acceptance["trading_blocked"] = not self._failure_monitor.can_open_positions
            
            # Resolve for next test
            self._failure_monitor.resolve_failure(
                FailureScenario.D1_MANUAL_OVERRIDE_ATTEMPT,
                "Test cleanup"
            )
            
            return all(acceptance.values()), acceptance
            
        except Exception as e:
            return False, {"error": False, "message": str(e)}
    
    async def _test_risk_per_trade(self) -> tuple[bool, Dict[str, bool]]:
        """TC-A2: Test risk per trade validation."""
        acceptance = {}
        
        if not self._governance_initialized:
            return False, {"governance_not_initialized": False}
        
        try:
            # Test risk kernel with excessive risk
            risk_kernel = self._grunnlov_registry.risk_kernel
            
            # Try 5% risk (should fail, max is 2%)
            allowed, reason = risk_kernel.check_trade_risk(
                risk_pct=5.0,  # 5% > 2% max
                leverage=10.0,
                position_pct=50.0,
            )
            acceptance["excessive_risk_blocked"] = not allowed
            
            # Try valid risk (should pass)
            allowed, _ = risk_kernel.check_trade_risk(
                risk_pct=1.5,  # 1.5% < 2% max
                leverage=5.0,
                position_pct=20.0,
            )
            acceptance["valid_risk_allowed"] = allowed
            
            return all(acceptance.values()), acceptance
            
        except Exception as e:
            return False, {"error": False, "message": str(e)}
    
    async def _test_daily_loss_limit(self) -> tuple[bool, Dict[str, bool]]:
        """TC-A3: Test daily loss limit kill-switch."""
        acceptance = {}
        
        if not self._governance_initialized:
            return False, {"governance_not_initialized": False}
        
        try:
            from backend.domains.governance import FailureScenario, SystemState
            
            # Trigger capital breach with -6% daily loss
            success, _ = self._failure_monitor.trigger_failure(
                FailureScenario.A1_CAPITAL_BREACH,
                {"daily_pnl_pct": -6.0, "reason": "Test daily loss"}
            )
            acceptance["failure_triggered"] = success
            
            # Check reduce-only mode
            acceptance["can_close"] = self._failure_monitor.can_close_positions
            acceptance["cannot_open"] = not self._failure_monitor.can_open_positions
            
            # Cleanup
            self._failure_monitor.resolve_failure(
                FailureScenario.A1_CAPITAL_BREACH,
                "Test cleanup"
            )
            
            return all(acceptance.values()), acceptance
            
        except Exception as e:
            return False, {"error": False, "message": str(e)}
    
    async def _test_data_inconsistency(self) -> tuple[bool, Dict[str, bool]]:
        """TC-A4: Test data inconsistency safe mode."""
        acceptance = {}
        
        if not self._governance_initialized:
            return False, {"governance_not_initialized": False}
        
        try:
            from backend.domains.governance import FailureScenario, SystemState
            
            # Trigger data collapse
            success, _ = self._failure_monitor.trigger_failure(
                FailureScenario.A3_DATA_COLLAPSE,
                {"latency_exceeded": True, "latency_ms": 6000}
            )
            acceptance["failure_triggered"] = success
            
            # Check safe mode
            state = self._failure_monitor.current_state
            acceptance["in_safe_mode"] = state == SystemState.SAFE_MODE
            
            # Exits should be allowed, entries blocked
            acceptance["can_exit"] = self._failure_monitor.can_close_positions
            acceptance["cannot_enter"] = not self._failure_monitor.can_open_positions
            
            # Cleanup
            self._failure_monitor.resolve_failure(
                FailureScenario.A3_DATA_COLLAPSE,
                "Test cleanup"
            )
            
            return all(acceptance.values()), acceptance
            
        except Exception as e:
            return False, {"error": False, "message": str(e)}
    
    async def _test_regime_uncertainty(self) -> tuple[bool, Dict[str, bool]]:
        """TC-B1: Test regime uncertainty pause."""
        acceptance = {}
        
        if not self._governance_initialized:
            return False, {"governance_not_initialized": False}
        
        try:
            from backend.domains.governance import FailureScenario, SystemState
            
            # Trigger regime uncertainty
            success, _ = self._failure_monitor.trigger_failure(
                FailureScenario.B1_REGIME_UNCERTAINTY,
                {"conflicting_signals": True, "confidence": 0.45}
            )
            acceptance["failure_triggered"] = success
            
            # Check pause/safe mode
            state = self._failure_monitor.current_state
            acceptance["in_safe_or_paused"] = state in [SystemState.SAFE_MODE, SystemState.PAUSED]
            
            # No new entries
            acceptance["entries_blocked"] = not self._failure_monitor.can_open_positions
            
            # Cleanup
            self._failure_monitor.resolve_failure(
                FailureScenario.B1_REGIME_UNCERTAINTY,
                "Test cleanup"
            )
            
            return all(acceptance.values()), acceptance
            
        except Exception as e:
            return False, {"error": False, "message": str(e)}
    
    async def _test_edge_decay(self) -> tuple[bool, Dict[str, bool]]:
        """TC-B2: Test edge decay strategy freeze."""
        acceptance = {}
        
        if not self._governance_initialized:
            return False, {"governance_not_initialized": False}
        
        try:
            from backend.domains.governance import FailureScenario, SystemState
            
            # Trigger edge decay
            success, _ = self._failure_monitor.trigger_failure(
                FailureScenario.B2_EDGE_DECAY,
                {"performance_drift": -15.0, "trades_analyzed": 50}
            )
            acceptance["failure_triggered"] = success
            
            # Check paused state
            state = self._failure_monitor.current_state
            acceptance["is_paused"] = state == SystemState.PAUSED
            
            # Cleanup
            self._failure_monitor.resolve_failure(
                FailureScenario.B2_EDGE_DECAY,
                "Test cleanup"
            )
            
            return all(acceptance.values()), acceptance
            
        except Exception as e:
            return False, {"error": False, "message": str(e)}
    
    async def _test_time_in_trade(self) -> tuple[bool, Dict[str, bool]]:
        """TC-C1: Test time-based stress forced exit."""
        acceptance = {}
        
        if not self._governance_initialized:
            return False, {"governance_not_initialized": False}
        
        try:
            from backend.domains.governance import FailureScenario, SystemState
            
            # Trigger time-based stress
            success, _ = self._failure_monitor.trigger_failure(
                FailureScenario.C2_TIME_BASED_STRESS,
                {"trade_duration_hours": 30, "pnl_stagnant": True}
            )
            acceptance["failure_triggered"] = success
            
            # Check reduce-only
            state = self._failure_monitor.current_state
            acceptance["is_reduce_only"] = state == SystemState.REDUCE_ONLY
            
            # Cleanup
            self._failure_monitor.resolve_failure(
                FailureScenario.C2_TIME_BASED_STRESS,
                "Test cleanup"
            )
            
            return all(acceptance.values()), acceptance
            
        except Exception as e:
            return False, {"error": False, "message": str(e)}
    
    async def _test_exchange_anomaly(self) -> tuple[bool, Dict[str, bool]]:
        """TC-E1: Test exchange anomaly isolation."""
        acceptance = {}
        
        if not self._governance_initialized:
            return False, {"governance_not_initialized": False}
        
        try:
            from backend.domains.governance import FailureScenario, SystemState
            
            # Trigger exchange risk
            success, _ = self._failure_monitor.trigger_failure(
                FailureScenario.E1_EXCHANGE_RISK,
                {"api_error": True, "balance_discrepancy": 50.0}
            )
            acceptance["failure_triggered"] = success
            
            # Check frozen state
            state = self._failure_monitor.current_state
            acceptance["is_frozen"] = state == SystemState.FROZEN
            
            # No trading allowed
            acceptance["trading_blocked"] = not self._failure_monitor.can_open_positions
            
            # Cleanup
            self._failure_monitor.resolve_failure(
                FailureScenario.E1_EXCHANGE_RISK,
                "Test cleanup"
            )
            
            return all(acceptance.values()), acceptance
            
        except Exception as e:
            return False, {"error": False, "message": str(e)}
    
    # ═══════════════════════════════════════════════════════════════════════════
    # REPORTING
    # ═══════════════════════════════════════════════════════════════════════════
    
    def print_report(self, results: Optional[List[TestResult]] = None):
        """Print test results report."""
        results = results or self._results
        
        print("\n" + "=" * 70)
        print("🧪 FAILURE SCENARIO TEST REPORT")
        print("=" * 70)
        
        passed = sum(1 for r in results if r.passed)
        failed = len(results) - passed
        
        print(f"\nSummary: {passed}/{len(results)} tests passed")
        print("-" * 70)
        
        for result in results:
            tc = TEST_CASES.get(result.test_id)
            if not tc:
                continue
            
            status = "✅ PASS" if result.passed else "❌ FAIL"
            print(f"\n{tc.severity.value} {result.test_id}: {tc.name}")
            print(f"   Status: {status} ({result.duration_ms:.1f}ms)")
            
            if not result.passed:
                print(f"   Message: {result.message}")
                if result.error:
                    print(f"   Error: {result.error}")
            
            if result.acceptance_results:
                print("   Acceptance Criteria:")
                for criterion, passed in result.acceptance_results.items():
                    icon = "✅" if passed else "❌"
                    print(f"     {icon} {criterion}")
        
        print("\n" + "=" * 70)
        
        if failed > 0:
            print(f"⚠️ {failed} TEST(S) FAILED - Review before going live!")
        else:
            print("✅ ALL TESTS PASSED - System ready for next phase")
        
        print("=" * 70 + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# PYTEST FIXTURES AND TESTS
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def test_runner():
    """Pytest fixture for test runner."""
    return FailureTestRunner()


@pytest.mark.asyncio
async def test_tc_a1_human_override(test_runner):
    """TC-A1: Human override should trigger freeze."""
    await test_runner.setup_governance()
    result = await test_runner.run_test(TEST_CASES["TC-A1"])
    assert result.passed, f"TC-A1 failed: {result.message}"


@pytest.mark.asyncio
async def test_tc_a2_risk_per_trade(test_runner):
    """TC-A2: Excessive risk per trade should be blocked."""
    await test_runner.setup_governance()
    result = await test_runner.run_test(TEST_CASES["TC-A2"])
    assert result.passed, f"TC-A2 failed: {result.message}"


@pytest.mark.asyncio
async def test_tc_a3_daily_loss(test_runner):
    """TC-A3: Daily loss limit should trigger kill-switch."""
    await test_runner.setup_governance()
    result = await test_runner.run_test(TEST_CASES["TC-A3"])
    assert result.passed, f"TC-A3 failed: {result.message}"


@pytest.mark.asyncio
async def test_tc_a4_data_inconsistency(test_runner):
    """TC-A4: Data inconsistency should trigger safe mode."""
    await test_runner.setup_governance()
    result = await test_runner.run_test(TEST_CASES["TC-A4"])
    assert result.passed, f"TC-A4 failed: {result.message}"


@pytest.mark.asyncio
async def test_all_failure_scenarios(test_runner):
    """Run all failure scenario tests."""
    results = await test_runner.run_all_tests()
    test_runner.print_report(results)
    
    failed = [r for r in results if not r.passed]
    assert len(failed) == 0, f"{len(failed)} tests failed"


# ═══════════════════════════════════════════════════════════════════════════════
# STANDALONE EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

async def main():
    """Run all tests standalone."""
    runner = FailureTestRunner()
    results = await runner.run_all_tests()
    runner.print_report(results)
    
    return all(r.passed for r in results)


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
