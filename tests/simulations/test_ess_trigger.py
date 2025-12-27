"""
Test: ESS Trigger & Recovery Scenario
Sprint 3 Part 3 - Failure Simulation

Simulates ESS triggering and recovery:
1. Trades causing drawdown > threshold
2. ESS trips (state=TRIPPED)
3. can_execute_orders() returns False
4. Orders blocked during trip
5. Manual reset after cooldown
6. System returns to ARMED
7. Orders allowed after reset

Run:
    pytest tests/simulations/test_ess_trigger.py -v -s
"""

import pytest
import asyncio
import logging
from datetime import datetime, timezone

from tests.simulations.harness import (
    FailureSimulationHarness,
    ESSTriggeredConfig,
    ScenarioStatus
)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)


class TestESSTrigger:
    """Test suite for ESS trigger and recovery scenario"""
    
    @pytest.fixture
    async def harness(self):
        """Create test harness"""
        harness = FailureSimulationHarness()
        yield harness
    
    @pytest.mark.asyncio
    async def test_ess_trigger_default_config(self, harness):
        """
        Test ESS trigger with default configuration (10% threshold).
        
        Expected:
        - ESS trips when drawdown > 10%
        - can_execute_orders() = False
        - Orders blocked
        - Manual reset works after cooldown
        - System returns to ARMED
        - Orders allowed after reset
        """
        result = await harness.run_ess_trigger_scenario()
        
        # Assertions
        assert result.status == ScenarioStatus.PASSED, f"Scenario failed: {result.errors}"
        assert result.checks_failed == 0, f"Checks failed: {result.checks_failed}"
        assert result.metrics["drawdown_percent"] > 10.0, "Drawdown should exceed threshold"
        assert result.metrics["ess_trips"] > 0, "ESS should have tripped"
        assert result.metrics["trades_blocked"] > 0, "Trades should be blocked"
        
        # Log results
        print(f"\n{'='*60}")
        print(f"ESS TRIGGER TEST RESULTS")
        print(f"{'='*60}")
        print(f"Status: {result.status.value}")
        print(f"Duration: {result.duration_seconds:.2f}s")
        print(f"Checks: {result.checks_passed}/{result.checks_passed + result.checks_failed}")
        print(f"Initial balance: ${result.metrics['initial_balance']}")
        print(f"Final PnL: ${result.metrics['final_pnl']}")
        print(f"Drawdown: {result.metrics['drawdown_percent']:.2f}%")
        print(f"ESS trips: {result.metrics['ess_trips']}")
        print(f"Trades blocked: {result.metrics['trades_blocked']}")
        print(f"\nObservations:")
        for obs in result.observations:
            print(f"  {obs}")
        print(f"{'='*60}\n")
    
    @pytest.mark.asyncio
    async def test_ess_trigger_severe_loss(self, harness):
        """
        Test ESS with severe loss (20% drawdown).
        
        Expected:
        - ESS trips quickly
        - Large drawdown recorded
        - Multiple alerts raised
        """
        config = ESSTriggeredConfig(
            initial_balance=10000.0,
            loss_amount=2500.0,  # 25% loss
            ess_threshold_percent=10.0
        )
        
        result = await harness.run_ess_trigger_scenario(config)
        
        assert result.status == ScenarioStatus.PASSED
        assert result.metrics["drawdown_percent"] > 20.0
        assert result.metrics["ess_trips"] > 0
        
        print(f"\nSevere loss: {result.metrics['drawdown_percent']:.2f}% drawdown")
    
    @pytest.mark.asyncio
    async def test_ess_trigger_manual_reset(self, harness):
        """
        Test manual ESS reset after cooldown.
        
        Expected:
        - Cooldown period enforced
        - Manual reset by operator succeeds
        - ESS returns to ARMED
        - Orders allowed after reset
        """
        config = ESSTriggeredConfig(
            cooldown_minutes=1,
            manual_reset_after_seconds=15.0  # Short cooldown for testing
        )
        
        result = await harness.run_ess_trigger_scenario(config)
        
        assert result.status == ScenarioStatus.PASSED
        
        # Check reset observations
        reset_obs = [obs for obs in result.observations if "reset" in obs.lower()]
        assert len(reset_obs) > 0, "Should have reset observations"
        
        print(f"\nManual reset observations:")
        for obs in reset_obs:
            print(f"  {obs}")
    
    @pytest.mark.asyncio
    async def test_ess_trigger_order_blocking(self, harness):
        """
        Test that orders are blocked when ESS is tripped.
        
        Expected:
        - Order execution attempted
        - Order blocked with clear message
        - can_execute_orders() returns False
        """
        result = await harness.run_ess_trigger_scenario()
        
        assert result.status == ScenarioStatus.PASSED
        assert harness.metrics["trades_blocked"] > 0
        
        # Check blocking observations
        block_obs = [obs for obs in result.observations if "block" in obs.lower()]
        assert len(block_obs) > 0, "Should have order blocking observations"
        
        print(f"\nOrder blocking: {len(block_obs)} observations")
    
    @pytest.mark.asyncio
    async def test_ess_trigger_monitoring_lifecycle(self, harness):
        """
        Test monitoring captures ESS lifecycle events.
        
        Expected:
        - Alert published when ESS trips
        - Alert includes trip reason and drawdown
        - Alert published when ESS reset
        - Full lifecycle tracked
        """
        result = await harness.run_ess_trigger_scenario()
        
        assert result.status == ScenarioStatus.PASSED
        assert harness.metrics["health_alerts"] > 0
        
        # Check lifecycle observations
        lifecycle_obs = [obs for obs in result.observations if "lifecycle" in obs.lower() or "alert" in obs.lower()]
        assert len(lifecycle_obs) > 0, "Should track ESS lifecycle"
        
        print(f"\nESS lifecycle events: {len(lifecycle_obs)}")
        for obs in lifecycle_obs:
            print(f"  {obs}")
    
    @pytest.mark.asyncio
    async def test_ess_trigger_post_reset_trading(self, harness):
        """
        Test that trading resumes normally after ESS reset.
        
        Expected:
        - ESS state = ARMED after reset
        - can_execute_orders() = True
        - Orders execute successfully
        - No lingering restrictions
        """
        result = await harness.run_ess_trigger_scenario()
        
        assert result.status == ScenarioStatus.PASSED
        
        # Check post-reset observations
        post_reset_obs = [obs for obs in result.observations if "after reset" in obs.lower()]
        assert len(post_reset_obs) > 0, "Should confirm post-reset functionality"
        
        print(f"\nPost-reset trading: {post_reset_obs}")
    
    @pytest.mark.asyncio
    async def test_ess_trigger_drawdown_calculation(self, harness):
        """
        Test accurate drawdown calculation.
        
        Expected:
        - Drawdown = |negative PnL| / initial_balance * 100
        - Calculation includes all losing trades
        - Threshold comparison accurate
        """
        config = ESSTriggeredConfig(
            initial_balance=10000.0,
            loss_amount=1500.0,  # 15% loss
            ess_threshold_percent=10.0
        )
        
        result = await harness.run_ess_trigger_scenario(config)
        
        assert result.status == ScenarioStatus.PASSED
        
        # Verify drawdown calculation
        expected_drawdown = (config.loss_amount / config.initial_balance) * 100
        actual_drawdown = result.metrics["drawdown_percent"]
        
        # Allow small tolerance for calculation differences
        assert abs(actual_drawdown - expected_drawdown) < 2.0, \
            f"Drawdown calculation error: expected ~{expected_drawdown:.2f}%, got {actual_drawdown:.2f}%"
        
        print(f"\nDrawdown calculation: {actual_drawdown:.2f}% (expected ~{expected_drawdown:.2f}%)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
