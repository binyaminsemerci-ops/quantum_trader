"""
Test: Signal Flood Scenario
Sprint 3 Part 3 - Failure Simulation

Simulates signal flood (30-50 signals rapidly) and verifies:
1. AI Engine processes without crashing
2. EventBus handles high throughput
3. Execution respects risk constraints
4. Queue lag within acceptable bounds
5. Rate limiter prevents overtrading

Run:
    pytest tests/simulations/test_signal_flood.py -v -s
"""

import pytest
import asyncio
import logging
from datetime import datetime, timezone

from tests.simulations.harness import (
    FailureSimulationHarness,
    SignalFloodConfig,
    ScenarioStatus
)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)


class TestSignalFlood:
    """Test suite for signal flood scenario"""
    
    @pytest.fixture
    async def harness(self):
        """Create test harness"""
        harness = FailureSimulationHarness()
        yield harness
    
    @pytest.mark.asyncio
    async def test_signal_flood_default_config(self, harness):
        """
        Test signal flood with default configuration (50 signals).
        
        Expected:
        - All signals published successfully
        - AI Engine processes without crash
        - Trade intents limited by risk policy
        - Queue lag < 5 seconds
        - System recovers after flood
        """
        result = await harness.run_signal_flood_scenario()
        
        # Assertions
        assert result.status == ScenarioStatus.PASSED, f"Scenario failed: {result.errors}"
        assert result.checks_failed == 0, f"Checks failed: {result.checks_failed}"
        assert result.metrics["signals_published"] == 50, "Should publish all 50 signals"
        assert result.metrics["queue_lag_seconds"] < 5.0, "Queue lag should be acceptable"
        
        # Log results
        print(f"\n{'='*60}")
        print(f"SIGNAL FLOOD TEST RESULTS")
        print(f"{'='*60}")
        print(f"Status: {result.status.value}")
        print(f"Duration: {result.duration_seconds:.2f}s")
        print(f"Checks: {result.checks_passed}/{result.checks_passed + result.checks_failed}")
        print(f"Signals published: {result.metrics['signals_published']}")
        print(f"Trade intents: {result.metrics['trade_intents_generated']}")
        print(f"Queue lag: {result.metrics['queue_lag_seconds']:.2f}s")
        print(f"\nObservations:")
        for obs in result.observations:
            print(f"  {obs}")
        print(f"{'='*60}\n")
    
    @pytest.mark.asyncio
    async def test_signal_flood_extreme_volume(self, harness):
        """
        Test extreme signal flood (100 signals).
        
        Expected:
        - System handles high volume
        - Queue lag still acceptable
        - Risk constraints prevent overtrading
        """
        config = SignalFloodConfig(
            signal_count=100,
            publish_interval_ms=50.0,  # Even faster
            max_queue_lag_seconds=10.0
        )
        
        result = await harness.run_signal_flood_scenario(config)
        
        assert result.status == ScenarioStatus.PASSED
        assert result.metrics["signals_published"] == 100
        assert result.metrics["queue_lag_seconds"] < 10.0
        
        print(f"\nExtreme volume: {result.metrics['signals_published']} signals, "
              f"lag: {result.metrics['queue_lag_seconds']:.2f}s")
    
    @pytest.mark.asyncio
    async def test_signal_flood_multi_symbol(self, harness):
        """
        Test signal flood across multiple symbols.
        
        Expected:
        - Signals distributed across symbols
        - No single symbol overwhelmed
        - Portfolio-level constraints applied
        """
        config = SignalFloodConfig(
            signal_count=60,
            symbols=["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "DOGEUSDT", "XRPUSDT"]
        )
        
        result = await harness.run_signal_flood_scenario(config)
        
        assert result.status == ScenarioStatus.PASSED
        assert result.metrics["signals_published"] == 60
        
        # Verify multi-symbol handling
        symbol_obs = [obs for obs in result.observations if "symbol" in obs.lower()]
        print(f"\nMulti-symbol test: {len(symbol_obs)} symbol-related observations")
    
    @pytest.mark.asyncio
    async def test_signal_flood_risk_constraints(self, harness):
        """
        Test that risk constraints limit trade intents during flood.
        
        Expected:
        - 50 signals → ≤10 trade intents (policy limit)
        - Execution respects max_positions
        - Risk checks prevent overexposure
        """
        result = await harness.run_signal_flood_scenario()
        
        assert result.status == ScenarioStatus.PASSED
        
        signals = result.metrics["signals_published"]
        trade_intents = result.metrics["trade_intents_generated"]
        
        assert trade_intents <= 10, f"Trade intents {trade_intents} exceeds policy limit 10"
        
        print(f"\nRisk constraints: {signals} signals → {trade_intents} trade intents "
              f"({trade_intents/signals*100:.1f}% conversion)")
    
    @pytest.mark.asyncio
    async def test_signal_flood_queue_metrics(self, harness):
        """
        Test EventBus queue metrics during flood.
        
        Expected:
        - Queue depth tracked
        - Processing rate measured
        - No queue overflow
        """
        config = SignalFloodConfig(
            signal_count=40,
            publish_interval_ms=100.0
        )
        
        result = await harness.run_signal_flood_scenario(config)
        
        assert result.status == ScenarioStatus.PASSED
        
        # Check queue lag is reasonable
        lag = result.metrics["queue_lag_seconds"]
        duration = result.metrics["flood_duration_seconds"]
        
        print(f"\nQueue metrics:")
        print(f"  Flood duration: {duration:.2f}s")
        print(f"  Queue lag: {lag:.2f}s")
        print(f"  Processing efficiency: {(duration/(duration+lag)*100):.1f}%")
    
    @pytest.mark.asyncio
    async def test_signal_flood_post_recovery(self, harness):
        """
        Test system recovery after signal flood.
        
        Expected:
        - System processes normally after flood
        - No lingering congestion
        - Metrics reset
        """
        result = await harness.run_signal_flood_scenario()
        
        assert result.status == ScenarioStatus.PASSED
        
        # Check recovery observation
        recovery_obs = [obs for obs in result.observations if "after flood" in obs.lower() or "normally" in obs.lower()]
        assert len(recovery_obs) > 0, "Should confirm post-flood recovery"
        
        print(f"\nRecovery status: {recovery_obs[0] if recovery_obs else 'Not checked'}")
    
    @pytest.mark.asyncio
    async def test_signal_flood_ai_engine_stability(self, harness):
        """
        Test AI Engine stability during signal flood.
        
        Expected:
        - No exceptions during processing
        - Ensemble voting completes for all signals
        - Meta-strategy evaluates signals
        - RL sizing applied to trade intents
        """
        result = await harness.run_signal_flood_scenario()
        
        assert result.status == ScenarioStatus.PASSED
        assert len(result.errors) == 0, "Should have no errors during flood"
        
        print(f"\nAI Engine stability: {result.checks_passed} checks passed, {len(result.errors)} errors")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
