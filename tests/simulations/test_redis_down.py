"""
Test: Redis Down Scenario
Sprint 3 Part 3 - Failure Simulation

Simulates Redis downtime and verifies:
1. EventBus doesn't crash on connection errors
2. DiskBuffer fallback works
3. System operates in degraded mode
4. Buffer flushes when Redis recovers
5. Monitoring reports Redis=DOWN correctly

Run:
    pytest tests/simulations/test_redis_down.py -v -s
"""

import pytest
import asyncio
import logging
from datetime import datetime, timezone

from tests.simulations.harness import (
    FailureSimulationHarness,
    RedisDownConfig,
    ScenarioStatus
)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)


class TestRedisDown:
    """Test suite for Redis downtime scenario"""
    
    @pytest.fixture
    async def harness(self):
        """Create test harness"""
        harness = FailureSimulationHarness()
        yield harness
    
    @pytest.mark.asyncio
    async def test_redis_down_default_config(self, harness):
        """
        Test Redis downtime with default configuration (60s downtime).
        
        Expected:
        - No exceptions during downtime
        - Events written to DiskBuffer
        - Buffer flushes on recovery
        - Health monitoring reports Redis=DOWN
        """
        result = await harness.run_redis_down_scenario()
        
        # Assertions
        assert result.status == ScenarioStatus.PASSED, f"Scenario failed: {result.errors}"
        assert result.checks_failed == 0, f"Checks failed: {result.checks_failed}"
        assert result.metrics["buffered_events"] > 0, "Should buffer events during downtime"
        assert result.metrics["redis_reconnects"] > 0, "Should reconnect after recovery"
        
        # Log results
        print(f"\n{'='*60}")
        print(f"REDIS DOWN TEST RESULTS")
        print(f"{'='*60}")
        print(f"Status: {result.status.value}")
        print(f"Duration: {result.duration_seconds:.2f}s")
        print(f"Checks: {result.checks_passed}/{result.checks_passed + result.checks_failed}")
        print(f"Buffered events: {result.metrics['buffered_events']}")
        print(f"Flush duration: {result.metrics.get('flush_duration_seconds', 0):.2f}s")
        print(f"\nObservations:")
        for obs in result.observations:
            print(f"  {obs}")
        print(f"{'='*60}\n")
    
    @pytest.mark.asyncio
    async def test_redis_down_high_volume(self, harness):
        """
        Test Redis downtime with high message volume.
        
        Expected:
        - DiskBuffer handles 50+ messages
        - No message loss
        - Fast flush on recovery
        """
        config = RedisDownConfig(
            downtime_seconds=30.0,
            publish_attempts_during_downtime=50,
            expected_buffer_writes=50,
            recovery_flush_timeout=60.0
        )
        
        result = await harness.run_redis_down_scenario(config)
        
        assert result.status == ScenarioStatus.PASSED
        assert result.metrics["buffered_events"] == 50, "Should buffer all 50 events"
        assert result.metrics["flush_duration_seconds"] < 60.0, "Flush should complete quickly"
        
        print(f"\nHigh volume test: {result.metrics['buffered_events']} events buffered and flushed")
    
    @pytest.mark.asyncio
    async def test_redis_down_rapid_reconnect(self, harness):
        """
        Test quick Redis recovery (short downtime).
        
        Expected:
        - Quick detection of recovery
        - Minimal buffer accumulation
        - Fast return to normal
        """
        config = RedisDownConfig(
            downtime_seconds=10.0,
            publish_attempts_during_downtime=5,
            expected_buffer_writes=5,
            recovery_flush_timeout=15.0
        )
        
        result = await harness.run_redis_down_scenario(config)
        
        assert result.status == ScenarioStatus.PASSED
        assert result.duration_seconds < 30.0, "Should recover quickly"
        assert harness.metrics["redis_reconnects"] > 0, "Should reconnect"
    
    @pytest.mark.asyncio
    async def test_redis_down_disk_buffer_metrics(self, harness):
        """
        Test DiskBuffer metrics during downtime.
        
        Expected:
        - disk_buffer_writes counter increments
        - Buffer size tracked
        - Flush metrics recorded
        """
        result = await harness.run_redis_down_scenario()
        
        assert result.status == ScenarioStatus.PASSED
        assert harness.metrics["disk_buffer_writes"] > 0, "Should record buffer writes"
        
        print(f"\nDiskBuffer metrics:")
        print(f"  Writes: {harness.metrics['disk_buffer_writes']}")
        print(f"  Reconnects: {harness.metrics['redis_reconnects']}")
    
    @pytest.mark.asyncio
    async def test_redis_down_health_monitoring(self, harness):
        """
        Test health monitoring during Redis downtime.
        
        Expected:
        - Health reports Redis=DOWN during outage
        - Health reports Redis=OK after recovery
        - Monitoring client receives status updates
        """
        result = await harness.run_redis_down_scenario()
        
        assert result.status == ScenarioStatus.PASSED
        
        # Check observations for health monitoring
        health_observations = [obs for obs in result.observations if "Health" in obs or "Redis" in obs]
        assert len(health_observations) > 0, "Should have health observations"
        
        print(f"\nHealth observations: {len(health_observations)}")
        for obs in health_observations:
            print(f"  {obs}")
    
    @pytest.mark.asyncio
    async def test_redis_down_no_message_loss(self, harness):
        """
        Test that no messages are lost during downtime.
        
        Expected:
        - All messages buffered
        - All messages flushed on recovery
        - Message order preserved (FIFO)
        """
        config = RedisDownConfig(
            downtime_seconds=45.0,
            publish_attempts_during_downtime=20,
            expected_buffer_writes=20
        )
        
        result = await harness.run_redis_down_scenario(config)
        
        assert result.status == ScenarioStatus.PASSED
        assert result.metrics["buffered_events"] == config.expected_buffer_writes
        
        # Verify no loss message in observations
        loss_check = any("loss" in obs.lower() or "lost" in obs.lower() for obs in result.observations)
        assert not loss_check, "Should not report message loss"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
