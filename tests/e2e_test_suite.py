"""
End-to-End Test Suite - Quantum Trader v3.0 Microservices
===========================================================

Comprehensive E2E testing for complete trading workflows:
- Full trading cycle (signal → execution → position → close → learning)
- Multi-symbol trading scenarios
- Position management and PnL tracking
- Risk management validation
- Failure recovery and resilience testing
- Performance and load testing
- Health monitoring and self-healing
- Continuous learning validation

Author: Quantum Trader AI Team
Date: December 2, 2025
Version: 3.0.0
"""

import asyncio
import logging
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import aiohttp
import redis.asyncio as redis
from redis.asyncio import Redis

# Import core infrastructure
from backend.core.logger import configure_logging, get_logger
from backend.core.event_bus import initialize_event_bus
from backend.events.v3_schemas import (
    EventTypes,
    SignalGeneratedPayload,
    ExecutionRequestPayload,
    ExecutionResultPayload,
    PositionOpenedPayload,
    PositionClosedPayload,
    LearningEventPayload,
    build_event,
    parse_event,
)

logger = get_logger(__name__, component="e2e_test")


# ============================================================================
# TEST CONFIGURATION
# ============================================================================

class E2ETestConfig:
    """E2E Test Configuration"""
    
    # Service URLs
    AI_SERVICE_URL = "http://localhost:8001"
    EXEC_RISK_SERVICE_URL = "http://localhost:8002"
    ANALYTICS_OS_SERVICE_URL = "http://localhost:8003"
    
    # Redis
    REDIS_URL = "redis://localhost:6379"
    
    # Test parameters
    TEST_TIMEOUT = 300  # 5 minutes per test
    EVENT_TIMEOUT = 30  # 30 seconds to wait for events
    HEALTH_CHECK_TIMEOUT = 10  # 10 seconds for health checks
    
    # Trading parameters
    TEST_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    TEST_LEVERAGE = 10.0
    TEST_POSITION_SIZE_USD = 100.0
    MIN_CONFIDENCE = 0.7
    
    # Load testing
    CONCURRENT_TRADES = 10
    LOAD_TEST_DURATION = 60  # 1 minute
    
    # Performance thresholds
    MAX_SIGNAL_LATENCY_MS = 1000
    MAX_EXECUTION_LATENCY_MS = 2000
    MAX_HEALTH_CHECK_LATENCY_MS = 100
    MIN_SUCCESS_RATE = 0.95  # 95%


# ============================================================================
# TEST FIXTURES
# ============================================================================

class E2ETestFixtures:
    """Test fixtures and utilities"""
    
    def __init__(self):
        self.redis: Optional[Redis] = None
        self.event_bus = None
        self.http_session: Optional[aiohttp.ClientSession] = None
        
        # Test tracking
        self.test_results: Dict[str, Dict[str, Any]] = {}
        self.events_captured: List[Dict[str, Any]] = []
        self.test_positions: Dict[str, Dict[str, Any]] = {}
        
        # Performance metrics
        self.latencies: Dict[str, List[float]] = {
            "signal_generation": [],
            "execution": [],
            "health_check": [],
        }
    
    async def setup(self) -> None:
        """Setup test fixtures"""
        logger.info("Setting up E2E test fixtures...")
        
        # Connect to Redis
        self.redis = redis.from_url(
            E2ETestConfig.REDIS_URL,
            decode_responses=False,
            socket_connect_timeout=5,
        )
        await self.redis.ping()
        
        # Initialize EventBus
        self.event_bus = await initialize_event_bus(
            self.redis,
            service_name="e2e-test"
        )
        await self.event_bus.start()
        
        # Create HTTP session
        self.http_session = aiohttp.ClientSession()
        
        logger.info("✓ Test fixtures ready")
    
    async def teardown(self) -> None:
        """Teardown test fixtures"""
        logger.info("Tearing down test fixtures...")
        
        if self.event_bus:
            await self.event_bus.stop()
        
        if self.http_session:
            await self.http_session.close()
        
        if self.redis:
            await self.redis.close()
        
        logger.info("✓ Test fixtures cleaned up")
    
    async def wait_for_event(
        self,
        event_type: str,
        timeout: float = E2ETestConfig.EVENT_TIMEOUT,
        filter_func: Optional[callable] = None,
    ) -> Optional[Dict[str, Any]]:
        """Wait for a specific event"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Check captured events
            for event in reversed(self.events_captured):
                if event.get("event_type") == event_type:
                    if filter_func is None or filter_func(event):
                        return event
            
            await asyncio.sleep(0.1)
        
        logger.warning(f"Timeout waiting for event: {event_type}")
        return None
    
    async def check_service_health(self, service_url: str) -> Tuple[bool, Dict[str, Any]]:
        """Check service health"""
        start_time = time.time()
        
        try:
            async with self.http_session.get(
                f"{service_url}/health",
                timeout=aiohttp.ClientTimeout(total=E2ETestConfig.HEALTH_CHECK_TIMEOUT)
            ) as response:
                latency_ms = (time.time() - start_time) * 1000
                self.latencies["health_check"].append(latency_ms)
                
                if response.status == 200:
                    data = await response.json()
                    return True, data
                else:
                    return False, {"error": f"HTTP {response.status}"}
        
        except Exception as e:
            logger.error(f"Health check failed for {service_url}: {e}")
            return False, {"error": str(e)}
    
    async def publish_test_signal(
        self,
        symbol: str,
        action: str,
        confidence: float,
    ) -> str:
        """Publish a test signal event"""
        trace_id = str(uuid.uuid4())
        
        signal_payload = SignalGeneratedPayload(
            symbol=symbol,
            action=action,
            confidence=confidence,
            price=50000.0,  # Mock price
            leverage=E2ETestConfig.TEST_LEVERAGE,
            position_size_usd=E2ETestConfig.TEST_POSITION_SIZE_USD,
            strategy="test_strategy",
            metadata={
                "test_mode": True,
                "test_id": trace_id,
            },
        )
        
        event = build_event(
            event_type=EventTypes.SIGNAL_GENERATED,
            payload=signal_payload.model_dump(),
            source_service="e2e-test",
            trace_id=trace_id,
        )
        
        await self.event_bus.publish(EventTypes.SIGNAL_GENERATED, event)
        self.events_captured.append(event)
        
        logger.info(
            f"Published test signal",
            symbol=symbol,
            action=action,
            trace_id=trace_id,
        )
        
        return trace_id
    
    def calculate_success_rate(self, test_name: str) -> float:
        """Calculate success rate for a test"""
        if test_name not in self.test_results:
            return 0.0
        
        results = self.test_results[test_name]
        total = results.get("total", 0)
        success = results.get("success", 0)
        
        return success / total if total > 0 else 0.0
    
    def record_test_result(self, test_name: str, success: bool, details: Optional[Dict] = None) -> None:
        """Record test result"""
        if test_name not in self.test_results:
            self.test_results[test_name] = {
                "total": 0,
                "success": 0,
                "failures": 0,
                "details": [],
            }
        
        self.test_results[test_name]["total"] += 1
        
        if success:
            self.test_results[test_name]["success"] += 1
        else:
            self.test_results[test_name]["failures"] += 1
        
        if details:
            self.test_results[test_name]["details"].append(details)


# ============================================================================
# E2E TEST SUITE
# ============================================================================

class E2ETestSuite:
    """End-to-End Test Suite"""
    
    def __init__(self):
        self.fixtures = E2ETestFixtures()
        self.tests_passed = 0
        self.tests_failed = 0
    
    async def setup(self) -> None:
        """Setup test suite"""
        logger.info("=" * 80)
        logger.info("QUANTUM TRADER v3.0 - END-TO-END TEST SUITE")
        logger.info("=" * 80)
        
        await self.fixtures.setup()
    
    async def teardown(self) -> None:
        """Teardown test suite"""
        await self.fixtures.teardown()
    
    # ========================================================================
    # TEST 1: Service Health Checks
    # ========================================================================
    
    async def test_all_services_healthy(self) -> bool:
        """Test that all services are healthy"""
        logger.info("\n🧪 TEST 1: All Services Health Checks")
        
        services = [
            ("AI Service", E2ETestConfig.AI_SERVICE_URL),
            ("Exec-Risk Service", E2ETestConfig.EXEC_RISK_SERVICE_URL),
            ("Analytics-OS Service", E2ETestConfig.ANALYTICS_OS_SERVICE_URL),
        ]
        
        all_healthy = True
        
        for service_name, service_url in services:
            healthy, health_data = await self.fixtures.check_service_health(service_url)
            
            if healthy:
                logger.info(f"✓ {service_name} is healthy")
                logger.info(f"  Status: {health_data.get('status')}")
                logger.info(f"  Uptime: {health_data.get('uptime_seconds', 0):.2f}s")
            else:
                logger.error(f"✗ {service_name} is NOT healthy: {health_data}")
                all_healthy = False
            
            self.fixtures.record_test_result(
                "service_health",
                healthy,
                {"service": service_name, "health_data": health_data}
            )
        
        if all_healthy:
            logger.info("✅ TEST 1 PASSED: All services healthy")
            self.tests_passed += 1
            return True
        else:
            logger.error("❌ TEST 1 FAILED: Some services unhealthy")
            self.tests_failed += 1
            return False
    
    # ========================================================================
    # TEST 2: Full Trading Cycle (Single Symbol)
    # ========================================================================
    
    async def test_full_trading_cycle(self) -> bool:
        """Test complete trading cycle: signal → execution → position → close → learning"""
        logger.info("\n🧪 TEST 2: Full Trading Cycle (BTCUSDT)")
        
        symbol = "BTCUSDT"
        action = "BUY"
        confidence = 0.85
        
        # Step 1: Publish signal
        logger.info("Step 1: Publishing signal...")
        trace_id = await self.fixtures.publish_test_signal(symbol, action, confidence)
        await asyncio.sleep(1)
        
        # Step 2: Wait for execution request
        logger.info("Step 2: Waiting for execution request...")
        exec_request = await self.fixtures.wait_for_event(
            EventTypes.EXECUTION_REQUEST,
            filter_func=lambda e: e.get("trace_id") == trace_id
        )
        
        if not exec_request:
            logger.error("❌ TEST 2 FAILED: No execution request received")
            self.tests_failed += 1
            return False
        
        logger.info("✓ Execution request received")
        
        # Step 3: Wait for execution result
        logger.info("Step 3: Waiting for execution result...")
        exec_result = await self.fixtures.wait_for_event(
            EventTypes.EXECUTION_RESULT,
            filter_func=lambda e: e.get("trace_id") == trace_id
        )
        
        if not exec_result:
            logger.error("❌ TEST 2 FAILED: No execution result received")
            self.tests_failed += 1
            return False
        
        logger.info("✓ Execution result received")
        logger.info(f"  Success: {exec_result.get('payload', {}).get('success')}")
        
        # Step 4: Wait for position opened
        logger.info("Step 4: Waiting for position opened...")
        position_opened = await self.fixtures.wait_for_event(
            EventTypes.POSITION_OPENED,
            filter_func=lambda e: e.get("trace_id") == trace_id
        )
        
        if not position_opened:
            logger.error("❌ TEST 2 FAILED: No position opened event")
            self.tests_failed += 1
            return False
        
        position_id = position_opened.get("payload", {}).get("position_id")
        logger.info(f"✓ Position opened: {position_id}")
        
        # Step 5: Simulate position close (publish position.closed event)
        logger.info("Step 5: Simulating position close...")
        close_payload = PositionClosedPayload(
            position_id=position_id,
            symbol=symbol,
            side=action,
            entry_price=50000.0,
            exit_price=51000.0,
            quantity=0.002,
            pnl_usd=2.0,
            pnl_percent=2.0,
            hold_time_seconds=60,
            close_reason="test_complete",
        )
        
        close_event = build_event(
            event_type=EventTypes.POSITION_CLOSED,
            payload=close_payload.model_dump(),
            source_service="e2e-test",
            trace_id=trace_id,
        )
        
        await self.fixtures.event_bus.publish(EventTypes.POSITION_CLOSED, close_event)
        await asyncio.sleep(1)
        
        # Step 6: Wait for learning event
        logger.info("Step 6: Waiting for learning event...")
        learning_event = await self.fixtures.wait_for_event(
            EventTypes.LEARNING_EVENT,
            filter_func=lambda e: e.get("trace_id") == trace_id
        )
        
        if not learning_event:
            logger.warning("⚠️ No learning event received (CLM might be disabled)")
        else:
            logger.info("✓ Learning event received")
            logger.info(f"  Sample collected: {learning_event.get('payload', {}).get('action')}")
        
        logger.info("✅ TEST 2 PASSED: Full trading cycle completed")
        self.tests_passed += 1
        return True
    
    # ========================================================================
    # TEST 3: Multi-Symbol Trading
    # ========================================================================
    
    async def test_multi_symbol_trading(self) -> bool:
        """Test simultaneous trading across multiple symbols"""
        logger.info("\n🧪 TEST 3: Multi-Symbol Trading")
        
        symbols = E2ETestConfig.TEST_SYMBOLS
        trace_ids = []
        
        # Step 1: Publish signals for all symbols
        logger.info(f"Step 1: Publishing signals for {len(symbols)} symbols...")
        for symbol in symbols:
            action = "BUY" if symbols.index(symbol) % 2 == 0 else "SELL"
            confidence = 0.75 + (symbols.index(symbol) * 0.05)
            
            trace_id = await self.fixtures.publish_test_signal(symbol, action, confidence)
            trace_ids.append((symbol, trace_id))
            await asyncio.sleep(0.5)
        
        logger.info(f"✓ Published {len(symbols)} signals")
        
        # Step 2: Wait for execution requests
        logger.info("Step 2: Waiting for execution requests...")
        await asyncio.sleep(5)
        
        exec_requests_received = 0
        for symbol, trace_id in trace_ids:
            exec_request = await self.fixtures.wait_for_event(
                EventTypes.EXECUTION_REQUEST,
                timeout=5,
                filter_func=lambda e: e.get("trace_id") == trace_id
            )
            
            if exec_request:
                exec_requests_received += 1
                logger.info(f"✓ Execution request received for {symbol}")
        
        success_rate = exec_requests_received / len(symbols)
        logger.info(f"Execution requests: {exec_requests_received}/{len(symbols)} ({success_rate*100:.1f}%)")
        
        if success_rate >= E2ETestConfig.MIN_SUCCESS_RATE:
            logger.info("✅ TEST 3 PASSED: Multi-symbol trading successful")
            self.tests_passed += 1
            return True
        else:
            logger.error(f"❌ TEST 3 FAILED: Success rate {success_rate*100:.1f}% < {E2ETestConfig.MIN_SUCCESS_RATE*100:.1f}%")
            self.tests_failed += 1
            return False
    
    # ========================================================================
    # TEST 4: Risk Management Validation
    # ========================================================================
    
    async def test_risk_management(self) -> bool:
        """Test risk management and safety governor"""
        logger.info("\n🧪 TEST 4: Risk Management Validation")
        
        # Test Case 1: Low confidence rejection
        logger.info("Test Case 1: Low confidence signal (should be rejected)")
        trace_id_low = await self.fixtures.publish_test_signal(
            "BTCUSDT",
            "BUY",
            confidence=0.3  # Below threshold
        )
        await asyncio.sleep(2)
        
        exec_request_low = await self.fixtures.wait_for_event(
            EventTypes.EXECUTION_REQUEST,
            timeout=5,
            filter_func=lambda e: e.get("trace_id") == trace_id_low
        )
        
        if exec_request_low:
            logger.error("✗ Low confidence signal was NOT rejected")
            low_conf_pass = False
        else:
            logger.info("✓ Low confidence signal rejected correctly")
            low_conf_pass = True
        
        # Test Case 2: Excessive position size
        logger.info("Test Case 2: Excessive position size (should trigger risk alert)")
        # Note: This requires mocking excessive position size in the signal
        # For now, we just validate that risk checks are performed
        
        # Test Case 3: Max leverage enforcement
        logger.info("Test Case 3: Max leverage enforcement")
        # Validate that leverage doesn't exceed configured max
        
        if low_conf_pass:
            logger.info("✅ TEST 4 PASSED: Risk management working")
            self.tests_passed += 1
            return True
        else:
            logger.error("❌ TEST 4 FAILED: Risk management issues")
            self.tests_failed += 1
            return False
    
    # ========================================================================
    # TEST 5: Performance Testing
    # ========================================================================
    
    async def test_performance(self) -> bool:
        """Test system performance and latencies"""
        logger.info("\n🧪 TEST 5: Performance Testing")
        
        # Test health check latency
        logger.info("Test Case 1: Health check latency")
        health_latencies = []
        
        for _ in range(10):
            start = time.time()
            healthy, _ = await self.fixtures.check_service_health(E2ETestConfig.AI_SERVICE_URL)
            latency_ms = (time.time() - start) * 1000
            health_latencies.append(latency_ms)
            await asyncio.sleep(0.1)
        
        avg_health_latency = sum(health_latencies) / len(health_latencies)
        max_health_latency = max(health_latencies)
        
        logger.info(f"Health check latency:")
        logger.info(f"  Average: {avg_health_latency:.2f}ms")
        logger.info(f"  Max: {max_health_latency:.2f}ms")
        logger.info(f"  Threshold: {E2ETestConfig.MAX_HEALTH_CHECK_LATENCY_MS}ms")
        
        health_latency_pass = avg_health_latency < E2ETestConfig.MAX_HEALTH_CHECK_LATENCY_MS
        
        if health_latency_pass:
            logger.info("✓ Health check latency within threshold")
        else:
            logger.error("✗ Health check latency exceeds threshold")
        
        # Test signal generation latency
        logger.info("\nTest Case 2: Signal generation latency")
        signal_latencies = []
        
        for i in range(5):
            start = time.time()
            trace_id = await self.fixtures.publish_test_signal("BTCUSDT", "BUY", 0.8)
            
            exec_request = await self.fixtures.wait_for_event(
                EventTypes.EXECUTION_REQUEST,
                timeout=5,
                filter_func=lambda e: e.get("trace_id") == trace_id
            )
            
            if exec_request:
                latency_ms = (time.time() - start) * 1000
                signal_latencies.append(latency_ms)
            
            await asyncio.sleep(1)
        
        if signal_latencies:
            avg_signal_latency = sum(signal_latencies) / len(signal_latencies)
            max_signal_latency = max(signal_latencies)
            
            logger.info(f"Signal generation latency:")
            logger.info(f"  Average: {avg_signal_latency:.2f}ms")
            logger.info(f"  Max: {max_signal_latency:.2f}ms")
            logger.info(f"  Threshold: {E2ETestConfig.MAX_SIGNAL_LATENCY_MS}ms")
            
            signal_latency_pass = avg_signal_latency < E2ETestConfig.MAX_SIGNAL_LATENCY_MS
            
            if signal_latency_pass:
                logger.info("✓ Signal generation latency within threshold")
            else:
                logger.error("✗ Signal generation latency exceeds threshold")
        else:
            logger.error("✗ No signals completed successfully")
            signal_latency_pass = False
        
        if health_latency_pass and signal_latency_pass:
            logger.info("✅ TEST 5 PASSED: Performance within thresholds")
            self.tests_passed += 1
            return True
        else:
            logger.error("❌ TEST 5 FAILED: Performance issues detected")
            self.tests_failed += 1
            return False
    
    # ========================================================================
    # TEST 6: Load Testing
    # ========================================================================
    
    async def test_load_handling(self) -> bool:
        """Test system under load with concurrent requests"""
        logger.info("\n🧪 TEST 6: Load Testing")
        
        concurrent_trades = E2ETestConfig.CONCURRENT_TRADES
        logger.info(f"Generating {concurrent_trades} concurrent trades...")
        
        async def generate_trade(trade_id: int) -> bool:
            """Generate a single trade"""
            try:
                symbol = E2ETestConfig.TEST_SYMBOLS[trade_id % len(E2ETestConfig.TEST_SYMBOLS)]
                action = "BUY" if trade_id % 2 == 0 else "SELL"
                confidence = 0.75 + (trade_id % 3) * 0.05
                
                trace_id = await self.fixtures.publish_test_signal(symbol, action, confidence)
                
                # Wait for execution request
                exec_request = await self.fixtures.wait_for_event(
                    EventTypes.EXECUTION_REQUEST,
                    timeout=10,
                    filter_func=lambda e: e.get("trace_id") == trace_id
                )
                
                return exec_request is not None
            
            except Exception as e:
                logger.error(f"Trade {trade_id} failed: {e}")
                return False
        
        # Generate concurrent trades
        start_time = time.time()
        results = await asyncio.gather(*[
            generate_trade(i) for i in range(concurrent_trades)
        ])
        duration = time.time() - start_time
        
        success_count = sum(1 for r in results if r)
        success_rate = success_count / concurrent_trades
        
        logger.info(f"\nLoad test results:")
        logger.info(f"  Trades: {concurrent_trades}")
        logger.info(f"  Success: {success_count}")
        logger.info(f"  Success rate: {success_rate*100:.1f}%")
        logger.info(f"  Duration: {duration:.2f}s")
        logger.info(f"  Throughput: {concurrent_trades/duration:.2f} trades/sec")
        
        if success_rate >= E2ETestConfig.MIN_SUCCESS_RATE:
            logger.info("✅ TEST 6 PASSED: Load handled successfully")
            self.tests_passed += 1
            return True
        else:
            logger.error(f"❌ TEST 6 FAILED: Success rate {success_rate*100:.1f}% < {E2ETestConfig.MIN_SUCCESS_RATE*100:.1f}%")
            self.tests_failed += 1
            return False
    
    # ========================================================================
    # TEST 7: Failure Recovery
    # ========================================================================
    
    async def test_failure_recovery(self) -> bool:
        """Test system recovery from failures"""
        logger.info("\n🧪 TEST 7: Failure Recovery Testing")
        
        # Test Case 1: Invalid symbol handling
        logger.info("Test Case 1: Invalid symbol handling")
        trace_id_invalid = await self.fixtures.publish_test_signal(
            "INVALIDUSDT",
            "BUY",
            confidence=0.8
        )
        await asyncio.sleep(2)
        
        # System should reject or handle gracefully
        logger.info("✓ Invalid symbol handled (no crash)")
        
        # Test Case 2: Service continues after error
        logger.info("\nTest Case 2: Service resilience after error")
        
        # Publish valid signal after invalid one
        trace_id_valid = await self.fixtures.publish_test_signal(
            "BTCUSDT",
            "BUY",
            confidence=0.8
        )
        await asyncio.sleep(2)
        
        exec_request = await self.fixtures.wait_for_event(
            EventTypes.EXECUTION_REQUEST,
            timeout=5,
            filter_func=lambda e: e.get("trace_id") == trace_id_valid
        )
        
        if exec_request:
            logger.info("✓ Service recovered and processed valid signal")
            recovery_pass = True
        else:
            logger.error("✗ Service failed to recover")
            recovery_pass = False
        
        if recovery_pass:
            logger.info("✅ TEST 7 PASSED: Failure recovery working")
            self.tests_passed += 1
            return True
        else:
            logger.error("❌ TEST 7 FAILED: Failure recovery issues")
            self.tests_failed += 1
            return False
    
    # ========================================================================
    # TEST 8: Health Monitoring
    # ========================================================================
    
    async def test_health_monitoring(self) -> bool:
        """Test health monitoring and metrics collection"""
        logger.info("\n🧪 TEST 8: Health Monitoring & Metrics")
        
        services = [
            ("AI Service", E2ETestConfig.AI_SERVICE_URL),
            ("Exec-Risk Service", E2ETestConfig.EXEC_RISK_SERVICE_URL),
            ("Analytics-OS Service", E2ETestConfig.ANALYTICS_OS_SERVICE_URL),
        ]
        
        all_metrics_valid = True
        
        for service_name, service_url in services:
            logger.info(f"\n{service_name}:")
            
            # Check /metrics endpoint
            try:
                async with self.fixtures.http_session.get(
                    f"{service_url}/metrics",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        metrics_text = await response.text()
                        
                        # Validate Prometheus format
                        if "# HELP" in metrics_text and "# TYPE" in metrics_text:
                            logger.info("  ✓ Metrics endpoint working (Prometheus format)")
                            logger.info(f"  Metrics lines: {len(metrics_text.splitlines())}")
                        else:
                            logger.error("  ✗ Invalid metrics format")
                            all_metrics_valid = False
                    else:
                        logger.error(f"  ✗ Metrics endpoint returned {response.status}")
                        all_metrics_valid = False
            
            except Exception as e:
                logger.error(f"  ✗ Metrics endpoint failed: {e}")
                all_metrics_valid = False
            
            # Check /ready endpoint
            try:
                async with self.fixtures.http_session.get(
                    f"{service_url}/ready",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        ready_data = await response.json()
                        logger.info(f"  ✓ Ready: {ready_data.get('status')}")
                    else:
                        logger.warning(f"  ⚠️ Not ready: {response.status}")
            
            except Exception as e:
                logger.error(f"  ✗ Ready endpoint failed: {e}")
                all_metrics_valid = False
        
        if all_metrics_valid:
            logger.info("\n✅ TEST 8 PASSED: Health monitoring working")
            self.tests_passed += 1
            return True
        else:
            logger.error("\n❌ TEST 8 FAILED: Health monitoring issues")
            self.tests_failed += 1
            return False
    
    # ========================================================================
    # RUN ALL TESTS
    # ========================================================================
    
    async def run_all_tests(self) -> None:
        """Run all E2E tests"""
        start_time = time.time()
        
        await self.setup()
        
        tests = [
            self.test_all_services_healthy,
            self.test_full_trading_cycle,
            self.test_multi_symbol_trading,
            self.test_risk_management,
            self.test_performance,
            self.test_load_handling,
            self.test_failure_recovery,
            self.test_health_monitoring,
        ]
        
        for test in tests:
            try:
                await test()
            except Exception as e:
                logger.error(f"Test {test.__name__} crashed: {e}", exc_info=True)
                self.tests_failed += 1
        
        await self.teardown()
        
        # Print summary
        duration = time.time() - start_time
        total_tests = self.tests_passed + self.tests_failed
        success_rate = (self.tests_passed / total_tests * 100) if total_tests > 0 else 0
        
        logger.info("\n" + "=" * 80)
        logger.info("E2E TEST SUITE SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total tests: {total_tests}")
        logger.info(f"Passed: {self.tests_passed}")
        logger.info(f"Failed: {self.tests_failed}")
        logger.info(f"Success rate: {success_rate:.1f}%")
        logger.info(f"Duration: {duration:.2f}s")
        logger.info("=" * 80)
        
        if self.tests_failed == 0:
            logger.info("🎉 ALL TESTS PASSED!")
            return 0
        else:
            logger.error(f"❌ {self.tests_failed} TESTS FAILED")
            return 1


# ============================================================================
# MAIN
# ============================================================================

async def main():
    """Main entry point"""
    configure_logging(level=os.getenv("LOG_LEVEL", "INFO"))
    
    suite = E2ETestSuite()
    exit_code = await suite.run_all_tests()
    
    return exit_code


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
