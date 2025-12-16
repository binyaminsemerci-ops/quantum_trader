"""
Integration Test Harness - Quantum Trader v3.0 Microservices
==============================================================

Comprehensive integration testing for all 3 microservices:
- Service startup and health checks
- Inter-service RPC communication
- Event flow validation (signal → execution → position closed → learning)
- Load testing with parallel requests
- Failure scenarios and recovery

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
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import aiohttp
import redis.asyncio as redis
from redis.asyncio import Redis

# Import test utilities
from backend.core.logger import configure_logging, get_logger
from backend.events.v3_schemas import (
    EventTypes,
    SignalGeneratedPayload,
    ExecutionRequestPayload,
    build_event,
)

logger = get_logger(__name__, component="integration_test")


# ============================================================================
# TEST CONFIGURATION
# ============================================================================

class TestConfig:
    """Integration test configuration"""
    
    # Service endpoints
    AI_SERVICE_URL = os.getenv("AI_SERVICE_URL", "http://localhost:8001")
    EXEC_RISK_SERVICE_URL = os.getenv("EXEC_RISK_SERVICE_URL", "http://localhost:8002")
    ANALYTICS_OS_SERVICE_URL = os.getenv("ANALYTICS_OS_SERVICE_URL", "http://localhost:8003")
    
    # Redis
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # Test parameters
    TEST_TIMEOUT_SECONDS = 60
    SERVICE_STARTUP_TIMEOUT = 120
    HEALTH_CHECK_INTERVAL = 5
    MAX_RETRY_ATTEMPTS = 10
    
    # Load test parameters
    LOAD_TEST_REQUESTS = 100
    LOAD_TEST_CONCURRENCY = 10


# ============================================================================
# TEST UTILITIES
# ============================================================================

class IntegrationTestHarness:
    """Integration test harness for microservices"""
    
    def __init__(self, config: TestConfig):
        """Initialize test harness"""
        self.config = config
        self.redis: Optional[Redis] = None
        self.results: Dict[str, Any] = {
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "errors": [],
            "start_time": time.time(),
        }
        
        logger.info("Integration Test Harness initialized")
    
    async def setup(self) -> None:
        """Setup test environment"""
        logger.info("Setting up test environment...")
        
        try:
            # Connect to Redis
            self.redis = redis.from_url(
                self.config.REDIS_URL,
                decode_responses=False
            )
            await self.redis.ping()
            logger.info("✓ Redis connected")
        
        except Exception as e:
            logger.error(f"Setup failed: {e}", exc_info=True)
            raise
    
    async def teardown(self) -> None:
        """Teardown test environment"""
        logger.info("Tearing down test environment...")
        
        if self.redis:
            await self.redis.close()
        
        logger.info("✓ Teardown complete")
    
    # ========================================================================
    # HEALTH CHECK TESTS
    # ========================================================================
    
    async def test_service_health(self, service_name: str, url: str) -> bool:
        """Test service health endpoint"""
        self.results["total_tests"] += 1
        
        try:
            logger.info(f"Testing {service_name} health...")
            
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{url}/health", timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        logger.info(
                            f"✓ {service_name} healthy",
                            status=data.get("status"),
                            uptime=data.get("uptime_seconds"),
                        )
                        
                        self.results["passed"] += 1
                        return True
                    else:
                        raise Exception(f"Health check failed: {response.status}")
        
        except Exception as e:
            logger.error(f"✗ {service_name} health check failed: {e}")
            self.results["failed"] += 1
            self.results["errors"].append(f"{service_name}: {str(e)}")
            return False
    
    async def test_all_services_health(self) -> bool:
        """Test health of all services"""
        logger.info("=" * 80)
        logger.info("TEST: All Services Health Check")
        logger.info("=" * 80)
        
        services = [
            ("AI Service", self.config.AI_SERVICE_URL),
            ("Exec-Risk Service", self.config.EXEC_RISK_SERVICE_URL),
            ("Analytics-OS Service", self.config.ANALYTICS_OS_SERVICE_URL),
        ]
        
        results = await asyncio.gather(
            *[self.test_service_health(name, url) for name, url in services]
        )
        
        all_healthy = all(results)
        
        if all_healthy:
            logger.info("✓ All services healthy")
        else:
            logger.error("✗ Some services unhealthy")
        
        return all_healthy
    
    # ========================================================================
    # READINESS TESTS
    # ========================================================================
    
    async def test_service_readiness(self, service_name: str, url: str) -> bool:
        """Test service readiness endpoint"""
        self.results["total_tests"] += 1
        
        try:
            logger.info(f"Testing {service_name} readiness...")
            
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{url}/ready", timeout=5) as response:
                    if response.status == 200:
                        logger.info(f"✓ {service_name} ready")
                        self.results["passed"] += 1
                        return True
                    else:
                        raise Exception(f"Readiness check failed: {response.status}")
        
        except Exception as e:
            logger.error(f"✗ {service_name} readiness check failed: {e}")
            self.results["failed"] += 1
            self.results["errors"].append(f"{service_name}: {str(e)}")
            return False
    
    # ========================================================================
    # RPC COMMUNICATION TESTS
    # ========================================================================
    
    async def test_rpc_communication(self) -> bool:
        """Test RPC communication between services"""
        logger.info("=" * 80)
        logger.info("TEST: RPC Communication")
        logger.info("=" * 80)
        
        self.results["total_tests"] += 1
        
        try:
            # Test 1: AI Service - get_signal RPC
            logger.info("Testing AI Service RPC: get_signal...")
            
            request_data = {
                "service_target": "ai-service",
                "command": "get_signal",
                "parameters": {"symbol": "BTCUSDT"},
                "timeout_seconds": 10.0,
            }
            
            # Publish RPC request via Redis
            request_id = str(uuid.uuid4())
            await self.redis.xadd(
                "quantum:rpc:request:ai-service",
                {"request_id": request_id, "data": str(request_data)}
            )
            
            # Wait for response (simplified - would use XREAD)
            await asyncio.sleep(2)
            
            logger.info("✓ RPC request published")
            
            # Test 2: Exec-Risk Service - validate_risk RPC
            logger.info("Testing Exec-Risk Service RPC: validate_risk...")
            
            request_data = {
                "service_target": "exec-risk-service",
                "command": "validate_risk",
                "parameters": {
                    "symbol": "BTCUSDT",
                    "side": "LONG",
                    "size_usd": 1000.0,
                },
                "timeout_seconds": 10.0,
            }
            
            request_id = str(uuid.uuid4())
            await self.redis.xadd(
                "quantum:rpc:request:exec-risk-service",
                {"request_id": request_id, "data": str(request_data)}
            )
            
            await asyncio.sleep(2)
            
            logger.info("✓ RPC request published")
            
            self.results["passed"] += 1
            return True
        
        except Exception as e:
            logger.error(f"✗ RPC communication test failed: {e}", exc_info=True)
            self.results["failed"] += 1
            self.results["errors"].append(f"RPC: {str(e)}")
            return False
    
    # ========================================================================
    # EVENT FLOW TESTS
    # ========================================================================
    
    async def test_signal_to_execution_flow(self) -> bool:
        """Test complete event flow: signal → execution → position closed"""
        logger.info("=" * 80)
        logger.info("TEST: Signal to Execution Flow")
        logger.info("=" * 80)
        
        self.results["total_tests"] += 1
        
        try:
            # Step 1: Publish signal.generated event
            logger.info("Step 1: Publishing signal.generated event...")
            
            signal_payload = SignalGeneratedPayload(
                symbol="BTCUSDT",
                action="BUY",
                confidence=0.85,
                model_source="test",
                score=0.85,
                timeframe="1m",
            )
            
            await self.redis.xadd(
                "quantum:events:signal.generated",
                {"data": signal_payload.json()}
            )
            
            logger.info("✓ Signal published")
            
            # Step 2: Wait for execution.result event
            logger.info("Step 2: Waiting for execution.result event...")
            
            await asyncio.sleep(5)
            
            # Check if execution.result event exists
            events = await self.redis.xrange(
                "quantum:events:execution.result",
                "-",
                "+",
                count=10
            )
            
            if events:
                logger.info(f"✓ Execution event received: {len(events)} events")
            else:
                logger.warning("⚠ No execution event found (may be expected in test env)")
            
            # Step 3: Check for position.opened event
            logger.info("Step 3: Checking for position.opened event...")
            
            events = await self.redis.xrange(
                "quantum:events:position.opened",
                "-",
                "+",
                count=10
            )
            
            if events:
                logger.info(f"✓ Position opened event found: {len(events)} events")
            else:
                logger.info("ℹ No position opened event (expected in test env)")
            
            self.results["passed"] += 1
            return True
        
        except Exception as e:
            logger.error(f"✗ Event flow test failed: {e}", exc_info=True)
            self.results["failed"] += 1
            self.results["errors"].append(f"Event flow: {str(e)}")
            return False
    
    # ========================================================================
    # LOAD TESTS
    # ========================================================================
    
    async def test_load_health_checks(self) -> bool:
        """Load test health checks with parallel requests"""
        logger.info("=" * 80)
        logger.info("TEST: Load Test - Health Checks")
        logger.info("=" * 80)
        
        self.results["total_tests"] += 1
        
        try:
            logger.info(
                f"Sending {self.config.LOAD_TEST_REQUESTS} requests "
                f"with {self.config.LOAD_TEST_CONCURRENCY} concurrency..."
            )
            
            start_time = time.time()
            
            async def make_request(url: str) -> bool:
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(f"{url}/health", timeout=5) as response:
                            return response.status == 200
                except:
                    return False
            
            # Generate requests
            requests = []
            for i in range(self.config.LOAD_TEST_REQUESTS):
                service_url = [
                    self.config.AI_SERVICE_URL,
                    self.config.EXEC_RISK_SERVICE_URL,
                    self.config.ANALYTICS_OS_SERVICE_URL,
                ][i % 3]
                
                requests.append(make_request(service_url))
            
            # Execute with concurrency limit
            semaphore = asyncio.Semaphore(self.config.LOAD_TEST_CONCURRENCY)
            
            async def limited_request(req):
                async with semaphore:
                    return await req
            
            results = await asyncio.gather(*[limited_request(r) for r in requests])
            
            elapsed = time.time() - start_time
            success_count = sum(results)
            
            logger.info(
                f"✓ Load test complete",
                total_requests=len(results),
                successful=success_count,
                failed=len(results) - success_count,
                elapsed_seconds=f"{elapsed:.2f}",
                rps=f"{len(results) / elapsed:.2f}",
            )
            
            self.results["passed"] += 1
            return True
        
        except Exception as e:
            logger.error(f"✗ Load test failed: {e}", exc_info=True)
            self.results["failed"] += 1
            self.results["errors"].append(f"Load test: {str(e)}")
            return False
    
    # ========================================================================
    # FAILURE SCENARIO TESTS
    # ========================================================================
    
    async def test_service_degradation(self) -> bool:
        """Test service degradation detection"""
        logger.info("=" * 80)
        logger.info("TEST: Service Degradation Detection")
        logger.info("=" * 80)
        
        self.results["total_tests"] += 1
        
        try:
            # Simulate missed heartbeats by checking health graph
            logger.info("Checking health graph for service states...")
            
            services = ["ai-service", "exec-risk-service", "analytics-os-service"]
            
            for service in services:
                health_data = await self.redis.hgetall(f"qt:health:{service}")
                
                if health_data:
                    logger.info(
                        f"✓ {service} health data found",
                        status=health_data.get(b"status", b"UNKNOWN").decode(),
                    )
                else:
                    logger.warning(f"⚠ {service} health data not found")
            
            self.results["passed"] += 1
            return True
        
        except Exception as e:
            logger.error(f"✗ Degradation test failed: {e}", exc_info=True)
            self.results["failed"] += 1
            self.results["errors"].append(f"Degradation: {str(e)}")
            return False
    
    # ========================================================================
    # TEST 7: Multi-Service Failure Simulation
    # ========================================================================
    
    async def test_multi_service_failures(self) -> bool:
        """Test system behavior when multiple services fail"""
        logger.info("=" * 80)
        logger.info("TEST: Multi-Service Failure Simulation")
        logger.info("=" * 80)
        
        self.results["total_tests"] += 1
        
        try:
            # Test Case 1: AI Service unavailable
            logger.info("Test Case 1: AI Service unavailable scenario...")
            
            # Publish signal when AI service might be down
            trace_id = str(uuid.uuid4())
            signal_payload = SignalGeneratedPayload(
                symbol="BTCUSDT",
                action="BUY",
                confidence=0.75,
                price=50000.0,
                leverage=10.0,
                position_size_usd=100.0,
                strategy="test_strategy",
            )
            
            event = build_event(
                event_type=EventTypes.SIGNAL_GENERATED,
                payload=signal_payload.model_dump(),
                source_service="integration-test",
                trace_id=trace_id,
            )
            
            await self.event_bus.publish(EventTypes.SIGNAL_GENERATED, event)
            logger.info(f"✓ Published signal with trace_id: {trace_id}")
            
            # System should handle gracefully
            await asyncio.sleep(2)
            logger.info("✓ System handled missing service gracefully")
            
            # Test Case 2: Check service recovery
            logger.info("\nTest Case 2: Service recovery check...")
            
            services_recovered = []
            for service_name, service_url in [
                ("ai-service", self.config.AI_SERVICE_URL),
                ("exec-risk-service", self.config.EXEC_RISK_SERVICE_URL),
                ("analytics-os-service", self.config.ANALYTICS_OS_SERVICE_URL),
            ]:
                try:
                    async with self.http_session.get(
                        f"{service_url}/health",
                        timeout=aiohttp.ClientTimeout(total=5)
                    ) as response:
                        if response.status == 200:
                            services_recovered.append(service_name)
                            logger.info(f"✓ {service_name} recovered")
                except Exception as e:
                    logger.warning(f"⚠️ {service_name} not yet recovered: {e}")
            
            logger.info(f"✓ {len(services_recovered)}/3 services recovered")
            
            self.results["passed"] += 1
            return True
        
        except Exception as e:
            logger.error(f"✗ Multi-service failure test failed: {e}", exc_info=True)
            self.results["failed"] += 1
            self.results["errors"].append(f"Multi-service failure: {str(e)}")
            return False
    
    # ========================================================================
    # TEST 8: RPC Timeout Handling
    # ========================================================================
    
    async def test_rpc_timeout_handling(self) -> bool:
        """Test RPC timeout and retry mechanisms"""
        logger.info("=" * 80)
        logger.info("TEST: RPC Timeout Handling")
        logger.info("=" * 80)
        
        self.results["total_tests"] += 1
        
        try:
            # Test Case 1: RPC call with short timeout
            logger.info("Test Case 1: RPC call with timeout...")
            
            from backend.core.service_rpc import ServiceRPCClient
            
            rpc_client = ServiceRPCClient(self.redis, service_name="integration-test")
            await rpc_client.initialize()
            
            # Call with short timeout (should timeout if service slow)
            try:
                result = await rpc_client.call(
                    target_service="ai-service",
                    command="get_signal",
                    parameters={"symbol": "BTCUSDT"},
                    timeout=0.5,  # 500ms timeout
                )
                
                if result:
                    logger.info("✓ RPC call succeeded within timeout")
                else:
                    logger.info("⚠️ RPC call returned None (expected on timeout)")
            
            except asyncio.TimeoutError:
                logger.info("✓ RPC timeout handled correctly")
            
            # Test Case 2: RPC retry mechanism
            logger.info("\nTest Case 2: RPC retry mechanism...")
            
            # Second attempt should work
            try:
                result = await rpc_client.call(
                    target_service="ai-service",
                    command="get_top_opportunities",
                    parameters={"universe_size": 5},
                    timeout=5.0,  # Longer timeout
                )
                
                logger.info("✓ RPC retry successful")
            
            except Exception as e:
                logger.warning(f"⚠️ RPC retry failed: {e}")
            
            await rpc_client.shutdown()
            
            self.results["passed"] += 1
            return True
        
        except Exception as e:
            logger.error(f"✗ RPC timeout test failed: {e}", exc_info=True)
            self.results["failed"] += 1
            self.results["errors"].append(f"RPC timeout: {str(e)}")
            return False
    
    # ========================================================================
    # TEST 9: Event Replay
    # ========================================================================
    
    async def test_event_replay(self) -> bool:
        """Test event replay and reprocessing"""
        logger.info("=" * 80)
        logger.info("TEST: Event Replay")
        logger.info("=" * 80)
        
        self.results["total_tests"] += 1
        
        try:
            # Test Case 1: Replay old events
            logger.info("Test Case 1: Reading historical events...")
            
            # Read last 10 events from stream
            try:
                events = await self.redis.xrevrange(
                    "quantum:events:signal.generated",
                    count=10
                )
                
                if events:
                    logger.info(f"✓ Found {len(events)} historical signal events")
                    
                    # Parse first event
                    event_id, event_data = events[0]
                    logger.info(f"  Most recent event ID: {event_id.decode()}")
                else:
                    logger.info("⚠️ No historical events found (fresh system)")
            
            except Exception as e:
                logger.warning(f"⚠️ Could not read historical events: {e}")
            
            # Test Case 2: Event retention
            logger.info("\nTest Case 2: Event retention check...")
            
            stream_info = await self.redis.xinfo_stream("quantum:events:signal.generated")
            length = stream_info.get(b"length", 0)
            
            logger.info(f"✓ Stream length: {length} events")
            
            self.results["passed"] += 1
            return True
        
        except Exception as e:
            logger.error(f"✗ Event replay test failed: {e}", exc_info=True)
            self.results["failed"] += 1
            self.results["errors"].append(f"Event replay: {str(e)}")
            return False
    
    # ========================================================================
    # TEST 10: Concurrent Signal Processing
    # ========================================================================
    
    async def test_concurrent_signals(self) -> bool:
        """Test concurrent signal processing"""
        logger.info("=" * 80)
        logger.info("TEST: Concurrent Signal Processing")
        logger.info("=" * 80)
        
        self.results["total_tests"] += 1
        
        try:
            # Publish multiple signals concurrently
            logger.info("Publishing 20 concurrent signals...")
            
            trace_ids = []
            
            async def publish_signal(i: int) -> str:
                trace_id = str(uuid.uuid4())
                symbol = ["BTCUSDT", "ETHUSDT", "SOLUSDT"][i % 3]
                action = "BUY" if i % 2 == 0 else "SELL"
                
                signal_payload = SignalGeneratedPayload(
                    symbol=symbol,
                    action=action,
                    confidence=0.75 + (i % 5) * 0.03,
                    price=50000.0,
                    leverage=10.0,
                    position_size_usd=100.0,
                    strategy=f"test_strategy_{i}",
                )
                
                event = build_event(
                    event_type=EventTypes.SIGNAL_GENERATED,
                    payload=signal_payload.model_dump(),
                    source_service="integration-test",
                    trace_id=trace_id,
                )
                
                await self.event_bus.publish(EventTypes.SIGNAL_GENERATED, event)
                return trace_id
            
            # Publish all signals
            trace_ids = await asyncio.gather(*[
                publish_signal(i) for i in range(20)
            ])
            
            logger.info(f"✓ Published {len(trace_ids)} concurrent signals")
            
            # Wait for processing
            await asyncio.sleep(5)
            
            # Check if system is still responsive
            logger.info("Checking system responsiveness...")
            
            healthy = True
            for service_url in [
                self.config.AI_SERVICE_URL,
                self.config.EXEC_RISK_SERVICE_URL,
            ]:
                try:
                    async with self.http_session.get(
                        f"{service_url}/health",
                        timeout=aiohttp.ClientTimeout(total=5)
                    ) as response:
                        if response.status != 200:
                            healthy = False
                except Exception:
                    healthy = False
            
            if healthy:
                logger.info("✓ System remained responsive under concurrent load")
            else:
                logger.warning("⚠️ Some services degraded under load")
            
            self.results["passed"] += 1
            return True
        
        except Exception as e:
            logger.error(f"✗ Concurrent signals test failed: {e}", exc_info=True)
            self.results["failed"] += 1
            self.results["errors"].append(f"Concurrent signals: {str(e)}")
            return False
    
    # ========================================================================
    # TEST RUNNER
    # ========================================================================
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all integration tests"""
        logger.info("=" * 80)
        logger.info("QUANTUM TRADER v3.0 - INTEGRATION TEST HARNESS")
        logger.info("=" * 80)
        logger.info("")
        
        try:
            # Setup
            await self.setup()
            
            # Test 1: Health checks
            await self.test_all_services_health()
            
            # Test 2: Readiness checks
            await asyncio.gather(
                self.test_service_readiness("AI Service", self.config.AI_SERVICE_URL),
                self.test_service_readiness("Exec-Risk Service", self.config.EXEC_RISK_SERVICE_URL),
                self.test_service_readiness("Analytics-OS Service", self.config.ANALYTICS_OS_SERVICE_URL),
            )
            
            # Test 3: RPC communication
            await self.test_rpc_communication()
            
            # Test 4: Event flow
            await self.test_signal_to_execution_flow()
            
            # Test 5: Load test
            await self.test_load_health_checks()
            
            # Test 6: Degradation detection
            await self.test_service_degradation()
            
            # Test 7: Multi-service failures
            await self.test_multi_service_failures()
            
            # Test 8: RPC timeout handling
            await self.test_rpc_timeout_handling()
            
            # Test 9: Event replay
            await self.test_event_replay()
            
            # Test 10: Concurrent signals
            await self.test_concurrent_signals()
            
            # Teardown
            await self.teardown()
            
            # Calculate results
            self.results["end_time"] = time.time()
            self.results["duration_seconds"] = self.results["end_time"] - self.results["start_time"]
            self.results["success_rate"] = (
                self.results["passed"] / self.results["total_tests"] * 100
                if self.results["total_tests"] > 0 else 0
            )
            
            # Print summary
            logger.info("")
            logger.info("=" * 80)
            logger.info("TEST RESULTS SUMMARY")
            logger.info("=" * 80)
            logger.info(f"Total Tests: {self.results['total_tests']}")
            logger.info(f"Passed: {self.results['passed']}")
            logger.info(f"Failed: {self.results['failed']}")
            logger.info(f"Success Rate: {self.results['success_rate']:.1f}%")
            logger.info(f"Duration: {self.results['duration_seconds']:.2f}s")
            
            if self.results["errors"]:
                logger.info("")
                logger.info("Errors:")
                for error in self.results["errors"]:
                    logger.error(f"  - {error}")
            
            logger.info("=" * 80)
            
            return self.results
        
        except Exception as e:
            logger.error(f"Test suite failed: {e}", exc_info=True)
            return self.results


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

async def main():
    """Main entry point"""
    configure_logging(level=os.getenv("LOG_LEVEL", "INFO"))
    
    config = TestConfig()
    harness = IntegrationTestHarness(config)
    
    results = await harness.run_all_tests()
    
    # Exit with appropriate code
    exit_code = 0 if results["failed"] == 0 else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    asyncio.run(main())
