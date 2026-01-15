#!/usr/bin/env python3
"""
Orchestrator Contract Tests - Clean deterministic version
Returns proper exit codes: 0 = success, 1 = failure
"""

import sys
import os
import asyncio
import httpx
import logging
from pathlib import Path

# Add project to path
sys.path.append("/home/qt/quantum_trader")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

class ContractTester:
    """Tests orchestrator-brain service contracts with proper exit codes."""
    
    def __init__(self):
        self.strategy_url = "http://127.0.0.1:8011"
        self.failures = []
        
    def fail(self, test_name, reason):
        """Record a test failure."""
        self.failures.append(f"{test_name}: {reason}")
        logger.error("FAIL %s: %s", test_name, reason)
    
    def test_schema_adapter(self):
        """Test schema adapter handles various signal formats."""
        logger.info("Testing Schema Adapter...")
        
        try:
            from backend.services.orchestration.schema_adapter import SchemaAdapter
            
            test_cases = [
                {"symbol": "BTCUSDT", "side": "LONG", "confidence": 0.7},
                {"symbol": "ETHUSDT", "direction": "SHORT", "confidence": 0.6},
                {"symbol": "ADAUSDT", "side": "BUY", "confidence": 0.8, "price": 1.20},
                {"symbol": "DOTUSDT", "confidence": 0.5}  # Missing direction
            ]
            
            for i, signal in enumerate(test_cases):
                payload = SchemaAdapter.normalize_strategy_payload(signal)
                
                # Validate required fields
                if "symbol" not in payload:
                    self.fail("SchemaAdapter", f"Case {i}: Missing symbol")
                    return False
                    
                if "direction" not in payload or payload["direction"] not in ["BUY", "SELL"]:
                    self.fail("SchemaAdapter", f"Case {i}: Invalid direction: {payload.get( direction)}")
                    return False
                    
                if "confidence" not in payload or not isinstance(payload["confidence"], float):
                    self.fail("SchemaAdapter", f"Case {i}: Invalid confidence: {payload.get(confidence)}")
                    return False
                    
                logger.info("✓ Case %d: %s %s", i, payload["direction"], payload["symbol"])
            
            logger.info("✓ Schema Adapter: PASS")
            return True
            
        except Exception as e:
            self.fail("SchemaAdapter", f"Exception: {e}")
            return False
    
    async def test_strategy_brain_contract(self):
        """Test Strategy Brain accepts contract and returns 200."""
        logger.info("Testing Strategy Brain Contract...")
        
        payload = {
            "symbol": "BTCUSDT",
            "direction": "BUY",
            "confidence": 0.75
        }
        
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(f"{self.strategy_url}/evaluate", json=payload)
                
                if response.status_code != 200:
                    self.fail("StrategyBrain", f"Expected 200, got {response.status_code}")
                    return False
                
                data = response.json()
                if "approved" not in data:
                    self.fail("StrategyBrain", "Missing approved field")
                    return False
                    
                if "reason" not in data:
                    self.fail("StrategyBrain", "Missing reason field")
                    return False
                
                logger.info("✓ Strategy Brain: 200 OK, approved=%s", data.get("approved"))
                return True
                
        except Exception as e:
            self.fail("StrategyBrain", f"Exception: {e}")
            return False
    
    async def test_orchestrator_integration(self):
        """Test orchestrator chain integration."""
        logger.info("Testing Orchestrator Integration...")
        
        try:
            from backend.services.orchestration.orchestrator import Orchestrator
            
            test_signal = {
                "symbol": "BTCUSDT",
                "side": "LONG",
                "confidence": 0.8,
                "price": 43000.0
            }
            
            orchestrator = Orchestrator()
            result = await orchestrator.evaluate_signal(test_signal)
            
            if result.strategy_approved is None:
                self.fail("Orchestrator", "Strategy approval is None")
                return False
                
            if result.operating_mode not in ["EXPANSION", "PRESERVATION", "EMERGENCY"]:
                self.fail("Orchestrator", f"Invalid operating mode: {result.operating_mode}")
                return False
                
            if result.final_decision not in ["EXECUTE", "SKIP", "DELAY"]:
                self.fail("Orchestrator", f"Invalid final decision: {result.final_decision}")
                return False
            
            logger.info("✓ Orchestrator: %s, mode=%s", result.final_decision, result.operating_mode)
            return True
            
        except Exception as e:
            self.fail("Orchestrator", f"Exception: {e}")
            return False
    
    async def run_all_tests(self):
        """Run all contract tests and return exit code."""
        logger.info("=== Starting Orchestrator Contract Tests ===")
        
        test_results = [
            self.test_schema_adapter(),
            await self.test_strategy_brain_contract(),
            await self.test_orchestrator_integration(),
        ]
        
        passed = sum(test_results)
        total = len(test_results)
        
        logger.info("=== RESULTS ===")
        if self.failures:
            for failure in self.failures:
                logger.error("FAILURE: %s", failure)
        
        logger.info("Tests passed: %d/%d", passed, total)
        
        if passed == total:
            logger.info("✓ ALL TESTS PASSED")
            return 0  # Success
        else:
            logger.error("✗ %d TESTS FAILED", total - passed)
            return 1  # Failure

async def main():
    """Main entry point."""
    tester = ContractTester()
    exit_code = await tester.run_all_tests()
    return exit_code

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
