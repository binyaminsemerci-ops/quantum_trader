"""
Quick validation test for Execution Service V2

Tests:
1. Import checks
2. RiskStub validation
3. BinanceAdapter initialization
4. Config loading
"""
import sys
import asyncio
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 60)
print("EXECUTION SERVICE V2 - VALIDATION TEST")
print("=" * 60)

# Test 1: Imports
print("\n[TEST 1] Checking imports...")
try:
    from microservices.execution.config import settings
    from microservices.execution.risk_stub import RiskStub
    from microservices.execution.binance_adapter import BinanceAdapter, ExecutionMode
    from microservices.execution.service_v2 import ExecutionService, SimpleRateLimiter
    print("✅ All imports successful")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test 2: Config
print("\n[TEST 2] Checking config...")
try:
    print(f"  Service: {settings.SERVICE_NAME} v{settings.VERSION}")
    print(f"  Port: {settings.PORT}")
    print(f"  Mode: {settings.EXECUTION_MODE}")
    print(f"  Max position: ${settings.MAX_POSITION_USD}")
    print(f"  Max leverage: {settings.MAX_LEVERAGE}x")
    print("✅ Config loaded")
except Exception as e:
    print(f"❌ Config failed: {e}")
    sys.exit(1)

# Test 3: RiskStub
print("\n[TEST 3] Testing RiskStub...")
async def test_risk_stub():
    stub = RiskStub(max_position_usd=1000, max_leverage=10)
    
    # Test 1: Valid trade
    result = await stub.validate_trade("BTCUSDT", "BUY", 500, 5, 50000)
    assert result["allowed"] == True, "Valid trade should be allowed"
    print("  ✓ Valid trade accepted")
    
    # Test 2: Invalid symbol
    result = await stub.validate_trade("XXXUSDT", "BUY", 500, 5, 100)
    assert result["allowed"] == False, "Invalid symbol should be rejected"
    print("  ✓ Invalid symbol rejected")
    
    # Test 3: Oversized position
    result = await stub.validate_trade("BTCUSDT", "BUY", 2000, 1, 50000)
    assert result["allowed"] == False, "Oversized position should be rejected"
    print("  ✓ Oversized position rejected")
    
    # Test 4: Excessive leverage
    result = await stub.validate_trade("BTCUSDT", "BUY", 500, 20, 50000)
    assert result["allowed"] == False, "Excessive leverage should be rejected"
    print("  ✓ Excessive leverage rejected")
    
    print("✅ RiskStub validation passed")

try:
    asyncio.run(test_risk_stub())
except Exception as e:
    print(f"❌ RiskStub test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: BinanceAdapter (PAPER mode)
print("\n[TEST 4] Testing BinanceAdapter (PAPER mode)...")
async def test_binance():
    adapter = BinanceAdapter(mode=ExecutionMode.PAPER)
    await adapter.connect()
    
    # Test paper order
    result = await adapter.place_market_order("BTCUSDT", "BUY", 0.01, 1)
    assert result["status"] == "FILLED", "Paper order should be filled"
    assert result["order_id"] is not None, "Paper order should have ID"
    print(f"  ✓ Paper order: {result['order_id']} - {result['symbol']} @ ${result['price']}")
    
    # Test price fetch
    price = await adapter.get_current_price("BTCUSDT")
    assert price > 0, "Price should be positive"
    print(f"  ✓ Price fetch: ${price}")
    
    # Test balance
    balance = await adapter.get_account_balance()
    assert balance == 10000.0, "Paper account should start with $10k"
    print(f"  ✓ Balance: ${balance}")
    
    await adapter.close()
    print("✅ BinanceAdapter test passed")

try:
    asyncio.run(test_binance())
except Exception as e:
    print(f"❌ BinanceAdapter test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: SimpleRateLimiter
print("\n[TEST 5] Testing SimpleRateLimiter...")
async def test_rate_limiter():
    limiter = SimpleRateLimiter(max_requests_per_minute=10)
    
    # Should allow first 10 requests immediately
    for i in range(10):
        await limiter.acquire()
    
    print("  ✓ Allowed 10 requests")
    print("✅ RateLimiter test passed")

try:
    asyncio.run(test_rate_limiter())
except Exception as e:
    print(f"❌ RateLimiter test failed: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED")
print("=" * 60)
print("\nExecution Service V2 is ready for deployment!")
print("")
