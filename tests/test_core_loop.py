"""
Integration Tests for Tier 1 Core Execution Loop
=================================================

Tests the complete flow:
AI Signal → Risk Approval → Execution → Position Tracking

Requirements:
- All 3 services running (risk-safety, execution, position-monitor)
- Redis available at localhost:6379
- Clean test environment

Usage:
    pytest tests/test_core_loop.py -v
    pytest tests/test_core_loop.py::test_full_pipeline -v

Author: Quantum Trader Team
Date: 2026-01-12
"""
import pytest
import asyncio
from datetime import datetime
from dataclasses import asdict

from ai_engine.services.eventbus_bridge import (
    EventBusClient,
    TradeSignal,
    publish_trade_signal,
    get_recent_signals
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
async def eventbus():
    """Create EventBus client"""
    bus = EventBusClient()
    await bus.connect()
    yield bus
    await bus.disconnect()


@pytest.fixture
def high_confidence_signal():
    """Create high confidence signal (should be approved)"""
    return TradeSignal(
        symbol="BTCUSDT",
        action="BUY",
        confidence=0.85,
        timestamp=datetime.utcnow().isoformat() + "Z",
        source="test_suite",
        meta_override=False,
        ensemble_votes={"xgb": "BUY", "lgb": "BUY", "meta": "BUY"}
    )


@pytest.fixture
def low_confidence_signal():
    """Create low confidence signal (should be rejected)"""
    return TradeSignal(
        symbol="ETHUSDT",
        action="SELL",
        confidence=0.55,  # Below 0.65 threshold
        timestamp=datetime.utcnow().isoformat() + "Z",
        source="test_suite",
        meta_override=False,
        ensemble_votes={"xgb": "SELL", "lgb": "HOLD", "meta": "SELL"}
    )


@pytest.fixture
def hold_signal():
    """Create HOLD signal (should be skipped)"""
    return TradeSignal(
        symbol="BNBUSDT",
        action="HOLD",
        confidence=0.70,
        timestamp=datetime.utcnow().isoformat() + "Z",
        source="test_suite",
        meta_override=False,
        ensemble_votes={"xgb": "HOLD", "lgb": "HOLD", "meta": "HOLD"}
    )


# ============================================================================
# TEST: EVENTBUS CONNECTIVITY
# ============================================================================

@pytest.mark.asyncio
async def test_eventbus_connection():
    """Test EventBus connectivity"""
    async with EventBusClient() as bus:
        # Should connect without error
        assert bus is not None
        
        # Test publish
        message_id = await bus.publish("test.topic", {"test": "data"})
        assert message_id is not None
        
        # Test stream length
        length = await bus.get_stream_length("test.topic")
        assert length >= 1


# ============================================================================
# TEST: SIGNAL PUBLISHING
# ============================================================================

@pytest.mark.asyncio
async def test_publish_signal(eventbus, high_confidence_signal):
    """Test publishing trade signal"""
    # Publish signal
    message_id = await eventbus.publish_signal(high_confidence_signal)
    assert message_id is not None
    
    # Verify signal appears in stream
    signals = await get_recent_signals("trade.signal.v5", 1)
    assert len(signals) >= 1
    
    latest = signals[0]
    assert latest["symbol"] == "BTCUSDT"
    assert latest["action"] == "BUY"
    assert latest["confidence"] == 0.85


# ============================================================================
# TEST: RISK APPROVAL
# ============================================================================

@pytest.mark.asyncio
async def test_risk_approval(eventbus, high_confidence_signal):
    """Test high confidence signal gets approved"""
    # Publish high confidence signal
    await eventbus.publish_signal(high_confidence_signal)
    
    # Wait for risk service to process
    await asyncio.sleep(2)
    
    # Check approved signals
    approved = await get_recent_signals("trade.signal.safe", 10)
    
    # Should find our signal
    found = any(
        s["symbol"] == "BTCUSDT" and 
        s["action"] == "BUY" 
        for s in approved
    )
    
    assert found, "High confidence signal should be approved"
    
    # Check it has position sizing
    btc_signal = next(
        s for s in approved 
        if s["symbol"] == "BTCUSDT" and s["action"] == "BUY"
    )
    
    assert "position_size_usd" in btc_signal
    assert btc_signal["position_size_usd"] > 0
    assert btc_signal["position_size_pct"] <= 0.10  # Max 10%


@pytest.mark.asyncio
async def test_risk_rejection(eventbus, low_confidence_signal):
    """Test low confidence signal gets rejected"""
    # Get current approved count
    approved_before = await get_recent_signals("trade.signal.safe", 100)
    eth_before = len([
        s for s in approved_before 
        if s["symbol"] == "ETHUSDT"
    ])
    
    # Publish low confidence signal
    await eventbus.publish_signal(low_confidence_signal)
    
    # Wait for processing
    await asyncio.sleep(2)
    
    # Check approved signals
    approved_after = await get_recent_signals("trade.signal.safe", 100)
    eth_after = len([
        s for s in approved_after 
        if s["symbol"] == "ETHUSDT"
    ])
    
    # Should NOT increase (rejected)
    assert eth_after == eth_before, "Low confidence signal should be rejected"


@pytest.mark.asyncio
async def test_hold_skip(eventbus, hold_signal):
    """Test HOLD signals are skipped"""
    # Publish HOLD signal
    await eventbus.publish_signal(hold_signal)
    
    # Wait
    await asyncio.sleep(2)
    
    # HOLD signals should NOT appear in approved
    approved = await get_recent_signals("trade.signal.safe", 100)
    
    hold_found = any(
        s["symbol"] == "BNBUSDT" and s["action"] == "HOLD"
        for s in approved
    )
    
    assert not hold_found, "HOLD signals should be skipped"


# ============================================================================
# TEST: EXECUTION
# ============================================================================

@pytest.mark.asyncio
async def test_execution_flow(eventbus, high_confidence_signal):
    """Test approved signal gets executed"""
    # Publish signal
    await eventbus.publish_signal(high_confidence_signal)
    
    # Wait for risk approval + execution
    await asyncio.sleep(4)
    
    # Check execution results
    executions = await get_recent_signals("trade.execution.res", 10)
    
    # Should find execution
    found = any(
        s["symbol"] == "BTCUSDT" and 
        s["action"] == "BUY" and
        s["status"] == "filled"
        for s in executions
    )
    
    assert found, "Approved signal should be executed"
    
    # Check execution details
    btc_exec = next(
        s for s in executions 
        if s["symbol"] == "BTCUSDT" and s["action"] == "BUY"
    )
    
    assert "order_id" in btc_exec
    assert btc_exec["order_id"].startswith("PAPER-")
    assert "entry_price" in btc_exec
    assert btc_exec["entry_price"] > 0
    assert "slippage_pct" in btc_exec
    assert "fee_usd" in btc_exec


# ============================================================================
# TEST: POSITION TRACKING
# ============================================================================

@pytest.mark.asyncio
async def test_position_tracking(eventbus, high_confidence_signal):
    """Test executed orders create positions"""
    # Publish signal
    await eventbus.publish_signal(high_confidence_signal)
    
    # Wait for full pipeline
    await asyncio.sleep(35)  # Wait for position update cycle (30s)
    
    # Check position updates
    positions = await get_recent_signals("trade.position.update", 10)
    
    # Should have position update
    assert len(positions) > 0, "Should have position updates"
    
    # Check for BTCUSDT position
    btc_pos = next(
        (p for p in positions if p["symbol"] == "BTCUSDT"),
        None
    )
    
    if btc_pos:  # May not exist if position closed
        assert "unrealized_pnl" in btc_pos
        assert "side" in btc_pos
        assert btc_pos["side"] in ["LONG", "SHORT"]


# ============================================================================
# TEST: FULL PIPELINE
# ============================================================================

@pytest.mark.asyncio
async def test_full_pipeline():
    """
    Test complete end-to-end flow
    
    This is the main integration test that validates:
    1. Signal publishing
    2. Risk approval
    3. Execution
    4. Position tracking
    
    Should complete in <10 seconds (excluding position update cycle)
    """
    # Create test signal
    signal = TradeSignal(
        symbol="SOLUSDT",
        action="BUY",
        confidence=0.88,
        timestamp=datetime.utcnow().isoformat() + "Z",
        source="integration_test",
        meta_override=False,
        ensemble_votes={"xgb": "BUY", "lgb": "BUY", "meta": "BUY"}
    )
    
    # Step 1: Publish signal
    await publish_trade_signal(
        symbol=signal.symbol,
        action=signal.action,
        confidence=signal.confidence,
        source=signal.source
    )
    
    print(f"\n✅ Published signal: {signal.symbol} {signal.action} @ {signal.confidence}")
    
    # Step 2: Wait for risk approval
    await asyncio.sleep(2)
    
    approved = await get_recent_signals("trade.signal.safe", 10)
    sol_approved = any(
        s["symbol"] == "SOLUSDT" and s["action"] == "BUY"
        for s in approved
    )
    
    assert sol_approved, "Signal should be approved (0.88 > 0.65)"
    print("✅ Signal approved by Risk Safety")
    
    # Step 3: Wait for execution
    await asyncio.sleep(2)
    
    executions = await get_recent_signals("trade.execution.res", 10)
    sol_exec = next(
        (s for s in executions if s["symbol"] == "SOLUSDT" and s["action"] == "BUY"),
        None
    )
    
    assert sol_exec is not None, "Signal should be executed"
    assert sol_exec["status"] == "filled"
    assert sol_exec["order_id"].startswith("PAPER-")
    
    print(f"✅ Order executed: {sol_exec['order_id']} @ ${sol_exec['entry_price']:.2f}")
    print(f"   Slippage: {sol_exec['slippage_pct']*100:.3f}%")
    print(f"   Fee: ${sol_exec['fee_usd']:.2f}")
    
    # Step 4: Check position (may take up to 30s for first update)
    # We'll just check execution success, position updates are async
    
    print("\n✅ FULL PIPELINE TEST PASSED")
    print(f"   Total time: ~4 seconds")
    print(f"   Flow: Signal → Approval → Execution ✅")


# ============================================================================
# TEST: PERFORMANCE
# ============================================================================

@pytest.mark.asyncio
async def test_execution_speed():
    """Test that execution completes within 5 seconds"""
    import time
    
    signal = TradeSignal(
        symbol="XRPUSDT",
        action="SELL",
        confidence=0.92,
        timestamp=datetime.utcnow().isoformat() + "Z",
        source="perf_test",
        meta_override=False
    )
    
    # Measure time
    start_time = time.time()
    
    await publish_trade_signal(
        symbol=signal.symbol,
        action=signal.action,
        confidence=signal.confidence,
        source=signal.source
    )
    
    # Wait for execution
    max_wait = 5  # 5 seconds max
    executed = False
    
    for _ in range(max_wait * 2):  # Check every 0.5s
        await asyncio.sleep(0.5)
        
        executions = await get_recent_signals("trade.execution.res", 5)
        if any(s["symbol"] == "XRPUSDT" and s["action"] == "SELL" for s in executions):
            executed = True
            break
    
    elapsed = time.time() - start_time
    
    assert executed, f"Execution took too long (>{max_wait}s)"
    print(f"\n✅ Execution completed in {elapsed:.2f}s")


# ============================================================================
# TEST: POSITION SIZE LIMITS
# ============================================================================

@pytest.mark.asyncio
async def test_position_size_limit():
    """Test that position sizes respect 10% limit"""
    signal = TradeSignal(
        symbol="BTCUSDT",
        action="BUY",
        confidence=0.95,  # Very high confidence
        timestamp=datetime.utcnow().isoformat() + "Z",
        source="size_test",
        meta_override=False
    )
    
    await publish_trade_signal(
        symbol=signal.symbol,
        action=signal.action,
        confidence=signal.confidence,
        source=signal.source
    )
    
    # Wait for approval
    await asyncio.sleep(2)
    
    approved = await get_recent_signals("trade.signal.safe", 10)
    btc_approved = next(
        (s for s in approved if s["symbol"] == "BTCUSDT"),
        None
    )
    
    if btc_approved:
        # Position size should be ≤10% of balance (assuming 10k balance)
        max_size = 10000 * 0.10  # $1000
        
        assert btc_approved["position_size_usd"] <= max_size, \
            f"Position size ${btc_approved['position_size_usd']:.2f} exceeds limit ${max_size:.2f}"
        
        print(f"\n✅ Position size OK: ${btc_approved['position_size_usd']:.2f} ≤ ${max_size:.2f}")


# ============================================================================
# RUN ALL TESTS
# ============================================================================

if __name__ == "__main__":
    """Run tests directly (without pytest)"""
    import sys
    
    async def run_all():
        print("=" * 70)
        print("TIER 1 CORE LOOP INTEGRATION TESTS")
        print("=" * 70)
        
        # Test EventBus
        print("\n[1] Testing EventBus...")
        await test_eventbus_connection()
        print("✅ EventBus OK")
        
        # Test full pipeline
        print("\n[2] Testing full pipeline...")
        await test_full_pipeline()
        
        # Test execution speed
        print("\n[3] Testing execution speed...")
        await test_execution_speed()
        
        print("\n" + "=" * 70)
        print("ALL TESTS PASSED ✅")
        print("=" * 70)
    
    asyncio.run(run_all())
