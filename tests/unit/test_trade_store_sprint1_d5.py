"""
SPRINT 1 - D5: TradeStore Tests
================================

Comprehensive tests for TradeStore migration:
- SQLite backend tests
- Redis backend tests (skip if unavailable)
- Trade lifecycle integration tests
- Recovery/restart simulation tests
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from typing import Optional
import os

import redis.asyncio as redis

from backend.core.trading import (
    Trade,
    TradeStatus,
    TradeSide,
    TradeStoreSQLite,
    TradeStoreRedis,
    get_trade_store,
    reset_trade_store,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
async def sqlite_store():
    """Create SQLite TradeStore for testing."""
    store = TradeStoreSQLite(db_path="test_trades.db")
    await store.initialize()
    yield store
    await store.close()
    # Cleanup
    if os.path.exists("test_trades.db"):
        os.remove("test_trades.db")


@pytest.fixture
async def redis_client():
    """Create Redis client for testing (skip if unavailable)."""
    try:
        client = redis.Redis(host="localhost", port=6379, decode_responses=False)
        await client.ping()
        yield client
        await client.close()
    except Exception:
        pytest.skip("Redis not available")


@pytest.fixture
async def redis_store(redis_client):
    """Create Redis TradeStore for testing."""
    store = TradeStoreRedis(redis_client)
    await store.initialize()
    
    # Cleanup before test
    pattern = "trade:test_*"
    async for key in redis_client.scan_iter(match=pattern):
        await redis_client.delete(key)
    
    yield store
    
    # Cleanup after test
    async for key in redis_client.scan_iter(match=pattern):
        await redis_client.delete(key)


@pytest.fixture
def sample_trade():
    """Create sample trade for testing."""
    return Trade(
        trade_id="test_trade_123",
        symbol="BTCUSDT",
        side=TradeSide.LONG,
        status=TradeStatus.OPEN,
        quantity=0.5,
        leverage=10.0,
        margin_usd=5000.0,
        entry_price=50000.0,
        entry_time=datetime.utcnow(),
        sl_price=48000.0,
        tp_price=55000.0,
        trail_percent=0.02,
        model="XGBoost_Ensemble",
        confidence=0.85,
        meta_strategy_id="momentum_breakout",
        regime="TRENDING",
        rl_state_key="state_abc123",
        rl_action_key="action_xyz789",
        entry_fee_usd=25.0,
        metadata={"test": "data"}
    )


# =============================================================================
# SQLITE BACKEND TESTS
# =============================================================================

class TestSQLiteBackend:
    """Test SQLite TradeStore backend."""
    
    @pytest.mark.asyncio
    async def test_save_and_get_trade(self, sqlite_store, sample_trade):
        """Test saving and retrieving a trade."""
        # Save trade
        await sqlite_store.save_new_trade(sample_trade)
        
        # Retrieve trade
        retrieved = await sqlite_store.get_trade_by_id(sample_trade.trade_id)
        
        assert retrieved is not None
        assert retrieved.trade_id == sample_trade.trade_id
        assert retrieved.symbol == sample_trade.symbol
        assert retrieved.side == TradeSide.LONG
        assert retrieved.status == TradeStatus.OPEN
        assert retrieved.quantity == sample_trade.quantity
        assert retrieved.leverage == sample_trade.leverage
        assert retrieved.margin_usd == sample_trade.margin_usd
        assert retrieved.entry_price == sample_trade.entry_price
        assert retrieved.sl_price == sample_trade.sl_price
        assert retrieved.tp_price == sample_trade.tp_price
        assert retrieved.model == sample_trade.model
        assert retrieved.confidence == sample_trade.confidence
    
    @pytest.mark.asyncio
    async def test_update_trade(self, sqlite_store, sample_trade):
        """Test updating trade fields."""
        # Save initial trade
        await sqlite_store.save_new_trade(sample_trade)
        
        # Update fields
        success = await sqlite_store.update_trade(
            sample_trade.trade_id,
            sl_price=49000.0,
            tp_price=56000.0,
            trail_percent=0.025
        )
        
        assert success is True
        
        # Verify updates
        updated = await sqlite_store.get_trade_by_id(sample_trade.trade_id)
        assert updated.sl_price == 49000.0
        assert updated.tp_price == 56000.0
        assert updated.trail_percent == 0.025
    
    @pytest.mark.asyncio
    async def test_get_open_trades(self, sqlite_store):
        """Test retrieving open trades."""
        # Create multiple trades
        trade1 = Trade(
            trade_id="test_btc_1",
            symbol="BTCUSDT",
            side=TradeSide.LONG,
            status=TradeStatus.OPEN,
            quantity=0.5,
            leverage=10.0,
            margin_usd=5000.0,
            entry_price=50000.0,
            entry_time=datetime.utcnow()
        )
        
        trade2 = Trade(
            trade_id="test_eth_1",
            symbol="ETHUSDT",
            side=TradeSide.SHORT,
            status=TradeStatus.OPEN,
            quantity=10.0,
            leverage=5.0,
            margin_usd=3000.0,
            entry_price=3000.0,
            entry_time=datetime.utcnow()
        )
        
        trade3 = Trade(
            trade_id="test_btc_2",
            symbol="BTCUSDT",
            side=TradeSide.LONG,
            status=TradeStatus.CLOSED,
            quantity=0.3,
            leverage=10.0,
            margin_usd=3000.0,
            entry_price=51000.0,
            entry_time=datetime.utcnow(),
            exit_price=52000.0,
            exit_time=datetime.utcnow()
        )
        
        # Save trades
        await sqlite_store.save_new_trade(trade1)
        await sqlite_store.save_new_trade(trade2)
        await sqlite_store.save_new_trade(trade3)
        
        # Get all open trades
        open_trades = await sqlite_store.get_open_trades()
        assert len(open_trades) == 2
        
        # Get open BTC trades only
        btc_trades = await sqlite_store.get_open_trades(symbol="BTCUSDT")
        assert len(btc_trades) == 1
        assert btc_trades[0].symbol == "BTCUSDT"
    
    @pytest.mark.asyncio
    async def test_mark_trade_closed(self, sqlite_store):
        """Test closing a trade with PnL calculation."""
        # Create LONG trade
        trade = Trade(
            trade_id="test_close_long",
            symbol="BTCUSDT",
            side=TradeSide.LONG,
            status=TradeStatus.OPEN,
            quantity=1.0,
            leverage=10.0,
            margin_usd=5000.0,
            entry_price=50000.0,
            entry_time=datetime.utcnow(),
            sl_price=48000.0,
            entry_fee_usd=25.0
        )
        
        await sqlite_store.save_new_trade(trade)
        
        # Close trade at profit
        exit_price = 52000.0
        exit_time = datetime.utcnow()
        success = await sqlite_store.mark_trade_closed(
            trade_id=trade.trade_id,
            exit_price=exit_price,
            exit_time=exit_time,
            close_reason="TP",
            exit_fee_usd=26.0
        )
        
        assert success is True
        
        # Verify closure
        closed_trade = await sqlite_store.get_trade_by_id(trade.trade_id)
        assert closed_trade.status == TradeStatus.CLOSED
        assert closed_trade.exit_price == exit_price
        assert closed_trade.close_reason == "TP"
        
        # Verify PnL calculation
        # Price change: 52000 - 50000 = +2000
        # PnL = (2000 / 50000) * 1.0 * 50000 * 10 - 25 - 26 = 4000 - 51 = 3949
        assert closed_trade.pnl_usd > 0
        assert closed_trade.pnl_pct > 0
        assert closed_trade.r_multiple > 0  # Should be ~1R (hit half way to TP from SL)
    
    @pytest.mark.asyncio
    async def test_get_stats(self, sqlite_store):
        """Test storage statistics."""
        # Create mix of open and closed trades
        for i in range(3):
            trade = Trade(
                trade_id=f"test_open_{i}",
                symbol="BTCUSDT",
                side=TradeSide.LONG,
                status=TradeStatus.OPEN,
                quantity=0.1,
                leverage=10.0,
                margin_usd=1000.0,
                entry_price=50000.0,
                entry_time=datetime.utcnow()
            )
            await sqlite_store.save_new_trade(trade)
        
        for i in range(2):
            trade = Trade(
                trade_id=f"test_closed_{i}",
                symbol="ETHUSDT",
                side=TradeSide.LONG,
                status=TradeStatus.CLOSED,
                quantity=0.1,
                leverage=5.0,
                margin_usd=500.0,
                entry_price=3000.0,
                entry_time=datetime.utcnow(),
                exit_price=3100.0,
                exit_time=datetime.utcnow(),
                pnl_usd=100.0 * (i + 1)
            )
            await sqlite_store.save_new_trade(trade)
        
        # Get stats
        stats = await sqlite_store.get_stats()
        
        assert stats["backend"] == "SQLite"
        assert stats["total_trades"] == 5
        assert stats["open_trades"] == 3
        assert stats["closed_trades"] == 2
        assert stats["total_pnl_usd"] > 0


# =============================================================================
# REDIS BACKEND TESTS
# =============================================================================

class TestRedisBackend:
    """Test Redis TradeStore backend."""
    
    @pytest.mark.asyncio
    async def test_save_and_get_trade(self, redis_store, sample_trade):
        """Test saving and retrieving a trade from Redis."""
        # Modify trade_id for Redis test
        sample_trade.trade_id = "test_redis_123"
        
        # Save trade
        await redis_store.save_new_trade(sample_trade)
        
        # Retrieve trade
        retrieved = await redis_store.get_trade_by_id(sample_trade.trade_id)
        
        assert retrieved is not None
        assert retrieved.trade_id == sample_trade.trade_id
        assert retrieved.symbol == sample_trade.symbol
        assert retrieved.side == TradeSide.LONG
        assert retrieved.margin_usd == sample_trade.margin_usd
    
    @pytest.mark.asyncio
    async def test_update_trade(self, redis_store, sample_trade):
        """Test updating trade in Redis."""
        sample_trade.trade_id = "test_redis_update"
        
        # Save initial
        await redis_store.save_new_trade(sample_trade)
        
        # Update
        success = await redis_store.update_trade(
            sample_trade.trade_id,
            sl_price=49500.0
        )
        
        assert success is True
        
        # Verify
        updated = await redis_store.get_trade_by_id(sample_trade.trade_id)
        assert updated.sl_price == 49500.0
    
    @pytest.mark.asyncio
    async def test_get_open_trades(self, redis_store):
        """Test retrieving open trades from Redis."""
        # Create trades
        for i in range(3):
            trade = Trade(
                trade_id=f"test_redis_open_{i}",
                symbol="BTCUSDT",
                side=TradeSide.LONG,
                status=TradeStatus.OPEN,
                quantity=0.1,
                leverage=10.0,
                margin_usd=1000.0,
                entry_price=50000.0,
                entry_time=datetime.utcnow()
            )
            await redis_store.save_new_trade(trade)
        
        # Get open trades
        open_trades = await redis_store.get_open_trades()
        assert len(open_trades) >= 3
    
    @pytest.mark.asyncio
    async def test_mark_trade_closed(self, redis_store):
        """Test closing trade in Redis."""
        trade = Trade(
            trade_id="test_redis_close",
            symbol="BTCUSDT",
            side=TradeSide.LONG,
            status=TradeStatus.OPEN,
            quantity=1.0,
            leverage=10.0,
            margin_usd=5000.0,
            entry_price=50000.0,
            entry_time=datetime.utcnow(),
            sl_price=48000.0
        )
        
        await redis_store.save_new_trade(trade)
        
        # Close
        success = await redis_store.mark_trade_closed(
            trade_id=trade.trade_id,
            exit_price=51000.0,
            exit_time=datetime.utcnow(),
            close_reason="TP"
        )
        
        assert success is True
        
        # Verify
        closed = await redis_store.get_trade_by_id(trade.trade_id)
        assert closed.status == TradeStatus.CLOSED
        assert closed.pnl_usd > 0


# =============================================================================
# FACTORY/SELECTOR TESTS
# =============================================================================

class TestFactory:
    """Test get_trade_store() factory."""
    
    @pytest.mark.asyncio
    async def test_force_sqlite(self):
        """Test forcing SQLite backend."""
        reset_trade_store()
        
        store = await get_trade_store(force_sqlite=True)
        
        assert store.backend_name == "SQLite"
        assert store.initialized
        
        # Cleanup
        if isinstance(store, TradeStoreSQLite):
            await store.close()
    
    @pytest.mark.asyncio
    async def test_redis_fallback_to_sqlite(self):
        """Test fallback to SQLite when Redis unavailable."""
        reset_trade_store()
        
        # Pass None as redis_client (simulates unavailable Redis)
        store = await get_trade_store(redis_client=None)
        
        assert store.backend_name == "SQLite"
        assert store.initialized
        
        # Cleanup
        if isinstance(store, TradeStoreSQLite):
            await store.close()
    
    @pytest.mark.asyncio
    async def test_singleton_pattern(self):
        """Test that get_trade_store() returns same instance."""
        reset_trade_store()
        
        store1 = await get_trade_store(force_sqlite=True)
        store2 = await get_trade_store(force_sqlite=True)
        
        assert store1 is store2  # Same instance
        
        # Cleanup
        if isinstance(store1, TradeStoreSQLite):
            await store1.close()


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestTradeLifecycle:
    """Test complete trade lifecycle."""
    
    @pytest.mark.asyncio
    async def test_full_trade_lifecycle(self, sqlite_store):
        """Test opening, updating, and closing a trade."""
        # 1. Open trade
        trade = Trade(
            trade_id="lifecycle_test_1",
            symbol="BTCUSDT",
            side=TradeSide.LONG,
            status=TradeStatus.OPEN,
            quantity=1.0,
            leverage=10.0,
            margin_usd=5000.0,
            entry_price=50000.0,
            entry_time=datetime.utcnow(),
            sl_price=48000.0,
            tp_price=55000.0,
            entry_fee_usd=25.0
        )
        
        await sqlite_store.save_new_trade(trade)
        
        # Verify open
        retrieved = await sqlite_store.get_trade_by_id(trade.trade_id)
        assert retrieved.status == TradeStatus.OPEN
        
        # 2. Update trailing stop
        await sqlite_store.update_trade(
            trade.trade_id,
            sl_price=49000.0,  # Move SL to breakeven
            status=TradeStatus.PARTIAL_TP.value
        )
        
        # Verify update
        updated = await sqlite_store.get_trade_by_id(trade.trade_id)
        assert updated.sl_price == 49000.0
        
        # 3. Close trade
        await sqlite_store.mark_trade_closed(
            trade_id=trade.trade_id,
            exit_price=54000.0,
            exit_time=datetime.utcnow(),
            close_reason="TP",
            exit_fee_usd=27.0
        )
        
        # Verify closed
        closed = await sqlite_store.get_trade_by_id(trade.trade_id)
        assert closed.status == TradeStatus.CLOSED
        assert closed.exit_price == 54000.0
        assert closed.pnl_usd > 0
        assert closed.r_multiple > 0
    
    @pytest.mark.asyncio
    async def test_recovery_simulation(self, sqlite_store):
        """Simulate restart and recovery of open trades."""
        # Simulate trades before "crash"
        for i in range(5):
            trade = Trade(
                trade_id=f"recovery_test_{i}",
                symbol="BTCUSDT",
                side=TradeSide.LONG,
                status=TradeStatus.OPEN,
                quantity=0.1 * (i + 1),
                leverage=10.0,
                margin_usd=1000.0 * (i + 1),
                entry_price=50000.0 + (i * 100),
                entry_time=datetime.utcnow() - timedelta(minutes=i)
            )
            await sqlite_store.save_new_trade(trade)
        
        # Simulate "restart" - close and reopen store
        await sqlite_store.close()
        
        # Create new store instance (simulates restart)
        new_store = TradeStoreSQLite(db_path="test_trades.db")
        await new_store.initialize()
        
        # Recover open trades
        open_trades = await new_store.get_open_trades()
        
        assert len(open_trades) == 5
        
        # Verify all trades recovered correctly
        for trade in open_trades:
            assert trade.status == TradeStatus.OPEN
            assert trade.symbol == "BTCUSDT"
            assert trade.margin_usd > 0
        
        # Cleanup
        await new_store.close()


# =============================================================================
# EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.mark.asyncio
    async def test_get_nonexistent_trade(self, sqlite_store):
        """Test getting a trade that doesn't exist."""
        result = await sqlite_store.get_trade_by_id("nonexistent_trade")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_update_nonexistent_trade(self, sqlite_store):
        """Test updating a trade that doesn't exist."""
        success = await sqlite_store.update_trade(
            "nonexistent_trade",
            sl_price=50000.0
        )
        assert success is False
    
    @pytest.mark.asyncio
    async def test_close_nonexistent_trade(self, sqlite_store):
        """Test closing a trade that doesn't exist."""
        success = await sqlite_store.mark_trade_closed(
            trade_id="nonexistent_trade",
            exit_price=50000.0,
            exit_time=datetime.utcnow(),
            close_reason="Test"
        )
        assert success is False
    
    @pytest.mark.asyncio
    async def test_empty_metadata(self, sqlite_store):
        """Test trade with empty metadata."""
        trade = Trade(
            trade_id="test_empty_metadata",
            symbol="BTCUSDT",
            side=TradeSide.LONG,
            status=TradeStatus.OPEN,
            quantity=1.0,
            leverage=10.0,
            margin_usd=5000.0,
            entry_price=50000.0,
            entry_time=datetime.utcnow(),
            metadata={}  # Empty metadata
        )
        
        await sqlite_store.save_new_trade(trade)
        
        retrieved = await sqlite_store.get_trade_by_id(trade.trade_id)
        assert retrieved is not None
        assert retrieved.metadata == {}
