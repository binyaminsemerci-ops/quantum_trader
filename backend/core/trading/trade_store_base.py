"""
SPRINT 1 - D5: TradeStore Base Abstraction
===========================================

Unified interface for trade persistence with support for:
- Redis backend (primary, high-performance)
- SQLite backend (fallback, always available)

Provides robust trade storage for:
- Position recovery after restarts
- Trade lifecycle tracking
- PnL calculation
- Risk management integration
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any, Protocol

logger = logging.getLogger(__name__)


class TradeStatus(str, Enum):
    """Trade lifecycle status."""
    PENDING = "PENDING"          # Order submitted, awaiting fill
    OPEN = "OPEN"                # Position active
    PARTIAL_TP = "PARTIAL_TP"    # Partial profit taken
    CLOSED = "CLOSED"            # Position fully closed
    CANCELLED = "CANCELLED"      # Order cancelled before fill
    FAILED = "FAILED"            # Order/execution failed


class TradeSide(str, Enum):
    """Trade direction."""
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class Trade:
    """
    Complete trade data model.
    
    Contains all information needed for:
    - Position tracking
    - PnL calculation
    - Risk management
    - Performance analysis
    - Recovery after restart
    """
    # Identification
    trade_id: str                        # Unique trade ID (typically exchange order ID)
    symbol: str                          # Trading pair (e.g., BTCUSDT)
    side: TradeSide                      # LONG or SHORT
    status: TradeStatus                  # Current lifecycle status
    
    # Position sizing
    quantity: float                      # Position size in base asset
    leverage: float                      # Applied leverage (1.0 = no leverage)
    margin_usd: float                    # Margin requirement in USD
    
    # Entry details
    entry_price: float                   # Actual fill price
    entry_time: datetime                 # When position was opened
    
    # Exit management
    sl_price: Optional[float] = None     # Stop loss price
    tp_price: Optional[float] = None     # Take profit price
    trail_percent: Optional[float] = None  # Trailing stop %
    
    # Exit details (when closed)
    exit_price: Optional[float] = None   # Actual exit price
    exit_time: Optional[datetime] = None  # When position was closed
    close_reason: Optional[str] = None   # Why trade closed (TP/SL/Manual/Time)
    
    # Performance
    pnl_usd: float = 0.0                 # Realized PnL in USD
    pnl_pct: float = 0.0                 # PnL as % of margin
    r_multiple: float = 0.0              # PnL as multiple of risk (SL distance)
    
    # Fees/costs
    entry_fee_usd: float = 0.0           # Entry execution fee
    exit_fee_usd: float = 0.0            # Exit execution fee
    funding_fees_usd: float = 0.0        # Accumulated funding fees
    
    # AI/Strategy context
    model: Optional[str] = None          # AI model that generated signal
    confidence: float = 0.0              # Signal confidence (0-1)
    meta_strategy_id: Optional[str] = None  # Meta-strategy that selected this
    regime: Optional[str] = None         # Market regime at entry
    
    # RL Position Sizing (for learning)
    rl_state_key: Optional[str] = None   # RL state hash for reward update
    rl_action_key: Optional[str] = None  # RL action taken
    rl_leverage_original: Optional[float] = None  # RL's original proposal
    
    # Exchange integration
    exchange_order_id: Optional[str] = None  # Primary order ID
    sl_order_id: Optional[str] = None    # Stop loss order ID
    tp_order_id: Optional[str] = None    # Take profit order ID
    
    # Metadata (flexible storage for additional fields)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.utcnow())
    updated_at: datetime = field(default_factory=lambda: datetime.utcnow())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        data = asdict(self)
        
        # Convert enums to strings
        data['side'] = self.side.value if isinstance(self.side, TradeSide) else self.side
        data['status'] = self.status.value if isinstance(self.status, TradeStatus) else self.status
        
        # Convert datetime to ISO strings
        for key in ['entry_time', 'exit_time', 'created_at', 'updated_at']:
            if data.get(key):
                data[key] = data[key].isoformat() if isinstance(data[key], datetime) else data[key]
        
        # Serialize metadata as JSON string
        data['metadata'] = json.dumps(data.get('metadata', {}))
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Trade":
        """Create Trade from dictionary."""
        # Convert string enums back to Enum objects
        if 'side' in data and isinstance(data['side'], str):
            data['side'] = TradeSide(data['side'])
        if 'status' in data and isinstance(data['status'], str):
            data['status'] = TradeStatus(data['status'])
        
        # Convert ISO strings back to datetime
        for key in ['entry_time', 'exit_time', 'created_at', 'updated_at']:
            if data.get(key) and isinstance(data[key], str):
                data[key] = datetime.fromisoformat(data[key])
        
        # Deserialize metadata
        if 'metadata' in data and isinstance(data['metadata'], str):
            data['metadata'] = json.loads(data['metadata'])
        
        return cls(**data)
    
    def update_exit(
        self,
        exit_price: float,
        exit_time: datetime,
        close_reason: str,
        exit_fee_usd: float = 0.0
    ) -> None:
        """Update trade with exit information and calculate PnL."""
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.close_reason = close_reason
        self.exit_fee_usd = exit_fee_usd
        self.status = TradeStatus.CLOSED
        self.updated_at = datetime.utcnow()
        
        # Calculate PnL
        if self.side == TradeSide.LONG:
            price_change = exit_price - self.entry_price
        else:  # SHORT
            price_change = self.entry_price - exit_price
        
        # PnL = (price change / entry price) * quantity * leverage - fees
        self.pnl_usd = (
            (price_change / self.entry_price) * self.quantity * self.entry_price * self.leverage
            - self.entry_fee_usd - self.exit_fee_usd - self.funding_fees_usd
        )
        self.pnl_pct = (self.pnl_usd / self.margin_usd) * 100.0 if self.margin_usd > 0 else 0.0
        
        # Calculate R-multiple if SL was set
        if self.sl_price:
            if self.side == TradeSide.LONG:
                risk_distance = self.entry_price - self.sl_price
            else:
                risk_distance = self.sl_price - self.entry_price
            
            if risk_distance > 0:
                self.r_multiple = price_change / risk_distance
        
        logger.info(
            f"[TradeStore] Updated exit for {self.trade_id}: "
            f"Exit=${exit_price:.2f}, PnL=${self.pnl_usd:.2f} ({self.pnl_pct:.2f}%), R={self.r_multiple:.2f}"
        )


class TradeStore(Protocol):
    """
    Abstract interface for trade persistence.
    
    Implementations:
    - TradeStoreRedis: High-performance Redis backend
    - TradeStoreSQLite: Reliable SQLite fallback
    """
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize storage backend (create tables, verify connection, etc.)."""
        ...
    
    @abstractmethod
    async def save_new_trade(self, trade: Trade) -> None:
        """
        Save a new trade to storage.
        
        Args:
            trade: Trade object to persist
        
        Raises:
            Exception: If save fails
        """
        ...
    
    @abstractmethod
    async def update_trade(self, trade_id: str, **fields) -> bool:
        """
        Update specific fields of an existing trade.
        
        Args:
            trade_id: Unique trade identifier
            **fields: Fields to update (e.g., status="CLOSED", pnl_usd=100.0)
        
        Returns:
            True if updated, False if trade not found
        """
        ...
    
    @abstractmethod
    async def get_trade_by_id(self, trade_id: str) -> Optional[Trade]:
        """
        Retrieve trade by ID.
        
        Args:
            trade_id: Unique trade identifier
        
        Returns:
            Trade object or None if not found
        """
        ...
    
    @abstractmethod
    async def get_open_trades(self, symbol: Optional[str] = None) -> List[Trade]:
        """
        Get all open trades, optionally filtered by symbol.
        
        Args:
            symbol: Optional symbol filter (e.g., "BTCUSDT")
        
        Returns:
            List of Trade objects with status=OPEN
        """
        ...
    
    @abstractmethod
    async def mark_trade_closed(
        self,
        trade_id: str,
        exit_price: float,
        exit_time: datetime,
        close_reason: str,
        exit_fee_usd: float = 0.0
    ) -> bool:
        """
        Mark trade as closed and calculate final PnL.
        
        Args:
            trade_id: Unique trade identifier
            exit_price: Final exit price
            exit_time: When trade was closed
            close_reason: Why trade closed (TP/SL/Manual/Time)
            exit_fee_usd: Exit execution fee
        
        Returns:
            True if closed, False if trade not found
        """
        ...
    
    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics.
        
        Returns:
            Dictionary with stats (total_trades, open_trades, backend, etc.)
        """
        ...


class TradeStoreBase(ABC, TradeStore):
    """
    Base implementation with common utilities.
    
    Concrete backends (Redis, SQLite) inherit from this.
    """
    
    def __init__(self):
        self._initialized = False
        self.backend_name = "Unknown"
    
    async def initialize(self) -> None:
        """Default initialization."""
        self._initialized = True
        logger.info(f"[TradeStore] {self.backend_name} initialized")
    
    @property
    def initialized(self) -> bool:
        """Check if store is initialized."""
        return self._initialized


# =============================================================================
# FACTORY & SINGLETON
# =============================================================================

_global_trade_store: Optional[TradeStore] = None


async def get_trade_store(
    redis_client: Optional[Any] = None,
    force_sqlite: bool = False
) -> TradeStore:
    """
    Get or create global TradeStore instance.
    
    Backend selection logic:
    1. If force_sqlite=True → Use SQLite
    2. If redis_client provided and connectable → Use Redis
    3. Otherwise → Use SQLite (fallback)
    
    Args:
        redis_client: Optional Redis client (if None, use SQLite)
        force_sqlite: Force SQLite backend regardless of Redis availability
    
    Returns:
        Initialized TradeStore instance
    
    Usage:
        # Auto-select backend
        store = await get_trade_store(redis_client=redis_client)
        
        # Force SQLite (e.g., for tests)
        store = await get_trade_store(force_sqlite=True)
    """
    global _global_trade_store
    
    # Return existing instance if already initialized
    if _global_trade_store is not None and _global_trade_store.initialized:
        return _global_trade_store
    
    # Import backends here to avoid circular imports
    from backend.core.trading.trade_store_sqlite import TradeStoreSQLite
    from backend.core.trading.trade_store_redis import TradeStoreRedis
    
    store: Optional[TradeStore] = None
    
    # Determine backend
    if force_sqlite:
        logger.info("[TradeStore] SQLite backend forced by configuration")
        store = TradeStoreSQLite()
        await store.initialize()  # <-- FIX: Initialize SQLite
    
    elif redis_client is not None:
        # Try Redis first
        try:
            logger.info("[TradeStore] Attempting to use Redis backend...")
            store = TradeStoreRedis(redis_client)
            await store.initialize()
            logger.info("[TradeStore] ✅ Redis backend selected and connected")
        except Exception as e:
            logger.warning(f"[TradeStore] Redis unavailable ({e}), falling back to SQLite")
            store = None
    
    # Fallback to SQLite
    if store is None:
        logger.info("[TradeStore] Using SQLite backend (fallback)")
        store = TradeStoreSQLite()
        await store.initialize()
    
    # Cache global instance
    _global_trade_store = store
    
    logger.info(
        f"[TradeStore] Global instance initialized: {store.backend_name}"
    )
    
    return store


def reset_trade_store() -> None:
    """
    Reset global TradeStore instance.
    
    Useful for testing or when switching backends.
    """
    global _global_trade_store
    _global_trade_store = None
    logger.info("[TradeStore] Global instance reset")
