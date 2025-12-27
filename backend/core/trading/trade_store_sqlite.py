"""
SPRINT 1 - D5: SQLite TradeStore Backend
========================================

Reliable, always-available trade persistence using SQLite.

Features:
- Local file storage (runtime/trades.db)
- Async operations via aiosqlite
- Automatic table creation
- Indexed for fast queries
- Survives restarts
- Used as fallback when Redis unavailable
"""

import aiosqlite
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

from backend.core.trading.trade_store_base import (
    Trade,
    TradeStore,
    TradeStoreBase,
    TradeStatus,
    TradeSide,
)

logger = logging.getLogger(__name__)


class TradeStoreSQLite(TradeStoreBase):
    """
    SQLite-backed trade storage.
    
    Always available, reliable fallback when Redis is not accessible.
    Stores trades in local SQLite database with full ACID guarantees.
    """
    
    DEFAULT_DB_PATH = "runtime/trades.db"
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize SQLite TradeStore.
        
        Args:
            db_path: Path to SQLite database file (default: runtime/trades.db)
        """
        super().__init__()
        self.backend_name = "SQLite"
        self.db_path = db_path or self.DEFAULT_DB_PATH
        self.db: Optional[aiosqlite.Connection] = None
        
        # Ensure runtime directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"[TradeStore] SQLite backend configured: {self.db_path}")
    
    async def initialize(self) -> None:
        """Initialize database connection and create tables."""
        try:
            # Connect to database
            self.db = await aiosqlite.connect(self.db_path)
            
            # Enable WAL mode for better concurrency
            await self.db.execute("PRAGMA journal_mode=WAL")
            
            # Create trades table
            await self.db.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    -- Identification
                    trade_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    status TEXT NOT NULL,
                    
                    -- Position sizing
                    quantity REAL NOT NULL,
                    leverage REAL NOT NULL,
                    margin_usd REAL NOT NULL,
                    
                    -- Entry details
                    entry_price REAL NOT NULL,
                    entry_time TEXT NOT NULL,
                    
                    -- Exit management
                    sl_price REAL,
                    tp_price REAL,
                    trail_percent REAL,
                    
                    -- Exit details
                    exit_price REAL,
                    exit_time TEXT,
                    close_reason TEXT,
                    
                    -- Performance
                    pnl_usd REAL DEFAULT 0.0,
                    pnl_pct REAL DEFAULT 0.0,
                    r_multiple REAL DEFAULT 0.0,
                    
                    -- Fees/costs
                    entry_fee_usd REAL DEFAULT 0.0,
                    exit_fee_usd REAL DEFAULT 0.0,
                    funding_fees_usd REAL DEFAULT 0.0,
                    
                    -- AI/Strategy context
                    model TEXT,
                    confidence REAL DEFAULT 0.0,
                    meta_strategy_id TEXT,
                    regime TEXT,
                    
                    -- RL Position Sizing
                    rl_state_key TEXT,
                    rl_action_key TEXT,
                    rl_leverage_original REAL,
                    
                    -- Exchange integration
                    exchange_order_id TEXT,
                    sl_order_id TEXT,
                    tp_order_id TEXT,
                    
                    -- Metadata & timestamps
                    metadata TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            
            # Create indexes for common queries
            await self.db.execute("""
                CREATE INDEX IF NOT EXISTS idx_trades_status 
                ON trades(status)
            """)
            
            await self.db.execute("""
                CREATE INDEX IF NOT EXISTS idx_trades_symbol_status 
                ON trades(symbol, status)
            """)
            
            await self.db.execute("""
                CREATE INDEX IF NOT EXISTS idx_trades_entry_time 
                ON trades(entry_time DESC)
            """)
            
            await self.db.commit()
            
            self._initialized = True
            logger.info(f"[TradeStore] SQLite initialized: {self.db_path}")
        
        except Exception as e:
            logger.error(f"[TradeStore] SQLite initialization failed: {e}")
            raise
    
    async def close(self) -> None:
        """Close database connection."""
        if self.db:
            await self.db.close()
            self.db = None
            logger.info("[TradeStore] SQLite connection closed")
    
    async def save_new_trade(self, trade: Trade) -> None:
        """
        Save a new trade to SQLite.
        
        Uses INSERT OR REPLACE for idempotency.
        """
        if not self._initialized:
            raise RuntimeError("TradeStore not initialized")
        
        try:
            data = trade.to_dict()
            
            # Build INSERT statement
            columns = ', '.join(data.keys())
            placeholders = ', '.join(['?' for _ in data])
            values = tuple(data.values())
            
            await self.db.execute(
                f"INSERT OR REPLACE INTO trades ({columns}) VALUES ({placeholders})",
                values
            )
            await self.db.commit()
            
            logger.debug(
                f"[TradeStore] Saved trade: {trade.trade_id} | "
                f"{trade.symbol} {trade.side.value} | ${trade.margin_usd:.2f}"
            )
        
        except Exception as e:
            logger.error(f"[TradeStore] Failed to save trade {trade.trade_id}: {e}")
            raise
    
    async def update_trade(self, trade_id: str, **fields) -> bool:
        """
        Update specific fields of an existing trade.
        
        Args:
            trade_id: Trade identifier
            **fields: Fields to update
        
        Returns:
            True if updated, False if not found
        """
        if not self._initialized:
            raise RuntimeError("TradeStore not initialized")
        
        if not fields:
            return False
        
        try:
            # Add updated_at timestamp
            fields['updated_at'] = datetime.utcnow().isoformat()
            
            # Convert datetime objects to ISO strings
            for key, value in fields.items():
                if isinstance(value, datetime):
                    fields[key] = value.isoformat()
                elif isinstance(value, dict):
                    fields[key] = json.dumps(value)
            
            # Build UPDATE statement
            set_clause = ', '.join([f"{k} = ?" for k in fields.keys()])
            values = list(fields.values()) + [trade_id]
            
            cursor = await self.db.execute(
                f"UPDATE trades SET {set_clause} WHERE trade_id = ?",
                values
            )
            await self.db.commit()
            
            if cursor.rowcount > 0:
                logger.debug(f"[TradeStore] Updated trade {trade_id}: {list(fields.keys())}")
                return True
            else:
                logger.warning(f"[TradeStore] Trade not found for update: {trade_id}")
                return False
        
        except Exception as e:
            logger.error(f"[TradeStore] Failed to update trade {trade_id}: {e}")
            return False
    
    async def get_trade_by_id(self, trade_id: str) -> Optional[Trade]:
        """
        Retrieve trade by ID.
        
        Returns:
            Trade object or None if not found
        """
        if not self._initialized:
            raise RuntimeError("TradeStore not initialized")
        
        try:
            cursor = await self.db.execute(
                "SELECT * FROM trades WHERE trade_id = ?",
                (trade_id,)
            )
            row = await cursor.fetchone()
            
            if not row:
                return None
            
            # Convert row to dict
            columns = [desc[0] for desc in cursor.description]
            data = dict(zip(columns, row))
            
            return Trade.from_dict(data)
        
        except Exception as e:
            logger.error(f"[TradeStore] Failed to get trade {trade_id}: {e}")
            return None
    
    async def get_open_trades(self, symbol: Optional[str] = None) -> List[Trade]:
        """
        Get all open trades, optionally filtered by symbol.
        
        Args:
            symbol: Optional symbol filter
        
        Returns:
            List of open Trade objects
        """
        if not self._initialized:
            raise RuntimeError("TradeStore not initialized")
        
        try:
            if symbol:
                cursor = await self.db.execute(
                    "SELECT * FROM trades WHERE status = ? AND symbol = ? ORDER BY entry_time DESC",
                    (TradeStatus.OPEN.value, symbol)
                )
            else:
                cursor = await self.db.execute(
                    "SELECT * FROM trades WHERE status = ? ORDER BY entry_time DESC",
                    (TradeStatus.OPEN.value,)
                )
            
            rows = await cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            
            trades = []
            for row in rows:
                data = dict(zip(columns, row))
                trades.append(Trade.from_dict(data))
            
            logger.debug(
                f"[TradeStore] Found {len(trades)} open trades" +
                (f" for {symbol}" if symbol else "")
            )
            return trades
        
        except Exception as e:
            logger.error(f"[TradeStore] Failed to get open trades: {e}")
            return []
    
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
        
        Returns:
            True if closed, False if not found
        """
        if not self._initialized:
            raise RuntimeError("TradeStore not initialized")
        
        try:
            # Get existing trade
            trade = await self.get_trade_by_id(trade_id)
            
            if not trade:
                logger.warning(f"[TradeStore] Trade not found for close: {trade_id}")
                return False
            
            # Update trade with exit info and calculate PnL
            trade.update_exit(exit_price, exit_time, close_reason, exit_fee_usd)
            
            # Save updated trade
            await self.save_new_trade(trade)
            
            logger.info(
                f"[TradeStore] Closed trade {trade_id}: "
                f"PnL=${trade.pnl_usd:.2f} ({trade.pnl_pct:.2f}%), R={trade.r_multiple:.2f}"
            )
            return True
        
        except Exception as e:
            logger.error(f"[TradeStore] Failed to close trade {trade_id}: {e}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        if not self._initialized:
            return {
                "error": "Not initialized",
                "backend": "SQLite",
                "initialized": False,
            }
        
        try:
            # Count total trades
            cursor = await self.db.execute("SELECT COUNT(*) FROM trades")
            total_count = (await cursor.fetchone())[0]
            
            # Count open trades
            cursor = await self.db.execute(
                "SELECT COUNT(*) FROM trades WHERE status = ?",
                (TradeStatus.OPEN.value,)
            )
            open_count = (await cursor.fetchone())[0]
            
            # Count closed trades
            cursor = await self.db.execute(
                "SELECT COUNT(*) FROM trades WHERE status = ?",
                (TradeStatus.CLOSED.value,)
            )
            closed_count = (await cursor.fetchone())[0]
            
            # Calculate total PnL
            cursor = await self.db.execute(
                "SELECT SUM(pnl_usd) FROM trades WHERE status = ?",
                (TradeStatus.CLOSED.value,)
            )
            total_pnl = (await cursor.fetchone())[0] or 0.0
            
            return {
                "total_trades": total_count,
                "open_trades": open_count,
                "closed_trades": closed_count,
                "total_pnl_usd": total_pnl,
                "backend": "SQLite",
                "db_path": self.db_path,
                "initialized": self._initialized,
            }
        
        except Exception as e:
            logger.error(f"[TradeStore] Failed to get stats: {e}")
            return {
                "error": str(e),
                "backend": "SQLite",
                "initialized": self._initialized,
            }
    
    async def list_all_trades(
        self,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[Trade]:
        """
        List all trades with pagination.
        
        Args:
            limit: Maximum number of trades to return
            offset: Number of trades to skip
        
        Returns:
            List of Trade objects
        """
        if not self._initialized:
            raise RuntimeError("TradeStore not initialized")
        
        try:
            query = "SELECT * FROM trades ORDER BY entry_time DESC"
            params = []
            
            if limit:
                query += " LIMIT ? OFFSET ?"
                params = [limit, offset]
            
            cursor = await self.db.execute(query, params)
            rows = await cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            
            trades = []
            for row in rows:
                data = dict(zip(columns, row))
                trades.append(Trade.from_dict(data))
            
            return trades
        
        except Exception as e:
            logger.error(f"[TradeStore] Failed to list trades: {e}")
            return []
