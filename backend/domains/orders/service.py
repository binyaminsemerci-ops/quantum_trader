"""
Order Service - Query layer for order history
EPIC: DASHBOARD-V3-TRADING-PANELS

Provides read-only access to order history from TradeLog table.
"""

import logging
from typing import List, Optional
from sqlalchemy.orm import Session
from datetime import datetime, timezone

from backend.domains.orders.models import OrderRecord, OrderStatus

logger = logging.getLogger(__name__)


class OrderService:
    """
    Service for querying order history.
    
    Reads from existing TradeLog table - no new storage needed.
    """
    
    def __init__(self, db_session: Session):
        """
        Initialize order service.
        
        Args:
            db_session: SQLAlchemy database session
        """
        self.db = db_session
    
    def get_recent_orders(self, limit: int = 50) -> List[OrderRecord]:
        """
        Get recent orders for dashboard display.
        
        Args:
            limit: Maximum number of orders to return (default 50)
            
        Returns:
            List of OrderRecord objects, most recent first
        """
        try:
            # Import TradeLog model
            from backend.database import TradeLog
            
            # Query recent trades/orders
            trades = (
                self.db.query(TradeLog)
                .order_by(TradeLog.timestamp.desc())
                .limit(limit)
                .all()
            )
            
            # Map to OrderRecord
            orders = []
            for trade in trades:
                # Map status from TradeLog to OrderStatus
                status_map = {
                    "FILLED": OrderStatus.FILLED,
                    "PENDING": OrderStatus.PENDING,
                    "NEW": OrderStatus.NEW,
                    "CANCELLED": OrderStatus.CANCELLED,
                    "REJECTED": OrderStatus.REJECTED,
                }
                status = status_map.get(trade.status.upper(), OrderStatus.FILLED)
                
                order = OrderRecord(
                    id=trade.id,
                    timestamp=trade.timestamp if trade.timestamp.tzinfo else trade.timestamp.replace(tzinfo=timezone.utc),
                    account="default",  # TODO: Add account field to TradeLog
                    exchange="binance_testnet",  # TODO: Add exchange field to TradeLog
                    symbol=trade.symbol,
                    side=trade.side.upper(),
                    order_type="MARKET",  # TradeLog doesn't store order type
                    size=trade.qty,
                    price=trade.price,
                    status=status,
                    strategy_id=getattr(trade, 'strategy_id', None)
                )
                orders.append(order)
            
            logger.info(f"[OrderService] Retrieved {len(orders)} recent orders")
            return orders
            
        except Exception as e:
            logger.error(f"[OrderService] Error retrieving orders: {e}")
            return []
    
    def get_orders_by_symbol(self, symbol: str, limit: int = 50) -> List[OrderRecord]:
        """
        Get recent orders for specific symbol.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            limit: Maximum number of orders
            
        Returns:
            List of OrderRecord objects for the symbol
        """
        try:
            from backend.database import TradeLog
            
            trades = (
                self.db.query(TradeLog)
                .filter(TradeLog.symbol == symbol)
                .order_by(TradeLog.timestamp.desc())
                .limit(limit)
                .all()
            )
            
            orders = []
            for trade in trades:
                status_map = {
                    "FILLED": OrderStatus.FILLED,
                    "PENDING": OrderStatus.PENDING,
                    "NEW": OrderStatus.NEW,
                }
                status = status_map.get(trade.status.upper(), OrderStatus.FILLED)
                
                order = OrderRecord(
                    id=trade.id,
                    timestamp=trade.timestamp if trade.timestamp.tzinfo else trade.timestamp.replace(tzinfo=timezone.utc),
                    account="default",
                    exchange="binance_testnet",
                    symbol=trade.symbol,
                    side=trade.side.upper(),
                    order_type="MARKET",
                    size=trade.qty,
                    price=trade.price,
                    status=status,
                    strategy_id=getattr(trade, 'strategy_id', None)
                )
                orders.append(order)
            
            return orders
            
        except Exception as e:
            logger.error(f"[OrderService] Error retrieving orders for {symbol}: {e}")
            return []
