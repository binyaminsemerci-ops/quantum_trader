"""
Tests for OrderService
EPIC: DASHBOARD-V3-TRADING-PANELS
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, MagicMock
from backend.domains.orders import OrderService, OrderRecord, OrderStatus


class MockTradeLog:
    """Mock TradeLog model for testing"""
    def __init__(self, id, symbol, side, qty, price, status, timestamp, strategy_id=None):
        self.id = id
        self.symbol = symbol
        self.side = side
        self.qty = qty
        self.price = price
        self.status = status
        self.timestamp = timestamp
        self.strategy_id = strategy_id


class TestOrderService:
    """Test suite for OrderService"""
    
    def test_get_recent_orders_success(self):
        """Test successful retrieval of recent orders"""
        # Mock database session
        mock_session = Mock()
        mock_query = Mock()
        mock_session.query.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        
        # Create mock trade logs
        mock_trades = [
            MockTradeLog(
                id=1,
                symbol="BTCUSDT",
                side="BUY",
                qty=0.1,
                price=50000.0,
                status="FILLED",
                timestamp=datetime(2025, 12, 5, 12, 0, 0, tzinfo=timezone.utc),
                strategy_id="test_strategy"
            ),
            MockTradeLog(
                id=2,
                symbol="ETHUSDT",
                side="SELL",
                qty=1.5,
                price=3000.0,
                status="NEW",
                timestamp=datetime(2025, 12, 5, 13, 0, 0, tzinfo=timezone.utc),
                strategy_id="test_strategy"
            ),
        ]
        mock_query.limit.return_value.all.return_value = mock_trades
        
        # Create service and get orders
        service = OrderService(mock_session)
        orders = service.get_recent_orders(limit=10)
        
        # Assertions
        assert len(orders) == 2
        assert isinstance(orders[0], OrderRecord)
        assert orders[0].symbol == "BTCUSDT"
        assert orders[0].side == "BUY"
        assert orders[0].status == OrderStatus.FILLED
        assert orders[1].symbol == "ETHUSDT"
        assert orders[1].side == "SELL"
        assert orders[1].status == OrderStatus.NEW
    
    def test_get_recent_orders_empty(self):
        """Test retrieval when no orders exist"""
        # Mock database session with empty result
        mock_session = Mock()
        mock_query = Mock()
        mock_session.query.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value.all.return_value = []
        
        # Create service and get orders
        service = OrderService(mock_session)
        orders = service.get_recent_orders(limit=10)
        
        # Assertions
        assert len(orders) == 0
        assert isinstance(orders, list)
    
    def test_get_recent_orders_error_handling(self):
        """Test error handling when database query fails"""
        # Mock database session that raises exception
        mock_session = Mock()
        mock_session.query.side_effect = Exception("Database connection error")
        
        # Create service and get orders
        service = OrderService(mock_session)
        orders = service.get_recent_orders(limit=10)
        
        # Should return empty list on error
        assert len(orders) == 0
        assert isinstance(orders, list)
    
    def test_get_orders_by_symbol(self):
        """Test filtering orders by symbol"""
        # Mock database session
        mock_session = Mock()
        mock_query = Mock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        
        # Create mock trade logs for specific symbol
        mock_trades = [
            MockTradeLog(
                id=1,
                symbol="BTCUSDT",
                side="BUY",
                qty=0.1,
                price=50000.0,
                status="FILLED",
                timestamp=datetime(2025, 12, 5, 12, 0, 0, tzinfo=timezone.utc)
            ),
        ]
        mock_query.limit.return_value.all.return_value = mock_trades
        
        # Create service and get orders
        service = OrderService(mock_session)
        orders = service.get_orders_by_symbol("BTCUSDT", limit=10)
        
        # Assertions
        assert len(orders) == 1
        assert orders[0].symbol == "BTCUSDT"
    
    def test_status_mapping(self):
        """Test status mapping from TradeLog to OrderStatus"""
        mock_session = Mock()
        mock_query = Mock()
        mock_session.query.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        
        # Test various status mappings (CLOSED and UNKNOWN both map to FILLED as default)
        test_statuses = [
            ("FILLED", OrderStatus.FILLED),
            ("NEW", OrderStatus.NEW),
            ("CANCELLED", OrderStatus.CANCELLED),
            ("CLOSED", OrderStatus.FILLED),  # Not in map, defaults to FILLED
            ("UNKNOWN", OrderStatus.FILLED),  # Not in map, defaults to FILLED
        ]
        
        for db_status, expected_status in test_statuses:
            mock_trades = [
                MockTradeLog(
                    id=1,
                    symbol="BTCUSDT",
                    side="BUY",
                    qty=0.1,
                    price=50000.0,
                    status=db_status,
                    timestamp=datetime(2025, 12, 5, 12, 0, 0, tzinfo=timezone.utc)
                ),
            ]
            mock_query.limit.return_value.all.return_value = mock_trades
            
            service = OrderService(mock_session)
            orders = service.get_recent_orders(limit=1)
            
            assert orders[0].status == expected_status
    
    def test_timestamp_handling(self):
        """Test proper handling of timestamps without timezone info"""
        mock_session = Mock()
        mock_query = Mock()
        mock_session.query.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        
        # Create mock trade with naive datetime
        naive_datetime = datetime(2025, 12, 5, 12, 0, 0)
        mock_trades = [
            MockTradeLog(
                id=1,
                symbol="BTCUSDT",
                side="BUY",
                qty=0.1,
                price=50000.0,
                status="FILLED",
                timestamp=naive_datetime
            ),
        ]
        mock_query.limit.return_value.all.return_value = mock_trades
        
        service = OrderService(mock_session)
        orders = service.get_recent_orders(limit=1)
        
        # Should add UTC timezone to naive datetime
        assert orders[0].timestamp.tzinfo is not None
        assert orders[0].timestamp.tzinfo == timezone.utc
