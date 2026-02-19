"""
Unit Tests for Multi-Exchange Abstraction (EPIC-EXCH-001)

Tests:
- Interface compliance (IExchangeClient Protocol)
- Factory creation (get_exchange_client)
- Symbol routing (resolve_exchange_for_symbol)
- Model validation (Pydantic models)
- Error handling (ExchangeAPIError)
"""

import pytest
from decimal import Decimal
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch

from backend.integrations.exchanges import (
    IExchangeClient,
    ExchangeAPIError,
    OrderSide,
    OrderType,
    OrderStatus,
    TimeInForce,
    PositionSide,
    OrderRequest,
    OrderResult,
    CancelResult,
    Position,
    Balance,
    Kline,
    ExchangeType,
    ExchangeConfig,
    get_exchange_client,
    resolve_exchange_for_symbol,
    set_symbol_exchange_mapping,
    BinanceAdapter,
    BybitAdapter,
    OKXAdapter,
)


# ============================================================================
# Test: Pydantic Model Validation
# ============================================================================

class TestModels:
    """Test Pydantic models and validators."""
    
    def test_order_request_creation(self):
        """Test OrderRequest creation with valid data."""
        request = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.5"),
            price=Decimal("50000.00"),
            leverage=10
        )
        
        assert request.symbol == "BTCUSDT"
        assert request.side == OrderSide.BUY
        assert request.order_type == OrderType.LIMIT
        assert request.quantity == Decimal("0.5")
        assert request.price == Decimal("50000.00")
        assert request.leverage == 10
    
    def test_symbol_uppercase_validator(self):
        """Test symbol is automatically uppercased."""
        request = OrderRequest(
            symbol="btcusdt",  # lowercase
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0")
        )
        
        assert request.symbol == "BTCUSDT"  # Should be uppercase
    
    def test_order_result_creation(self):
        """Test OrderResult creation."""
        result = OrderResult(
            order_id="12345",
            symbol="ETHUSDT",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal("5.0"),
            filled_quantity=Decimal("5.0"),
            status=OrderStatus.FILLED,
            timestamp=datetime.now(),
            exchange="binance"
        )
        
        assert result.order_id == "12345"
        assert result.status == OrderStatus.FILLED
        assert result.exchange == "binance"
    
    def test_position_creation(self):
        """Test Position model creation."""
        position = Position(
            symbol="SOLUSDT",
            side=OrderSide.BUY,
            quantity=Decimal("100.0"),
            entry_price=Decimal("25.50"),
            mark_price=Decimal("26.00"),
            unrealized_pnl=Decimal("50.00"),
            leverage=5,
            margin=Decimal("510.00"),
            exchange="binance"
        )
        
        assert position.symbol == "SOLUSDT"
        assert position.side == OrderSide.BUY
        assert position.leverage == 5
        assert position.exchange == "binance"
    
    def test_balance_creation(self):
        """Test Balance model creation."""
        balance = Balance(
            asset="USDT",
            free=Decimal("10000.00"),
            locked=Decimal("500.00"),
            total=Decimal("10500.00"),
            exchange="binance"
        )
        
        assert balance.asset == "USDT"
        assert balance.total == Decimal("10500.00")
        assert balance.exchange == "binance"


# ============================================================================
# Test: Factory & Routing
# ============================================================================

class TestFactory:
    """Test exchange factory and symbol routing."""
    
    def test_binance_adapter_creation(self):
        """Test BinanceAdapter creation via factory."""
        mock_client = Mock()
        mock_wrapper = Mock()
        
        config = ExchangeConfig(
            exchange=ExchangeType.BINANCE,
            api_key="test_key",
            api_secret="test_secret",
            client=mock_client,
            wrapper=mock_wrapper,
            testnet=False
        )
        
        adapter = get_exchange_client(config)
        
        assert isinstance(adapter, BinanceAdapter)
        assert adapter.get_exchange_name() == "binance"
        assert adapter.client == mock_client
        assert adapter.wrapper == mock_wrapper
    
    def test_bybit_adapter_creation(self):
        """Test BybitAdapter creation via factory."""
        config = ExchangeConfig(
            exchange=ExchangeType.BYBIT,
            api_key="test_key",
            api_secret="test_secret",
            testnet=True
        )
        
        adapter = get_exchange_client(config)
        
        assert isinstance(adapter, BybitAdapter)
        assert adapter.get_exchange_name() == "bybit"
    
    def test_okx_adapter_creation(self):
        """Test OKXAdapter creation via factory."""
        config = ExchangeConfig(
            exchange=ExchangeType.OKX,
            api_key="test_key",
            api_secret="test_secret",
            passphrase="test_passphrase",
            testnet=False
        )
        
        adapter = get_exchange_client(config)
        
        assert isinstance(adapter, OKXAdapter)
        assert adapter.get_exchange_name() == "okx"
    
    def test_okx_requires_passphrase(self):
        """Test OKX adapter requires passphrase."""
        config = ExchangeConfig(
            exchange=ExchangeType.OKX,
            api_key="test_key",
            api_secret="test_secret",
            # Missing passphrase
        )
        
        with pytest.raises(ValueError, match="passphrase"):
            get_exchange_client(config)
    
    def test_binance_requires_client(self):
        """Test Binance adapter requires client instance."""
        config = ExchangeConfig(
            exchange=ExchangeType.BINANCE,
            api_key="test_key",
            api_secret="test_secret",
            # Missing client
        )
        
        with pytest.raises(ValueError, match="client"):
            get_exchange_client(config)
    
    def test_symbol_routing_default(self):
        """Test default symbol routing (all to Binance)."""
        exchange = resolve_exchange_for_symbol("BTCUSDT")
        assert exchange == ExchangeType.BINANCE
        
        exchange = resolve_exchange_for_symbol("ETHUSDT")
        assert exchange == ExchangeType.BINANCE
    
    def test_symbol_routing_custom(self):
        """Test custom symbol routing."""
        # Set custom mapping
        set_symbol_exchange_mapping({
            "BTCUSDT": ExchangeType.BINANCE,
            "ETHUSDT": ExchangeType.BYBIT,
            "SOLUSDT": ExchangeType.OKX,
        })
        
        assert resolve_exchange_for_symbol("BTCUSDT") == ExchangeType.BINANCE
        assert resolve_exchange_for_symbol("ETHUSDT") == ExchangeType.BYBIT
        assert resolve_exchange_for_symbol("SOLUSDT") == ExchangeType.OKX
        
        # Unmapped symbol defaults to Binance
        assert resolve_exchange_for_symbol("ADAUSDT") == ExchangeType.BINANCE
    
    def test_symbol_routing_case_insensitive(self):
        """Test symbol routing is case-insensitive."""
        set_symbol_exchange_mapping({
            "BTCUSDT": ExchangeType.BYBIT,
        })
        
        # All should resolve to Bybit
        assert resolve_exchange_for_symbol("BTCUSDT") == ExchangeType.BYBIT
        assert resolve_exchange_for_symbol("btcusdt") == ExchangeType.BYBIT
        assert resolve_exchange_for_symbol("BtCuSdT") == ExchangeType.BYBIT


# ============================================================================
# Test: Adapter Interface Compliance
# ============================================================================

class TestAdapterCompliance:
    """Test adapters implement IExchangeClient Protocol."""
    
    @pytest.mark.asyncio
    async def test_binance_adapter_implements_protocol(self):
        """Test BinanceAdapter implements all IExchangeClient methods."""
        mock_client = Mock()
        adapter = BinanceAdapter(client=mock_client, testnet=False)
        
        # Check all required methods exist
        assert hasattr(adapter, 'place_order')
        assert hasattr(adapter, 'cancel_order')
        assert hasattr(adapter, 'get_open_positions')
        assert hasattr(adapter, 'get_balances')
        assert hasattr(adapter, 'get_klines')
        assert hasattr(adapter, 'get_order_status')
        assert hasattr(adapter, 'set_leverage')
        assert hasattr(adapter, 'close_position')
        assert hasattr(adapter, 'get_exchange_name')
        
        # Check they're callable
        assert callable(adapter.place_order)
        assert callable(adapter.cancel_order)
        assert callable(adapter.get_open_positions)
        assert callable(adapter.get_balances)
        assert callable(adapter.get_klines)
        assert callable(adapter.get_order_status)
        assert callable(adapter.set_leverage)
        assert callable(adapter.close_position)
        assert callable(adapter.get_exchange_name)
    
    @pytest.mark.asyncio
    async def test_bybit_adapter_raises_not_implemented(self):
        """Test BybitAdapter raises NotImplementedError (skeleton)."""
        adapter = BybitAdapter(api_key="test", api_secret="test")
        
        request = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0")
        )
        
        with pytest.raises(NotImplementedError):
            await adapter.place_order(request)
        
        with pytest.raises(NotImplementedError):
            await adapter.cancel_order("BTCUSDT", "12345")
        
        with pytest.raises(NotImplementedError):
            await adapter.get_open_positions()
        
        with pytest.raises(NotImplementedError):
            await adapter.get_balances()
        
        with pytest.raises(NotImplementedError):
            await adapter.get_klines("BTCUSDT", "1h")
        
        # get_exchange_name should work (not async)
        assert adapter.get_exchange_name() == "bybit"
    
    @pytest.mark.asyncio
    async def test_okx_adapter_raises_not_implemented(self):
        """Test OKXAdapter raises NotImplementedError (skeleton)."""
        adapter = OKXAdapter(
            api_key="test",
            api_secret="test",
            passphrase="test"
        )
        
        request = OrderRequest(
            symbol="ETHUSDT",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal("2.0")
        )
        
        with pytest.raises(NotImplementedError):
            await adapter.place_order(request)
        
        # get_exchange_name should work
        assert adapter.get_exchange_name() == "okx"


# ============================================================================
# Test: BinanceAdapter Integration (Mocked)
# ============================================================================

class TestBinanceAdapterIntegration:
    """Test BinanceAdapter with mocked Binance client."""
    
    @pytest.mark.asyncio
    async def test_place_order_success(self):
        """Test successful order placement."""
        mock_client = Mock()
        mock_client.futures_create_order = Mock(return_value={
            'orderId': 12345,
            'clientOrderId': 'test_order_1',
            'symbol': 'BTCUSDT',
            'side': 'BUY',
            'type': 'LIMIT',
            'origQty': '0.5',
            'executedQty': '0.5',
            'price': '50000.00',
            'avgPrice': '50000.00',
            'status': 'FILLED',
            'updateTime': 1700000000000
        })
        
        adapter = BinanceAdapter(client=mock_client, testnet=False)
        
        request = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.5"),
            price=Decimal("50000.00")
        )
        
        result = await adapter.place_order(request)
        
        assert result.order_id == "12345"
        assert result.symbol == "BTCUSDT"
        assert result.status == OrderStatus.FILLED
        assert result.exchange == "binance"
        assert mock_client.futures_create_order.called
    
    @pytest.mark.asyncio
    async def test_get_positions_filters_zero(self):
        """Test get_open_positions filters zero-quantity positions."""
        mock_client = Mock()
        mock_client.futures_position_information = Mock(return_value=[
            {
                'symbol': 'BTCUSDT',
                'positionAmt': '1.0',  # Active position
                'entryPrice': '50000.00',
                'markPrice': '51000.00',
                'unRealizedProfit': '1000.00',
                'leverage': '10',
                'isolatedMargin': '5100.00',
                'positionSide': 'BOTH'
            },
            {
                'symbol': 'ETHUSDT',
                'positionAmt': '0.0',  # Should be filtered
                'entryPrice': '0',
                'markPrice': '3000.00',
                'unRealizedProfit': '0',
                'leverage': '10',
                'isolatedMargin': '0',
                'positionSide': 'BOTH'
            }
        ])
        
        adapter = BinanceAdapter(client=mock_client, testnet=False)
        positions = await adapter.get_open_positions()
        
        # Should only return BTCUSDT (non-zero position)
        assert len(positions) == 1
        assert positions[0].symbol == "BTCUSDT"
        assert positions[0].quantity == Decimal("1.0")
        assert positions[0].exchange == "binance"
    
    @pytest.mark.asyncio
    async def test_cancel_order_success(self):
        """Test successful order cancellation."""
        mock_client = Mock()
        mock_client.futures_cancel_order = Mock(return_value={
            'orderId': 12345,
            'symbol': 'BTCUSDT',
            'status': 'CANCELED'
        })
        
        adapter = BinanceAdapter(client=mock_client, testnet=False)
        result = await adapter.cancel_order("BTCUSDT", "12345")
        
        assert result.order_id == "12345"
        assert result.symbol == "BTCUSDT"
        assert result.status == OrderStatus.CANCELED
        assert result.success is True
        assert result.exchange == "binance"
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test ExchangeAPIError wrapping."""
        mock_client = Mock()
        mock_client.futures_create_order = Mock(
            side_effect=Exception("API Error: Invalid symbol")
        )
        
        adapter = BinanceAdapter(client=mock_client, testnet=False)
        
        request = OrderRequest(
            symbol="INVALID",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0")
        )
        
        with pytest.raises(ExchangeAPIError) as exc_info:
            await adapter.place_order(request)
        
        assert exc_info.value.exchange == "binance"
        assert "API Error" in str(exc_info.value.message)


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
