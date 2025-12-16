"""
Tests for SignalService
EPIC: DASHBOARD-V3-TRADING-PANELS
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch
from backend.domains.signals import SignalService, SignalRecord


class TestSignalService:
    """Test suite for SignalService"""
    
    @pytest.mark.asyncio
    async def test_get_recent_signals_success(self):
        """Test successful retrieval of recent signals"""
        # Mock response data
        mock_response_data = [
            {
                "id": "sig_1",
                "timestamp": "2025-12-05T12:00:00Z",
                "symbol": "BTCUSDT",
                "side": "BUY",
                "confidence": 0.85,
                "price": 50000.0,
                "source": "ai_engine"
            },
            {
                "id": "sig_2",
                "timestamp": "2025-12-05T13:00:00Z",
                "symbol": "ETHUSDT",
                "side": "SELL",
                "confidence": 0.72,
                "price": 3000.0,
                "source": "ai_engine"
            },
        ]
        
        # Mock httpx.AsyncClient
        with patch('backend.domains.signals.service.httpx.AsyncClient') as mock_client_class:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_response_data
            
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__.return_value = mock_client
            mock_client_class.return_value = mock_client
            
            # Create service and get signals
            service = SignalService()
            signals = await service.get_recent_signals(limit=10)
            
            # Assertions
            assert len(signals) == 2
            assert isinstance(signals[0], SignalRecord)
            assert signals[0].symbol == "BTCUSDT"
            assert signals[0].direction == "LONG"  # BUY mapped to LONG
            assert signals[0].confidence == 0.85
            assert signals[1].symbol == "ETHUSDT"
            assert signals[1].direction == "SHORT"  # SELL mapped to SHORT
    
    @pytest.mark.asyncio
    async def test_get_recent_signals_empty(self):
        """Test retrieval when no signals exist"""
        # Mock empty response
        with patch('backend.domains.signals.service.httpx.AsyncClient') as mock_client_class:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = []
            
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__.return_value = mock_client
            mock_client_class.return_value = mock_client
            
            service = SignalService()
            signals = await service.get_recent_signals(limit=10)
            
            assert len(signals) == 0
            assert isinstance(signals, list)
    
    @pytest.mark.asyncio
    async def test_get_recent_signals_object_response(self):
        """Test handling of object response with 'signals' key"""
        # Mock response as object
        mock_response_data = {
            "signals": [
                {
                    "id": "sig_1",
                    "timestamp": "2025-12-05T12:00:00Z",
                    "symbol": "BTCUSDT",
                    "side": "BUY",
                    "confidence": 0.85,
                    "source": "ai_engine"
                },
            ]
        }
        
        with patch('backend.domains.signals.service.httpx.AsyncClient') as mock_client_class:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_response_data
            
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__.return_value = mock_client
            mock_client_class.return_value = mock_client
            
            service = SignalService()
            signals = await service.get_recent_signals(limit=10)
            
            assert len(signals) == 1
            assert signals[0].symbol == "BTCUSDT"
    
    @pytest.mark.asyncio
    async def test_get_recent_signals_direction_normalization(self):
        """Test normalization of BUY/SELL to LONG/SHORT"""
        test_cases = [
            ("BUY", "LONG"),
            ("SELL", "SHORT"),
            ("buy", "LONG"),
            ("sell", "SHORT"),
            ("LONG", "LONG"),
            ("SHORT", "SHORT"),
            ("HOLD", "HOLD"),
        ]
        
        for input_side, expected_direction in test_cases:
            mock_response_data = [
                {
                    "id": "sig_1",
                    "timestamp": "2025-12-05T12:00:00Z",
                    "symbol": "BTCUSDT",
                    "side": input_side,
                    "confidence": 0.85,
                    "source": "ai_engine"
                },
            ]
            
            with patch('backend.domains.signals.service.httpx.AsyncClient') as mock_client_class:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = mock_response_data
                
                mock_client = AsyncMock()
                mock_client.get = AsyncMock(return_value=mock_response)
                mock_client.__aenter__.return_value = mock_client
                mock_client_class.return_value = mock_client
                
                service = SignalService()
                signals = await service.get_recent_signals(limit=1)
                
                assert signals[0].direction == expected_direction
    
    @pytest.mark.asyncio
    async def test_get_recent_signals_http_error(self):
        """Test error handling for HTTP errors"""
        with patch('backend.domains.signals.service.httpx.AsyncClient') as mock_client_class:
            mock_response = Mock()
            mock_response.status_code = 500
            
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__.return_value = mock_client
            mock_client_class.return_value = mock_client
            
            service = SignalService()
            signals = await service.get_recent_signals(limit=10)
            
            # Should return empty list on HTTP error
            assert len(signals) == 0
            assert isinstance(signals, list)
    
    @pytest.mark.asyncio
    async def test_get_recent_signals_timeout(self):
        """Test error handling for timeouts"""
        with patch('backend.domains.signals.service.httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=TimeoutError("Request timeout"))
            mock_client.__aenter__.return_value = mock_client
            mock_client_class.return_value = mock_client
            
            service = SignalService()
            signals = await service.get_recent_signals(limit=10)
            
            # Should return empty list on timeout
            assert len(signals) == 0
            assert isinstance(signals, list)
    
    @pytest.mark.asyncio
    async def test_get_signals_by_symbol(self):
        """Test filtering signals by symbol"""
        mock_response_data = [
            {
                "id": "sig_1",
                "timestamp": "2025-12-05T12:00:00Z",
                "symbol": "BTCUSDT",
                "side": "BUY",
                "confidence": 0.85,
                "source": "ai_engine"
            },
        ]
        
        with patch('backend.domains.signals.service.httpx.AsyncClient') as mock_client_class:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_response_data
            
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__.return_value = mock_client
            mock_client_class.return_value = mock_client
            
            service = SignalService()
            signals = await service.get_signals_by_symbol("BTCUSDT", limit=10)
            
            assert len(signals) == 1
            assert signals[0].symbol == "BTCUSDT"
    
    @pytest.mark.asyncio
    async def test_custom_endpoint(self):
        """Test using custom signals endpoint"""
        custom_endpoint = "http://custom-ai-engine:8001/signals/recent"
        
        with patch('backend.domains.signals.service.httpx.AsyncClient') as mock_client_class:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = []
            
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__.return_value = mock_client
            mock_client_class.return_value = mock_client
            
            service = SignalService(signals_endpoint=custom_endpoint)
            await service.get_recent_signals(limit=10)
            
            # Verify custom endpoint was called
            call_args = mock_client.get.call_args
            assert call_args[0][0] == custom_endpoint
