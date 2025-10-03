# Price stream service
import asyncio
from typing import Any, Callable, Optional


class PriceStreamManager:
    """Price stream manager stub."""
    
    def __init__(self) -> None:
        self.subscribers: list[Callable[[dict[str, Any]], None]] = []
    
    def subscribe(self, callback: Callable[[dict[str, Any]], None]) -> None:
        """Subscribe to price updates."""
        self.subscribers.append(callback)
    
    def unsubscribe(self, callback: Callable[[dict[str, Any]], None]) -> None:
        """Unsubscribe from price updates."""
        if callback in self.subscribers:
            self.subscribers.remove(callback)
    
    async def start_stream(self) -> None:
        """Start the price stream."""
        pass
    
    async def stop_stream(self) -> None:
        """Stop the price stream."""
        pass


_price_stream_manager = PriceStreamManager()


def get_price_stream_manager() -> PriceStreamManager:
    """Get the price stream manager instance."""
    return _price_stream_manager


def ensure_price_stream() -> None:
    """Ensure price stream is running."""
    pass


def get_price_snapshot() -> dict[str, Any]:
    """Get current price snapshot."""
    return {"BTCUSDT": 50000.0, "ETHUSDT": 3000.0}


def get_last_error() -> Optional[str]:
    """Get last error message."""
    return None


def get_orderbook_snapshot() -> dict[str, Any]:
    """Get orderbook snapshot."""
    return {
        "bids": [[50000.0, 0.1], [49999.0, 0.2]], 
        "asks": [[50001.0, 0.1], [50002.0, 0.2]]
    }