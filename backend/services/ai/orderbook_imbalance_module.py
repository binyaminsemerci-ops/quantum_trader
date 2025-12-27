"""
🔥 PHASE 2B: ORDERBOOK IMBALANCE MODULE

Real-time orderbook depth analysis for detecting aggressive buying/selling pressure.
Provides 5 key metrics:
1. Orderflow imbalance (-1 to 1)
2. Delta volume (cumulative aggressive trades)
3. Bid/ask spread percentage
4. Order book depth ratio
5. Large order presence detection

Uses WebSocket for real-time orderbook updates (10-100 updates/sec).

Author: Quantum Trader AI
Date: December 2025
"""

import asyncio
import structlog
from datetime import datetime, timezone
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Deque
from enum import Enum

logger = structlog.get_logger(__name__)


class OrderType(Enum):
    """Order aggressiveness classification"""
    AGGRESSIVE_BUY = "aggressive_buy"    # Market buy or aggressive limit buy
    AGGRESSIVE_SELL = "aggressive_sell"  # Market sell or aggressive limit sell
    PASSIVE = "passive"                   # Passive limit order


@dataclass
class OrderbookSnapshot:
    """Snapshot of orderbook at a point in time"""
    timestamp: datetime
    symbol: str
    bids: List[Tuple[float, float]]  # [(price, quantity), ...]
    asks: List[Tuple[float, float]]  # [(price, quantity), ...]
    best_bid: float
    best_ask: float
    mid_price: float
    spread_pct: float


@dataclass
class OrderbookMetrics:
    """Calculated orderbook imbalance metrics"""
    timestamp: datetime
    symbol: str
    
    # Core metrics (5 total)
    orderflow_imbalance: float        # -1 to 1 (negative = sell pressure)
    delta_volume: float               # Cumulative aggressive buy/sell delta
    bid_ask_spread_pct: float         # Spread as % of mid-price
    order_book_depth_ratio: float     # Bid depth / ask depth
    large_order_presence: float       # 0-1 score for large orders (>1% of volume)
    
    # Additional context
    bid_notional: float               # Total bid liquidity
    ask_notional: float               # Total ask liquidity
    total_depth: float                # Combined depth
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for feature extraction"""
        return {
            "orderflow_imbalance": self.orderflow_imbalance,
            "delta_volume": self.delta_volume,
            "bid_ask_spread_pct": self.bid_ask_spread_pct,
            "order_book_depth_ratio": self.order_book_depth_ratio,
            "large_order_presence": self.large_order_presence,
            "bid_notional": self.bid_notional,
            "ask_notional": self.ask_notional,
            "total_depth": self.total_depth
        }


class OrderbookImbalanceModule:
    """
    Analyzes orderbook depth to detect buying/selling pressure.
    
    Key Features:
    - Real-time orderflow imbalance calculation
    - Delta volume tracking (aggressive trades)
    - Spread analysis
    - Depth ratio monitoring
    - Large order detection
    
    Usage:
        module = OrderbookImbalanceModule(depth_levels=20, delta_volume_window=100)
        
        # Update with orderbook snapshot
        module.update_orderbook(symbol, bids, asks)
        
        # Get metrics
        metrics = module.get_metrics(symbol)
        print(f"Imbalance: {metrics.orderflow_imbalance:.3f}")
    """
    
    def __init__(
        self,
        depth_levels: int = 20,
        delta_volume_window: int = 100,
        large_order_threshold_pct: float = 0.01,  # 1% of total volume
        history_size: int = 50
    ):
        """
        Initialize orderbook imbalance module.
        
        Args:
            depth_levels: Number of orderbook levels to analyze (default: 20)
            delta_volume_window: Window for delta volume calculation (default: 100)
            large_order_threshold_pct: Threshold for large order detection (default: 0.01 = 1%)
            history_size: Max orderbook snapshots to store (default: 50)
        """
        self.depth_levels = depth_levels
        self.delta_volume_window = delta_volume_window
        self.large_order_threshold_pct = large_order_threshold_pct
        self.history_size = history_size
        
        # Per-symbol storage
        self._orderbook_history: Dict[str, Deque[OrderbookSnapshot]] = defaultdict(
            lambda: deque(maxlen=history_size)
        )
        self._delta_volume: Dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=delta_volume_window)
        )
        self._large_orders: Dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=20)  # Track last 20 large orders
        )
        
        # Current best bid/ask for each symbol
        self._current_best_bid: Dict[str, float] = {}
        self._current_best_ask: Dict[str, float] = {}
        
        logger.info(
            "[PHASE 2B] OrderbookImbalanceModule initialized",
            depth_levels=depth_levels,
            delta_volume_window=delta_volume_window,
            large_order_threshold=large_order_threshold_pct
        )
    
    def update_orderbook(
        self,
        symbol: str,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]]
    ) -> None:
        """
        Update orderbook snapshot for a symbol.
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            bids: List of (price, quantity) tuples for bids
            asks: List of (price, quantity) tuples for asks
        """
        if not bids or not asks:
            logger.warning(f"[OrderbookImbalance] Empty orderbook for {symbol}")
            return
        
        # Extract best bid/ask
        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        mid_price = (best_bid + best_ask) / 2.0
        spread_pct = ((best_ask - best_bid) / mid_price) * 100.0
        
        # Create snapshot
        snapshot = OrderbookSnapshot(
            timestamp=datetime.now(timezone.utc),
            symbol=symbol,
            bids=bids[:self.depth_levels],  # Limit to configured depth
            asks=asks[:self.depth_levels],
            best_bid=best_bid,
            best_ask=best_ask,
            mid_price=mid_price,
            spread_pct=spread_pct
        )
        
        # Store snapshot
        self._orderbook_history[symbol].append(snapshot)
        
        # Detect aggressive orders (compare with previous best bid/ask)
        if symbol in self._current_best_bid and symbol in self._current_best_ask:
            prev_best_bid = self._current_best_bid[symbol]
            prev_best_ask = self._current_best_ask[symbol]
            
            # Aggressive buy: price crossed previous best ask
            if best_bid > prev_best_ask:
                # Estimate volume based on price movement
                aggressive_volume = best_bid - prev_best_ask
                self._delta_volume[symbol].append(aggressive_volume)
            
            # Aggressive sell: price crossed previous best bid
            elif best_ask < prev_best_bid:
                aggressive_volume = prev_best_bid - best_ask
                self._delta_volume[symbol].append(-aggressive_volume)
        
        # Update current best bid/ask
        self._current_best_bid[symbol] = best_bid
        self._current_best_ask[symbol] = best_ask
    
    def calculate_orderflow_imbalance(self, symbol: str) -> float:
        """
        Calculate orderflow imbalance based on bid/ask volumes.
        
        Imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
        
        Returns:
            -1.0 to 1.0 where:
            - Negative = sell pressure (more asks than bids)
            - Positive = buy pressure (more bids than asks)
            - 0.0 = balanced
        """
        if symbol not in self._orderbook_history or len(self._orderbook_history[symbol]) == 0:
            return 0.0
        
        snapshot = self._orderbook_history[symbol][-1]
        
        # Calculate total bid volume
        bid_volume = sum(qty for price, qty in snapshot.bids)
        
        # Calculate total ask volume
        ask_volume = sum(qty for price, qty in snapshot.asks)
        
        if bid_volume + ask_volume == 0:
            return 0.0
        
        imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
        return float(imbalance)
    
    def calculate_delta_volume(self, symbol: str) -> float:
        """
        Calculate cumulative delta volume (aggressive buy volume - aggressive sell volume).
        
        Delta volume tracks the net aggressive order flow over the configured window.
        
        Returns:
            Cumulative delta volume (positive = net buying, negative = net selling)
        """
        if symbol not in self._delta_volume or len(self._delta_volume[symbol]) == 0:
            return 0.0
        
        return float(sum(self._delta_volume[symbol]))
    
    def calculate_depth_ratio(self, symbol: str) -> float:
        """
        Calculate bid depth / ask depth ratio.
        
        Returns:
            Ratio > 1.0 = more bid liquidity (buying pressure)
            Ratio < 1.0 = more ask liquidity (selling pressure)
            Ratio = 1.0 = balanced
        """
        if symbol not in self._orderbook_history or len(self._orderbook_history[symbol]) == 0:
            return 1.0
        
        snapshot = self._orderbook_history[symbol][-1]
        
        # Calculate bid notional (price * quantity)
        bid_notional = sum(price * qty for price, qty in snapshot.bids)
        
        # Calculate ask notional
        ask_notional = sum(price * qty for price, qty in snapshot.asks)
        
        if ask_notional == 0:
            return 2.0  # Cap at 2.0 to avoid infinity
        
        ratio = bid_notional / ask_notional
        return float(min(ratio, 2.0))  # Cap at 2.0
    
    def detect_large_orders(self, symbol: str) -> float:
        """
        Detect presence of large orders in the orderbook.
        
        Large orders are defined as orders > large_order_threshold_pct of total volume.
        
        Returns:
            0.0 to 1.0 score indicating large order presence
        """
        if symbol not in self._orderbook_history or len(self._orderbook_history[symbol]) == 0:
            return 0.0
        
        snapshot = self._orderbook_history[symbol][-1]
        
        # Calculate total volume
        total_volume = sum(qty for _, qty in snapshot.bids) + sum(qty for _, qty in snapshot.asks)
        
        if total_volume == 0:
            return 0.0
        
        # Find large orders
        large_order_threshold = total_volume * self.large_order_threshold_pct
        large_orders = []
        
        for price, qty in snapshot.bids + snapshot.asks:
            if qty > large_order_threshold:
                large_orders.append(qty)
        
        if not large_orders:
            return 0.0
        
        # Store large order detection
        self._large_orders[symbol].append(len(large_orders))
        
        # Calculate score (0-1) based on number of large orders
        # More large orders = higher score
        score = min(len(large_orders) / 5.0, 1.0)  # Cap at 5 large orders = 1.0
        return float(score)
    
    def get_metrics(self, symbol: str) -> Optional[OrderbookMetrics]:
        """
        Get comprehensive orderbook metrics for a symbol.
        
        Args:
            symbol: Trading pair
        
        Returns:
            OrderbookMetrics object with all 5 key metrics + context, or None if no data
        """
        if symbol not in self._orderbook_history or len(self._orderbook_history[symbol]) == 0:
            return None
        
        snapshot = self._orderbook_history[symbol][-1]
        
        # Calculate all metrics
        orderflow_imbalance = self.calculate_orderflow_imbalance(symbol)
        delta_volume = self.calculate_delta_volume(symbol)
        depth_ratio = self.calculate_depth_ratio(symbol)
        large_order_presence = self.detect_large_orders(symbol)
        
        # Calculate notional values
        bid_notional = sum(price * qty for price, qty in snapshot.bids)
        ask_notional = sum(price * qty for price, qty in snapshot.asks)
        total_depth = bid_notional + ask_notional
        
        return OrderbookMetrics(
            timestamp=snapshot.timestamp,
            symbol=symbol,
            orderflow_imbalance=orderflow_imbalance,
            delta_volume=delta_volume,
            bid_ask_spread_pct=snapshot.spread_pct,
            order_book_depth_ratio=depth_ratio,
            large_order_presence=large_order_presence,
            bid_notional=bid_notional,
            ask_notional=ask_notional,
            total_depth=total_depth
        )
    
    def get_summary_stats(self, symbol: str) -> Dict:
        """
        Get summary statistics for debugging/monitoring.
        
        Returns:
            Dictionary with snapshot count, delta volume window size, etc.
        """
        return {
            "symbol": symbol,
            "snapshots_stored": len(self._orderbook_history.get(symbol, [])),
            "delta_volume_window": len(self._delta_volume.get(symbol, [])),
            "large_orders_detected": len(self._large_orders.get(symbol, [])),
            "current_best_bid": self._current_best_bid.get(symbol, 0.0),
            "current_best_ask": self._current_best_ask.get(symbol, 0.0)
        }


# Example usage
if __name__ == "__main__":
    # Initialize module
    module = OrderbookImbalanceModule(depth_levels=20, delta_volume_window=100)
    
    # Simulate orderbook update
    bids = [(100.0, 1.5), (99.9, 2.0), (99.8, 1.0)]
    asks = [(100.1, 1.0), (100.2, 1.5), (100.3, 2.0)]
    
    module.update_orderbook("BTCUSDT", bids, asks)
    
    # Get metrics
    metrics = module.get_metrics("BTCUSDT")
    if metrics:
        print(f"Orderflow Imbalance: {metrics.orderflow_imbalance:.3f}")
        print(f"Delta Volume: {metrics.delta_volume:.2f}")
        print(f"Spread: {metrics.bid_ask_spread_pct:.4f}%")
        print(f"Depth Ratio: {metrics.order_book_depth_ratio:.3f}")
        print(f"Large Orders: {metrics.large_order_presence:.2f}")
