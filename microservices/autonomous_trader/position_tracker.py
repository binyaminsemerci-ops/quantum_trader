"""
Position Tracker - Real-time position state management
"""
import asyncio
import logging
import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Active position state"""
    symbol: str
    side: str  # LONG/SHORT
    entry_price: float
    current_price: float
    position_qty: float
    entry_timestamp: int
    leverage: float
    stop_loss: float
    take_profit: float
    entry_regime: str
    entry_confidence: float
    peak_price: float = 0.0
    R_history: List[float] = field(default_factory=list)
    
    @property
    def age_sec(self) -> int:
        return int(time.time() - self.entry_timestamp)
    
    @property
    def R_net(self) -> float:
        """Calculate R multiple
        
        If stop_loss is not set (=0), uses default 2% risk
        """
        if self.side == "LONG":
            risk = self.entry_price - self.stop_loss if self.stop_loss > 0 else self.entry_price * 0.02
            profit = self.current_price - self.entry_price
        else:
            risk = self.stop_loss - self.entry_price if self.stop_loss > 0 else self.entry_price * 0.02
            profit = self.entry_price - self.current_price
        
        return profit / risk if risk > 0 else 0.0
    
    @property
    def pnl_usd(self) -> float:
        """Calculate unrealized PnL in USD"""
        if self.side == "LONG":
            return (self.current_price - self.entry_price) * self.position_qty
        else:
            return (self.entry_price - self.current_price) * self.position_qty
    
    @property
    def pnl_pct(self) -> float:
        """Calculate PnL percentage"""
        if self.side == "LONG":
            return ((self.current_price - self.entry_price) / self.entry_price) * 100
        else:
            return ((self.entry_price - self.current_price) / self.entry_price) * 100
    
    def update_price(self, price: float):
        """Update current price and peak tracking"""
        self.current_price = price
        
        if self.side == "LONG":
            if price > self.peak_price:
                self.peak_price = price
        else:
            if self.peak_price == 0 or price < self.peak_price:
                self.peak_price = price
        
        # Update R history (keep last 10)
        current_r = self.R_net
        if len(self.R_history) == 0 or abs(current_r - self.R_history[-1]) > 0.1:
            self.R_history.append(current_r)
            if len(self.R_history) > 10:
                self.R_history.pop(0)


class PositionTracker:
    """
    Tracks active positions and updates their state in real-time
    """
    
    def __init__(self, redis_client):
        self.redis = redis_client
        self.positions: Dict[str, Position] = {}
        self._running = False
        self._update_task: Optional[asyncio.Task] = None
        
        logger.info("[PositionTracker] Initialized")
    
    async def start(self):
        """Start position tracking"""
        self._running = True
        self._update_task = asyncio.create_task(self._update_loop())
        logger.info("[PositionTracker] Started")
    
    async def stop(self):
        """Stop position tracking"""
        self._running = False
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
        logger.info("[PositionTracker] Stopped")
    
    async def _update_loop(self):
        """Update position states every 10 seconds"""
        while self._running:
            try:
                await self._refresh_positions()
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[PositionTracker] Update error: {e}", exc_info=True)
                await asyncio.sleep(10)
    
    async def _refresh_positions(self):
        """Fetch current positions from Redis"""
        try:
            # Read position snapshot stream (only recent messages, max 60 seconds old)
            messages = await self.redis.xrevrange(
                "quantum:stream:position.snapshot",
                count=100
            )
            
            current_symbols = set()
            seen_symbols = set()
            current_time = time.time()
            
            for msg_id, data in messages:
                symbol = data.get("symbol")
                if not symbol:
                    continue
                
                # Skip if we already processed this symbol (only use latest message per symbol)
                if symbol in seen_symbols:
                    continue
                seen_symbols.add(symbol)
                
                # Skip old messages (>60 seconds old)
                timestamp = int(data.get("timestamp", 0))
                if timestamp > 0 and (current_time - timestamp) > 60:
                    continue
                
                # Parse position data
                side = data.get("side", "LONG")
                entry_price = float(data.get("entry_price", 0))
                current_price = float(data.get("mark_price", entry_price))
                qty = float(data.get("position_qty", 0))
                entry_timestamp = int(data.get("entry_timestamp", time.time()))
                leverage = float(data.get("leverage", 1.0))
                stop_loss = float(data.get("stop_loss", 0))
                take_profit = float(data.get("take_profit", 0))
                entry_regime = data.get("entry_regime", "UNKNOWN")
                entry_confidence = float(data.get("entry_confidence", 0.5))
                
                if qty > 0:
                    current_symbols.add(symbol)
                    
                    if symbol in self.positions:
                        # Update existing position
                        self.positions[symbol].update_price(current_price)
                    else:
                        # New position
                        pos = Position(
                            symbol=symbol,
                            side=side,
                            entry_price=entry_price,
                            current_price=current_price,
                            position_qty=qty,
                            entry_timestamp=entry_timestamp,
                            leverage=leverage,
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                            entry_regime=entry_regime,
                            entry_confidence=entry_confidence,
                            peak_price=current_price
                        )
                        self.positions[symbol] = pos
                        logger.info(f"[PositionTracker] New position: {symbol} {side} @ {entry_price}")
            
            # Remove closed positions
            closed = [s for s in self.positions.keys() if s not in current_symbols]
            for symbol in closed:
                logger.info(f"[PositionTracker] Position closed: {symbol}")
                del self.positions[symbol]
        
        except Exception as e:
            logger.error(f"[PositionTracker] Refresh error: {e}", exc_info=True)
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position by symbol"""
        return self.positions.get(symbol)
    
    def get_all_positions(self) -> List[Position]:
        """Get all active positions"""
        return list(self.positions.values())
    
    def has_position(self, symbol: str) -> bool:
        """Check if position exists"""
        return symbol in self.positions
    
    def get_total_exposure_usd(self) -> float:
        """Calculate total exposure across all positions"""
        return sum(
            pos.position_qty * pos.current_price
            for pos in self.positions.values()
        )
    
    def get_total_pnl_usd(self) -> float:
        """Calculate total unrealized PnL"""
        return sum(pos.pnl_usd for pos in self.positions.values())
