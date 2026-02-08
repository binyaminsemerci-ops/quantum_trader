"""
Entry Scanner - Finds entry opportunities using market analysis
"""
import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EntryOpportunity:
    """Potential entry signal"""
    symbol: str
    side: str  # LONG/SHORT
    confidence: float
    volatility: float
    regime: str
    price: float
    timestamp: int
    reason: str


class EntryScanner:
    """
    Scans market for entry opportunities
    
    Uses:
    - Market data (price, volume, volatility)
    - Regime detection
    - Confidence thresholds
    """
    
    def __init__(
        self,
        redis_client,
        min_confidence: float = 0.65,
        max_positions: int = 5,
        symbols: List[str] = None
    ):
        self.redis = redis_client
        self.min_confidence = min_confidence
        self.max_positions = max_positions
        self.symbols = symbols or ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        
        logger.info(f"[EntryScanner] Initialized: {len(self.symbols)} symbols ({{}}), min_conf={min_confidence}".format(", ".join(self.symbols[:5])))
    
    async def scan_for_entries(
        self,
        current_position_count: int,
        max_exposure_usd: float = 2500.0
    ) -> List[EntryOpportunity]:
        """
        Scan for entry opportunities
        
        Args:
            current_position_count: Number of active positions
            max_exposure_usd: Maximum total exposure allowed
        
        Returns:
            List of entry opportunities sorted by confidence
        """
        if current_position_count >= self.max_positions:
            return []
        
        opportunities = []
        
        for symbol in self.symbols:
            try:
                opportunity = await self._evaluate_symbol(symbol)
                if opportunity and opportunity.confidence >= self.min_confidence:
                    opportunities.append(opportunity)
            except Exception as e:
                logger.error(f"[EntryScanner] Error evaluating {symbol}: {e}")
        
        # Sort by confidence descending
        opportunities.sort(key=lambda x: x.confidence, reverse=True)
        
        # Limit to available slots
        available_slots = self.max_positions - current_position_count
        opportunities = opportunities[:available_slots]
        
        if opportunities:
            logger.info(
                f"[EntryScanner] Found {len(opportunities)} opportunities: " +
                ", ".join([f"{o.symbol}({o.confidence:.2f})" for o in opportunities])
            )
        
        return opportunities
    
    async def _evaluate_symbol(self, symbol: str) -> Optional[EntryOpportunity]:
        """
        Evaluate single symbol for entry
        
        Checks:
        1. AI Engine signal (from Redis stream)
        2. Market regime
        3. Volatility
        4. Recent performance
        """
        try:
            # Check for recent AI signals
            messages = await self.redis.xrevrange(
                "quantum:stream:ai.signal_generated",
                count=50
            )
            
            # Find latest signal for this symbol
            for msg_id, data in messages:
                # Parse payload JSON
                payload_str = data.get("payload", "{}")
                try:
                    payload = json.loads(payload_str)
                except:
                    continue
                
                if payload.get("symbol") != symbol:
                    continue
                
                # Check signal age (max 5 minutes old)
                timestamp_str = data.get("timestamp", "")
                try:
                    dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    timestamp = int(dt.timestamp())
                except:
                    timestamp = int(time.time())
                
                age_sec = int(time.time()) - timestamp
                if age_sec > 300:
                    logger.debug(f"[Scanner] Signal for {symbol} too old: {age_sec}s")
                    continue
                
                # Parse signal
                action = payload.get("action", "").upper()
                confidence = float(payload.get("confidence", 0))
                regime = payload.get("regime", "UNKNOWN")
                price = float(payload.get("price", 0))
                
                if action not in ["BUY", "SELL"]:
                    continue
                
                if confidence < self.min_confidence:
                    continue
                
                # Get volatility from VSE if available
                volatility = await self._get_volatility(symbol)
                
                side = "LONG" if action == "BUY" else "SHORT"
                
                return EntryOpportunity(
                    symbol=symbol,
                    side=side,
                    confidence=confidence,
                    volatility=volatility,
                    regime=regime,
                    price=price,
                    timestamp=timestamp,
                    reason=f"AI signal: {action} conf={confidence:.2f} regime={regime}"
                )
            
            return None
        
        except Exception as e:
            logger.error(f"[EntryScanner] Error evaluating {symbol}: {e}", exc_info=True)
            return None
    
    async def _get_volatility(self, symbol: str) -> float:
        """Get current volatility for symbol"""
        try:
            # Try to get from VSE stream
            messages = await self.redis.xrevrange(
                "quantum:stream:vse.structure",
                count=20
            )
            
            for msg_id, data in messages:
                if data.get("symbol") == symbol:
                    atr = float(data.get("atr", 0))
                    return atr
            
            # Fallback: estimate from price history
            return 0.02  # 2% default
        
        except Exception:
            return 0.02
    
    def update_symbols(self, symbols: List[str]):
        """Update watchlist symbols"""
        self.symbols = symbols
        logger.info(f"[EntryScanner] Updated symbols: {len(symbols)}")
    
    def set_min_confidence(self, confidence: float):
        """Update minimum confidence threshold"""
        self.min_confidence = confidence
        logger.info(f"[EntryScanner] Updated min_confidence: {confidence}")
