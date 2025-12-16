"""
Signal Service - Query layer for AI signals
EPIC: DASHBOARD-V3-TRADING-PANELS

Provides access to recently generated AI signals.
"""

import logging
import httpx
from typing import List, Optional
from datetime import datetime, timezone

from backend.domains.signals.models import SignalRecord

logger = logging.getLogger(__name__)


class SignalService:
    """
    Service for querying AI-generated signals.
    
    Fetches signals from live AI signal endpoint.
    """
    
    def __init__(self, signals_endpoint: str = "http://localhost:8000/signals/recent"):
        """
        Initialize signal service.
        
        Args:
            signals_endpoint: URL for signals API
        """
        self.signals_endpoint = signals_endpoint
    
    async def get_recent_signals(self, limit: int = 20) -> List[SignalRecord]:
        """
        Get recent AI signals for dashboard display.
        
        Args:
            limit: Maximum number of signals to return (default 20)
            
        Returns:
            List of SignalRecord objects, most recent first
        """
        try:
            # Fetch signals from live AI endpoint
            async with httpx.AsyncClient(timeout=2.0) as client:
                response = await client.get(
                    self.signals_endpoint,
                    params={"limit": limit, "profile": "mixed"}
                )
                
                if response.status_code != 200:
                    logger.warning(f"[SignalService] Signals endpoint returned {response.status_code}")
                    return []
                
                data = response.json()
                
                # Handle both array and object response formats
                if isinstance(data, dict):
                    signal_list = data.get("signals", [])
                elif isinstance(data, list):
                    signal_list = data
                else:
                    logger.warning(f"[SignalService] Unexpected response format: {type(data)}")
                    return []
                
                # Map to SignalRecord
                signals = []
                for sig in signal_list[:limit]:
                    # Parse timestamp
                    ts_str = sig.get("timestamp", "")
                    try:
                        if ts_str:
                            timestamp = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                        else:
                            timestamp = datetime.now(timezone.utc)
                    except:
                        timestamp = datetime.now(timezone.utc)
                    
                    # Ensure timezone awareness
                    if timestamp.tzinfo is None:
                        timestamp = timestamp.replace(tzinfo=timezone.utc)
                    
                    # Map side to direction (normalize BUY/SELL to LONG/SHORT)
                    side = sig.get("side", sig.get("direction", "")).upper()
                    if side == "BUY":
                        direction = "LONG"
                    elif side == "SELL":
                        direction = "SHORT"
                    else:
                        direction = side
                    
                    signal = SignalRecord(
                        id=sig.get("id", f"sig_{sig.get('symbol', '')}_{int(timestamp.timestamp())}"),
                        timestamp=timestamp,
                        account=sig.get("account"),
                        symbol=sig.get("symbol", ""),
                        direction=direction,
                        confidence=float(sig.get("confidence", 0.5)),
                        strategy_id=sig.get("strategy_id"),
                        price=sig.get("price"),
                        source=sig.get("source", "AI")
                    )
                    signals.append(signal)
                
                logger.info(f"[SignalService] Retrieved {len(signals)} recent signals")
                return signals
                
        except httpx.TimeoutException:
            logger.warning("[SignalService] Timeout fetching signals")
            return []
        except Exception as e:
            logger.error(f"[SignalService] Error retrieving signals: {e}")
            return []
    
    async def get_signals_by_symbol(self, symbol: str, limit: int = 20) -> List[SignalRecord]:
        """
        Get recent signals for specific symbol.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            limit: Maximum number of signals
            
        Returns:
            List of SignalRecord objects for the symbol
        """
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                response = await client.get(
                    self.signals_endpoint,
                    params={"limit": limit, "symbol": symbol, "profile": "mixed"}
                )
                
                if response.status_code != 200:
                    return []
                
                data = response.json()
                signal_list = data.get("signals", []) if isinstance(data, dict) else data
                
                signals = []
                for sig in signal_list:
                    if sig.get("symbol") != symbol:
                        continue
                    
                    ts_str = sig.get("timestamp", "")
                    try:
                        timestamp = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                    except:
                        timestamp = datetime.now(timezone.utc)
                    
                    if timestamp.tzinfo is None:
                        timestamp = timestamp.replace(tzinfo=timezone.utc)
                    
                    side = sig.get("side", sig.get("direction", "")).upper()
                    direction = "LONG" if side == "BUY" else "SHORT" if side == "SELL" else side
                    
                    signal = SignalRecord(
                        id=sig.get("id", f"sig_{symbol}_{int(timestamp.timestamp())}"),
                        timestamp=timestamp,
                        account=sig.get("account"),
                        symbol=sig.get("symbol", ""),
                        direction=direction,
                        confidence=float(sig.get("confidence", 0.5)),
                        strategy_id=sig.get("strategy_id"),
                        price=sig.get("price"),
                        source=sig.get("source", "AI")
                    )
                    signals.append(signal)
                
                return signals[:limit]
                
        except Exception as e:
            logger.error(f"[SignalService] Error retrieving signals for {symbol}: {e}")
            return []
