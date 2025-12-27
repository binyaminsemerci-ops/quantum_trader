"""Market data helpers for risk management calculations."""

import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


def calculate_atr(bars: List[Dict], period: int = 14) -> float:
    """
    Calculate Average True Range from OHLCV bars.
    
    Args:
        bars: List of OHLCV candles with 'high', 'low', 'close' keys
        period: ATR period (default 14)
    
    Returns:
        ATR value or 0.0 if insufficient data
    """
    if len(bars) < period + 1:
        return 0.0
    
    tr_values = []
    for i in range(1, len(bars)):
        high = float(bars[i].get('high', 0))
        low = float(bars[i].get('low', 0))
        prev_close = float(bars[i-1].get('close', 0))
        
        high_low = high - low
        high_close = abs(high - prev_close)
        low_close = abs(low - prev_close)
        
        tr = max(high_low, high_close, low_close)
        tr_values.append(tr)
    
    if len(tr_values) < period:
        return 0.0
    
    atr = sum(tr_values[-period:]) / period
    return atr


def calculate_ema(prices: List[float], period: int = 200) -> float:
    """
    Calculate Exponential Moving Average.
    
    Args:
        prices: List of prices (oldest to newest)
        period: EMA period (default 200)
    
    Returns:
        EMA value or 0.0 if insufficient data
    """
    if len(prices) < period:
        return 0.0
    
    multiplier = 2 / (period + 1)
    ema = prices[0]
    
    for price in prices[1:]:
        ema = (price * multiplier) + (ema * (1 - multiplier))
    
    return ema


def get_consensus_type(model_votes: Dict[str, str]) -> str:
    """
    Calculate consensus type from model votes.
    
    Args:
        model_votes: {model_name: action} e.g., {"XGBoost": "LONG", "LightGBM": "LONG", ...}
    
    Returns:
        "UNANIMOUS", "STRONG", "WEAK", or "SPLIT"
    """
    if not model_votes:
        return "WEAK"
    
    actions = list(model_votes.values())
    long_votes = actions.count("LONG") + actions.count("BUY")
    short_votes = actions.count("SHORT") + actions.count("SELL")
    hold_votes = actions.count("HOLD")
    
    total_votes = len(actions)
    max_votes = max(long_votes, short_votes, hold_votes)
    
    # All models agree
    if max_votes == total_votes:
        return "UNANIMOUS"
    
    # 3 out of 4 agree
    if max_votes >= 3 and total_votes == 4:
        return "STRONG"
    
    # Check for 2-2 split
    if total_votes == 4 and max_votes == 2:
        high_vote_actions = sum(1 for count in [long_votes, short_votes, hold_votes] if count >= 2)
        if high_vote_actions == 2:
            return "SPLIT"
    
    return "WEAK"


def normalize_action(action: str) -> str:
    """
    Normalize action to LONG/SHORT/HOLD.
    
    Args:
        action: "BUY", "SELL", "LONG", "SHORT", or "HOLD"
    
    Returns:
        "LONG", "SHORT", or "HOLD"
    """
    action = action.upper()
    if action in ("BUY", "LONG"):
        return "LONG"
    elif action in ("SELL", "SHORT"):
        return "SHORT"
    else:
        return "HOLD"


async def fetch_market_conditions(symbol: str, adapter, lookback: int = 200):
    """
    Fetch market conditions for risk management.
    
    Args:
        symbol: Trading pair
        adapter: Execution adapter with API access
        lookback: Number of candles to fetch
    
    Returns:
        Dict with price, atr, ema_200, volume_24h, spread_bps
    """
    try:
        # Import here to avoid circular dependency
        from backend.routes.external_data import binance_ohlcv
        
        # Get OHLCV data
        ohlcv_data = await binance_ohlcv(symbol=symbol, limit=lookback)
        candles = ohlcv_data.get("candles", [])
        
        if not candles or len(candles) < 2:
            logger.warning(f"Insufficient OHLCV data for {symbol}")
            return None
        
        # Current price (last close)
        current_price = float(candles[-1].get('close', 0))
        
        # Calculate ATR (14 periods)
        atr = calculate_atr(candles, period=14)
        
        # Calculate EMA 200
        closes = [float(c.get('close', 0)) for c in candles]
        ema_200 = calculate_ema(closes, period=200) if len(closes) >= 200 else closes[0]
        
        # Get 24h volume
        volume_24h = float(candles[-1].get('volume_usdt', 0))
        
        # Get spread from ticker (fallback to default if not available)
        spread_bps = 10  # Default spread
        
        try:
            if hasattr(adapter, 'get_ticker'):
                ticker = await adapter.get_ticker(symbol)
                if ticker and 'bid' in ticker and 'ask' in ticker:
                    best_bid = float(ticker['bid'])
                    best_ask = float(ticker['ask'])
                    spread = (best_ask - best_bid) / best_bid if best_bid > 0 else 0
                    spread_bps = int(spread * 10000)
        except Exception as ticker_err:
            logger.debug(f"Could not get ticker for {symbol}, using default spread: {ticker_err}")
        
        return {
            'price': current_price,
            'atr': atr,
            'ema_200': ema_200,
            'volume_24h': volume_24h,
            'spread_bps': spread_bps
        }
    
    except Exception as e:
        logger.error(f"Error fetching market conditions for {symbol}: {e}", exc_info=True)
        return None
