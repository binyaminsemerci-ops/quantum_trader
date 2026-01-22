"""
Bootstrap tracked_positions from Binance open positions.

This ensures exit-monitor protects ALL open positions, not just ones it "saw" via execution.res.
Run at startup and periodically (every 60s).
"""
import asyncio
import os
from typing import Dict, Optional
from dataclasses import dataclass
from binance.client import AsyncClient
from binance import BinanceSocketManager
import logging

logger = logging.getLogger(__name__)


@dataclass
class TrackedPosition:
    """Position being tracked for exit (matches exit_monitor_service.py)"""
    symbol: str
    side: str  # "BUY" or "SELL"
    entry_price: float
    quantity: float
    leverage: float
    take_profit: Optional[float]
    stop_loss: Optional[float]
    order_id: str
    opened_at: str
    highest_price: float = 0.0
    lowest_price: float = 999999.0


async def bootstrap_from_binance(
    client: AsyncClient,
    tracked_positions: Dict[str, TrackedPosition]
) -> int:
    """
    Bootstrap tracked_positions from Binance open positions.
    
    Args:
        client: Binance AsyncClient (testnet or mainnet)
        tracked_positions: Dict to update with open positions
        
    Returns:
        Number of positions bootstrapped
    """
    try:
        # Get all open positions from Binance
        positions = await client.futures_position_information()
        
        bootstrapped = 0
        for pos in positions:
            symbol = pos['symbol']
            position_amt = float(pos['positionAmt'])
            
            # Skip positions with no size
            if abs(position_amt) == 0:
                continue
                
            entry_price = float(pos['entryPrice'])
            notional = abs(float(pos['notional']))
            leverage = int(pos['leverage'])
            
            # Determine side
            side = "BUY" if position_amt > 0 else "SELL"
            
            # Calculate TP/SL (simple fixed percentages for now)
            if side == "BUY":
                tp = entry_price * 1.025  # +2.5%
                sl = entry_price * 0.985  # -1.5%
            else:
                tp = entry_price * 0.975  # -2.5%
                sl = entry_price * 1.015  # +1.5%
            
            # Create tracked position
            tracked_pos = TrackedPosition(
                symbol=symbol,
                side=side,
                entry_price=entry_price,
                quantity=abs(position_amt),
                leverage=leverage,
                take_profit=tp,
                stop_loss=sl,
                order_id=f"BOOTSTRAP_{symbol}",  # Placeholder order ID
                opened_at="",  # Unknown timestamp
                highest_price=entry_price if side == "BUY" else 999999.0,
                lowest_price=entry_price if side == "SELL" else 0.0
            )
            
            tracked_positions[symbol] = tracked_pos
            bootstrapped += 1
            
            logger.info(
                f"🔄 BOOTSTRAP: {symbol} {side} | "
                f"Entry=${entry_price:.4f} | "
                f"Size=${notional:.2f} | "
                f"Lev={leverage}x | "
                f"TP=${tp:.4f} | SL=${sl:.4f}"
            )
        
        logger.info(f"✅ BOOTSTRAP COMPLETE: {bootstrapped} positions tracked from Binance")
        return bootstrapped
        
    except Exception as e:
        logger.error(f"❌ Bootstrap failed: {e}")
        return 0


async def run_bootstrap_loop(
    client: AsyncClient,
    tracked_positions: Dict[str, TrackedPosition],
    interval_seconds: int = 60
):
    """
    Run bootstrap periodically.
    
    Args:
        client: Binance AsyncClient
        tracked_positions: Shared dict to update
        interval_seconds: How often to re-bootstrap
    """
    while True:
        try:
            await bootstrap_from_binance(client, tracked_positions)
            await asyncio.sleep(interval_seconds)
        except Exception as e:
            logger.error(f"❌ Bootstrap loop error: {e}")
            await asyncio.sleep(30)  # Retry sooner on error
