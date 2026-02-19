#!/usr/bin/env python3
"""
Exit Logic Tester - Test exit formulas on live position data
===========================================================

Dette scriptet:
1. ⬇️  Henter live posisjoner fra Binance Testnet
2. 🔬 Tester compute_dynamic_stop og evaluate_exit på reelle data
3. 📊 Sammenligner med exit_monitor_service_patched.py
4. 📝 Logger alle beregninger detaljert

Formål: Validere at exit-formelen fungerer korrekt.

Author: Exit Logic Testing
Date: 2026-02-18
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from binance.client import Client
from binance.exceptions import BinanceAPIException

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from common.exit_math import (
    Position, Account, Market, RiskSettings,
    compute_dynamic_stop, evaluate_exit, get_exit_metrics,
    compute_R, should_activate_trailing
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s"
)
logger = logging.getLogger(__name__)

# ============================================================================
# CREDENTIALS & CLIENT
# ============================================================================

# Load environment from .env file
from dotenv import load_dotenv
load_dotenv()

BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")

if not BINANCE_API_KEY or not BINANCE_API_SECRET:
    logger.error("❌ Missing Binance credentials")
    sys.exit(1)

# Initialize Binance Testnet client
binance_client = Client(BINANCE_API_KEY, BINANCE_API_SECRET, testnet=True)
binance_client.FUTURES_URL = "https://testnet.binancefuture.com"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_account_equity() -> float:
    """Get account total equity"""
    try:
        account_info = binance_client.futures_account()
        equity = float(account_info['totalWalletBalance'])
        logger.info(f"📊 Account equity: ${equity:.2f} USDT")
        return equity
    except Exception as e:
        logger.error(f"❌ Failed to get account equity: {e}")
        return 3785.35  # Fallback to known value

def get_current_price(symbol: str) -> float:
    """Get current market price"""
    try:
        ticker = binance_client.futures_symbol_ticker(symbol=symbol)
        price = float(ticker["price"])
        return price
    except Exception as e:
        logger.error(f"❌ Failed to get price for {symbol}: {e}")
        return 0.0

def get_atr_estimate(symbol: str) -> float:
    """
    Estimate ATR using 24h price range.
    Real implementation would use proper ATR calculation.
    """
    try:
        # Use get_24hr_ticker instead of futures_24hr_ticker
        ticker = binance_client.get_24hr_ticker(symbol=symbol)
        high_24h = float(ticker['highPrice'])
        low_24h = float(ticker['lowPrice'])
        current_price = float(ticker['lastPrice'])
        
        # Calculate percentage-based ATR estimate
        daily_range = (high_24h - low_24h)
        atr_estimate = daily_range * 0.2  # ~20% of daily range as ATR proxy
        
        # Minimum ATR relative to price (0.1% of current price)
        min_atr = current_price * 0.001
        atr_final = max(atr_estimate, min_atr)
        
        logger.debug(f"ATR for {symbol}: daily_range={daily_range:.6f}, atr_est={atr_final:.6f}")
        return atr_final
        
    except Exception as e:
        logger.error(f"❌ Failed to get ATR estimate for {symbol}: {e}")
        # Return reasonable fallback based on typical crypto volatility
        current_price = get_current_price(symbol)
        if current_price > 0:
            return current_price * 0.02  # 2% of price as fallback
        return 0.001  # Last resort fallback

def get_live_positions():
    """Fetch current open positions from Binance"""
    try:
        positions = binance_client.futures_position_information()
        
        # Filter open positions (positionAmt != 0)
        open_positions = []
        for pos in positions:
            position_amt = float(pos['positionAmt'])
            if position_amt != 0:
                open_positions.append(pos)
        
        logger.info(f"📍 Found {len(open_positions)} open positions")
        return open_positions
        
    except Exception as e:
        logger.error(f"❌ Failed to fetch positions: {e}")
        return []

# ============================================================================
# EXIT LOGIC TESTING
# ============================================================================

def test_exit_logic_on_position(binance_pos, account_equity: float):
    """Test exit logic on a single position"""
    
    symbol = binance_pos['symbol']
    position_amt = float(binance_pos['positionAmt'])
    entry_price = float(binance_pos['entryPrice'])
    mark_price = float(binance_pos['markPrice'])
    leverage = float(binance_pos['leverage'])
    
    # Handle missing unRealizedPnl field
    pnl = float(binance_pos.get('unRealizedPnl', 0))
    if pnl == 0:
        # Calculate PnL manually if missing
        if position_amt > 0:  # LONG
            pnl = (mark_price - entry_price) * abs(position_amt)
        else:  # SHORT
            pnl = (entry_price - mark_price) * abs(position_amt)
    
    # Determine side
    side = "BUY" if position_amt > 0 else "SELL"
    size = abs(position_amt)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"🧪 TESTING EXIT LOGIC: {symbol}")
    logger.info(f"{'='*60}")
    logger.info(f"📊 Position: {side} {size:.4f} @ {entry_price:.4f}")
    logger.info(f"💰 Mark Price: {mark_price:.4f} | PnL: {pnl:+.4f} USDT")
    logger.info(f"⚡ Leverage: {leverage}x")
    
    # Get current market data
    current_price = get_current_price(symbol)
    atr = get_atr_estimate(symbol)
    
    # Create data structures
    position = Position(
        symbol=symbol,
        side=side,
        entry_price=entry_price,
        size=size,
        leverage=leverage,
        highest_price=max(entry_price, current_price) if side == "BUY" else entry_price,
        lowest_price=min(entry_price, current_price) if side == "SELL" else entry_price,
        time_in_trade=3600,  # Assume 1 hour for testing
        distance_to_liq=0.05  # Assume 5% from liquidation
    )
    
    account = Account(equity=account_equity)
    market = Market(current_price=current_price, atr=atr)
    settings = RiskSettings()
    
    # Test exit math functions
    logger.info(f"\n🔬 TESTING EXIT MATH:")
    logger.info(f"Current Price: ${current_price:.4f}")
    logger.info(f"ATR Estimate: ${atr:.4f}")
    
    # Dynamic stop calculation
    dynamic_stop = compute_dynamic_stop(position, account, market, settings)
    stop_distance = abs(entry_price - dynamic_stop)
    
    logger.info(f"\n📐 DYNAMIC STOP CALCULATION:")
    logger.info(f"Risk Capital: ${account_equity * settings.RISK_FRACTION:.2f}")
    logger.info(f"Dynamic Stop: ${dynamic_stop:.4f}")
    logger.info(f"Stop Distance: ${stop_distance:.4f}")
    
    # R calculation
    current_r = compute_R(position, current_price, stop_distance)
    logger.info(f"Current R: {current_r:+.2f}R")
    
    # Trailing stop check
    trailing_active = should_activate_trailing(position, market, stop_distance, settings)
    logger.info(f"Trailing Active: {trailing_active}")
    
    # Main exit evaluation
    exit_reason = evaluate_exit(position, account, market, settings)
    
    logger.info(f"\n🚦 EXIT DECISION:")
    if exit_reason:
        logger.info(f"🔴 EXIT TRIGGERED: {exit_reason}")
    else:
        logger.info(f"🟢 HOLD POSITION")
    
    # Detailed metrics
    metrics = get_exit_metrics(position, account, market, settings)
    logger.info(f"\n📊 DETAILED METRICS:")
    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"{key}: {value:.4f}")
        else:
            logger.info(f"{key}: {value}")
    
    return {
        "symbol": symbol,
        "side": side,
        "current_price": current_price,
        "dynamic_stop": dynamic_stop,
        "current_r": current_r,
        "exit_reason": exit_reason,
        "trailing_active": trailing_active,
        "pnl": pnl
    }

def main():
    """Main testing function"""
    logger.info("🚀 STARTING EXIT LOGIC TEST")
    logger.info("="*60)
    
    # Get account equity
    account_equity = get_account_equity()
    
    # Get live positions
    positions = get_live_positions()
    
    if not positions:
        logger.warning("⚠️  No open positions found")
        return
    
    # Test exit logic on each position
    results = []
    for pos in positions[:5]:  # Limit to first 5 positions
        try:
            result = test_exit_logic_on_position(pos, account_equity)
            results.append(result)
        except Exception as e:
            logger.error(f"❌ Failed to test position {pos.get('symbol', 'UNKNOWN')}: {e}")
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"📈 EXIT LOGIC TEST SUMMARY")
    logger.info(f"{'='*60}")
    
    exit_signals = [r for r in results if r['exit_reason']]
    hold_signals = [r for r in results if not r['exit_reason']]
    
    logger.info(f"Total Positions Tested: {len(results)}")
    logger.info(f"Exit Signals: {len(exit_signals)}")
    logger.info(f"Hold Signals: {len(hold_signals)}")
    
    if exit_signals:
        logger.info(f"\n🔴 POSITIONS SHOULD EXIT:")
        for result in exit_signals:
            logger.info(f"  {result['symbol']} {result['side']} -> {result['exit_reason']}")
    
    if hold_signals:
        logger.info(f"\n🟢 POSITIONS SHOULD HOLD:")
        for result in hold_signals:
            logger.info(f"  {result['symbol']} {result['side']} -> {result['current_r']:+.2f}R")

if __name__ == "__main__":
    main()