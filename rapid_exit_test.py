#!/usr/bin/env python3
"""
Rapid Exit Logic Test - Test exit formulas with simulated data
============================================================

Dette scriptet tester exit-formelen direkte uten API-kall,
med simulerte markedsdata som matcher de reelle posisjonene
vi observerte tidligere.

Author: Exit Logic Testing
Date: 2026-02-18
"""

import sys
from pathlib import Path
import logging
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from common.exit_math import (
    Position, Account, Market, RiskSettings,
    compute_dynamic_stop, evaluate_exit, get_exit_metrics,
    compute_R, should_activate_trailing, compute_trailing_hit
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s"
)
logger = logging.getLogger(__name__)

# ============================================================================
# SIMULATED MARKET DATA
# ============================================================================

SIMULATED_POSITIONS = [
    {
        "symbol": "ALTUSDT",
        "side": "SELL",
        "size": 68807.0,
        "entry_price": 0.0087,
        "current_price": 0.0087,
        "leverage": 30.0,
        "pnl": 2.6656,
        "atr_pct": 0.15  # 15% volatility
    },
    {
        "symbol": "ALICEUSDT", 
        "side": "SELL",
        "size": 5309.7,
        "entry_price": 0.1110,
        "current_price": 0.1125,
        "leverage": 20.0,
        "pnl": -7.8766,
        "atr_pct": 0.08  # 8% volatility
    },
    {
        "symbol": "AEVOUSDT",
        "side": "BUY",
        "size": 3472.7,
        "entry_price": 0.0288,
        "current_price": 0.0285,
        "leverage": 20.0,
        "pnl": -0.9029,
        "atr_pct": 0.12  # 12% volatility
    },
    {
        "symbol": "BTCUSDT",
        "side": "SELL",
        "size": 0.5,
        "entry_price": 45000.0,
        "current_price": 44950.0,
        "leverage": 10.0,
        "pnl": 25.0,
        "atr_pct": 0.035  # 3.5% volatility
    }
]

ACCOUNT_EQUITY = 3785.31  # From earlier test

# ============================================================================
# TESTING FUNCTIONS
# ============================================================================

def test_position(pos_data):
    """Test exit logic on simulated position"""
    
    symbol = pos_data["symbol"]
    side = pos_data["side"]
    entry_price = pos_data["entry_price"]
    current_price = pos_data["current_price"]
    leverage = pos_data["leverage"]
    size = pos_data["size"]
    atr_pct = pos_data["atr_pct"]
    
    # Calculate ATR based on percentage of price
    atr = current_price * atr_pct
    
    logger.info(f"\n{'='*60}")
    logger.info(f"🧪 TESTING: {symbol} ({side})")
    logger.info(f"{'='*60}")
    logger.info(f"📊 Size: {size:.2f} | Entry: ${entry_price:.4f}")
    logger.info(f"💰 Current: ${current_price:.4f} | Leverage: {leverage}x")
    logger.info(f"📈 ATR: ${atr:.6f} ({atr_pct*100:.1f}% of price)")
    
    # Create data structures
    position = Position(
        symbol=symbol,
        side=side,
        entry_price=entry_price,
        size=size,
        leverage=leverage,
        highest_price=max(entry_price, current_price) if side == "BUY" else entry_price,
        lowest_price=min(entry_price, current_price) if side == "SELL" else entry_price,
        time_in_trade=1800,  # 30 minutes
        distance_to_liq=0.15  # 15% from liquidation
    )
    
    account = Account(equity=ACCOUNT_EQUITY)
    market = Market(current_price=current_price, atr=atr)
    settings = RiskSettings(
        RISK_FRACTION=0.005,  # 0.5% risk per trade
        STOP_ATR_MULT=1.2,
        TRAILING_ATR_MULT=1.5,
        TRAILING_ACTIVATION_R=1.0,
        MAX_HOLD_TIME=3600,
        LIQ_BUFFER_PCT=0.05
    )
    
    # Test calculations
    logger.info(f"\n🔬 RISK CALCULATIONS:")
    risk_capital = account.equity * settings.RISK_FRACTION
    logger.info(f"Risk Capital: ${risk_capital:.2f} ({settings.RISK_FRACTION*100:.1f}% of ${account.equity:.2f})")
    
    # Dynamic stop
    dynamic_stop = compute_dynamic_stop(position, account, market, settings)
    stop_distance = abs(entry_price - dynamic_stop)
    
    logger.info(f"\n📐 DYNAMIC STOP:")
    logger.info(f"Dynamic Stop Price: ${dynamic_stop:.6f}")
    logger.info(f"Stop Distance: ${stop_distance:.6f}")
    logger.info(f"Stop Distance %: {(stop_distance/entry_price)*100:.2f}%")
    
    # R calculation
    current_r = compute_R(position, current_price, stop_distance)
    logger.info(f"Current R-Multiple: {current_r:+.3f}R")
    
    # Trailing stop
    trailing_active = should_activate_trailing(position, market, stop_distance, settings)
    logger.info(f"Trailing Active: {trailing_active}")
    
    if trailing_active:
        trailing_hit = compute_trailing_hit(position, market, settings)
        logger.info(f"Trailing Hit: {trailing_hit}")
    
    # Main exit evaluation
    exit_reason = evaluate_exit(position, account, market, settings)
    
    logger.info(f"\n🚦 EXIT DECISION:")
    if exit_reason:
        logger.info(f"🔴 EXIT: {exit_reason}")
    else:
        logger.info(f"🟢 HOLD")
    
    # Test different price scenarios
    logger.info(f"\n🎯 SCENARIO TESTING:")
    
    # Test stop loss trigger
    if side == "BUY":
        test_price = dynamic_stop - 0.0001  # Slightly below stop
    else:
        test_price = dynamic_stop + 0.0001  # Slightly above stop
        
    test_market = Market(current_price=test_price, atr=atr)
    exit_at_stop = evaluate_exit(position, account, test_market, settings)
    logger.info(f"At ${test_price:.6f} -> {exit_at_stop or 'HOLD'}")
    
    return {
        "symbol": symbol,
        "side": side,
        "dynamic_stop": dynamic_stop,
        "current_r": current_r,
        "exit_reason": exit_reason,
        "trailing_active": trailing_active,
        "stop_distance_pct": (stop_distance/entry_price)*100
    }

def main():
    """Run comprehensive exit logic test"""
    logger.info("🚀 RAPID EXIT LOGIC TEST")
    logger.info("="*60)
    
    results = []
    for pos_data in SIMULATED_POSITIONS:
        try:
            result = test_position(pos_data)
            results.append(result)
        except Exception as e:
            logger.error(f"❌ Error testing {pos_data['symbol']}: {e}")
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"📈 SUMMARY")
    logger.info(f"{'='*60}")
    
    for result in results:
        action = "🔴 EXIT" if result['exit_reason'] else "🟢 HOLD"
        logger.info(f"{result['symbol']:10} {result['side']:4} | {action:8} | "
                   f"R={result['current_r']:+.3f} | "
                   f"Stop={result['stop_distance_pct']:.2f}%")
    
    exit_count = len([r for r in results if r['exit_reason']])
    hold_count = len([r for r in results if not r['exit_reason']])
    
    logger.info(f"\nTotal: {len(results)} | Exit: {exit_count} | Hold: {hold_count}")
    
    # Test with more volatile market
    logger.info(f"\n🌪️ HIGH VOLATILITY TEST:")
    logger.info(f"Testing with 2x ATR to see sensitivity...")
    
    high_vol_pos = SIMULATED_POSITIONS[0].copy()
    high_vol_pos["atr_pct"] = 0.30  # 30% volatility
    high_vol_pos["symbol"] = "ALTUSDT_HIGHVOL"
    
    test_position(high_vol_pos)

if __name__ == "__main__":
    main()