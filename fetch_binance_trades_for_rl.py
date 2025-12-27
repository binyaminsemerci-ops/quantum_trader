"""
Fetch Binance Futures Trade History for RL Training

Henter alle lukkede positions fra Binance Futures og replayer dem til RL agent.

Usage:
    python fetch_binance_trades_for_rl.py
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict
import json

# Load .env file
from dotenv import load_dotenv
load_dotenv()

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from binance.client import Client
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_binance_client() -> Client:
    """Initialize Binance client with testnet or production credentials."""
    use_testnet = os.getenv("USE_BINANCE_TESTNET", "false").lower() == "true"
    
    if use_testnet:
        api_key = os.getenv("BINANCE_TESTNET_API_KEY")
        api_secret = os.getenv("BINANCE_TESTNET_SECRET_KEY")
        logger.info("🧪 Using TESTNET credentials")
    else:
        api_key = os.getenv("BINANCE_API_KEY")
        api_secret = os.getenv("BINANCE_SECRET_KEY")
        logger.info("🔴 Using PRODUCTION credentials")
    
    if not api_key or not api_secret:
        raise ValueError("❌ Binance API credentials not found in environment!")
    
    client = Client(api_key, api_secret, testnet=use_testnet)
    
    if use_testnet:
        client.API_URL = 'https://testnet.binancefuture.com'
    
    return client


def fetch_futures_trades(client: Client, days_back: int = 7) -> List[Dict]:
    """
    Fetch all futures trades from last N days.
    
    Returns list of trades with:
    - symbol, side (BUY/SELL), qty, price, commission, time
    """
    logger.info(f"📡 Fetching futures trades from last {days_back} days...")
    
    # Get all positions we've traded
    all_trades = []
    
    # Binance only allows fetching trades by symbol, so we need to check common symbols
    # You can add more symbols from your trading universe
    symbols = [
        'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT',
        'XRPUSDT', 'DOTUSDT', 'DOGEUSDT', 'AVAXUSDT', 'MATICUSDT',
        'LINKUSDT', 'UNIUSDT', 'ATOMUSDT', 'LTCUSDT', 'ETCUSDT',
        'NEARUSDT', 'APTUSDT', 'ARBUSDT', 'OPUSDT', 'SUIUSDT',
        'AAVEUSDT', 'FILUSDT', 'INJUSDT', 'RNDRUSDT', 'STXUSDT',
        # Add more from your universe
    ]
    
    # Calculate start time (N days ago)
    start_time = int((datetime.now() - timedelta(days=days_back)).timestamp() * 1000)
    
    for symbol in symbols:
        try:
            # Fetch account trades for this symbol
            trades = client.futures_account_trades(
                symbol=symbol,
                startTime=start_time,
                limit=1000  # Max 1000 per request
            )
            
            if trades:
                logger.info(f"  ✅ {symbol}: {len(trades)} trades")
                all_trades.extend(trades)
            
        except Exception as e:
            # Symbol might not exist or we have no trades
            if "Invalid symbol" not in str(e):
                logger.debug(f"  ⚠️ {symbol}: {e}")
            continue
    
    logger.info(f"✅ Total trades fetched: {len(all_trades)}")
    return all_trades


def group_trades_into_positions(trades: List[Dict]) -> List[Dict]:
    """
    Group individual BUY/SELL trades into complete positions (entry + exit).
    
    A position is considered closed when:
    - LONG: Buy opens, Sell closes
    - SHORT: Sell opens, Buy closes
    
    Returns list of closed positions with PnL calculated.
    """
    logger.info("🔄 Grouping trades into closed positions...")
    
    # Group by symbol
    by_symbol = {}
    for trade in trades:
        symbol = trade['symbol']
        if symbol not in by_symbol:
            by_symbol[symbol] = []
        by_symbol[symbol].append(trade)
    
    # Sort each symbol's trades by time
    for symbol in by_symbol:
        by_symbol[symbol].sort(key=lambda t: t['time'])
    
    # Track open positions and closed positions
    closed_positions = []
    
    for symbol, symbol_trades in by_symbol.items():
        position_qty = 0.0
        position_entry_price = 0.0
        position_entry_time = None
        position_side = None
        
        for trade in symbol_trades:
            qty = float(trade['qty'])
            price = float(trade['price'])
            side = trade['side']  # BUY or SELL
            time = trade['time']
            commission = float(trade['commission'])
            
            # Update position
            if side == 'BUY':
                position_qty += qty
            else:  # SELL
                position_qty -= qty
            
            # Check if position just opened
            if position_entry_time is None and abs(position_qty) > 0:
                position_entry_time = time
                position_entry_price = price
                position_side = "LONG" if side == "BUY" else "SHORT"
            
            # Check if position closed (qty back to 0)
            if position_entry_time is not None and abs(position_qty) < 0.001:
                # Position closed!
                exit_price = price
                exit_time = time
                duration_seconds = (exit_time - position_entry_time) / 1000
                
                # Calculate PnL
                if position_side == "LONG":
                    pnl_pct = (exit_price - position_entry_price) / position_entry_price
                else:  # SHORT
                    pnl_pct = (position_entry_price - exit_price) / position_entry_price
                
                # Estimate position size (we don't have exact notional value)
                # Assume avg position size from your config (e.g., $300)
                estimated_position_size = 300.0
                pnl_usd = estimated_position_size * pnl_pct
                
                closed_positions.append({
                    'symbol': symbol,
                    'side': position_side,
                    'entry_price': position_entry_price,
                    'exit_price': exit_price,
                    'entry_time': datetime.fromtimestamp(position_entry_time / 1000),
                    'exit_time': datetime.fromtimestamp(exit_time / 1000),
                    'duration_hours': duration_seconds / 3600,
                    'pnl_pct': pnl_pct,
                    'pnl_usd': pnl_usd,
                    'r_multiple': pnl_pct / 0.025 if pnl_pct < 0 else pnl_pct / 0.06,  # Estimate R based on TP/SL
                })
                
                # Reset position tracking
                position_qty = 0.0
                position_entry_time = None
                position_side = None
    
    logger.info(f"✅ Found {len(closed_positions)} closed positions")
    return closed_positions


def replay_to_rl(positions: List[Dict]) -> int:
    """
    Replay positions to RL agent using same logic as position_monitor.
    """
    logger.info("🎓 Replaying positions to RL agent...")
    
    from services.rl_position_sizing_agent import RLPositionSizingAgent
    
    rl_agent = RLPositionSizingAgent()
    
    successful = 0
    
    for i, pos in enumerate(positions, 1):
        try:
            # Estimate state (same heuristics as before)
            r_mult = pos['r_multiple']
            pnl = pos['pnl_usd']
            
            if r_mult > 2.0 and pnl > 50:
                state_key = "vol_low_trend_strong_mom_bullish"
            elif r_mult > 0.5 and pnl > 0:
                state_key = "vol_medium_trend_moderate_mom_bullish"
            elif r_mult < -0.5:
                state_key = "vol_high_trend_weak_mom_bearish"
            else:
                state_key = "vol_medium_trend_weak_mom_neutral"
            
            # Estimate action
            pnl_pct = pos['pnl_pct']
            if abs(pnl_pct) > 0.03:
                action_key = "size_100%_lev_5x_tp_wide"
            elif abs(pnl_pct) > 0.015:
                action_key = "size_75%_lev_5x_tp_medium"
            else:
                action_key = "size_50%_lev_5x_tp_tight"
            
            # Create outcome data
            outcome_data = {
                'symbol': pos['symbol'],
                'pnl_usd': pos['pnl_usd'],
                'pnl_pct': pos['pnl_pct'],
                'duration_seconds': pos['duration_hours'] * 3600,
                'r_multiple': pos['r_multiple'],
                'exit_reason': 'BINANCE_HISTORY',
                'state_key': state_key,
                'action_key': action_key,
            }
            
            # Update RL
            rl_agent.update_from_outcome(outcome_data)
            
            logger.info(
                f"  [{i}/{len(positions)}] {pos['symbol']}: "
                f"PnL=${pos['pnl_usd']:.2f} ({pos['pnl_pct']*100:.2f}%), "
                f"R={pos['r_multiple']:.2f}R"
            )
            
            successful += 1
            
        except Exception as e:
            logger.error(f"❌ Failed to replay position {i}: {e}")
            continue
    
    # Save state
    try:
        rl_agent._save_state()
        logger.info("💾 RL state saved")
    except Exception as e:
        logger.error(f"⚠️ Failed to save RL state: {e}")
    
    return successful


def main():
    """Main execution."""
    print("\n" + "="*70)
    print("📡 BINANCE FUTURES TRADE HISTORY → RL TRAINING")
    print("="*70 + "\n")
    
    try:
        # Step 1: Connect to Binance
        client = get_binance_client()
        logger.info("✅ Connected to Binance\n")
        
        # Step 2: Fetch trades
        trades = fetch_futures_trades(client, days_back=7)
        
        if not trades:
            print("\n❌ No trades found in last 7 days!")
            print("   Try increasing days_back or check if you have trades on Binance")
            sys.exit(1)
        
        # Step 3: Group into positions
        positions = group_trades_into_positions(trades)
        
        if not positions:
            print("\n❌ No closed positions found!")
            print("   Found trades but couldn't pair them into complete positions")
            sys.exit(1)
        
        # Step 4: Show summary
        total_pnl = sum(p['pnl_usd'] for p in positions)
        winners = len([p for p in positions if p['pnl_usd'] > 0])
        
        print(f"\n📊 BINANCE HISTORY SUMMARY:")
        print(f"   Total closed positions: {len(positions)}")
        print(f"   Winners: {winners} ({winners/len(positions)*100:.1f}%)")
        print(f"   Total PnL: ${total_pnl:.2f}")
        print(f"   Avg PnL: ${total_pnl/len(positions):.2f}")
        
        # Step 5: Ask confirmation
        print(f"\n⚠️  NOTE: State/Action estimation is heuristic-based")
        print(f"   Real future trades will have accurate state/action data")
        print(f"\n🎯 Replay these {len(positions)} positions to RL agent? (y/n): ", end='')
        
        response = input().strip().lower()
        
        if response != 'y':
            print("❌ Cancelled")
            sys.exit(0)
        
        # Step 6: Replay
        print("\n" + "-"*70)
        replayed = replay_to_rl(positions)
        print("-"*70)
        
        # Step 7: Summary
        print(f"\n✅ REPLAY COMPLETE!")
        print(f"   Positions processed: {replayed}/{len(positions)}")
        print(f"   RL agent bootstrapped with real trading data!")
        print(f"   Q-table now has learned values from Binance history")
        print(f"\n💡 RL will continue learning from NEW closed positions\n")
        
    except Exception as e:
        logger.error(f"\n❌ ERROR: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
