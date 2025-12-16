"""
RL Replay Training - Bootstrap RL Agent from Historical Logs

This script parses closed position logs and replays them to the RL agent
to bootstrap learning from historical trades.

Usage:
    docker exec quantum_backend python /app/scripts/rl_replay_training.py
"""

import re
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.rl_position_sizing_agent import RLPositionSizingAgent
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_trade_logs_from_docker() -> List[Dict]:
    """
    Parse closed trade logs from current container logs.
    
    Returns:
        List of dicts with trade info: {symbol, pnl, duration, entry_price, exit_price}
    """
    import subprocess
    
    logger.info("📖 Reading Docker logs for closed positions...")
    
    try:
        # Get logs from current container
        result = subprocess.run(
            ["docker", "logs", "quantum_backend"],
            capture_output=True,
            text=True,
            timeout=30
        )
        logs = result.stdout + result.stderr
    except Exception as e:
        logger.error(f"❌ Failed to read Docker logs: {e}")
        return []
    
    # Parse MEMO CLOSE LOG entries
    trades = []
    lines = logs.split('\n')
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Look for MEMO CLOSE LOG marker
        if '[MEMO] CLOSE LOG:' in line:
            trade_data = {}
            
            # Parse next 10 lines for trade details
            for j in range(i, min(i + 15, len(lines))):
                log_line = lines[j]
                
                # Extract symbol
                symbol_match = re.search(r'Symbol:\s+(\w+)\s+(LONG|SHORT)', log_line)
                if symbol_match:
                    trade_data['symbol'] = symbol_match.group(1)
                    trade_data['action'] = symbol_match.group(2)
                
                # Extract PnL
                pnl_match = re.search(r'PnL:\s+\$([+-]?\d+\.?\d*)\s+\(([+-]?\d+\.?\d*)%\)', log_line)
                if pnl_match:
                    trade_data['pnl_usd'] = float(pnl_match.group(1))
                    trade_data['pnl_pct'] = float(pnl_match.group(2)) / 100
                
                # Extract duration
                duration_match = re.search(r'Duration:\s+(\d+\.?\d*)h', log_line)
                if duration_match:
                    trade_data['duration_hours'] = float(duration_match.group(1))
                
                # Extract R-multiple
                r_match = re.search(r'R-multiple:\s+([+-]?\d+\.?\d*)R', log_line)
                if r_match:
                    trade_data['r_multiple'] = float(r_match.group(1))
                
                # Extract exit reason
                exit_match = re.search(r'Exit Reason:\s+(.+)$', log_line)
                if exit_match:
                    trade_data['exit_reason'] = exit_match.group(1).strip()
            
            # Add trade if we got minimum data
            if 'symbol' in trade_data and 'pnl_usd' in trade_data:
                trades.append(trade_data)
            
            i += 15  # Skip ahead
        else:
            i += 1
    
    logger.info(f"✅ Found {len(trades)} closed trades in logs")
    return trades


def estimate_state_from_trade(trade: Dict) -> str:
    """
    Estimate market state from trade data.
    
    Since we don't have actual market state saved, we make educated guesses:
    - High R-multiple + WIN = good conditions (low vol, strong trend)
    - Low/negative R-multiple = bad conditions (high vol, weak trend)
    - Duration can hint at volatility
    """
    r_mult = trade.get('r_multiple', 0)
    duration = trade.get('duration_hours', 0)
    pnl = trade.get('pnl_usd', 0)
    
    # Heuristic state estimation
    if r_mult > 2.0 and pnl > 50:
        # Strong winner = probably low vol + strong trend
        volatility = "low"
        trend = "strong"
    elif r_mult > 0.5 and pnl > 0:
        # Small winner = moderate conditions
        volatility = "medium"
        trend = "moderate"
    elif r_mult < -0.5:
        # Big loser = probably high vol or wrong trend
        volatility = "high"
        trend = "weak"
    else:
        # Breakeven/small loss = ranging
        volatility = "medium"
        trend = "weak"
    
    # Duration hints
    if duration < 1.0:
        volatility = "high"  # Quick exit = volatility
    elif duration > 8.0:
        volatility = "low"   # Long hold = stable
    
    momentum = "bullish" if pnl > 0 else "bearish"
    
    return f"vol_{volatility}_trend_{trend}_mom_{momentum}"


def estimate_action_from_trade(trade: Dict) -> str:
    """
    Estimate which action was taken based on trade outcome.
    
    We don't know the actual action, but we can make educated guesses:
    - Big win = probably full size with good TP
    - Small win = probably reduced size or tight TP
    - Big loss = probably full size hit SL
    """
    r_mult = trade.get('r_multiple', 0)
    pnl_pct = trade.get('pnl_pct', 0)
    
    # Estimate position size
    if abs(pnl_pct) > 0.03:  # >3% move
        size = "100%"
    elif abs(pnl_pct) > 0.015:
        size = "75%"
    else:
        size = "50%"
    
    # Estimate leverage (assuming 5x default)
    leverage = "5x"
    
    # Estimate TP/SL strategy
    if r_mult > 2.0:
        strategy = "wide"  # Wide TP worked
    elif 0.5 < r_mult < 2.0:
        strategy = "medium"
    else:
        strategy = "tight"
    
    return f"size_{size}_lev_{leverage}_tp_{strategy}"


def replay_trades_to_rl(trades: List[Dict]) -> int:
    """
    Replay historical trades to RL agent for learning.
    
    Returns:
        Number of trades successfully replayed
    """
    logger.info("🎓 Initializing RL Agent for replay training...")
    
    # Initialize RL agent
    rl_agent = RLPositionSizingAgent()
    
    logger.info(f"📚 Replaying {len(trades)} trades for learning...")
    
    successful_replays = 0
    
    for i, trade in enumerate(trades, 1):
        try:
            # Estimate state and action
            state_key = estimate_state_from_trade(trade)
            action_key = estimate_action_from_trade(trade)
            
            # Get reward (PnL in USD)
            reward = trade.get('pnl_usd', 0)
            
            # Create next_state (simplified - assume same state)
            next_state_key = state_key
            
            # Update RL agent
            logger.info(
                f"  [{i}/{len(trades)}] {trade['symbol']}: "
                f"PnL=${reward:.2f}, R={trade.get('r_multiple', 0):.2f}R, "
                f"State={state_key[:30]}..., Action={action_key}"
            )
            
            # Call update_from_outcome (same method Position Monitor uses)
            outcome_data = {
                'symbol': trade['symbol'],
                'pnl_usd': reward,
                'pnl_pct': trade.get('pnl_pct', 0),
                'duration_seconds': trade.get('duration_hours', 0) * 3600,
                'r_multiple': trade.get('r_multiple', 0),
                'exit_reason': trade.get('exit_reason', 'REPLAY'),
                'state_key': state_key,
                'action_key': action_key,
            }
            
            rl_agent.update_from_outcome(outcome_data)
            successful_replays += 1
            
        except Exception as e:
            logger.error(f"❌ Failed to replay trade {i}: {e}")
            continue
    
    logger.info(f"✅ Successfully replayed {successful_replays}/{len(trades)} trades")
    
    # Save updated Q-table
    try:
        rl_agent._save_state()
        logger.info("💾 RL state saved to disk")
    except Exception as e:
        logger.error(f"⚠️ Failed to save RL state: {e}")
    
    return successful_replays


def main():
    """Main execution."""
    print("\n" + "="*70)
    print("🎓 RL REPLAY TRAINING - Bootstrap from Historical Logs")
    print("="*70 + "\n")
    
    # Step 1: Parse logs
    trades = parse_trade_logs_from_docker()
    
    if not trades:
        print("❌ No closed trades found in logs!")
        print("\nPossible reasons:")
        print("  1. Backend recently restarted (logs cleared)")
        print("  2. No positions have closed yet")
        print("  3. Position Monitor not detecting closes")
        sys.exit(1)
    
    # Step 2: Show summary
    total_pnl = sum(t.get('pnl_usd', 0) for t in trades)
    winners = len([t for t in trades if t.get('pnl_usd', 0) > 0])
    
    print(f"\n📊 HISTORICAL TRADES SUMMARY:")
    print(f"   Total trades: {len(trades)}")
    print(f"   Winners: {winners} ({winners/len(trades)*100:.1f}%)")
    print(f"   Total PnL: ${total_pnl:.2f}")
    print(f"   Avg PnL: ${total_pnl/len(trades):.2f}")
    
    # Step 3: Ask for confirmation
    print(f"\n⚠️  NOTE: State/Action estimation is heuristic-based")
    print(f"   Real future trades will have accurate state/action data")
    print(f"\n🎯 Replay these {len(trades)} trades to RL agent? (y/n): ", end='')
    
    response = input().strip().lower()
    
    if response != 'y':
        print("❌ Cancelled by user")
        sys.exit(0)
    
    # Step 4: Replay
    print("\n" + "-"*70)
    replayed = replay_trades_to_rl(trades)
    print("-"*70)
    
    # Step 5: Summary
    print(f"\n✅ REPLAY COMPLETE!")
    print(f"   Trades processed: {replayed}/{len(trades)}")
    print(f"   RL agent now has bootstrapped learning")
    print(f"   Q-table is no longer empty!")
    print(f"\n💡 TIP: RL will continue learning from NEW closed positions")
    print(f"   Expected: Much faster convergence with bootstrap data\n")


if __name__ == "__main__":
    main()
