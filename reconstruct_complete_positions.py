#!/usr/bin/env python3
"""
COMPLETE Position Reconstruction from Full Trade History
Uses complete_trade_history.json (2593 trades)
"""

import json
from datetime import datetime
from collections import defaultdict
import statistics

print("="*80)
print("LOADING COMPLETE TRADE HISTORY...")
print("="*80)

with open("complete_trade_history.json", "r") as f:
    all_data = json.load(f)

total_trades = sum(len(trades) for trades in all_data.values())
print(f"Loaded {total_trades} trades across {len(all_data)} symbols\n")

# Reconstruct closed positions per symbol
all_closed_positions = []
symbol_stats = {}

for symbol, trades in sorted(all_data.items()):
    if not trades:
        continue
    
    print(f"\n{symbol}:")
    print(f"  Raw trades: {len(trades)}")
    
    # Sort chronologically
    trades.sort(key=lambda x: x["time"])
    
    closed_positions = []
    position_qty = 0.0
    entry_price_sum = 0.0
    entry_qty = 0.0
    entry_commissions = 0.0
    
    for trade in trades:
        qty = float(trade["qty"])
        price = float(trade["price"])
        commission = float(trade["commission"])
        is_buyer = trade["buyer"]
        
        # Determine trade direction
        trade_qty = qty if is_buyer else -qty
        
        # Check if this closes or reduces a position
        if position_qty != 0 and ((position_qty > 0 and trade_qty < 0) or (position_qty < 0 and trade_qty > 0)):
            # Partial or full close
            close_qty = min(abs(trade_qty), abs(position_qty))
            
            # Calculate average entry price
            avg_entry = entry_price_sum / entry_qty if entry_qty > 0 else 0
            
            # Calculate PnL
            if position_qty > 0:  # Long position
                pnl = (price - avg_entry) * close_qty
            else:  # Short position
                pnl = (avg_entry - price) * close_qty
            
            # Deduct commissions proportionally
            entry_comm_portion = (close_qty / entry_qty) * entry_commissions if entry_qty > 0 else 0
            exit_comm = commission
            total_comm = entry_comm_portion + exit_comm
            pnl_net = pnl - total_comm
            
            closed_positions.append({
                "symbol": symbol,
                "side": "LONG" if position_qty > 0 else "SHORT",
                "entry_price": avg_entry,
                "exit_price": price,
                "qty": close_qty,
                "pnl_gross": pnl,
                "pnl_net": pnl_net,
                "commissions": total_comm,
                "exit_time": trade["time"]
            })
            
            # Update position
            if abs(trade_qty) >= abs(position_qty):
                # Full close or reversal
                remaining = abs(trade_qty) - abs(position_qty)
                if remaining > 0.001:
                    # Reversal - new position in opposite direction
                    position_qty = remaining if trade_qty > 0 else -remaining
                    entry_price_sum = price * remaining
                    entry_qty = remaining
                    entry_commissions = commission
                else:
                    # Exact close
                    position_qty = 0
                    entry_price_sum = 0
                    entry_qty = 0
                    entry_commissions = 0
            else:
                # Partial close
                position_qty += trade_qty
                entry_price_sum -= (close_qty / entry_qty) * entry_price_sum
                entry_qty -= close_qty
                entry_commissions -= entry_comm_portion
        
        else:
            # Adding to position or opening new
            position_qty += trade_qty
            entry_price_sum += price * qty
            entry_qty += qty
            entry_commissions += commission
    
    # Statistics for this symbol
    if closed_positions:
        pnls = [p["pnl_net"] for p in closed_positions]
        winners = [p for p in closed_positions if p["pnl_net"] > 0]
        losers = [p for p in closed_positions if p["pnl_net"] <= 0]
        
        win_rate = len(winners) / len(closed_positions) * 100 if closed_positions else 0
        total_pnl = sum(pnls)
        avg_win = statistics.mean([w["pnl_net"] for w in winners]) if winners else 0
        avg_loss = statistics.mean([l["pnl_net"] for l in losers]) if losers else 0
        expectancy = statistics.mean(pnls) if pnls else 0
        
        symbol_stats[symbol] = {
            "closed_positions": len(closed_positions),
            "winners": len(winners),
            "losers": len(losers),
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "expectancy": expectancy
        }
        
        print(f"  Closed positions: {len(closed_positions)}")
        print(f"  Win rate: {win_rate:.1f}% ({len(winners)}W / {len(losers)}L)")
        print(f"  Total PnL: ${total_pnl:+.2f}")
        print(f"  Expectancy: ${expectancy:+.4f}")
        
        all_closed_positions.extend(closed_positions)

# OVERALL STATISTICS
print("\n" + "="*80)
print("COMPLETE AUDIT RESULTS")
print("="*80)

if all_closed_positions:
    all_pnls = [p["pnl_net"] for p in all_closed_positions]
    all_winners = [p for p in all_closed_positions if p["pnl_net"] > 0]
    all_losers = [p for p in all_closed_positions if p["pnl_net"] <= 0]
    
    total_pnl = sum(all_pnls)
    win_rate = len(all_winners) / len(all_closed_positions) * 100
    
    avg_win = statistics.mean([w["pnl_net"] for w in all_winners]) if all_winners else 0
    avg_loss = statistics.mean([l["pnl_net"] for l in all_losers]) if all_losers else 0
    
    expectancy = statistics.mean(all_pnls)
    
    # Profit factor
    total_wins = sum([w["pnl_net"] for w in all_winners])
    total_losses = abs(sum([l["pnl_net"] for l in all_losers]))
    profit_factor = total_wins / total_losses if total_losses > 0 else 0
    
    print(f"\nSample Size: {len(all_closed_positions)} closed positions")
    print(f"Data Source: {total_trades} trades from Binance Futures Testnet API")
    print(f"")
    print(f"Win Rate: {win_rate:.2f}% ({len(all_winners)} wins / {len(all_losers)} losses)")
    print(f"")
    print(f"Total Realized PnL: ${total_pnl:+.2f} USDT")
    print(f"Average Win: ${avg_win:+.2f}")
    print(f"Average Loss: ${avg_loss:+.2f}")
    print(f"")
    print(f"Expectancy: ${expectancy:+.4f} per trade")
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"")
    
    # Statistical significance
    if len(all_closed_positions) >= 30:
        stderr = statistics.stdev(all_pnls) / (len(all_closed_positions) ** 0.5)
        ci_95 = 1.96 * stderr
        print(f"95% Confidence Interval: ${expectancy - ci_95:.4f} to ${expectancy + ci_95:.4f}")
        print(f"Statistical Validity: ✅ CONFIRMED (n={len(all_closed_positions)} > 30)")
    
    print("\n" + "="*80)
    
    # Verdict
    if expectancy < -0.01:
        print("VERDICT: ❌ STRUCTURALLY NEGATIVE (Do not deploy)")
        
        # Bankruptcy projection
        balance = 3878.54  # From account info
        if expectancy < 0:
            trades_to_zero = abs(balance / expectancy)
            print(f"Projected bankruptcy in ~{int(trades_to_zero)} trades")
    elif expectancy > 0.01:
        print("VERDICT: ✅ STRUCTURALLY POSITIVE (Potential for live deployment)")
    else:
        print("VERDICT: ⚠️ BREAK-EVEN (Insufficient edge)")
    
    print("="*80)
    
    # Top losers
    print("\nWORST PERFORMING SYMBOLS:")
    worst = sorted(symbol_stats.items(), key=lambda x: x[1]["total_pnl"])[:5]
    for symbol, stats in worst:
        print(f"  {symbol:12} ${stats['total_pnl']:+8.2f} | {stats['win_rate']:5.1f}% WR | {stats['closed_positions']:3} closes")
    
    # Save to JSON
    output = {
        "summary": {
            "total_trades": total_trades,
            "closed_positions": len(all_closed_positions),
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "expectancy": expectancy,
            "profit_factor": profit_factor,
            "avg_win": avg_win,
            "avg_loss": avg_loss
        },
        "per_symbol": symbol_stats,
        "all_positions": all_closed_positions
    }
    
    with open("COMPLETE_AUDIT_RESULTS.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print("\n✓ Saved detailed results to COMPLETE_AUDIT_RESULTS.json")

else:
    print("\n⚠️ No closed positions found")
