from binance.client import Client
from datetime import datetime
from collections import defaultdict

API_KEY = "w2W60kzuCfPJKGIqSvmp0pqISUO8XKICjc5sD8QyJuJpp9LKQgvXKhtd09Ii3rwg"
API_SECRET = "QI18cg4zcbApc9uaDL8ZUmoAJQthQczZ9cKzORlSJfnK2zBEdLvSLb5ZEgZ6R1Kg"

print("=" * 80)
print("TRADE CYCLE RECONSTRUCTION - FINDING CLOSED POSITIONS")
print("=" * 80)

client = Client(API_KEY, API_SECRET, testnet=True)

# Get all positions
positions = client.futures_position_information()
symbols = [p['symbol'] for p in positions if float(p.get('notional', 0)) != 0 or float(p.get('positionAmt', 0)) != 0]

print(f"Analyzing {len(symbols)} symbols...")
print()

all_closed_positions = []
position_state = {}

for symbol in sorted(symbols):
    trades = client.futures_account_trades(symbol=symbol, limit=1000)
    
    # Track position lifecycle
    position_qty = 0.0
    entry_price_weighted = 0.0
    entry_value = 0.0
    entry_commission = 0.0
    closes = []
    
    for trade in sorted(trades, key=lambda x: x['time']):
        price = float(trade['price'])
        qty = float(trade['qty'])
        commission = float(trade['commission'])
        side = trade['side']
        
        # Convert to signed quantity
        if side == 'SELL':
            signed_qty = -qty
        else:
            signed_qty = qty
        
        prev_position = position_qty
        position_qty += signed_qty
        
        # Check if this trade closes or reduces position
        if prev_position != 0 and ((prev_position > 0 and signed_qty < 0) or (prev_position < 0 and signed_qty > 0)):
            # This is a close/reduce trade
            closed_qty = min(abs(signed_qty), abs(prev_position))
            
            if entry_value != 0:
                avg_entry = abs(entry_value / prev_position)
                
                # Calculate PnL for closed portion
                if prev_position > 0:  # Was long
                    pnl = closed_qty * (price - avg_entry)
                else:  # Was short
                    pnl = closed_qty * (avg_entry - price)
                
                # Subtract commissions
                close_commission = (commission / qty) * closed_qty
                pnl -= (entry_commission * (closed_qty / abs(prev_position)) + close_commission)
                
                if abs(signed_qty) >= abs(prev_position):
                    # Full close
                    closes.append({
                        'symbol': symbol,
                        'pnl': pnl,
                        'entry': avg_entry,
                        'exit': price,
                        'qty': closed_qty,
                        'time': trade['time']
                    })
                    all_closed_positions.append(closes[-1])
                    
                    # Reset for new position
                    entry_value = 0
                    entry_commission = 0
                    if abs(signed_qty) > abs(prev_position):
                        # Reversal - new position starts
                        remaining_qty = abs(signed_qty) - abs(prev_position)
                        entry_value = remaining_qty * price * (1 if signed_qty > 0 else -1)
                        entry_commission = (commission / qty) * remaining_qty
                else:
                    # Partial close - reduce entry value proportionally
                    entry_value *= (abs(position_qty) / abs(prev_position))
                    entry_commission *= (abs(position_qty) / abs(prev_position))
        else:
            # Adding to position or opening new
            entry_value += signed_qty * price
            entry_commission += commission
    
    if closes:
        total_pnl = sum(c['pnl'] for c in closes)
        print(f"{symbol:15} {len(closes):3} closed positions  PnL: ${total_pnl:8.2f}")
    
    # Track current state
    position_state[symbol] = position_qty

print()
print("=" * 80)
print("GROUND-TRUTH EXPECTANCY AUDIT")
print("=" * 80)

if all_closed_positions:
    pnls = [p['pnl'] for p in all_closed_positions]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    
    total_pnl = sum(pnls)
    win_rate = len(wins) / len(pnls) * 100
    avg_win = sum(wins) / len(wins) if wins else 0
    avg_loss = abs(sum(losses) / len(losses)) if losses else 0
    profit_factor = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else 999
    expectancy = total_pnl / len(pnls)
    
    print(f"\nClosed Positions:      {len(all_closed_positions)}")
    print(f"Total Realized PnL:    ${total_pnl:.2f}")
    print(f"Win Rate:              {win_rate:.1f}%")
    print(f"Wins / Losses:         {len(wins)} / {len(losses)}")
    print(f"Average Win:           ${avg_win:.2f}")
    print(f"Average Loss:          ${avg_loss:.2f}")
    print(f"Profit Factor:         {profit_factor:.2f}")
    print(f"Expectancy per Trade:  ${expectancy:.2f}")
    print()
    
    # Statistical significance
    min_required = 174
    percent_complete = (len(all_closed_positions) / min_required) * 100
    print(f"Sample Size Progress:  {len(all_closed_positions)}/{min_required} ({percent_complete:.1f}%)")
    
    if len(all_closed_positions) < min_required:
        print(f"⚠️  NOT statistically significant (need {min_required - len(all_closed_positions)} more)")
    else:
        print(f"✓ Statistically significant")
    
    print()
    
    # Verdict
    if expectancy > 0 and profit_factor > 1.5:
        verdict = "✓ STRUCTURALLY PROFITABLE"
    elif expectancy > 0 and profit_factor > 1.0:
        verdict = "⚠️  POTENTIALLY POSITIVE"
    elif expectancy < 0:
        verdict = "❌ STRUCTURALLY NEGATIVE"
    else:
        verdict = "⚠️  INCONCLUSIVE"
    
    print(f"VERDICT: {verdict}")
    print("=" * 80)
    
    # Show sample
    print("\nSample closed positions:")
    for i, pos in enumerate(sorted(all_closed_positions, key=lambda x: x['time'], reverse=True)[:15]):
        ts = datetime.fromtimestamp(pos['time']/1000).strftime("%Y-%m-%d %H:%M")
        print(f"  {i+1:2}. {pos['symbol']:15} ${pos['pnl']:8.2f}  "
              f"Entry: ${pos['entry']:8.4f}  Exit: ${pos['exit']:8.4f}  {ts}")
else:
    print("\n⚠️  NO CLOSED POSITIONS FOUND")
    print("All trades are in currently open positions!")

print("\nCurrent open positions:")
for symbol, qty in position_state.items():
    if qty != 0:
        print(f"  {symbol:15} Position: {qty:12.2f}")
