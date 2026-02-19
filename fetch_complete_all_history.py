from binance.client import Client
from datetime import datetime, timedelta
import json
import time

API_KEY = "w2W60kzuCfPJKGIqSvmp0pqISUO8XKICjc5sD8QyJuJpp9LKQgvXKhtd09Ii3rwg"
API_SECRET = "QI18cg4zcbApc9uaDL8ZUmoAJQthQczZ9cKzORlSJfnK2zBEdLvSLb5ZEgZ6R1Kg"

client = Client(API_KEY, API_SECRET, testnet=True)

# Get all trading symbols
positions = client.futures_position_information()
symbols = sorted(set([p["symbol"] for p in positions if float(p.get("positionAmt", 0)) != 0 or float(p.get("unrealizedProfit", 0)) != 0]))

print(f"Fetching COMPLETE history for {len(symbols)} symbols...")
print("="*80)

all_data = {}
total_trades = 0

for symbol in symbols:
    print(f"\n{symbol}:")
    
    all_trades_for_symbol = []
    oldest_time = datetime.now()
    iterations = 0
    max_iterations = 50
    
    while iterations < max_iterations:
        iterations += 1
        
        # Fetch batch ending before current oldest
        end_ms = int((oldest_time - timedelta(seconds=1)).timestamp() * 1000)
        
        try:
            batch = client.futures_account_trades(
                symbol=symbol,
                endTime=end_ms,
                limit=1000
            )
        except Exception as e:
            print(f"  Error iteration {iterations}: {e}")
            break
        
        if not batch:
            break
        
        # Sort batch chronologically
        batch.sort(key=lambda x: x["time"])
        
        batch_first = datetime.fromtimestamp(batch[0]["time"]/1000)
        batch_last = datetime.fromtimestamp(batch[-1]["time"]/1000)
        
        first_str = batch_first.strftime('%m-%d %H:%M')
        last_str = batch_last.strftime('%m-%d %H:%M')
        
        print(f"  Iter {iterations:2}: {len(batch):4} trades | {first_str} -> {last_str}")
        
        all_trades_for_symbol.extend(batch)
        oldest_time = batch_first
        
        if len(batch) < 1000:
            break
        
        time.sleep(0.05)
    
    # Remove duplicates and sort
    unique_trades = {t["id"]: t for t in all_trades_for_symbol}
    sorted_trades = sorted(unique_trades.values(), key=lambda x: x["time"])
    
    all_data[symbol] = sorted_trades
    total_trades += len(sorted_trades)
    
    if sorted_trades:
        first = datetime.fromtimestamp(sorted_trades[0]["time"]/1000)
        last = datetime.fromtimestamp(sorted_trades[-1]["time"]/1000)
        duration = last - first
        
        first_str = first.strftime('%m-%d %H:%M')
        last_str = last.strftime('%m-%d %H:%M')
        dur_str = f"{duration.days}d {duration.seconds//3600}h"
        
        print(f"  Total: {len(sorted_trades)} trades | {first_str} -> {last_str} ({dur_str})")

print("\n" + "="*80)
print(f"GRAND TOTAL: {total_trades} trades across {len(all_data)} symbols")
print("="*80)

# Save to file
with open("complete_trade_history.json", "w") as f:
    json.dump(all_data, f, indent=2)

print("\nSaved to complete_trade_history.json")

# Summary per symbol
print("\nPer-symbol summary:")
for symbol in sorted(all_data.keys(), key=lambda s: len(all_data[s]), reverse=True):
    count = len(all_data[symbol])
    print(f"  {symbol:12} {count:5} trades")
