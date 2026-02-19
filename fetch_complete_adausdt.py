from binance.client import Client
from datetime import datetime
import time

API_KEY = "w2W60kzuCfPJKGIqSvmp0pqISUO8XKICjc5sD8QyJuJpp9LKQgvXKhtd09Ii3rwg"
API_SECRET = "QI18cg4zcbApc9uaDL8ZUmoAJQthQczZ9cKzORlSJfnK2zBEdLvSLb5ZEgZ6R1Kg"

print("=" * 80)
print("FETCHING COMPLETE ADAUSDT HISTORY (PAGINATED)")
print("=" * 80)

client = Client(API_KEY, API_SECRET, testnet=True)

# Fetch ALL ADAUSDT trades by paginating backwards
symbol = "ADAUSDT"
all_trades = []
batch_size = 1000
oldest_id = None

print(f"\nFetching {symbol} trades (this may take a minute)...\n")

while True:
    params = {'symbol': symbol, 'limit': batch_size}
    if oldest_id:
        params['fromId'] = oldest_id
    
    batch = client.futures_account_trades(**params)
    
    if not batch:
        break
    
    all_trades.extend(batch)
    oldest_id = batch[-1]['id']
    
    oldest_time = datetime.fromtimestamp(batch[-1]['time']/1000).strftime("%Y-%m-%d %H:%M")
    print(f"  Fetched {len(batch)} trades, oldest: {oldest_time}, total so far: {len(all_trades)}")
    
    # If we got less than requested, we're at the end
    if len(batch) < batch_size:
        break
    
    # Check if last trade in this batch has same ID as first in previous
    # (Binance sometimes returns duplicates at boundaries)
    if len(all_trades) > batch_size and all_trades[-1]['id'] == all_trades[-batch_size-1]['id']:
        break
    
    time.sleep(0.2)  # Rate limit protection

# Remove duplicates by ID
unique_trades = {t['id']: t for t in all_trades}
all_trades = sorted(unique_trades.values(), key=lambda x: x['time'])

print(f"\n{'=' * 80}")
print(f"COMPLETE {symbol} HISTORY")
print(f"{'=' * 80}")
print(f"Total unique trades: {len(all_trades)}")

if all_trades:
    first = datetime.fromtimestamp(all_trades[0]['time']/1000).strftime("%Y-%m-%d %H:%M:%S")
    last = datetime.fromtimestamp(all_trades[-1]['time']/1000).strftime("%Y-%m-%d %H:%M:%S")
    print(f"Date range: {first} → {last}")
    
    # Trading period
    from datetime import datetime as dt
    start_dt = dt.fromtimestamp(all_trades[0]['time']/1000)
    end_dt = dt.fromtimestamp(all_trades[-1]['time']/1000)
    duration = end_dt - start_dt
    
    print(f"Duration: {duration.days} days, {duration.seconds // 3600} hours")
    print(f"Average: {len(all_trades) / max(duration.total_seconds() / 3600, 1):.1f} trades/hour")
    
    # Calculate total commission paid
    total_commission = sum(float(t['commission']) for t in all_trades)
    print(f"\nTotal commission paid: ${total_commission:.2f} USDT")
    
    # Side distribution
    buys = sum(1 for t in all_trades if t['side'] == 'BUY')
    sells = len(all_trades) - buys
    print(f"Buy trades: {buys} ({buys/len(all_trades)*100:.1f}%)")
    print(f"Sell trades: {sells} ({sells/len(all_trades)*100:.1f}%)")

print(f"\n{'=' * 80}")
print(f"IMPACT ON AUDIT")
print(f"{'=' * 80}")
print(f"Previous audit used: 1,000 trades")
print(f"Actual total trades: {len(all_trades)}")
print(f"Missing trades:      {len(all_trades) - 1000}")
print(f"\n⚠️  The audit was incomplete! Need to rerun with full history.")
