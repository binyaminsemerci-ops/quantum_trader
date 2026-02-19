from binance.client import Client
from datetime import datetime, timedelta

API_KEY = "w2W60kzuCfPJKGIqSvmp0pqISUO8XKICjc5sD8QyJuJpp9LKQgvXKhtd09Ii3rwg"
API_SECRET = "QI18cg4zcbApc9uaDL8ZUmoAJQthQczZ9cKzORlSJfnK2zBEdLvSLb5ZEgZ6R1Kg"

print("=" * 80)
print("COMPREHENSIVE HISTORY CHECK - ALL SYMBOLS")
print("=" * 80)

client = Client(API_KEY, API_SECRET, testnet=True)

# Check account age by trying to fetch from very old timestamp
print("\n1. Checking account age...")
very_old_timestamp = int((datetime.now() - timedelta(days=365)).timestamp() * 1000)

try:
    # Try ADAUSDT first
    old_trades = client.futures_account_trades(symbol='ADAUSDT', startTime=very_old_timestamp, limit=1)
    if old_trades:
        oldest_date = datetime.fromtimestamp(old_trades[0]['time']/1000).strftime("%Y-%m-%d %H:%M:%S")
        print(f"   Oldest ADAUSDT trade: {oldest_date}")
        
        account_age_days = (datetime.now() - datetime.fromtimestamp(old_trades[0]['time']/1000)).days
        print(f"   Account age: ~{account_age_days} days")
    else:
        print("   No old trades found - account likely created recently")
except Exception as e:
    print(f"   Error checking old trades: {e}")

# Get all symbols with positions
print("\n2. Fetching complete history for ALL symbols...")
positions = client.futures_position_information()
symbols = [p['symbol'] for p in positions if float(p.get('notional', 0)) != 0 or float(p.get('positionAmt', 0)) != 0]

total_all_trades = 0
summary = []

for symbol in sorted(symbols):
    # Fetch with very old startTime to ensure we get everything
    start_time = int((datetime.now() - timedelta(days=365)).timestamp() * 1000)
    
    all_symbol_trades = []
    while True:
        batch = client.futures_account_trades(
            symbol=symbol,
            startTime=start_time,
            limit=1000
        )
        
        if not batch:
            break
        
        all_symbol_trades.extend(batch)
        
        # If less than limit, we got everything
        if len(batch) < 1000:
            break
        
        # Move startTime forward to last trade + 1ms
        start_time = batch[-1]['time'] + 1
    
    # Remove duplicates
    unique_trades = {t['id']: t for t in all_symbol_trades}
    count = len(unique_trades)
    total_all_trades += count
    
    if count > 0:
        first = datetime.fromtimestamp(list(unique_trades.values())[0]['time']/1000).strftime("%Y-%m-%d %H:%M")
        summary.append({
            'symbol': symbol,
            'count': count,
            'first': first
        })

print(f"\n{'=' * 80}")
print("COMPLETE TRADE COUNT BY SYMBOL")
print(f"{'=' * 80}\n")

for s in summary:
    print(f"{s['symbol']:15} {s['count']:5} trades (oldest: {s['first']})")

print(f"\n{'=' * 80}")
print(f"TOTAL TRADES (COMPLETE): {total_all_trades}")
print(f"{'=' * 80}")

print(f"\nPrevious audit count: 1,665 trades")
print(f"Complete history:     {total_all_trades} trades")
print(f"Difference:           {total_all_trades - 1665} trades")

if total_all_trades > 1665:
    print(f"\n⚠️  AUDIT WAS INCOMPLETE - {total_all_trades - 1665} trades were missed!")
    print("   Need to rerun full reconstruction with complete data.")
else:
    print("\n✓ Audit included complete history - no additional trades found.")
