from binance.client import Client
from datetime import datetime

API_KEY = "w2W60kzuCfPJKGIqSvmp0pqISUO8XKICjc5sD8QyJuJpp9LKQgvXKhtd09Ii3rwg"
API_SECRET = "QI18cg4zcbApc9uaDL8ZUmoAJQthQczZ9cKzORlSJfnK2zBEdLvSLb5ZEgZ6R1Kg"

print("=" * 80)
print("CHECKING IF WE HIT API LIMITS")
print("=" * 80)

client = Client(API_KEY, API_SECRET, testnet=True)

# Get all symbols
positions = client.futures_position_information()
symbols = [p['symbol'] for p in positions if float(p.get('notional', 0)) != 0 or float(p.get('positionAmt', 0)) != 0]

print(f"\nChecking {len(symbols)} symbols for complete history...\n")

total_trades = 0
symbols_at_limit = []

for symbol in sorted(symbols):
    # Fetch with limit=1000
    trades = client.futures_account_trades(symbol=symbol, limit=1000)
    total_trades += len(trades)
    
    status = "✓ Complete"
    if len(trades) >= 1000:
        status = "⚠️  AT LIMIT (likely more history)"
        symbols_at_limit.append(symbol)
    
    if trades:
        first_trade = datetime.fromtimestamp(trades[0]['time']/1000).strftime("%Y-%m-%d %H:%M")
        last_trade = datetime.fromtimestamp(trades[-1]['time']/1000).strftime("%Y-%m-%d %H:%M")
        print(f"{symbol:15} {len(trades):4} trades  {first_trade} → {last_trade}  {status}")

print(f"\n{'=' * 80}")
print(f"TOTAL TRADES FETCHED: {total_trades}")
print(f"{'=' * 80}\n")

if symbols_at_limit:
    print(f"⚠️  WARNING: {len(symbols_at_limit)} symbols hit 1000-trade limit:")
    for sym in symbols_at_limit:
        print(f"    - {sym}")
    print("\n   There is likely MORE history that was NOT included in audit!")
    print("   Need to fetch in batches using 'startTime' parameter.\n")
else:
    print("✓ All symbols show complete history (no limits hit)\n")

# Check oldest trade timestamp
print("Checking account creation / first trade...")
all_first_trades = []
for symbol in symbols:
    trades = client.futures_account_trades(symbol=symbol, limit=1)
    if trades:
        all_first_trades.append((symbol, trades[0]['time']))

if all_first_trades:
    oldest = min(all_first_trades, key=lambda x: x[1])
    oldest_date = datetime.fromtimestamp(oldest[1]/1000).strftime("%Y-%m-%d %H:%M:%S")
    print(f"Oldest trade found: {oldest[0]} at {oldest_date}")
    
    # Calculate days of trading
    from datetime import datetime as dt
    days_trading = (dt.now() - dt.fromtimestamp(oldest[1]/1000)).days
    print(f"Trading period: ~{days_trading} days")
    print(f"Average trades per day: {total_trades / days_trading:.1f}")
