from binance.client import Client
from datetime import datetime

API_KEY = "w2W60kzuCfPJKGIqSvmp0pqISUO8XKICjc5sD8QyJuJpp9LKQgvXKhtd09Ii3rwg"
API_SECRET = "QI18cg4zcbApc9uaDL8ZUmoAJQthQczZ9cKzORlSJfnK2zBEdLvSLb5ZEgZ6R1Kg"

print("=" * 80)
print("COMPREHENSIVE EXCHANGE STATE ANALYSIS")
print("=" * 80)

client = Client(API_KEY, API_SECRET, testnet=True)

# Account overview
account = client.futures_account()
print(f"\nAccount Balance: {account['totalWalletBalance']} USDT")
print(f"Total Margin Balance: {account['totalMarginBalance']} USDT")
print(f"Available Balance: {account['availableBalance']} USDT")
print(f"Total Unrealized Profit: {account['totalUnrealizedProfit']} USDT")
print(f"Total Position Initial Margin: {account['totalPositionInitialMargin']} USDT")

# Check all income types
print("\n" + "=" * 80)
print("INCOME HISTORY BY TYPE")
print("=" * 80)

income_types = ['REALIZED_PNL', 'FUNDING_FEE', 'COMMISSION', 'TRANSFER', 'WELCOME_BONUS']
for inc_type in income_types:
    try:
        records = client.futures_income_history(incomeType=inc_type, limit=1000)
        if records:
            total = sum(float(r['income']) for r in records)
            print(f"{inc_type:20} {len(records):5} records  Total: ${total:.2f}")
    except Exception as e:
        print(f"{inc_type:20} Error: {e}")

# All income (no filter)
print("\nFetching ALL income (no filter)...")
all_income = client.futures_income_history(limit=1000)
print(f"Total income records: {len(all_income)}")

if all_income:
    by_type = {}
    for rec in all_income:
        t = rec['incomeType']
        if t not in by_type:
            by_type[t] = []
        by_type[t].append(float(rec['income']))
    
    print("\nBreakdown:")
    for t, vals in by_type.items():
        print(f"  {t:20} {len(vals):5} records  ${sum(vals):.2f}")

# Current positions
print("\n" + "=" * 80)
print("CURRENT POSITIONS")
print("=" * 80)

positions = client.futures_position_information()
active_positions = [p for p in positions if float(p['positionAmt']) != 0]

if active_positions:
    print(f"Found {len(active_positions)} active positions:\n")
    for p in active_positions:
        print(f"  {p['symbol']:12} Qty: {float(p['positionAmt']):10.4f}  "
              f"Entry: ${float(p['entryPrice']):10.2f}  "
              f"UnrealizedPnL: ${float(p['unRealizedProfit']):8.2f}")
else:
    print("No active positions")

# Account trades for all symbols
print("\n" + "=" * 80)
print("ACCOUNT TRADES")
print("=" * 80)

# Get list of symbols that have been traded
traded_symbols = set()
for pos in positions:
    if float(pos.get('notional', 0)) != 0 or float(pos.get('positionAmt', 0)) != 0:
        traded_symbols.add(pos['symbol'])

print(f"Checking {len(traded_symbols)} symbols for trade history...")

all_trades = []
for symbol in sorted(traded_symbols):
    try:
        trades = client.futures_account_trades(symbol=symbol, limit=1000)
        if trades:
            all_trades.extend(trades)
            print(f"  {symbol:12} {len(trades):5} trades")
    except Exception as e:
        print(f"  {symbol:12} Error: {e}")

print(f"\nTotal trades found: {len(all_trades)}")

if all_trades:
    print("\nFirst 10 trades:")
    for i, t in enumerate(all_trades[:10]):
        ts = datetime.fromtimestamp(t['time']/1000).strftime("%Y-%m-%d %H:%M")
        print(f"  {i+1}. {t['symbol']:12} {t['side']:4} {float(t['qty']):10.4f} @ ${float(t['price']):10.2f}  "
              f"Commission: ${float(t['commission']):6.4f} {t['commissionAsset']:5}  {ts}")

print("\n" + "=" * 80)
