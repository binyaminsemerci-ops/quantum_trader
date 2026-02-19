from binance.client import Client
from datetime import datetime

API_KEY = "w2W60kzuCfPJKGIqSvmp0pqISUO8XKICjc5sD8QyJuJpp9LKQgvXKhtd09Ii3rwg"
API_SECRET = "QI18cg4zcbApc9uaDL8ZUmoAJQthQczZ9cKzORlSJfnK2zBEdLvSLb5ZEgZ6R1Kg"

print("=" * 80)
print("QUANTUM TRADER - EXCHANGE GROUND-TRUTH AUDIT")
print("=" * 80)
print(f"Timestamp: {datetime.now().isoformat()}")
print()

client = Client(API_KEY, API_SECRET, testnet=True)
print("Connected to Binance Futures TESTNET")

account = client.futures_account()
print(f"Account Balance: {account['totalWalletBalance']} USDT")
print()

print("Fetching income history...")
income = client.futures_income_history(incomeType="REALIZED_PNL", limit=1000)
print(f"Found {len(income)} income records")

pnl_records = [rec for rec in income if float(rec["income"]) != 0]
print(f"Non-zero PnL: {len(pnl_records)} trades")
print()

if len(pnl_records) > 0:
    pnls = [float(rec["income"]) for rec in pnl_records]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    
    total_pnl = sum(pnls)
    win_rate = len(wins) / len(pnls) * 100
    avg_win = sum(wins) / len(wins) if wins else 0
    avg_loss = abs(sum(losses) / len(losses)) if losses else 0
    profit_factor = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else 999
    expectancy = total_pnl / len(pnls)
    
    print("=" * 80)
    print("EXCHANGE GROUND-TRUTH METRICS")
    print("=" * 80)
    print(f"Sample Size:           {len(pnl_records)} trades")
    print(f"Total PnL:             ${total_pnl:.2f} USDT")
    print(f"Win Rate:              {win_rate:.1f}%")
    print(f"Wins / Losses:         {len(wins)} / {len(losses)}")
    print(f"Average Win:           ${avg_win:.2f}")
    print(f"Average Loss:          ${avg_loss:.2f}")
    print(f"Profit Factor:         {profit_factor:.2f}")
    print(f"Expectancy per Trade:  ${expectancy:.2f}")
    print()
    
    min_required = 174
    if len(pnl_records) < min_required:
        print(f"WARNING: Sample size ({len(pnl_records)}) < required ({min_required})")
        print("Results NOT statistically significant")
        print()
    
    if expectancy > 0 and profit_factor > 1.5:
        verdict = "STRUCTURALLY PROFITABLE"
    elif expectancy > 0 and profit_factor > 1.0:
        verdict = "POTENTIALLY POSITIVE"
    elif expectancy < 0:
        verdict = "STRUCTURALLY NEGATIVE"
    else:
        verdict = "INCONCLUSIVE"
    
    print(f"VERDICT: {verdict}")
    print("=" * 80)
    print()
    print("Sample trades:")
    for i, rec in enumerate(pnl_records[:10]):
        ts = datetime.fromtimestamp(rec["time"]/1000).strftime("%Y-%m-%d %H:%M:%S")
        print(f"  {i+1}. {rec['symbol']:12} ${float(rec['income']):8.2f} @ {ts}")
else:
    print("No trades found")
