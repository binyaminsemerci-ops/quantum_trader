import os
os.environ["QT_PAPER_TRADING"] = "true"
from backend.services.binance.binance_client import BinanceClient
client = BinanceClient()
positions = client.futures_position_information()
active = [p for p in positions if float(p["positionAmt"]) != 0]
for p in active:
    amt = float(p["positionAmt"])
    side = "LONG" if amt > 0 else "SHORT"
    pnl = float(p["unrealizedProfit"])
    entry = p["entryPrice"]
    print(f"{p[''symbol'']}: {side} {abs(amt):.4f} @ {entry} PnL={pnl:.2f}")
