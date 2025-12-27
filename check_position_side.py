import sys
sys.path.insert(0, '/app')
from backend.utils.binance_client import BinanceClient

c = BinanceClient()
positions = c.client.futures_position_information(symbol='SOLUSDT')
for p in positions:
    print(f"positionSide: {p.get('positionSide')}, positionAmt: {p.get('positionAmt')}")
