"""Check current leverage for all positions"""
import os
from binance.client import Client
from dotenv import load_dotenv

load_dotenv('backend/.env.live')

client = Client(os.getenv("BINANCE_API_KEY"), os.getenv("BINANCE_API_SECRET"))

# Get position info
positions = client.futures_position_information()

print("\n[CHART] ALL POSITION LEVERAGE INFO:")
print("=" * 100)

for pos in positions:
    symbol = pos['symbol']
    amt = float(pos.get('positionAmt', 0))
    leverage = pos.get('leverage', 'N/A')
    margin_type = pos.get('marginType', 'N/A')
    
    if amt != 0 or symbol.endswith('USDT') or symbol.endswith('USDC'):
        status = "[GREEN_CIRCLE]" if amt != 0 else "⚪"
        print(f"{status} {symbol:15} | Amt: {amt:>12.4f} | Leverage: {leverage:>3} | Margin: {margin_type}")

print("=" * 100)

# Try to get account info too
try:
    account = client.futures_account()
    print(f"\n[MONEY] Total Wallet Balance: ${float(account['totalWalletBalance']):.2f}")
    print(f"💵 Available Balance: ${float(account['availableBalance']):.2f}")
    print(f"[CHART] Total Position Initial Margin: ${float(account['totalPositionInitialMargin']):.2f}")
except Exception as e:
    print(f"[WARNING] Could not get account info: {e}")
