"""Check Binance account balance."""
from binance.client import Client
import os

client = Client(
    os.getenv('BINANCE_API_KEY'),
    os.getenv('BINANCE_API_SECRET'),
    testnet=True
)

print("\n" + "="*80)
print("BINANCE FUTURES ACCOUNT BALANCE")
print("="*80)

try:
    account = client.futures_account()
    
    print(f"\nTotal Wallet Balance: ${float(account['totalWalletBalance']):.2f} USDT")
    print(f"Total Unrealized PnL: ${float(account['totalUnrealizedProfit']):.2f} USDT")
    print(f"Total Margin Balance: ${float(account['totalMarginBalance']):.2f} USDT")
    print(f"Available Balance:    ${float(account['availableBalance']):.2f} USDT")
    print(f"Total Position Initial Margin: ${float(account['totalPositionInitialMargin']):.2f} USDT")
    print(f"Total Open Order Initial Margin: ${float(account['totalOpenOrderInitialMargin']):.2f} USDT")
    
    print(f"\n{'─'*80}")
    print("ASSETS:")
    print(f"{'─'*80}")
    
    for asset in account['assets']:
        wallet_balance = float(asset['walletBalance'])
        unrealized_profit = float(asset['unrealizedProfit'])
        
        if wallet_balance != 0 or unrealized_profit != 0:
            print(f"{asset['asset']}: Wallet=${wallet_balance:.2f}, Unrealized=${unrealized_profit:+.2f}")
    
    print(f"\n{'─'*80}")
    print("OPEN POSITIONS:")
    print(f"{'─'*80}")
    
    positions = client.futures_position_information()
    open_count = 0
    
    for pos in positions:
        position_amt = float(pos['positionAmt'])
        if position_amt != 0:
            open_count += 1
            side = 'LONG' if position_amt > 0 else 'SHORT'
            entry_price = float(pos['entryPrice'])
            mark_price = float(pos['markPrice'])
            unrealized_pnl = float(pos['unRealizedProfit'])
            
            print(f"{pos['symbol']} {side}: {abs(position_amt)} @ ${entry_price:.4f}")
            print(f"  Mark: ${mark_price:.4f}, PnL: ${unrealized_pnl:+.2f}")
    
    print(f"\nTotal open positions: {open_count}")
    
except Exception as e:
    print(f"❌ Error: {e}")

print("="*80 + "\n")
