#!/usr/bin/env python3
"""
Check current Binance account status and positions
"""
from binance.client import Client
import os

client = Client(os.getenv('BINANCE_API_KEY'), os.getenv('BINANCE_API_SECRET'))

# Get account info
account = client.futures_account()
positions = client.futures_position_information()

# Filter open positions
open_positions = [p for p in positions if float(p['positionAmt']) != 0]

print('=' * 80)
print('💰 ACCOUNT BALANCE')
print('=' * 80)
print(f"Total Wallet Balance: ${float(account['totalWalletBalance']):.2f}")
print(f"Unrealized PnL: ${float(account['totalUnrealizedProfit']):.2f}")
print(f"Available Balance: ${float(account['availableBalance']):.2f}")
print(f"Total Margin Balance: ${float(account['totalMarginBalance']):.2f}")

print('\n' + '=' * 80)
print(f'📊 OPEN POSITIONS: {len(open_positions)}')
print('=' * 80)

if not open_positions:
    print('\n✅ Ingen åpne posisjoner')
else:
    total_unrealized = 0
    for pos in open_positions:
        symbol = pos['symbol']
        amt = float(pos['positionAmt'])
        side = 'LONG' if amt > 0 else 'SHORT'
        qty = abs(amt)
        entry = float(pos['entryPrice'])
        mark = float(pos['markPrice'])
        unrealized = float(pos['unRealizedProfit'])
        leverage = pos['leverage']
        
        # Calculate PnL %
        if side == 'LONG':
            pnl_pct = ((mark - entry) / entry) * 100
        else:
            pnl_pct = ((entry - mark) / entry) * 100
        
        emoji = '💚' if unrealized > 0 else '❌'
        
        print(f'\n🔹 {symbol} {side} ({leverage}x)')
        print(f'   Entry: ${entry:.4f}')
        print(f'   Mark:  ${mark:.4f}')
        print(f'   Qty:   {qty:.4f}')
        print(f'   {emoji} PnL: ${unrealized:.2f} ({pnl_pct:+.2f}%)')
        
        total_unrealized += unrealized
    
    print('\n' + '=' * 80)
    emoji = '💚' if total_unrealized > 0 else '❌'
    print(f'{emoji} TOTAL UNREALIZED PNL: ${total_unrealized:.2f}')
    print('=' * 80)
