"""Force close all positions via direct Binance API."""
from binance.client import Client
import os

API_KEY = os.getenv('BINANCE_API_KEY')
API_SECRET = os.getenv('BINANCE_API_SECRET')

if not API_KEY or not API_SECRET:
    print('❌ Missing API credentials')
    exit(1)

client = Client(API_KEY, API_SECRET)

print('\n🛑 CLOSING ALL LIVE BINANCE POSITIONS...\n')
print('═' * 80)

try:
    # Get all positions
    positions = client.futures_position_information()
    
    closed_count = 0
    total_pnl = 0
    
    for pos in positions:
        amt = float(pos['positionAmt'])
        if amt == 0:
            continue
            
        symbol = pos['symbol']
        side = 'LONG' if amt > 0 else 'SHORT'
        entry = float(pos['entryPrice'])
        pnl = float(pos['unRealizedProfit'])
        total_pnl += pnl
        
        print(f'\n[CHART] {symbol} {side}')
        print(f'   Entry: ${entry:.4f}')
        print(f'   Amount: {abs(amt)}')
        print(f'   P&L: ${pnl:.2f}')
        
        # Close with MARKET order, reduceOnly
        close_side = 'SELL' if amt > 0 else 'BUY'
        
        try:
            order = client.futures_create_order(
                symbol=symbol,
                side=close_side,
                type='MARKET',
                quantity=abs(amt),
                reduceOnly=True
            )
            print(f'   [OK] CLOSED via {close_side} MARKET order')
            print(f'   Order ID: {order["orderId"]}')
            closed_count += 1
        except Exception as e:
            print(f'   ❌ Error closing: {e}')
    
    print('\n' + '═' * 80)
    print(f'\n[OK] Closed {closed_count} positions')
    print(f'[MONEY] Total P&L: ${total_pnl:.2f}')
    print('\n' + '═' * 80)
    
except Exception as e:
    print(f'❌ Fatal error: {e}')
