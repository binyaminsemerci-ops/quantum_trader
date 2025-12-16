"""Close all open positions safely."""
import requests
import time

try:
    print('\n🛑 CLOSING ALL LIVE POSITIONS...\n')
    print('═' * 80)
    
    # Get current positions
    r = requests.get('http://localhost:8000/api/positions', timeout=10)
    data = r.json()
    
    positions = data.get('positions', [])
    active_positions = [p for p in positions if float(p.get('positionAmt', 0)) != 0]
    
    if not active_positions:
        print('[OK] No active positions to close')
    else:
        print(f'Found {len(active_positions)} positions to close:\n')
        
        for pos in active_positions:
            symbol = pos.get('symbol')
            amt = float(pos.get('positionAmt', 0))
            side = 'LONG' if amt > 0 else 'SHORT'
            entry = float(pos.get('entryPrice', 0))
            pnl = float(pos.get('unrealizedProfit', 0))
            
            print(f'[CHART] {symbol} {side}')
            print(f'   Entry: ${entry:.4f}')
            print(f'   P&L: ${pnl:.2f}')
            
            # Close position via API
            close_side = 'SELL' if amt > 0 else 'BUY'
            close_qty = abs(amt)
            
            payload = {
                'symbol': symbol,
                'side': close_side,
                'type': 'MARKET',
                'quantity': close_qty,
                'reduceOnly': True
            }
            
            print(f'   Closing with {close_side} {close_qty}...')
            
            close_response = requests.post(
                'http://localhost:8000/api/orders',
                json=payload,
                timeout=10
            )
            
            if close_response.status_code == 200:
                print(f'   [OK] Closed successfully')
            else:
                print(f'   ❌ Error: {close_response.text}')
            
            print()
            time.sleep(0.5)
        
        # Summary
        total_pnl = sum(float(p.get('unrealizedProfit', 0)) for p in active_positions)
        print('═' * 80)
        print(f'\n[MONEY] Total P&L from closed positions: ${total_pnl:.2f}')
    
    print('\n[OK] All positions closed')
    print('═' * 80)
    
except Exception as e:
    print(f'❌ Error: {e}')
