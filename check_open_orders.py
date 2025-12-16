"""Check open orders to see SL/TP levels."""
import requests

try:
    r = requests.get('http://localhost:8000/api/orders/open', timeout=10)
    data = r.json()
    
    print('\n[CLIPBOARD] OPEN STOP LOSS / TAKE PROFIT ORDERS:\n')
    print('═' * 80)
    
    if 'error' in data:
        print(f'❌ Error: {data["error"]}')
    else:
        orders = data.get('orders', data)  # Handle both structures
        
        if not orders or len(orders) == 0:
            print('❌ INGEN ÅPNE ORDRER FUNNET!')
            print('[WARNING]  Dette betyr at posisjonene IKKE har stop loss beskyttelse!')
        else:
            for order in orders:
                symbol = order.get('symbol')
                order_type = order.get('type')
                side = order.get('side')
                price = order.get('stopPrice') or order.get('price')
                
                print(f'{symbol:12s} | {order_type:20s} | {side:4s} @ ${price}')
    
    print('\n' + '═' * 80)
    
except Exception as e:
    print(f'❌ Feil: {e}')
