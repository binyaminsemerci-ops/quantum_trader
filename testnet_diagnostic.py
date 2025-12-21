#!/usr/bin/env python3
import os
import sys
from binance.client import Client
from binance.exceptions import BinanceAPIException

# Get API credentials
api_key = os.getenv('BINANCE_API_KEY', 'IsY3mFpko7Z8joZr8clWwpJZuZcFdAtnDBy4g4ULQu827Gf6kJPAPS9VyVOVrR0r')
api_secret = os.getenv('BINANCE_API_SECRET')

if not api_secret:
    print('ERROR: BINANCE_API_SECRET not found')
    sys.exit(1)

print('=' * 70)
print('BINANCE TESTNET COMPREHENSIVE DIAGNOSTIC')
print('=' * 70)

try:
    # Initialize testnet client
    client = Client(api_key, api_secret, testnet=True)
    print('✅ Testnet client initialized')
    
    # Test 1: Server connectivity
    print('\n[TEST 1] Server Connectivity')
    try:
        server_time = client.get_server_time()
        st = server_time['serverTime']
        print(f'✅ Server reachable - Time: {st}')
    except Exception as e:
        print(f'❌ Server unreachable: {e}')
        sys.exit(1)
    
    # Test 2: API Key validity
    print('\n[TEST 2] API Key Validity')
    try:
        account = client.get_account()
        can_trade = account['canTrade']
        print(f'✅ API Key valid - Can Trade: {can_trade}')
    except BinanceAPIException as e:
        print(f'❌ API Key error: {e}')
        sys.exit(1)
    
    # Test 3: Futures account status
    print('\n[TEST 3] Futures Account Status')
    try:
        futures_account = client.futures_account()
        print(f'✅ Futures account accessible')
        total_wallet = futures_account['totalWalletBalance']
        avail_balance = futures_account['availableBalance']
        max_withdraw = futures_account['maxWithdrawAmount']
        pos_margin = futures_account['totalPositionInitialMargin']
        order_margin = futures_account['totalOpenOrderInitialMargin']
        
        print(f'   Total Wallet Balance: {total_wallet} USDT')
        print(f'   Available Balance: {avail_balance} USDT')
        print(f'   Max Withdraw Amount: {max_withdraw} USDT')
        print(f'   Total Position Initial Margin: {pos_margin}')
        print(f'   Total Open Order Initial Margin: {order_margin}')
    except BinanceAPIException as e:
        print(f'❌ Futures account error: {e}')
        print('   This may mean futures trading is not enabled')
        sys.exit(1)
    
    # Test 4: Asset balances
    print('\n[TEST 4] Asset Balances')
    try:
        balances = client.futures_account_balance()
        usdt_balance = None
        for bal in balances:
            if bal['asset'] == 'USDT':
                usdt_balance = bal
                break
        
        if usdt_balance:
            print(f'✅ USDT Balance Found:')
            balance = usdt_balance['balance']
            available = usdt_balance['availableBalance']
            cross_wallet = usdt_balance['crossWalletBalance']
            print(f'   Balance: {balance} USDT')
            print(f'   Available: {available} USDT')
            print(f'   Cross Wallet Balance: {cross_wallet} USDT')
        else:
            print('⚠️  No USDT balance found - account may be empty')
    except Exception as e:
        print(f'❌ Balance check error: {e}')
    
    # Test 5: Current positions
    print('\n[TEST 5] Open Positions')
    try:
        positions = client.futures_position_information()
        open_positions = [p for p in positions if float(p['positionAmt']) != 0]
        if open_positions:
            print(f'⚠️  {len(open_positions)} open position(s) found:')
            for pos in open_positions[:5]:
                symbol = pos['symbol']
                pos_amt = pos['positionAmt']
                entry_price = pos['entryPrice']
                print(f'   {symbol}: {pos_amt} @ {entry_price}')
        else:
            print('✅ No open positions')
    except Exception as e:
        print(f'❌ Position check error: {e}')
    
    # Test 6: Exchange info for BTCUSDT
    print('\n[TEST 6] Symbol Information (BTCUSDT)')
    try:
        exchange_info = client.futures_exchange_info()
        btc_symbol = None
        for symbol in exchange_info['symbols']:
            if symbol['symbol'] == 'BTCUSDT':
                btc_symbol = symbol
                break
        
        if btc_symbol:
            print(f'✅ BTCUSDT tradeable')
            status = btc_symbol['status']
            contract_type = btc_symbol['contractType']
            print(f'   Status: {status}')
            print(f'   Contract Type: {contract_type}')
            for filter_item in btc_symbol['filters']:
                if filter_item['filterType'] == 'MIN_NOTIONAL':
                    notional = filter_item['notional']
                    print(f'   Min Notional: {notional} USDT')
                if filter_item['filterType'] == 'MARKET_LOT_SIZE':
                    min_qty = filter_item['minQty']
                    print(f'   Min Qty: {min_qty}')
        else:
            print('❌ BTCUSDT not found')
    except Exception as e:
        print(f'❌ Exchange info error: {e}')
    
    # Test 7: Try to get current price
    print('\n[TEST 7] Market Data Access')
    try:
        ticker = client.futures_symbol_ticker(symbol='BTCUSDT')
        price = ticker['price']
        print(f'✅ Market data accessible')
        print(f'   BTCUSDT Price: ${price}')
    except Exception as e:
        print(f'❌ Market data error: {e}')
    
    print('\n' + '=' * 70)
    print('DIAGNOSTIC COMPLETE')
    print('=' * 70)
    
except Exception as e:
    print(f'\n❌ FATAL ERROR: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
