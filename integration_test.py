#!/usr/bin/env python3
"""
Quantum Trader - Full System Integration Test
Tests all critical components and their connectivity
"""
import requests
import sys
from datetime import datetime
import hmac
import hashlib
import time

def run_tests():
    tests = []
    print('='*70)
    print('🧪 QUANTUM TRADER - FULL SYSTEM INTEGRATION TEST')
    print(f'⏰ Time: {datetime.now()}')
    print('='*70)
    print()

    # Test 1: Backend API
    print("Testing Backend API...")
    try:
        r = requests.get('http://localhost:8000/', timeout=5)
        tests.append(('Backend API (Port 8000)', r.status_code == 200, r.status_code))
    except Exception as e:
        tests.append(('Backend API (Port 8000)', False, str(e)[:50]))

    # Test 2: Execution Service (via host port)
    print("Testing Execution Service...")
    try:
        # Try host network first
        try:
            r = requests.get('http://host.docker.internal:8002/health', timeout=3)
        except:
            r = requests.get('http://172.17.0.1:8002/health', timeout=3)
        tests.append(('Execution Service (Port 8002)', r.status_code == 200, r.status_code))
    except Exception as e:
        tests.append(('Execution Service (Port 8002)', False, str(e)[:50]))

    # Test 3: Portfolio Intelligence (via host port - root endpoint)
    print("Testing Portfolio Intelligence...")
    try:
        try:
            r = requests.get('http://host.docker.internal:8004/', timeout=3)
        except:
            r = requests.get('http://172.17.0.1:8004/', timeout=3)
        tests.append(('Portfolio Intelligence (Port 8004)', r.status_code == 200, r.status_code))
    except Exception as e:
        tests.append(('Portfolio Intelligence (Port 8004)', False, str(e)[:50]))

    # Test 4: Trading Bot (via host port)
    print("Testing Trading Bot...")
    try:
        try:
            r = requests.get('http://host.docker.internal:8003/health', timeout=3)
        except:
            r = requests.get('http://172.17.0.1:8003/health', timeout=3)
        tests.append(('Trading Bot (Port 8003)', r.status_code == 200, r.status_code))
    except Exception as e:
        tests.append(('Trading Bot (Port 8003)', False, str(e)[:50]))

    # Test 5: Binance Testnet Connectivity
    print("Testing Binance Testnet...")
    try:
        r = requests.get('https://testnet.binancefuture.com/fapi/v1/time', timeout=5)
        tests.append(('Binance Testnet API', r.status_code == 200, r.status_code))
    except Exception as e:
        tests.append(('Binance Testnet API', False, str(e)[:50]))

    # Test 6: Binance Account Access
    print("Testing Binance Account Authentication...")
    try:
        api_key = 'IsY3mFpko7Z8joZr8clWwpJZuZcFdAtnDBy4g4ULQu827Gf6kJPAPS9VyVOVrR0r'
        api_secret = 'tEKYWf77tqSOArfgeSqgVwiO62bjro8D1VMaEvXBwcUOQuRxmCdxrtTvAXy7ZKSE'
        timestamp = int(time.time() * 1000)
        query = f'timestamp={timestamp}'
        signature = hmac.new(api_secret.encode(), query.encode(), hashlib.sha256).hexdigest()
        headers = {'X-MBX-APIKEY': api_key}
        r = requests.get(
            f'https://testnet.binancefuture.com/fapi/v2/account?{query}&signature={signature}', 
            headers=headers, 
            timeout=10
        )
        
        if r.status_code == 200:
            data = r.json()
            # Find USDT balance (try both 'balance' and 'availableBalance')
            usdt_asset = next((a for a in data.get('assets', []) if a['asset'] == 'USDT'), {})
            balance = usdt_asset.get('balance') or usdt_asset.get('availableBalance', '0')
            positions = [p for p in data.get('positions', []) if float(p.get('positionAmt', 0)) != 0]
            print(f'   💰 Balance: {balance} USDT')
            print(f'   📊 Active Positions: {len(positions)}')
            tests.append(('Binance Account Access', True, f'{balance} USDT, {len(positions)} positions'))
        else:
            tests.append(('Binance Account Access', False, f'HTTP {r.status_code}'))
    except Exception as e:
        tests.append(('Binance Account Access', False, str(e)[:50]))

    # Print Results
    print()
    print('='*70)
    print('📊 TEST RESULTS')
    print('='*70)
    
    passed = 0
    failed = 0
    
    for name, result, detail in tests:
        status = '✅' if result else '❌'
        print(f'{status} {name:<45} {detail}')
        if result:
            passed += 1
        else:
            failed += 1
    
    print('='*70)
    print(f'📈 SUMMARY: {passed}/{len(tests)} tests passed, {failed} failed')
    print('='*70)
    
    return failed == 0

if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
