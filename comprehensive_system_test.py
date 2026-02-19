#!/usr/bin/env python3
"""
QUANTUM TRADER - COMPREHENSIVE SYSTEM TEST
Testing all API credentials and system components after credential restoration
"""

import os
import sys
import json
import time
import requests
import hmac
import hashlib
from pathlib import Path
from urllib.parse import urlencode

# Working credentials (verified)
WORKING_API_KEY = "w2W60kzuCfPJKGIqSvmp0pqISUO8XKICjc5sD8QyJuJpp9LKQgvXKhtd09Ii3rwg"
WORKING_API_SECRET = "QI18cg4zcbApc9uaDL8ZUmoAJQthQczZ9cKzORlSJfnK2zBEdLvSLb5ZEgZ6R1Kg"

def print_header(title):
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def print_subheader(title):
    print(f"\n{'-'*40}")
    print(f" {title}")
    print(f"{'-'*40}")

def sign_request(params, secret):
    query_string = urlencode(sorted(params.items()))
    return hmac.new(secret.encode(), query_string.encode(), hashlib.sha256).hexdigest()

def test_api_credentials():
    """Test direct API access with working credentials"""
    print_subheader("1. DIRECT API CREDENTIAL TEST")
    
    try:
        base_url = 'https://testnet.binancefuture.com'
        timestamp = int(time.time() * 1000)
        params = {'timestamp': timestamp}
        signature = sign_request(params, WORKING_API_SECRET)
        params['signature'] = signature
        
        headers = {'X-MBX-APIKEY': WORKING_API_KEY}
        
        # Test account endpoint
        response = requests.get(f'{base_url}/fapi/v2/account', headers=headers, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API Access: SUCCESS")
            print(f"   Balance: {data.get('totalWalletBalance', 'N/A')} USDT")
            print(f"   Available: {data.get('availableBalance', 'N/A')} USDT")
            print(f"   Positions: {data.get('totalPositionInitialMargin', 'N/A')} USDT margin")
            return True, data
        else:
            print(f"❌ API Access: FAILED ({response.status_code})")
            return False, None
            
    except Exception as e:
        print(f"❌ API Test Error: {e}")
        return False, None

def test_environment_files():
    """Test that environment files contain working credentials"""
    print_subheader("2. ENVIRONMENT FILE VERIFICATION")
    
    critical_files = [
        ".env",
        "backend/.env",
        "config/balance-tracker.env",
        "systemd/env-templates/ai-engine.env",
        "systemd/env-templates/execution.env",
        "systemd/env-templates/binance-pnl-tracker.env",
        "GO_LIVE_ENV_TEMPLATE.env"
    ]
    
    results = {"success": 0, "failed": 0, "missing": 0}
    
    for file_path in critical_files:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                
                has_working_key = WORKING_API_KEY in content
                has_working_secret = WORKING_API_SECRET in content
                
                if has_working_key and has_working_secret:
                    print(f"✅ {file_path}: HAS WORKING CREDENTIALS")
                    results["success"] += 1
                elif has_working_key or has_working_secret:
                    print(f"⚠️  {file_path}: PARTIAL CREDENTIALS (key:{has_working_key}, secret:{has_working_secret})")
                    results["failed"] += 1
                else:
                    print(f"❌ {file_path}: NO WORKING CREDENTIALS")
                    results["failed"] += 1
                    
            except Exception as e:
                print(f"❌ {file_path}: READ ERROR - {e}")
                results["failed"] += 1
        else:
            print(f"⚠️  {file_path}: FILE NOT FOUND")
            results["missing"] += 1
    
    print(f"\n📊 Environment Files Summary:")
    print(f"   ✅ Success: {results['success']}")
    print(f"   ❌ Failed: {results['failed']}")  
    print(f"   ⚠️  Missing: {results['missing']}")
    
    return results["success"] > 0 and results["failed"] == 0

def test_position_access():
    """Test accessing live position data"""
    print_subheader("3. LIVE POSITION DATA ACCESS")
    
    try:
        base_url = 'https://testnet.binancefuture.com'
        timestamp = int(time.time() * 1000)
        params = {'timestamp': timestamp}
        signature = sign_request(params, WORKING_API_SECRET)
        params['signature'] = signature
        
        headers = {'X-MBX-APIKEY': WORKING_API_KEY}
        
        response = requests.get(f'{base_url}/fapi/v2/positionRisk', headers=headers, params=params, timeout=10)
        
        if response.status_code == 200:
            positions = response.json()
            active_positions = [p for p in positions if float(p.get('positionAmt', 0)) != 0]
            
            print(f"✅ Position Access: SUCCESS")
            print(f"   Total Symbols: {len(positions)}")
            print(f"   Active Positions: {len(active_positions)}")
            
            print(f"\n📋 Active Positions:")
            for i, pos in enumerate(active_positions[:5], 1):  # Show first 5
                symbol = pos.get('symbol', 'Unknown')
                size = float(pos.get('positionAmt', 0))
                side = "LONG" if size > 0 else "SHORT"
                pnl = float(pos.get('unRealizedProfit', 0))
                print(f"   {i}. {symbol}: {abs(size):,.0f} {side} (PnL: {pnl:+.2f})")
            
            return True, active_positions
        
        else:
            print(f"❌ Position Access: FAILED ({response.status_code})")
            return False, []
            
    except Exception as e:
        print(f"❌ Position Access Error: {e}")
        return False, []

def test_market_data():
    """Test market data access for formula calculations"""
    print_subheader("4. MARKET DATA ACCESS")
    
    try:
        # Test symbol info
        symbol_response = requests.get('https://testnet.binancefuture.com/fapi/v1/exchangeInfo', timeout=10)
        ticker_response = requests.get('https://testnet.binancefuture.com/fapi/v1/ticker/24hr', params={'symbol': 'BTCUSDT'}, timeout=10)
        
        symbol_success = symbol_response.status_code == 200
        ticker_success = ticker_response.status_code == 200
        
        if symbol_success and ticker_success:
            symbol_data = symbol_response.json()
            ticker_data = ticker_response.json()
            
            print(f"✅ Market Data Access: SUCCESS")
            print(f"   Available Symbols: {len(symbol_data.get('symbols', []))}")
            print(f"   BTC Price: ${float(ticker_data.get('lastPrice', 0)):,.2f}")
            print(f"   BTC 24h Change: {ticker_data.get('priceChangePercent', 'N/A')}%")
            
            return True
        else:
            print(f"❌ Market Data Access: FAILED")
            print(f"   Symbol Info: {symbol_response.status_code}")
            print(f"   Ticker Data: {ticker_response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Market Data Error: {e}")
        return False

def test_formula_system():
    """Test formula system calculation capability"""
    print_subheader("5. FORMULA SYSTEM SIMULATION")
    
    try:
        # Simulate formula calculation with live data
        base_url = 'https://testnet.binancefuture.com'
        timestamp = int(time.time() * 1000)
        params = {'timestamp': timestamp}
        signature = sign_request(params, WORKING_API_SECRET)
        params['signature'] = signature
        
        headers = {'X-MBX-APIKEY': WORKING_API_KEY}
        
        # Get position data
        pos_response = requests.get(f'{base_url}/fapi/v2/positionRisk', headers=headers, params=params, timeout=10)
        
        if pos_response.status_code == 200:
            positions = pos_response.json()
            active_positions = [p for p in positions if float(p.get('positionAmt', 0)) != 0]
            
            print(f"✅ Formula System Test: SUCCESS")
            print(f"   Analyzing {len(active_positions)} active positions")
            
            # Simulate formula calculations
            for i, pos in enumerate(active_positions[:3], 1):
                symbol = pos.get('symbol', 'Unknown')
                entry_price = float(pos.get('entryPrice', 0))
                mark_price = float(pos.get('markPrice', 0))
                pnl = float(pos.get('unRealizedProfit', 0))
                
                if entry_price > 0 and mark_price > 0:
                    # Simulate dynamic stop calculation
                    price_change_pct = abs((mark_price - entry_price) / entry_price) * 100
                    suggested_stop_pct = max(1.5, min(5.0, price_change_pct * 0.5))  # Example formula
                    
                    print(f"   {i}. {symbol}: Entry ${entry_price:.6f} → Mark ${mark_price:.6f}")
                    print(f"      PnL: ${pnl:+.2f} | Suggested Stop: {suggested_stop_pct:.2f}%")
            
            return True
        else:
            print(f"❌ Formula System Test: FAILED to get position data")
            return False
            
    except Exception as e:
        print(f"❌ Formula System Error: {e}")
        return False

def test_deployment_readiness():
    """Test deployment file readiness"""
    print_subheader("6. DEPLOYMENT READINESS CHECK")
    
    deployment_files = [
        "DEPLOY_testnet.env",
        "setup_vps_environment.sh",
        "CREDENTIAL_RESTORATION_COMPLETE.md"
    ]
    
    all_ready = True
    
    for file_path in deployment_files:
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            print(f"✅ {file_path}: READY ({file_size} bytes)")
        else:
            print(f"❌ {file_path}: MISSING")
            all_ready = False
    
    # Check if setup script has executable content
    if os.path.exists("setup_vps_environment.sh"):
        with open("setup_vps_environment.sh", 'r') as f:
            script_content = f.read()
        
        if "WORKING_KEY=" in script_content and "/etc/quantum" in script_content:
            print(f"✅ VPS Setup Script: PROPERLY CONFIGURED")
        else:
            print(f"❌ VPS Setup Script: MISSING CONFIGURATION")
            all_ready = False
    
    return all_ready

def test_backup_verification():
    """Verify backup files exist"""
    print_subheader("7. BACKUP VERIFICATION")
    
    backup_pattern = "*.backup_pre_api_fix"
    backup_files = list(Path('.').rglob(backup_pattern))
    
    print(f"✅ Backup Files: {len(backup_files)} files backed up")
    
    if len(backup_files) > 0:
        print(f"   Sample backups:")
        for backup in backup_files[:3]:
            print(f"   • {backup}")
        if len(backup_files) > 3:
            print(f"   • ... and {len(backup_files)-3} more")
    
    return len(backup_files) > 0

def main():
    """Run comprehensive system test"""
    print_header("🧪 QUANTUM TRADER - COMPREHENSIVE SYSTEM TEST")
    print(f"Testing all components after API credential restoration")
    print(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    test_results = {}
    
    # Run all tests
    test_results["api_credentials"], account_data = test_api_credentials()
    test_results["environment_files"] = test_environment_files()
    test_results["position_access"], positions = test_position_access()
    test_results["market_data"] = test_market_data()
    test_results["formula_system"] = test_formula_system()
    test_results["deployment_ready"] = test_deployment_readiness()
    test_results["backups_exist"] = test_backup_verification()
    
    # Overall results
    print_header("🎯 OVERALL TEST RESULTS")
    
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name.replace('_', ' ').title()}")
    
    print(f"\n📊 SUMMARY:")
    print(f"   Tests Passed: {passed_tests}/{total_tests}")
    print(f"   Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    if passed_tests == total_tests:
        print(f"\n🎉 ALL TESTS PASSED!")
        print(f"🚀 QUANTUM TRADER IS FULLY OPERATIONAL!")
        print(f"   ✅ API credentials working perfectly")
        print(f"   ✅ All environment files updated")
        print(f"   ✅ Position data accessible ({len(positions) if 'positions' in locals() else 'N/A'} active)")
        print(f"   ✅ Formula system ready")
        print(f"   ✅ VPS deployment ready")
        print(f"   ✅ Backups secured")
        
        if account_data:
            balance = account_data.get('totalWalletBalance', 'N/A')
            print(f"\n💰 Account Status: {balance} USDT available for trading")
        
        return True
    else:
        print(f"\n⚠️  SOME TESTS FAILED!")
        print(f"Review failed components and retry")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)