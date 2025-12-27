#!/usr/bin/env python3
"""
Enable Hedge Mode on Binance Testnet
Sets dualSidePosition=true to allow LONG and SHORT simultaneously
"""

import os
import sys
import time
import hmac
import hashlib
import requests
from urllib.parse import urlencode

# Binance Testnet API
BASE_URL = "https://testnet.binancefuture.com"

def get_binance_credentials():
    """Get Binance API credentials from environment"""
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    
    if not api_key or not api_secret:
        print("❌ Error: BINANCE_API_KEY and BINANCE_API_SECRET must be set")
        sys.exit(1)
    
    return api_key, api_secret

def create_signature(params, api_secret):
    """Create HMAC SHA256 signature"""
    query_string = urlencode(params)
    signature = hmac.new(
        api_secret.encode('utf-8'),
        query_string.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    return signature

def get_current_position_mode(api_key, api_secret):
    """Check current position mode"""
    timestamp = int(time.time() * 1000)
    params = {
        'timestamp': timestamp,
        'recvWindow': 10000
    }
    params['signature'] = create_signature(params, api_secret)
    
    headers = {'X-MBX-APIKEY': api_key}
    url = f"{BASE_URL}/fapi/v1/positionSide/dual"
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        
        dual_side = data.get('dualSidePosition', False)
        mode = "Hedge Mode (Dual-Side)" if dual_side else "One-Way Mode"
        
        print(f"✅ Current position mode: {mode}")
        print(f"   dualSidePosition: {dual_side}")
        
        return dual_side
    except Exception as e:
        print(f"❌ Failed to get position mode: {e}")
        if hasattr(e, 'response') and e.response:
            print(f"   Response: {e.response.text}")
        sys.exit(1)

def enable_hedge_mode(api_key, api_secret):
    """Enable Hedge Mode (dualSidePosition=true)"""
    timestamp = int(time.time() * 1000)
    params = {
        'dualSidePosition': 'true',
        'timestamp': timestamp,
        'recvWindow': 10000
    }
    params['signature'] = create_signature(params, api_secret)
    
    headers = {'X-MBX-APIKEY': api_key}
    url = f"{BASE_URL}/fapi/v1/positionSide/dual"
    
    try:
        print(f"\n🔧 Enabling Hedge Mode...")
        response = requests.post(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        
        if data.get('code') == 200 or 'msg' in data:
            print(f"✅ Hedge Mode enabled successfully!")
            print(f"   Response: {data}")
            return True
        else:
            print(f"⚠️  Unexpected response: {data}")
            return False
            
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 400:
            error_data = e.response.json()
            if error_data.get('code') == -4059:
                print(f"✅ Hedge Mode already enabled!")
                return True
        
        print(f"❌ Failed to enable Hedge Mode: {e}")
        if hasattr(e, 'response') and e.response:
            print(f"   Response: {e.response.text}")
        return False
    except Exception as e:
        print(f"❌ Failed to enable Hedge Mode: {e}")
        return False

def main():
    print("=" * 60)
    print("🔧 ENABLE HEDGE MODE ON BINANCE TESTNET")
    print("=" * 60)
    print()
    
    # Get credentials
    api_key, api_secret = get_binance_credentials()
    print(f"API Key: {api_key[:10]}...")
    print()
    
    # Check current mode
    print("Step 1: Checking current position mode...")
    is_hedge = get_current_position_mode(api_key, api_secret)
    
    if is_hedge:
        print("\n✅ Already in Hedge Mode - no changes needed!")
        return
    
    # Enable Hedge Mode
    print("\nStep 2: Enabling Hedge Mode...")
    success = enable_hedge_mode(api_key, api_secret)
    
    if success:
        print("\n✅ Hedge Mode enabled successfully!")
        print("\nVerifying...")
        time.sleep(1)
        get_current_position_mode(api_key, api_secret)
    else:
        print("\n❌ Failed to enable Hedge Mode")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("✅ DONE! Hedge Mode is now active")
    print("=" * 60)

if __name__ == "__main__":
    main()
