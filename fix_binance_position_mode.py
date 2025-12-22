#!/usr/bin/env python3
"""
Fix Binance Position Mode for Position Monitor

Changes Binance Futures position mode to One-Way Mode (dualSidePosition=false)
This allows Position Monitor to place TP/SL orders correctly.
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
        print("Set them in your environment or .env file")
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

def set_position_mode(api_key, api_secret, dual_side_position=False):
    """
    Set position mode
    
    Args:
        dual_side_position: False = One-Way Mode, True = Hedge Mode
    """
    timestamp = int(time.time() * 1000)
    params = {
        'dualSidePosition': 'true' if dual_side_position else 'false',
        'timestamp': timestamp,
        'recvWindow': 10000
    }
    params['signature'] = create_signature(params, api_secret)
    
    headers = {'X-MBX-APIKEY': api_key}
    url = f"{BASE_URL}/fapi/v1/positionSide/dual"
    
    mode = "Hedge Mode (Dual-Side)" if dual_side_position else "One-Way Mode"
    
    try:
        print(f"\n🔧 Changing position mode to: {mode}...")
        response = requests.post(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        
        if data.get('code') == 200 or 'msg' in data:
            print(f"✅ Position mode changed successfully!")
            print(f"   New mode: {mode}")
            print(f"   Response: {data}")
            return True
        else:
            print(f"⚠️  Unexpected response: {data}")
            return False
            
    except requests.exceptions.HTTPError as e:
        # Check if error is "No need to change position side"
        if e.response.status_code == 400:
            error_data = e.response.json()
            if error_data.get('code') == -4059:
                print(f"✅ Position mode already set to {mode}")
                return True
        
        print(f"❌ Failed to change position mode: {e}")
        if hasattr(e, 'response') and e.response:
            print(f"   Response: {e.response.text}")
        return False
    except Exception as e:
        print(f"❌ Failed to change position mode: {e}")
        return False

def main():
    print("=" * 60)
    print("🔧 BINANCE POSITION MODE FIXER")
    print("=" * 60)
    print()
    
    # Get credentials
    api_key, api_secret = get_binance_credentials()
    print(f"✅ Binance API credentials loaded")
    print(f"   API Key: {api_key[:8]}...")
    print()
    
    # Check current mode
    print("📊 Checking current position mode...")
    current_dual_side = get_current_position_mode(api_key, api_secret)
    print()
    
    # Determine if we need to change
    target_mode = False  # One-Way Mode (recommended for Position Monitor)
    
    if current_dual_side == target_mode:
        print("✅ Position mode is already correct!")
        print("   No changes needed.")
        print()
        print("🎯 Position Monitor should now be able to place TP/SL orders.")
        return
    
    # Change mode
    print("⚠️  Position mode needs to be changed")
    print(f"   Current: {'Hedge Mode' if current_dual_side else 'One-Way Mode'}")
    print(f"   Target:  One-Way Mode (dualSidePosition=false)")
    print()
    
    success = set_position_mode(api_key, api_secret, dual_side_position=target_mode)
    
    if success:
        print()
        print("=" * 60)
        print("✅ SUCCESS!")
        print("=" * 60)
        print()
        print("🎯 Position Monitor can now place TP/SL orders!")
        print()
        print("📊 To verify, check Position Monitor logs:")
        print("   docker logs -f quantum_backend | grep POSITION-MONITOR")
        print()
    else:
        print()
        print("=" * 60)
        print("❌ FAILED")
        print("=" * 60)
        print()
        print("Please change position mode manually:")
        print("1. Go to: https://testnet.binancefuture.com")
        print("2. Settings → Preferences → Position Mode")
        print("3. Select: One-Way Mode")
        print()

if __name__ == "__main__":
    main()
