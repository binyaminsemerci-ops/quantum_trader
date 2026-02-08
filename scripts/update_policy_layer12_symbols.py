#!/usr/bin/env python3
"""
Update Policy with 12 Layer 1/2 High-Volume Symbols

Overwrites quantum:policy:current with Layer 1/2 symbols for testnet trading.
Disables AI universe generator to prevent overwrite.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.policy_store import save_policy
import time
from datetime import datetime

# 12 Layer 1/2 High-Volume Symbols (testnet compatible, high liquidity)
LAYER12_SYMBOLS = [
    "ETHUSDT",     # Layer 1 - Ethereum
    "BTCUSDT",     # Layer 1 - Bitcoin
    "SOLUSDT",     # Layer 1 - Solana
    "XRPUSDT",     # Layer 1 - XRP
    "BNBUSDT",     # Layer 1 - BNB Chain
    "ADAUSDT",     # Layer 1 - Cardano
    "SUIUSDT",     # Layer 1 - Sui
    "LINKUSDT",    # Layer 2 - Chainlink
    "AVAXUSDT",    # Layer 1 - Avalanche
    "LTCUSDT",     # Layer 1 - Litecoin
    "DOTUSDT",     # Layer 1 - Polkadot
    "NEARUSDT"     # Layer 1 - Near Protocol
]

def update_policy():
    """Update policy with Layer 1/2 symbols"""
    
    print("[POLICY-UPDATE] Updating policy with 12 Layer 1/2 symbols...")
    print(f"[POLICY-UPDATE] Symbols: {', '.join(LAYER12_SYMBOLS)}")
    
    # Leverage configuration (conservative for testnet)
    leverage_by_symbol = {symbol: 3.0 for symbol in LAYER12_SYMBOLS}
    
    # Harvest parameters (from existing policy)
    harvest_params = {
        "T1_R": 1.8,
        "T2_R": 3.5,
        "T3_R": 5.8,
        "lock_R": 1.2,
        "be_plus_pct": 0.003,
        "cost_bps": 12.0,
        "k_regime_flip": 0.8,
        "k_sigma_spike": 0.6,
        "k_ts_drop": 0.4,
        "k_age_penalty": 0.2,
        "kill_threshold": 0.55
    }
    
    # Kill parameters (from existing policy)
    kill_params = {
        "trend_min": 0.25,
        "sigma_ref": 0.012,
        "sigma_spike_cap": 1.8,
        "ts_ref": 0.35,
        "ts_drop_cap": 0.45,
        "max_age_sec": 72000.0,
        "k_close_threshold": 0.62
    }
    
    # Save policy (valid for 1 hour instead of 5 minutes)
    success = save_policy(
        universe_symbols=LAYER12_SYMBOLS,
        leverage_by_symbol=leverage_by_symbol,
        harvest_params=harvest_params,
        kill_params=kill_params,
        valid_for_seconds=3600,  # 1 hour validity
        policy_version="1.0.0-layer12-override",
        generator="manual_layer12_v1",
        features_window="15m,1h",
        universe_hash="layer12_override",
        market="futures",
        stats_endpoint="fapi/v1/ticker/24hr",
        venue="binance-futures",
        exchange_info_endpoint="fapi/v1/exchangeInfo",
        ticker_24h_endpoint="fapi/v1/ticker/24hr"
    )
    
    if success:
        print(f"\n[POLICY-UPDATE] ✅ Policy updated successfully!")
        print(f"[POLICY-UPDATE] Universe: {len(LAYER12_SYMBOLS)} Layer 1/2 symbols")
        print(f"[POLICY-UPDATE] Leverage: 3.0x (uniform, conservative)")
        print(f"[POLICY-UPDATE] Valid for: 1 hour")
        print(f"[POLICY-UPDATE] Next expiry: {datetime.fromtimestamp(time.time() + 3600).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\n[POLICY-UPDATE] 🔥 IMPORTANT: Disable AI universe timer to prevent overwrite:")
        print(f"[POLICY-UPDATE]   systemctl stop quantum-ai-universe.timer")
        print(f"[POLICY-UPDATE]   systemctl disable quantum-ai-universe.timer")
        return True
    else:
        print(f"\n[POLICY-UPDATE] ❌ Failed to save policy to PolicyStore")
        return False


if __name__ == "__main__":
    try:
        success = update_policy()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n[POLICY-UPDATE] FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
