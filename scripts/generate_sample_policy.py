#!/usr/bin/env python3
"""
Generate sample AI policy for testing PolicyStore fail-closed behavior.

This simulates what an AI/RL model would produce.

Usage:
    python scripts/generate_sample_policy.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.policy_store import save_policy
import time


def generate_sample_policy():
    """Generate realistic AI policy"""
    
    # AI-selected universe (10 best symbols)
    universe_symbols = [
        "BTCUSDT",
        "ETHUSDT",
        "BNBUSDT",
        "SOLUSDT",
        "ADAUSDT",
        "DOTUSDT",
        "AVAXUSDT",
        "MATICUSDT",
        "LINKUSDT",
        "ATOMUSDT"
    ]
    
    # AI-decided leverage per symbol (varies 5x - 20x)
    leverage_by_symbol = {
        "BTCUSDT": 8.0,    # AI: Low volatility → higher leverage
        "ETHUSDT": 10.0,   # AI: Medium volatility
        "BNBUSDT": 12.0,   # AI: Strong trend → higher leverage
        "SOLUSDT": 6.0,    # AI: High volatility → lower leverage
        "ADAUSDT": 15.0,   # AI: Low volatility + strong signal
        "DOTUSDT": 7.0,    # AI: Medium risk
        "AVAXUSDT": 9.0,   # AI: Medium confidence
        "MATICUSDT": 11.0, # AI: Good trend strength
        "LINKUSDT": 8.0,   # AI: Conservative
        "ATOMUSDT": 10.0   # AI: Balanced
    }
    
    # AI-generated harvest parameters (NO hardcoded 2.0, 4.0, 6.0!)
    # These are formula parameters, not thresholds
    harvest_params = {
        # Partial close triggers (in R units)
        "T1_R": 1.8,       # AI: Tighter first partial (not 2.0!)
        "T2_R": 3.5,       # AI: Second partial (not 4.0!)
        "T3_R": 5.8,       # AI: Third partial (not 6.0!)
        
        # Profit lock
        "lock_R": 1.2,     # AI: Lock profit earlier (not 1.5!)
        "be_plus_pct": 0.003,  # AI: 0.3% above BE (not 0.2%!)
        
        # Cost estimation
        "cost_bps": 12.0,  # AI: Higher slippage estimate (not 10!)
        
        # Kill score weights
        "k_regime_flip": 0.8,   # AI: Lower regime weight (not 1.0!)
        "k_sigma_spike": 0.6,   # AI: Higher sigma weight (not 0.5!)
        "k_ts_drop": 0.4,       # AI: Lower ts weight (not 0.5!)
        "k_age_penalty": 0.2,   # AI: Lower age weight (not 0.3!)
        
        # Kill threshold
        "kill_threshold": 0.55  # AI: More aggressive close (not 0.6!)
    }
    
    # AI-generated kill score parameters
    kill_params = {
        "trend_min": 0.25,        # AI: Lower trend threshold (not 0.3!)
        "sigma_ref": 0.012,       # AI: Higher vol reference (not 0.01!)
        "sigma_spike_cap": 1.8,   # AI: Lower spike cap (not 2.0!)
        "ts_ref": 0.35,           # AI: Higher ts reference (not 0.3!)
        "ts_drop_cap": 0.45,      # AI: Lower drop cap (not 0.5!)
        "max_age_sec": 72000.0,   # AI: 20h max (not 24h!)
        "k_close_threshold": 0.62 # AI: Lower threshold (not 0.65!)
    }
    
    # Save policy (valid for 1 hour)
    success = save_policy(
        universe_symbols=universe_symbols,
        leverage_by_symbol=leverage_by_symbol,
        harvest_params=harvest_params,
        kill_params=kill_params,
        valid_for_seconds=3600,
        policy_version="1.0.0-ai-sample"
    )
    
    if success:
        print("\n✅ Sample AI policy generated successfully!")
        print(f"   Universe: {len(universe_symbols)} symbols")
        print(f"   Leverage range: {min(leverage_by_symbol.values()):.1f}x - {max(leverage_by_symbol.values()):.1f}x")
        print(f"   Harvest T1_R: {harvest_params['T1_R']:.1f} (AI decided, not 2.0!)")
        print(f"   Valid for: 1 hour")
        print(f"\n   Redis key: quantum:policy:current")
        print(f"   Expires: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time() + 3600))}")
    else:
        print("\n❌ Failed to generate policy!")
        sys.exit(1)


if __name__ == "__main__":
    generate_sample_policy()
