#!/usr/bin/env python3
"""
Apply Profit Maximization Strategy

Quick script to update .env with aggressive profit targets.

Author: Quantum Trader Team
Date: 2025-11-26
"""

import os
from pathlib import Path


STRATEGIES = {
    "ultra": {
        "name": "ULTRA AGGRESSIVE (Recommended for AI)",
        "description": "3x-8x risk targets, tight stop",
        "settings": {
            "TP_ATR_MULT_SL": "1.0",
            "TP_ATR_MULT_TP1": "3.0",
            "TP_ATR_MULT_TP2": "5.0",
            "TP_ATR_MULT_TP3": "8.0",
        },
        "expected": "6-8% margin profit per win (180-240% with 30x leverage)"
    },
    "balanced": {
        "name": "BALANCED AGGRESSIVE",
        "description": "4x-10x risk targets, wider stop",
        "settings": {
            "TP_ATR_MULT_SL": "1.5",
            "TP_ATR_MULT_TP1": "4.0",
            "TP_ATR_MULT_TP2": "6.0",
            "TP_ATR_MULT_TP3": "10.0",
        },
        "expected": "8-10% margin profit per win (240-300% with 30x leverage)"
    },
    "moderate": {
        "name": "MODERATE AGGRESSIVE",
        "description": "2.5x-6x risk targets, tight stop",
        "settings": {
            "TP_ATR_MULT_SL": "1.0",
            "TP_ATR_MULT_TP1": "2.5",
            "TP_ATR_MULT_TP2": "4.0",
            "TP_ATR_MULT_TP3": "6.0",
        },
        "expected": "4-6% margin profit per win (120-180% with 30x leverage)"
    }
}


def read_env_file(path: Path) -> dict:
    """Read .env file into dict."""
    env_vars = {}
    if path.exists():
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip()
    return env_vars


def write_env_file(path: Path, env_vars: dict):
    """Write dict to .env file."""
    with open(path, 'w') as f:
        f.write("# Quantum Trader Configuration\n")
        f.write("# Updated with Profit Maximization Strategy\n\n")
        
        # Group by prefix
        groups = {}
        for key, value in sorted(env_vars.items()):
            prefix = key.split('_')[0]
            if prefix not in groups:
                groups[prefix] = []
            groups[prefix].append((key, value))
        
        # Write grouped
        for prefix, items in groups.items():
            f.write(f"# {prefix} Settings\n")
            for key, value in items:
                f.write(f"{key}={value}\n")
            f.write("\n")


def apply_strategy(strategy_name: str):
    """Apply profit maximization strategy to .env file."""
    
    if strategy_name not in STRATEGIES:
        print(f"❌ Unknown strategy: {strategy_name}")
        print(f"Available: {', '.join(STRATEGIES.keys())}")
        return False
    
    strategy = STRATEGIES[strategy_name]
    env_path = Path(__file__).parent / '.env'
    
    print(f"\n🚀 APPLYING PROFIT MAXIMIZATION STRATEGY")
    print(f"=" * 60)
    print(f"Strategy: {strategy['name']}")
    print(f"Description: {strategy['description']}")
    print(f"Expected: {strategy['expected']}")
    print(f"=" * 60)
    
    # Read current .env
    print(f"\n📖 Reading {env_path}...")
    env_vars = read_env_file(env_path)
    
    # Show current values
    print(f"\n📊 CURRENT VALUES:")
    for key in strategy['settings'].keys():
        current = env_vars.get(key, 'NOT SET')
        print(f"   {key}: {current}")
    
    # Update values
    print(f"\n✏️  NEW VALUES:")
    for key, value in strategy['settings'].items():
        old = env_vars.get(key, 'NOT SET')
        env_vars[key] = value
        print(f"   {key}: {old} → {value}")
    
    # Write updated .env
    print(f"\n💾 Writing updated {env_path}...")
    write_env_file(env_path, env_vars)
    
    print(f"\n✅ SUCCESS! Profit maximization strategy applied!")
    print(f"\n📋 NEXT STEPS:")
    print(f"   1. Restart backend: sudo systemctl restart quantum-backend.service")
    print(f"   2. Verify settings: python verify_backend_tpsl.py")
    print(f"   3. Monitor results: python quick_status.py")
    
    return True


def show_comparison():
    """Show comparison of all strategies."""
    print(f"\n🎯 PROFIT MAXIMIZATION STRATEGIES\n")
    print(f"{'Strategy':<20} {'SL':<6} {'TP1':<6} {'TP2':<6} {'TP3':<6} {'Expected Profit'}")
    print(f"{'-'*80}")
    
    # Current (defensive)
    print(f"{'CURRENT (Defensive)':<20} {'1.0R':<6} {'1.5R':<6} {'2.5R':<6} {'4.0R':<6} 2-3% margin")
    print()
    
    # New strategies
    for key, strat in STRATEGIES.items():
        settings = strat['settings']
        name = strat['name'][:19]
        sl = settings['TP_ATR_MULT_SL'] + 'R'
        tp1 = settings['TP_ATR_MULT_TP1'] + 'R'
        tp2 = settings['TP_ATR_MULT_TP2'] + 'R'
        tp3 = settings['TP_ATR_MULT_TP3'] + 'R'
        exp = strat['expected'][:25]
        print(f"{name:<20} {sl:<6} {tp1:<6} {tp2:<6} {tp3:<6} {exp}")


if __name__ == '__main__':
    import sys
    
    print(f"\n🚀 QUANTUM TRADER - PROFIT MAXIMIZATION")
    
    if len(sys.argv) < 2:
        show_comparison()
        print(f"\n💡 USAGE:")
        print(f"   python apply_profit_maximization.py <strategy>")
        print(f"\n   Available strategies:")
        for key, strat in STRATEGIES.items():
            print(f"      {key:<10} - {strat['name']}")
        print(f"\n   Example:")
        print(f"      python apply_profit_maximization.py ultra")
        sys.exit(0)
    
    strategy_name = sys.argv[1].lower()
    
    if strategy_name == 'compare':
        show_comparison()
        sys.exit(0)
    
    success = apply_strategy(strategy_name)
    sys.exit(0 if success else 1)
