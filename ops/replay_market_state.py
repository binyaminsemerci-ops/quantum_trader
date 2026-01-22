#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P0 MarketState Replay Script - SPEC v1.0
Demonstrates market state calculations on synthetic/historical data
"""

import sys
import argparse
import numpy as np
from pathlib import Path

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_engine.market_state import MarketState


def generate_trend(n=300, drift=0.002, vol=0.01):
    """Generate trending price series"""
    np.random.seed(42)
    returns = np.random.randn(n) * vol + drift
    prices = 100 * np.exp(np.cumsum(returns))
    return prices


def generate_mean_revert(n=300, speed=0.05, vol=0.015):
    """Generate mean-reverting price series (Ornstein-Uhlenbeck)"""
    np.random.seed(42)
    prices = np.zeros(n)
    prices[0] = 100.0
    mean_price = 100.0
    
    for i in range(1, n):
        reversion = speed * (mean_price - prices[i-1])
        shock = np.random.randn() * vol
        prices[i] = prices[i-1] + reversion + shock
    
    return prices


def generate_chop(n=300, vol=0.02):
    """Generate choppy/sideways series"""
    np.random.seed(42)
    returns = np.random.randn(n) * vol
    prices = 100 + np.cumsum(returns)
    return prices


def replay_synthetic(regime='trend'):
    """Run synthetic scenario"""
    print(f"\n{'='*80}")
    print(f"P0 MarketState Replay - Synthetic: {regime.upper()}")
    print(f"{'='*80}\n")
    
    # Generate prices
    if regime == 'trend':
        prices = generate_trend()
    elif regime == 'mean_revert':
        prices = generate_mean_revert()
    elif regime == 'chop':
        prices = generate_chop()
    else:
        print(f"ERROR: Unknown regime '{regime}'")
        sys.exit(1)
    
    print(f"Generated {len(prices)} prices (range: ${prices.min():.2f} - ${prices.max():.2f})\n")
    
    # Compute state
    ms = MarketState()
    state = ms.get_state("SYNTHETIC", prices)
    
    if not state:
        print("ERROR: Could not compute state (insufficient data)")
        sys.exit(1)
    
    # Print results
    print(f"RESULTS:")
    print(f"{'='*80}")
    print(f"Sigma (volatility): {state['sigma']:.8f}")
    print(f"Mu (trend):         {state['mu']:.8f}")
    print(f"TS (trend strength): {state['ts']:.6f}")
    
    print(f"\nRegime Probabilities:")
    for regime_name, prob in sorted(state['regime_probs'].items(), key=lambda x: -x[1]):
        bar = '#' * int(prob * 50)
        print(f"  {regime_name:5s}: {prob:6.2%} {bar}")
    
    print(f"\nFeatures:")
    print(f"  dp (directional persistence): {state['features']['dp']:.4f}")
    print(f"  vr (variance ratio):          {state['features']['vr']:.4f}")
    print(f"  spike_proxy:                  {state['features']['spike_proxy']:.4f}")
    
    print(f"\nWindows:")
    print(f"  returns_n:      {state['windows']['returns_n']}")
    print(f"  trend_windows:  {state['windows']['trend_windows']}")
    print(f"  regime_window:  {state['windows']['regime_window']}")
    
    # Interpretation
    print(f"\n{'='*80}")
    dominant = max(state['regime_probs'].items(), key=lambda x: x[1])
    print(f"INTERPRETATION: Dominant regime is {dominant[0].upper()} ({dominant[1]:.1%})")
    
    if dominant[0] == 'trend':
        if state['mu'] > 0:
            print("  → Uptrend detected")
            print("  → Consider momentum strategies")
        else:
            print("  → Downtrend detected")
            print("  → Consider short positions or defensive stance")
    elif dominant[0] == 'mr':
        print("  → Mean-reverting market")
        print("  → Consider counter-trend strategies")
    else:  # chop
        print("  → Choppy/sideways market (no clear direction)")
        print("  → Consider staying out or using tight stops")
    
    print(f"{'='*80}\n")


def replay_file(filepath):
    """Run replay from CSV file"""
    print(f"\n{'='*80}")
    print(f"P0 MarketState Replay - File: {filepath}")
    print(f"{'='*80}\n")
    
    try:
        prices = np.loadtxt(filepath, delimiter=',')
        if prices.ndim != 1:
            prices = prices.flatten()
    except Exception as e:
        print(f"ERROR: Could not load file: {e}")
        sys.exit(1)
    
    print(f"Loaded {len(prices)} prices (range: ${prices.min():.2f} - ${prices.max():.2f})\n")
    
    # Compute state
    ms = MarketState()
    state = ms.get_state("FILE", prices)
    
    if not state:
        print("ERROR: Could not compute state (insufficient data)")
        sys.exit(1)
    
    # Print results
    print(f"RESULTS:")
    print(f"{'='*80}")
    print(f"Sigma: {state['sigma']:.8f}")
    print(f"Mu:    {state['mu']:.8f}")
    print(f"TS:    {state['ts']:.6f}")
    print(f"\nRegime Probabilities:")
    for regime_name, prob in sorted(state['regime_probs'].items(), key=lambda x: -x[1]):
        bar = '#' * int(prob * 50)
        print(f"  {regime_name:5s}: {prob:6.2%} {bar}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="P0 MarketState Replay Script",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--synthetic', action='store_true',
                        help='Generate synthetic price series')
    parser.add_argument('--regime', type=str, default='trend',
                        choices=['trend', 'mean_revert', 'chop'],
                        help='Synthetic regime (default: trend)')
    parser.add_argument('--file', type=str,
                        help='Path to CSV file with prices (one per line)')
    
    args = parser.parse_args()
    
    if args.synthetic:
        replay_synthetic(args.regime)
    elif args.file:
        replay_file(args.file)
    else:
        print("ERROR: Must specify either --synthetic or --file")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
