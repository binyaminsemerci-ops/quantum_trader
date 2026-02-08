#!/usr/bin/env python3
"""
COMPREHENSIVE AGENT VERIFICATION TEST
Tests all 4 AI prediction agents (XGBoost, LightGBM, N-HiTS, PatchTST)
with realistic scenarios and edge cases.
"""
import sys
sys.path.insert(0, '/home/qt/quantum_trader')

from ai_engine.agents.unified_agents import XGBoostAgent, LightGBMAgent, NHiTSAgent, PatchTSTAgent
import numpy as np

# Test scenarios with realistic feature values
test_scenarios = {
    "bullish_breakout": {
        'price_change': 0.025, 'rsi_14': 68.5, 'macd': 25.3, 'volume_ratio': 2.1,
        'momentum_10': 0.035, 'high_low_range': 0.04, 'volume_change': 0.45,
        'volume_ma_ratio': 1.8, 'ema_10': 96000, 'ema_20': 95200, 'ema_50': 93800,
        'ema_10_20_cross': 1.0, 'ema_10_50_cross': 1.0, 'volatility_20': 0.032,
        'macd_signal': 20.1, 'macd_hist': 5.2, 'bb_position': 0.85, 'momentum_20': 0.028
    },
    "bearish_breakdown": {
        'price_change': -0.032, 'rsi_14': 28.3, 'macd': -18.7, 'volume_ratio': 1.9,
        'momentum_10': -0.028, 'high_low_range': 0.038, 'volume_change': 0.35,
        'volume_ma_ratio': 1.5, 'ema_10': 94200, 'ema_20': 94800, 'ema_50': 95500,
        'ema_10_20_cross': -1.0, 'ema_10_50_cross': -1.0, 'volatility_20': 0.029,
        'macd_signal': -15.2, 'macd_hist': -3.5, 'bb_position': 0.15, 'momentum_20': -0.025
    },
    "sideways_neutral": {
        'price_change': 0.002, 'rsi_14': 48.5, 'macd': 1.2, 'volume_ratio': 1.05,
        'momentum_10': 0.003, 'high_low_range': 0.015, 'volume_change': 0.08,
        'volume_ma_ratio': 1.02, 'ema_10': 95000, 'ema_20': 94950, 'ema_50': 94800,
        'ema_10_20_cross': 0.0, 'ema_10_50_cross': 0.0, 'volatility_20': 0.012,
        'macd_signal': 0.8, 'macd_hist': 0.4, 'bb_position': 0.52, 'momentum_20': 0.004
    },
    "high_volatility": {
        'price_change': -0.018, 'rsi_14': 42.8, 'macd': -8.3, 'volume_ratio': 3.2,
        'momentum_10': -0.015, 'high_low_range': 0.055, 'volume_change': 0.85,
        'volume_ma_ratio': 2.4, 'ema_10': 94500, 'ema_20': 94600, 'ema_50': 94700,
        'ema_10_20_cross': -0.5, 'ema_10_50_cross': -0.5, 'volatility_20': 0.048,
        'macd_signal': -5.1, 'macd_hist': -3.2, 'bb_position': 0.25, 'momentum_20': -0.012
    },
}

def test_agent(agent_name, agent_class, scenarios):
    """Test a single agent with all scenarios"""
    print(f"\n{'='*70}")
    print(f"TESTING {agent_name}")
    print(f"{'='*70}")
    
    # Initialize agent
    print(f"\n[1] Initializing {agent_name}...")
    try:
        agent = agent_class()
        if not agent.is_ready():
            print(f"  ❌ Agent not ready!")
            return False
        print(f"  ✅ Agent ready: {type(agent.model).__name__}, features={len(agent.features)}")
    except Exception as e:
        print(f"  ❌ Initialization failed: {e}")
        return False
    
    # Test all scenarios
    print(f"\n[2] Testing scenarios...")
    results = []
    for scenario_name, features in scenarios.items():
        try:
            result = agent.predict('BTCUSDT', features)
            action = result['action']
            conf = result['confidence']
            results.append({
                'scenario': scenario_name,
                'action': action,
                'conf': conf,
                'success': True
            })
            emoji = "🟢" if action == "BUY" else "🔴" if action == "SELL" else "⚪"
            print(f"  {emoji} {scenario_name:20s} → {action:4s} (conf={conf:.3f})")
        except Exception as e:
            print(f"  ❌ {scenario_name:20s} → FAILED: {str(e)[:50]}")
            results.append({'scenario': scenario_name, 'success': False, 'error': str(e)})
    
    # Summary
    success_count = sum(1 for r in results if r.get('success', False))
    print(f"\n[3] Summary: {success_count}/{len(scenarios)} scenarios passed")
    
    # Check diversity
    if success_count > 0:
        actions = [r['action'] for r in results if r.get('success')]
        unique_actions = set(actions)
        print(f"    Action diversity: {len(unique_actions)}/3 classes ({', '.join(sorted(unique_actions))})")
        
        confs = [r['conf'] for r in results if r.get('success')]
        print(f"    Confidence range: {min(confs):.3f} - {max(confs):.3f}")
    
    return success_count == len(scenarios)

def main():
    print("="*70)
    print("COMPREHENSIVE AI AGENT VERIFICATION")
    print("Testing: XGBoost, LightGBM, N-HiTS, PatchTST")
    print("Scenarios: Bullish, Bearish, Sideways, High-Volatility")
    print("="*70)
    
    agents = [
        ("XGBoost Agent", XGBoostAgent),
        ("LightGBM Agent", LightGBMAgent),
        ("N-HiTS Agent", NHiTSAgent),
        ("PatchTST Agent", PatchTSTAgent),
    ]
    
    results = {}
    for agent_name, agent_class in agents:
        try:
            success = test_agent(agent_name, agent_class, test_scenarios)
            results[agent_name] = "✅ PASSED" if success else "⚠️ PARTIAL"
        except Exception as e:
            print(f"\n❌ {agent_name} CRASHED: {e}")
            results[agent_name] = f"❌ FAILED: {str(e)[:50]}"
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL VERIFICATION SUMMARY")
    print("="*70)
    for agent_name, status in results.items():
        print(f"  {agent_name:20s} {status}")
    
    all_passed = all("PASSED" in status for status in results.values())
    print(f"\n{'🎉 ALL AGENTS OPERATIONAL!' if all_passed else '⚠️ Some agents need attention'}")
    print("="*70)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    import os
    os.chdir("/home/qt/quantum_trader")
    sys.exit(main())
