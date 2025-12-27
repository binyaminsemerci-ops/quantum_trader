"""Check Continuous Learning and Training Activity"""

import sys
sys.path.insert(0, '/app')

import json
import subprocess
import re
from datetime import datetime, timedelta
from pathlib import Path

print("🧠 CONTINUOUS LEARNING & TRAINING CHECK")
print("=" * 80)

# Check RL v3 training activity
print("\n1️⃣ RL V3 TRAINING ACTIVITY:")
print("-" * 80)

result = subprocess.run(
    ['docker', 'logs', 'quantum_backend', '--tail', '1000'],
    capture_output=True, text=True
)
logs = result.stdout + result.stderr

# Check for training episodes
training_episodes = re.findall(r'Episode (\d+).*reward.*?(-?[\d.]+)', logs)
if training_episodes:
    print(f"   ✅ Found {len(training_episodes)} training episodes")
    if len(training_episodes) > 0:
        last_ep = training_episodes[-1]
        print(f"   📊 Latest: Episode {last_ep[0]}, Reward: {last_ep[1]}")
else:
    print("   ⚠️  No training episodes found in recent logs")

# Check for model updates
model_updates = re.findall(r'Model.*updated|Weights.*saved|Training.*complete', logs, re.IGNORECASE)
print(f"\n   Model updates: {len(model_updates)}")

# Check for learning rate adjustments
lr_adjustments = re.findall(r'learning.*rate|LR.*adjust', logs, re.IGNORECASE)
if lr_adjustments:
    print(f"   Learning rate adjustments: {len(lr_adjustments)}")

print("\n2️⃣ ENSEMBLE LEARNING:")
print("-" * 80)

# Check for strategy performance updates
strategy_signals = re.findall(r'Strategy (loadtest_\d+).*signal', logs)
if strategy_signals:
    strategies = set(strategy_signals)
    print(f"   ✅ Active strategies: {', '.join(sorted(strategies))}")
    print(f"   📊 Total signals generated: {len(strategy_signals)}")
else:
    print("   ⚠️  No strategy signals found")

# Check for model retraining
retraining = re.findall(r'Retrain|Model.*update|Performance.*evaluation', logs, re.IGNORECASE)
print(f"\n   Retraining events: {len(retraining)}")

print("\n3️⃣ POSITION SIZING LEARNING:")
print("-" * 80)

# Check for position sizing adjustments
sizing_events = re.findall(r'Position.*size|Sizing.*agent|RL.*sizing', logs, re.IGNORECASE)
print(f"   Position sizing events: {len(sizing_events)}")

# Check for risk adjustments
risk_adjustments = re.findall(r'Risk.*adjust|Exposure.*update', logs, re.IGNORECASE)
print(f"   Risk adjustments: {len(risk_adjustments)}")

print("\n4️⃣ MODEL PERFORMANCE TRACKING:")
print("-" * 80)

# Check for performance metrics
win_rate = re.findall(r'win.*rate.*?(\d+\.?\d*)%', logs, re.IGNORECASE)
if win_rate:
    print(f"   Win rate mentions: {len(win_rate)}")

profit_factor = re.findall(r'profit.*factor.*?(\d+\.?\d*)', logs, re.IGNORECASE)
if profit_factor:
    print(f"   Profit factor mentions: {len(profit_factor)}")

# Check for model evaluation
evaluations = re.findall(r'Evaluat|Validat|Test.*performance', logs, re.IGNORECASE)
print(f"   Model evaluations: {len(evaluations)}")

print("\n5️⃣ LEARNING DATA SOURCES:")
print("-" * 80)

# Check if learning from live trades
live_feedback = re.findall(r'Trade.*feedback|Position.*closed|PnL.*recorded', logs, re.IGNORECASE)
print(f"   Live trade feedback: {len(live_feedback)}")

# Check market data updates
market_updates = re.findall(r'Market data|Price.*update|Candle.*fetch', logs, re.IGNORECASE)
print(f"   Market data updates: {len(market_updates)}")

print("\n" + "=" * 80)
print("✅ CONTINUOUS LEARNING CHECK COMPLETE")
