#!/usr/bin/env python3
"""
Learning Systems Diagnostic
Check status of all AI training and learning systems
"""

print("🧠 QUANTUM TRADER - LEARNING SYSTEMS STATUS")
print("=" * 70)

# 1. Meta-Strategy Selector
print("\n1️⃣ META-STRATEGY SELECTOR:")
try:
    from backend.services.meta_strategy_integration import get_meta_strategy_integration
    meta = get_meta_strategy_integration()
    if meta and meta.enabled:
        print("   ✅ ENABLED - Real-time strategy reward tracking")
    else:
        print("   ⚠️  Available but DISABLED")
except Exception as e:
    print(f"   ❌ Error: {e}")

# 2. Position Intelligence Layer (PIL)
print("\n2️⃣ POSITION INTELLIGENCE LAYER (PIL):")
try:
    from backend.services.position_intelligence_layer import get_pil
    pil = get_pil()
    if pil:
        print("   ✅ ENABLED - Classifies WINNERS/LOSERS")
    else:
        print("   ⚠️  Not active")
except Exception as e:
    print(f"   ❌ Error: {e}")

# 3. Profit Amplification Layer (PAL)
print("\n3️⃣ PROFIT AMPLIFICATION LAYER (PAL):")
try:
    from backend.services.profit_amplification import get_profit_amplification
    pal = get_profit_amplification()
    if pal:
        print("   ✅ ENABLED - Analyzes amplification opportunities")
    else:
        print("   ⚠️  Not active")
except Exception as e:
    print(f"   ❌ Error: {e}")

# 4. RL v3 Environment with TP Accuracy Tracking
print("\n4️⃣ RL v3 WITH TP ACCURACY REWARD:")
try:
    from backend.domains.learning.rl_v3.env_v3 import TradingEnvV3
    from backend.domains.learning.rl_v3.config_v3 import DEFAULT_CONFIG

    env = TradingEnvV3(DEFAULT_CONFIG)
    has_tp = hasattr(env, 'tp_zone_accuracy')
    print(f"   ✅ Environment ready")
    print(f"   📈 TP accuracy tracking: {has_tp}")
    if has_tp:
        print("   💡 New reward function active (+5.0 max bonus)")
except Exception as e:
    print(f"   ❌ Error: {e}")

# 5. Continuous Learning System
print("\n5️⃣ CONTINUOUS LEARNING:")
try:
    try:
        from backend.services.learning import ContinuousLearningSystem, get_continuous_learning
    except Exception:
        from backend.services.learning.continuous_learning_system import (  # Backward compatibility
            ContinuousLearningSystem,  # type: ignore
            get_continuous_learning  # type: ignore
        )

    # Prefer factory to keep singleton behavior
    cl = get_continuous_learning()
    print("   ✅ Module available")
    print("   ℹ️  Auto-triggers on position close events")
    print(f"   🛰️  Instance created: {cl is not None}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# 6. Model Status
print("\n6️⃣ AI MODELS:")
from pathlib import Path
model_path = Path('/app/ai_engine/models')
if model_path.exists():
    models = list(model_path.glob('*.pkl'))
    print(f"   ✅ {len(models)} trained models found")
    
    # Latest model
    if models:
        latest = max(models, key=lambda x: x.stat().st_mtime)
        from datetime import datetime
        age = datetime.now() - datetime.fromtimestamp(latest.stat().st_mtime)
        print(f"   📅 Latest: {latest.name}")
        print(f"      Age: {age.days} days, {age.seconds//3600} hours")
else:
    print("   ⚠️  Model directory not found")

# 7. TP Performance Tracker
print("\n7️⃣ TP PERFORMANCE TRACKING:")
try:
    from backend.services.monitoring.tp_performance_tracker import get_tp_performance_tracker
    tracker = get_tp_performance_tracker()
    summary = tracker.get_summary()
    print(f"   ✅ Tracker active")
    print(f"   📊 Tracking: {summary['tracked_pairs']} strategy/symbol pairs")
    print(f"   🎯 Total TP attempts: {summary['total_attempts']}")
    if summary['total_attempts'] > 0:
        print(f"   📈 Hit rate: {summary['overall_hit_rate']:.1%}")
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n" + "=" * 70)
print("\n💡 RECOMMENDATIONS:")
print("   • RL v3 retraining: Run 'python activate_retraining_system.py'")
print("   • Monitor TP metrics: Check /app/tmp/tp_metrics.json")
print("   • Learning is EVENT-DRIVEN: Triggers on position closes")
print("=" * 70)
