"""Comprehensive AI Modules Health Check"""

import sys
sys.path.insert(0, '/app')

print("🏥 QUANTUM TRADER - AI MODULES HEALTH CHECK")
print("=" * 80)

# 1. Check AI Engine availability
print("\n1️⃣ AI ENGINE STATUS:")
try:
    from ai_engine.ensemble_manager import EnsembleManager
    print("   ✅ EnsembleManager: Available")
    
    from ai_engine.agent import Agent
    print("   ✅ Agent: Available")
    
    from ai_engine.continuous_learning_manager import ContinuousLearningManager
    print("   ✅ ContinuousLearningManager: Available")
except Exception as e:
    print(f"   ❌ Error: {e}")

# 2. Check RL v2 modules
print("\n2️⃣ RL V2 MODULES:")
try:
    from backend.services.ai.rl_v2.meta_strategy_agent_v2 import MetaStrategyAgentV2
    print("   ✅ MetaStrategyAgentV2: Available")
    
    from backend.services.ai.rl_v2.position_sizing_agent_v2 import PositionSizingAgentV2
    print("   ✅ PositionSizingAgentV2: Available")
    
    from backend.services.ai.rl_v2.rl_subscriber_v2 import RLSubscriberV2
    print("   ✅ RLSubscriberV2: Available")
except Exception as e:
    print(f"   ❌ Error: {e}")

# 3. Check RL v3 modules
print("\n3️⃣ RL V3 MODULES:")
try:
    from backend.services.ai.rl_v3.training_daemon_v3 import TrainingDaemonV3
    print("   ✅ TrainingDaemonV3: Available")
    
    from backend.services.ai.rl_v3.rl_v3_subscriber import RLv3Subscriber
    print("   ✅ RLv3Subscriber: Available")
except Exception as e:
    print(f"   ❌ Error: {e}")

# 4. Check AI Trading Engine
print("\n4️⃣ AI TRADING ENGINE:")
try:
    from backend.services.ai.ai_trading_engine import AITradingEngine
    print("   ✅ AITradingEngine: Available")
except Exception as e:
    print(f"   ❌ Error: {e}")

# 5. Check Model Supervisor
print("\n5️⃣ MODEL SUPERVISOR:")
try:
    from backend.services.ai.model_supervisor import ModelSupervisor
    print("   ✅ ModelSupervisor: Available")
except Exception as e:
    print(f"   ❌ Error: {e}")

# 6. Check Orchestrator
print("\n6️⃣ ORCHESTRATOR:")
try:
    from backend.services.governance.orchestrator_policy import OrchestratorPolicy
    print("   ✅ OrchestratorPolicy: Available")
except Exception as e:
    print(f"   ❌ Error: {e}")

# 7. Check Position Sizing
print("\n7️⃣ POSITION SIZING:")
try:
    from backend.services.ai.rl_position_sizing_agent import RLPositionSizingAgent
    print("   ✅ RLPositionSizingAgent: Available")
    
    from backend.services.ai.trading_mathematician import TradingMathematician
    print("   ✅ TradingMathematician: Available")
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n" + "=" * 80)
print("✅ MODULE AVAILABILITY CHECK COMPLETE")
