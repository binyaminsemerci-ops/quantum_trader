"""Check Runtime Status of AI Modules"""

import sys
sys.path.insert(0, '/app')

print("🔍 AI MODULES RUNTIME STATUS CHECK")
print("=" * 80)

# Check logs for active modules
import subprocess
import re

def check_module_status(module_name, log_patterns):
    """Check if module is active in logs"""
    result = subprocess.run(
        ['docker', 'logs', 'quantum_backend', '--tail', '500'],
        capture_output=True, text=True
    )
    logs = result.stdout + result.stderr
    
    found = False
    for pattern in log_patterns:
        if re.search(pattern, logs, re.IGNORECASE):
            found = True
            break
    
    return found

print("\n1️⃣ CHECKING ACTIVE MODULES IN LOGS:")
print("-" * 80)

modules = {
    "RLv3 Training Daemon": [
        r"\[v3\].*Training Daemon",
        r"RLv3.*started",
        r"Training.*v3"
    ],
    "RLv3 Subscriber": [
        r"RLv3 Subscriber",
        r"RL.*v3.*subscribed"
    ],
    "RL Subscriber v2": [
        r"RL Subscriber v2",
        r"RLv2.*started",
        r"Meta.*strategy.*agent"
    ],
    "Ensemble Manager": [
        r"EnsembleManager",
        r"ensemble.*initialized",
        r"Strategy.*loadtest"
    ],
    "Continuous Learning": [
        r"Continuous.*learning",
        r"Learning.*manager",
        r"Model.*training"
    ],
    "AI Trading Engine": [
        r"AITradingEngine",
        r"get_trading_signals",
        r"Strategy.*generated"
    ],
    "Orchestrator Policy": [
        r"Orchestrator.*LIVE",
        r"orchestrator.*enforcing",
        r"OrchestratorPolicy"
    ],
    "Position Sizing": [
        r"Position.*sizing",
        r"TradingMathematician",
        r"RL.*sizing"
    ]
}

for module, patterns in modules.items():
    status = "✅ ACTIVE" if check_module_status(module, patterns) else "❌ INACTIVE"
    print(f"{module:30} {status}")

print("\n" + "=" * 80)
print("✅ RUNTIME STATUS CHECK COMPLETE")
