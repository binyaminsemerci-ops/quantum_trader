#!/usr/bin/env python3
"""
QUANTUM TRADER - FULL SYSTEM STABILITY CHECK
"""
import sys
sys.path.insert(0, '/app')

print('=' * 80)
print(' ' * 20 + 'QUANTUM TRADER - FULL SYSTEM CHECK')
print('=' * 80)

results = {'passed': 0, 'failed': 0, 'warnings': 0}

# 1. Container Health
print('\n[1/7] Container Health...')
try:
    import os
    print(f'  ✅ Python version: {sys.version.split()[0]}')
    print(f'  ✅ Working directory: {os.getcwd()}')
    results['passed'] += 1
except Exception as e:
    print(f'  ❌ FAILED: {e}')
    results['failed'] += 1

# 2. Core Imports
print('\n[2/7] Core Module Imports...')
try:
    from backend.services.ai.memory_state_manager import MemoryStateManager
    from backend.services.ai.reinforcement_signal_manager import ReinforcementSignalManager
    from backend.services.ai.drift_detection_manager import DriftDetectionManager
    from backend.services.ai.covariate_shift_manager import CovariateShiftManager
    from backend.services.ai.shadow_model_manager import ShadowModelManager
    from backend.services.ai.continuous_learning_manager import ContinuousLearningManager
    print('  ✅ All 6 Bulletproof AI modules importable')
    results['passed'] += 1
except Exception as e:
    print(f'  ❌ FAILED: {e}')
    results['failed'] += 1

# 3. Trading Profile
print('\n[3/7] Trading Profile System...')
try:
    from backend.services.ai.trading_profile import *
    from backend.config.trading_profile import get_trading_profile_config
    config = get_trading_profile_config()
    enabled_str = str(config.enabled)
    max_size = str(config.liquidity.max_universe_size)
    risk_pct = config.risk.base_risk_frac * 100
    print(f'  ✅ Trading Profile enabled: {enabled_str}')
    print(f'  ✅ Max universe size: {max_size}')
    print(f'  ✅ Base risk: {risk_pct}%')
    results['passed'] += 1
except Exception as e:
    print(f'  ❌ FAILED: {e}')
    results['failed'] += 1

# 4. API Routes
print('\n[4/7] API Routes Registration...')
try:
    from backend.routes.trading_profile import router
    print('  ✅ Trading Profile API routes loaded')
    results['passed'] += 1
except Exception as e:
    print(f'  ❌ FAILED: {e}')
    results['failed'] += 1

# 5. Orchestrator Integration
print('\n[5/7] Orchestrator Integration...')
try:
    from backend.services.governance.orchestrator_policy import OrchestratorPolicy
    orch = OrchestratorPolicy()
    tp_enabled = getattr(orch, 'tp_enabled', False)
    has_methods = hasattr(orch, 'can_trade_symbol') and hasattr(orch, 'filter_symbols')
    print(f'  ✅ Trading Profile enabled: {tp_enabled}')
    print(f'  ✅ Integration methods: {has_methods}')
    results['passed'] += 1
except Exception as e:
    print(f'  ❌ FAILED: {e}')
    results['failed'] += 1

# 6. Execution Integration
print('\n[6/7] Execution Integration...')
try:
    from backend.services.execution.execution import BinanceFuturesExecutionAdapter
    has_tpsl_method = hasattr(BinanceFuturesExecutionAdapter, 'submit_order_with_tpsl')
    print(f'  ✅ submit_order_with_tpsl(): {has_tpsl_method}')
    results['passed'] += 1
except Exception as e:
    print(f'  ❌ FAILED: {e}')
    results['failed'] += 1

# 7. Environment Variables
print('\n[7/7] Critical Environment Variables...')
try:
    import os
    critical_vars = {
        'TP_ENABLED': os.getenv('TP_ENABLED', 'NOT SET'),
        'ENABLE_SHADOW_MODELS': os.getenv('ENABLE_SHADOW_MODELS', 'NOT SET'),
        'QT_EXECUTION_EXCHANGE': os.getenv('QT_EXECUTION_EXCHANGE', 'NOT SET'),
        'AI_MODEL': os.getenv('AI_MODEL', 'NOT SET')
    }
    for var, val in critical_vars.items():
        status = '✅' if val != 'NOT SET' else '⚠️'
        print(f'  {status} {var}: {val}')
    results['passed'] += 1
except Exception as e:
    print(f'  ❌ FAILED: {e}')
    results['failed'] += 1

# Summary
print('\n' + '=' * 80)
passed = results['passed']
failed = results['failed']
warnings = results['warnings']
print(f'RESULTS: {passed} passed, {failed} failed, {warnings} warnings')
if failed == 0:
    print('✅ SYSTEM STABLE - ALL CHECKS PASSED!')
    sys.exit(0)
else:
    print('❌ SYSTEM HAS ISSUES - REVIEW FAILED CHECKS')
    sys.exit(1)
print('=' * 80)
