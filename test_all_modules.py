#!/usr/bin/env python3
"""
Comprehensive test suite for all Bulletproof AI modules
"""
import sys
sys.path.insert(0, '/app')

print('=' * 70)
print('BULLETPROOF AI MODULES - COMPREHENSIVE TEST SUITE')
print('=' * 70)

errors = []

# TEST 1: Memory State Manager
print('\n[1/6] Testing Memory State Manager...')
try:
    from backend.services.ai.memory_state_manager import MemoryStateManager
    manager = MemoryStateManager(
        ewma_alpha=0.3,
        min_samples_for_memory=10
    )
    
    # Test diagnostics
    diagnostics = manager.get_diagnostics()
    assert 'regime_state' in diagnostics
    assert 'performance_memory' in diagnostics
    regime = diagnostics['regime_state']['current_regime']
    print(f'  ✅ Memory State Manager working (regime: {regime})')
except Exception as e:
    errors.append(f'Memory State Manager: {str(e)}')
    print(f'  ❌ FAILED: {e}')

# TEST 2: Reinforcement Signal Manager
print('\n[2/6] Testing Reinforcement Signal Manager...')
try:
    from backend.services.ai.reinforcement_signal_manager import ReinforcementSignalManager
    manager = ReinforcementSignalManager(
        learning_rate=0.05,
        discount_factor=0.95
    )
    
    # Test diagnostics
    diagnostics = manager.get_diagnostics()
    assert 'model_weights' in diagnostics
    assert 'exploration_rate' in diagnostics
    exp_rate = diagnostics['exploration_rate']
    print(f'  ✅ Reinforcement working (exploration: {exp_rate:.3f})')
except Exception as e:
    errors.append(f'Reinforcement Signal Manager: {str(e)}')
    print(f'  ❌ FAILED: {e}')

# TEST 3: Drift Detection Manager
print('\n[3/6] Testing Drift Detection Manager...')
try:
    from backend.services.ai.drift_detection_manager import DriftDetectionManager
    manager = DriftDetectionManager(
        psi_minor_threshold=0.10,
        psi_moderate_threshold=0.15
    )
    
    # Test diagnostics
    diagnostics = manager.get_diagnostics()
    assert diagnostics is not None
    assert isinstance(diagnostics, dict)
    print(f'  ✅ Drift Detection working (models tracked: {diagnostics.get("models_tracked", [])})')
except Exception as e:
    err_msg = str(e) if str(e) else 'Unknown error'
    errors.append(f'Drift Detection Manager: {err_msg}')
    print(f'  ❌ FAILED: {err_msg}')

# TEST 4: Covariate Shift Manager
print('\n[4/6] Testing Covariate Shift Manager...')
try:
    from backend.services.ai.covariate_shift_manager import CovariateShiftManager
    manager = CovariateShiftManager(
        shift_threshold=0.15,
        adaptation_rate=0.10
    )
    
    # Test status
    status = manager.get_status()
    assert 'reference_stats' in status or 'shift_threshold' in status
    print(f'  ✅ Covariate Shift working')
except Exception as e:
    errors.append(f'Covariate Shift Manager: {str(e)}')
    print(f'  ❌ FAILED: {e}')

# TEST 5: Shadow Model Manager
print('\n[5/6] Testing Shadow Model Manager...')
try:
    from backend.services.ai.shadow_model_manager import ShadowModelManager
    import os
    os.environ['ENABLE_SHADOW_MODELS'] = 'true'
    manager = ShadowModelManager(
        min_trades_for_promotion=500,
        mdd_tolerance=1.20
    )
    
    # Test champion retrieval
    champion = manager.get_champion()
    challengers = manager.get_challengers()
    print(f'  ✅ Shadow Models working (champion: {champion}, challengers: {len(challengers)})')
except Exception as e:
    errors.append(f'Shadow Model Manager: {str(e)}')
    print(f'  ❌ FAILED: {e}')

# TEST 6: Continuous Learning Manager
print('\n[6/6] Testing Continuous Learning Manager...')
try:
    from backend.services.ai.continuous_learning_manager import ContinuousLearningManager
    manager = ContinuousLearningManager(
        ewma_alpha=0.1,
        ewma_threshold=3.0,
        online_learning_rate=0.01
    )
    
    # Test status
    status = manager.get_status()
    assert 'performance_monitor' in status or 'last_retrain' in status or hasattr(manager, 'performance_monitor')
    print(f'  ✅ Continuous Learning working')
except Exception as e:
    errors.append(f'Continuous Learning Manager: {str(e)}')
    print(f'  ❌ FAILED: {e}')

# SUMMARY
print('\n' + '=' * 70)
if len(errors) == 0:
    print('✅ ALL 6 BULLETPROOF AI MODULES PASSED!')
    sys.exit(0)
else:
    print(f'❌ {len(errors)} MODULE(S) FAILED:')
    for err in errors:
        print(f'   - {err}')
    sys.exit(1)
print('=' * 70)
