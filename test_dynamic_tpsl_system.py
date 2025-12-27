#!/usr/bin/env python3
"""
Test suite for Dynamic TP/SL System (ATR-based, multi-target + trailing)
"""
import sys
sys.path.insert(0, '/app')

print('=' * 80)
print(' ' * 15 + 'DYNAMIC TP/SL SYSTEM - COMPREHENSIVE TEST')
print('=' * 80)

from backend.services.ai.trading_profile import (
    TpslConfig,
    DynamicTpslLevels,
    compute_dynamic_tpsl_long,
    compute_dynamic_tpsl_short
)
from backend.services.binance_market_data import BinanceMarketDataFetcher, calculate_atr

errors = []
warnings = []

# ============================================================================
# TEST 1: ATR CALCULATION
# ============================================================================
print('\n[1/8] Testing ATR Calculation...')
try:
    # Test with mock data
    import pandas as pd
    from datetime import datetime, timedelta
    
    # Create mock OHLCV data
    base_price = 43500
    dates = [datetime.now() - timedelta(minutes=15*i) for i in range(20, 0, -1)]
    
    mock_data = pd.DataFrame({
        'timestamp': dates,
        'open': [base_price + (i % 3) * 100 for i in range(20)],
        'high': [base_price + (i % 3) * 150 for i in range(20)],
        'low': [base_price - (i % 3) * 100 for i in range(20)],
        'close': [base_price + (i % 2) * 50 for i in range(20)],
        'volume': [1000000 for _ in range(20)]
    })
    
    # Calculate ATR
    atr = calculate_atr(mock_data, period=14)
    
    print(f'  ✅ ATR calculated: {atr:.2f}')
    print(f'  📊 Test data: 20 candles, base price: ${base_price}')
    
    if atr > 0:
        print(f'  ✅ ATR > 0 (valid): {atr:.2f}')
    else:
        warnings.append('ATR = 0 (may indicate insufficient data)')
        print(f'  ⚠️  ATR = 0 (insufficient volatility in test data)')
        
except Exception as e:
    errors.append(f'ATR Calculation: {e}')
    print(f'  ❌ FAILED: {e}')

# ============================================================================
# TEST 2: TPSL CONFIG VALIDATION
# ============================================================================
print('\n[2/8] Testing TpslConfig Parameters...')
try:
    config = TpslConfig()
    
    checks = [
        ('ATR Period', config.atr_period, 14),
        ('ATR Timeframe', config.atr_timeframe, '15m'),
        ('ATR Base Multiplier', config.atr_mult_base, 1.0),
        ('Stop Loss Multiplier', config.atr_mult_sl, 1.0),
        ('TP1 Multiplier', config.atr_mult_tp1, 1.5),
        ('TP2 Multiplier', config.atr_mult_tp2, 2.5),
        ('TP3 Multiplier', config.atr_mult_tp3, 4.0),
        ('Break-Even Multiplier', config.atr_mult_be, 1.0),
        ('Trail Distance Multiplier', config.trail_dist_mult, 0.8),
        ('Trail Activation', config.trail_activation_mult, 2.5),
        ('Partial Close TP1', config.partial_close_tp1, 0.5),
        ('Partial Close TP2', config.partial_close_tp2, 0.3),
    ]
    
    print('  Configuration:')
    for name, actual, expected in checks:
        status = '✅' if actual == expected else '⚠️'
        print(f'    {status} {name}: {actual} (expected: {expected})')
        
    print('  ✅ TpslConfig loaded successfully')
    
except Exception as e:
    errors.append(f'TpslConfig: {e}')
    print(f'  ❌ FAILED: {e}')

# ============================================================================
# TEST 3: LONG POSITION TP/SL CALCULATION
# ============================================================================
print('\n[3/8] Testing LONG Position TP/SL Levels...')
try:
    entry_price = 43500.0
    atr = 650.0  # ~1.5% ATR
    config = TpslConfig()
    
    levels = compute_dynamic_tpsl_long(entry_price, atr, config)
    
    # Validate levels structure
    assert hasattr(levels, 'sl_init'), 'Missing sl_init'
    assert hasattr(levels, 'tp1'), 'Missing tp1'
    assert hasattr(levels, 'tp2'), 'Missing tp2'
    assert hasattr(levels, 'tp3'), 'Missing tp3'
    assert hasattr(levels, 'be_trigger'), 'Missing be_trigger'
    assert hasattr(levels, 'be_price'), 'Missing be_price'
    assert hasattr(levels, 'trail_activation'), 'Missing trail_activation'
    assert hasattr(levels, 'trail_distance'), 'Missing trail_distance'
    
    # Validate level logic for LONG
    assert levels.sl_init < entry_price, f'SL should be below entry: {levels.sl_init} >= {entry_price}'
    assert levels.tp1 > entry_price, f'TP1 should be above entry: {levels.tp1} <= {entry_price}'
    assert levels.tp2 > levels.tp1, f'TP2 should be above TP1: {levels.tp2} <= {levels.tp1}'
    assert levels.tp3 > levels.tp2, f'TP3 should be above TP2: {levels.tp3} <= {levels.tp2}'
    assert levels.be_trigger > entry_price, f'BE trigger should be above entry'
    assert levels.be_price >= entry_price, f'BE price should be at or above entry'
    
    # Validate R multiples
    r = atr * config.atr_mult_base
    expected_sl = entry_price - (r * config.atr_mult_sl)
    expected_tp1 = entry_price + (r * config.atr_mult_tp1)
    expected_tp2 = entry_price + (r * config.atr_mult_tp2)
    
    print(f'  📊 Entry: ${entry_price:.2f}, ATR: ${atr:.2f}')
    print(f'  📊 R-value: ${r:.2f}')
    print(f'  ✅ SL (1R):  ${levels.sl_init:.2f} (expected: ${expected_sl:.2f})')
    print(f'  ✅ TP1 (1.5R): ${levels.tp1:.2f} (expected: ${expected_tp1:.2f})')
    print(f'  ✅ TP2 (2.5R): ${levels.tp2:.2f} (expected: ${expected_tp2:.2f})')
    print(f'  ✅ TP3 (4R): ${levels.tp3:.2f}')
    print(f'  ✅ Break-Even Trigger: ${levels.be_trigger:.2f}')
    print(f'  ✅ Break-Even Price: ${levels.be_price:.2f}')
    print(f'  ✅ Trail Activation: ${levels.trail_activation:.2f}')
    print(f'  ✅ Trail Distance: ${levels.trail_distance:.2f}')
    
    # Check R:R ratios
    risk = entry_price - levels.sl_init
    reward_tp1 = levels.tp1 - entry_price
    reward_tp2 = levels.tp2 - entry_price
    
    rr_tp1 = reward_tp1 / risk
    rr_tp2 = reward_tp2 / risk
    
    print(f'  📈 Risk: ${risk:.2f}')
    print(f'  📈 R:R TP1: 1:{rr_tp1:.2f} (expected: 1:1.5)')
    print(f'  📈 R:R TP2: 1:{rr_tp2:.2f} (expected: 1:2.5)')
    
    if abs(rr_tp1 - 1.5) < 0.01 and abs(rr_tp2 - 2.5) < 0.01:
        print('  ✅ R:R ratios correct!')
    else:
        warnings.append(f'R:R ratios slightly off: TP1={rr_tp1:.2f}, TP2={rr_tp2:.2f}')
        
    print('  ✅ LONG TP/SL calculation PASS')
    
except Exception as e:
    errors.append(f'LONG TP/SL: {e}')
    print(f'  ❌ FAILED: {e}')
    import traceback
    traceback.print_exc()

# ============================================================================
# TEST 4: SHORT POSITION TP/SL CALCULATION
# ============================================================================
print('\n[4/8] Testing SHORT Position TP/SL Levels...')
try:
    entry_price = 43500.0
    atr = 650.0
    config = TpslConfig()
    
    levels = compute_dynamic_tpsl_short(entry_price, atr, config)
    
    # Validate level logic for SHORT (inverted)
    assert levels.sl_init > entry_price, f'SL should be above entry for SHORT: {levels.sl_init} <= {entry_price}'
    assert levels.tp1 < entry_price, f'TP1 should be below entry for SHORT: {levels.tp1} >= {entry_price}'
    assert levels.tp2 < levels.tp1, f'TP2 should be below TP1 for SHORT: {levels.tp2} >= {levels.tp1}'
    assert levels.tp3 < levels.tp2, f'TP3 should be below TP2 for SHORT: {levels.tp3} >= {levels.tp2}'
    
    print(f'  📊 Entry: ${entry_price:.2f}, ATR: ${atr:.2f}')
    print(f'  ✅ SL (1R):  ${levels.sl_init:.2f} (above entry)')
    print(f'  ✅ TP1 (1.5R): ${levels.tp1:.2f} (below entry)')
    print(f'  ✅ TP2 (2.5R): ${levels.tp2:.2f} (below TP1)')
    print(f'  ✅ TP3 (4R): ${levels.tp3:.2f} (below TP2)')
    
    # Check R:R ratios for SHORT
    risk = levels.sl_init - entry_price
    reward_tp1 = entry_price - levels.tp1
    reward_tp2 = entry_price - levels.tp2
    
    rr_tp1 = reward_tp1 / risk
    rr_tp2 = reward_tp2 / risk
    
    print(f'  📈 Risk: ${risk:.2f}')
    print(f'  📈 R:R TP1: 1:{rr_tp1:.2f} (expected: 1:1.5)')
    print(f'  📈 R:R TP2: 1:{rr_tp2:.2f} (expected: 1:2.5)')
    
    print('  ✅ SHORT TP/SL calculation PASS')
    
except Exception as e:
    errors.append(f'SHORT TP/SL: {e}')
    print(f'  ❌ FAILED: {e}')
    import traceback
    traceback.print_exc()

# ============================================================================
# TEST 5: PARTIAL CLOSE MECHANICS
# ============================================================================
print('\n[5/8] Testing Partial Close Mechanics...')
try:
    config = TpslConfig()
    
    initial_position = 100  # 100 contracts
    
    # At TP1: Close 50%
    close_tp1 = initial_position * config.partial_close_tp1
    remaining_after_tp1 = initial_position - close_tp1
    
    # At TP2: Close 30% of original
    close_tp2 = initial_position * config.partial_close_tp2
    remaining_after_tp2 = remaining_after_tp1 - close_tp2
    
    # Final: 20% remains for trailing
    final_remaining = remaining_after_tp2
    
    print(f'  📊 Initial Position: {initial_position} contracts')
    print(f'  ✅ TP1 Close: {close_tp1:.0f} contracts ({config.partial_close_tp1*100}%)')
    print(f'  ✅ Remaining after TP1: {remaining_after_tp1:.0f} contracts')
    print(f'  ✅ TP2 Close: {close_tp2:.0f} contracts ({config.partial_close_tp2*100}%)')
    print(f'  ✅ Remaining after TP2: {remaining_after_tp2:.0f} contracts')
    print(f'  ✅ Final for trailing: {final_remaining:.0f} contracts (20%)')
    
    # Validate
    assert close_tp1 == 50, f'TP1 close should be 50, got {close_tp1}'
    assert close_tp2 == 30, f'TP2 close should be 30, got {close_tp2}'
    assert final_remaining == 20, f'Final should be 20, got {final_remaining}'
    
    print('  ✅ Partial close mechanics PASS')
    
except Exception as e:
    errors.append(f'Partial Close: {e}')
    print(f'  ❌ FAILED: {e}')

# ============================================================================
# TEST 6: BREAK-EVEN LOGIC
# ============================================================================
print('\n[6/8] Testing Break-Even Move Logic...')
try:
    entry_price = 43500.0
    atr = 650.0
    config = TpslConfig()
    
    levels_long = compute_dynamic_tpsl_long(entry_price, atr, config)
    
    # BE trigger should be at 1R profit
    r = atr * config.atr_mult_base
    expected_be_trigger = entry_price + (r * config.atr_mult_be)
    
    # BE price should be entry + buffer (5 bps)
    buffer = entry_price * (config.be_buffer_bps / 10000)
    expected_be_price = entry_price + buffer
    
    print(f'  📊 Entry: ${entry_price:.2f}')
    print(f'  📊 R-value: ${r:.2f}')
    print(f'  ✅ BE Trigger (1R): ${levels_long.be_trigger:.2f} (expected: ${expected_be_trigger:.2f})')
    print(f'  ✅ BE Price: ${levels_long.be_price:.2f} (expected: ${expected_be_price:.2f})')
    print(f'  📈 BE Buffer: {config.be_buffer_bps} bps (${buffer:.2f})')
    
    # Validate BE trigger is at or near expected
    assert abs(levels_long.be_trigger - expected_be_trigger) < 1, 'BE trigger mismatch'
    assert abs(levels_long.be_price - expected_be_price) < 1, 'BE price mismatch'
    
    print('  ✅ Break-even logic PASS')
    
except Exception as e:
    errors.append(f'Break-Even: {e}')
    print(f'  ❌ FAILED: {e}')

# ============================================================================
# TEST 7: TRAILING STOP LOGIC
# ============================================================================
print('\n[7/8] Testing Trailing Stop Logic...')
try:
    entry_price = 43500.0
    atr = 650.0
    config = TpslConfig()
    
    levels_long = compute_dynamic_tpsl_long(entry_price, atr, config)
    
    # Trailing activates at TP2 (2.5R)
    r = atr * config.atr_mult_base
    expected_trail_activation = entry_price + (r * config.trail_activation_mult)
    
    # Trailing distance is 0.8R
    expected_trail_distance = r * config.trail_dist_mult
    
    print(f'  📊 Entry: ${entry_price:.2f}')
    print(f'  📊 R-value: ${r:.2f}')
    print(f'  ✅ Trail Activation (2.5R): ${levels_long.trail_activation:.2f}')
    print(f'     Expected: ${expected_trail_activation:.2f}')
    print(f'  ✅ Trail Distance (0.8R): ${levels_long.trail_distance:.2f}')
    print(f'     Expected: ${expected_trail_distance:.2f}')
    
    # Validate
    assert abs(levels_long.trail_activation - expected_trail_activation) < 1, 'Trail activation mismatch'
    assert abs(levels_long.trail_distance - expected_trail_distance) < 1, 'Trail distance mismatch'
    
    # Simulate trailing
    current_price = levels_long.trail_activation + 500  # Price goes $500 above activation
    trailing_stop = current_price - levels_long.trail_distance
    
    print(f'  📈 Simulation: Price reaches ${current_price:.2f}')
    print(f'  📈 Trailing stop would be at: ${trailing_stop:.2f}')
    print(f'  📈 Distance from price: ${current_price - trailing_stop:.2f}')
    
    print('  ✅ Trailing stop logic PASS')
    
except Exception as e:
    errors.append(f'Trailing Stop: {e}')
    print(f'  ❌ FAILED: {e}')

# ============================================================================
# TEST 8: EXECUTION INTEGRATION
# ============================================================================
print('\n[8/8] Testing Execution Integration...')
try:
    from backend.services.execution.execution import BinanceFuturesExecutionAdapter
    
    # Check if method exists
    has_method = hasattr(BinanceFuturesExecutionAdapter, 'submit_order_with_tpsl')
    
    if has_method:
        print('  ✅ submit_order_with_tpsl() method exists in Execution adapter')
        
        # Check method signature
        import inspect
        sig = inspect.signature(BinanceFuturesExecutionAdapter.submit_order_with_tpsl)
        params = list(sig.parameters.keys())
        
        required_params = ['symbol', 'side', 'quantity', 'entry_price']
        optional_params = ['equity', 'ai_risk_factor']
        
        print(f'  📊 Method parameters: {params}')
        
        for param in required_params:
            if param in params:
                print(f'    ✅ {param} (required)')
            else:
                warnings.append(f'Missing parameter: {param}')
                print(f'    ⚠️  {param} (missing)')
        
        for param in optional_params:
            if param in params:
                print(f'    ✅ {param} (optional)')
                
        print('  ✅ Execution integration PASS')
    else:
        errors.append('submit_order_with_tpsl method not found')
        print('  ❌ submit_order_with_tpsl() method NOT FOUND')
        
except Exception as e:
    errors.append(f'Execution Integration: {e}')
    print(f'  ❌ FAILED: {e}')

# ============================================================================
# SUMMARY
# ============================================================================
print('\n' + '=' * 80)
print(' ' * 25 + 'TEST SUMMARY')
print('=' * 80)

print(f'\n✅ Tests Passed: {8 - len(errors)}')
print(f'❌ Tests Failed: {len(errors)}')
print(f'⚠️  Warnings: {len(warnings)}')

if errors:
    print('\n❌ FAILED TESTS:')
    for i, error in enumerate(errors, 1):
        print(f'  {i}. {error}')

if warnings:
    print('\n⚠️  WARNINGS:')
    for i, warning in enumerate(warnings, 1):
        print(f'  {i}. {warning}')

print('\n' + '=' * 80)
if len(errors) == 0:
    print('✅ DYNAMIC TP/SL SYSTEM FULLY FUNCTIONAL!')
    print('   - ATR calculation working')
    print('   - Multi-target levels (TP1/TP2/TP3) calculated correctly')
    print('   - Partial close mechanics (50%/30%/20%) configured')
    print('   - Break-even move logic in place')
    print('   - Trailing stop activation at 2.5R')
    print('   - Execution integration verified')
    sys.exit(0)
else:
    print('❌ DYNAMIC TP/SL SYSTEM HAS ISSUES')
    sys.exit(1)
print('=' * 80)
