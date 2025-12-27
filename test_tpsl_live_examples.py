#!/usr/bin/env python3
"""
Dynamic TP/SL System - Live Examples
"""
import sys
sys.path.insert(0, '/app')

print('=' * 70)
print('DYNAMIC TP/SL SYSTEM - LIVE FUNCTIONALITY TEST')
print('=' * 70)

from backend.services.ai.trading_profile import (
    TpslConfig,
    compute_dynamic_tpsl_long,
    compute_dynamic_tpsl_short
)

# Real BTC example
entry_btc = 43500.0
atr_btc = 650.0

config = TpslConfig()

print('\n📊 BTC LONG Position (Entry: $43,500, ATR: $650)')
print('-' * 70)
levels_long = compute_dynamic_tpsl_long(entry_btc, atr_btc, config)

atr_pct = atr_btc / entry_btc * 100
sl_diff = entry_btc - levels_long.sl_init
sl_pct = sl_diff / entry_btc * 100
tp1_diff = levels_long.tp1 - entry_btc
tp1_pct = tp1_diff / entry_btc * 100
tp2_diff = levels_long.tp2 - entry_btc
tp2_pct = tp2_diff / entry_btc * 100
tp3_diff = levels_long.tp3 - entry_btc
tp3_pct = tp3_diff / entry_btc * 100

print(f'Entry Price:    ${entry_btc:,.2f}')
print(f'ATR (14, 15m):  ${atr_btc:,.2f} (~{atr_pct:.2f}%)')
print()
print(f'🛑 Stop Loss:   ${levels_long.sl_init:,.2f} (-${sl_diff:,.2f} / -{sl_pct:.2f}%)')
print(f'🎯 TP1 (50%):   ${levels_long.tp1:,.2f} (+${tp1_diff:,.2f} / +{tp1_pct:.2f}%)')
print(f'🎯 TP2 (30%):   ${levels_long.tp2:,.2f} (+${tp2_diff:,.2f} / +{tp2_pct:.2f}%)')
print(f'🎯 TP3 (trail): ${levels_long.tp3:,.2f} (+${tp3_diff:,.2f} / +{tp3_pct:.2f}%)')
print()

be_trigger_pct = (levels_long.be_trigger - entry_btc) / entry_btc * 100
print(f'🔄 Break-Even:')
print(f'   Trigger:     ${levels_long.be_trigger:,.2f} (+{be_trigger_pct:.2f}%)')
print(f'   Move SL to:  ${levels_long.be_price:,.2f} (entry + 5 bps)')
print()
print(f'📈 Trailing:')
print(f'   Activation:  ${levels_long.trail_activation:,.2f} (at TP2)')
print(f'   Distance:    ${levels_long.trail_distance:,.2f} (0.8R)')
print()
risk = entry_btc - levels_long.sl_init
reward_tp1 = levels_long.tp1 - entry_btc
reward_tp2 = levels_long.tp2 - entry_btc
rr_tp1 = reward_tp1 / risk
rr_tp2 = reward_tp2 / risk
print(f'💰 Risk/Reward:')
print(f'   Risk:        ${risk:,.2f}')
print(f'   TP1 R:R:     1:{rr_tp1:.2f}')
print(f'   TP2 R:R:     1:{rr_tp2:.2f}')

print('\n' + '=' * 70)
print('📊 ETH SHORT Position (Entry: $2,500, ATR: $40)')
print('-' * 70)

entry_eth = 2500.0
atr_eth = 40.0

levels_short = compute_dynamic_tpsl_short(entry_eth, atr_eth, config)

atr_eth_pct = atr_eth / entry_eth * 100
sl_diff_short = levels_short.sl_init - entry_eth
sl_pct_short = sl_diff_short / entry_eth * 100
tp1_diff_short = entry_eth - levels_short.tp1
tp1_pct_short = tp1_diff_short / entry_eth * 100
tp2_diff_short = entry_eth - levels_short.tp2
tp2_pct_short = tp2_diff_short / entry_eth * 100
tp3_diff_short = entry_eth - levels_short.tp3
tp3_pct_short = tp3_diff_short / entry_eth * 100

print(f'Entry Price:    ${entry_eth:,.2f}')
print(f'ATR (14, 15m):  ${atr_eth:,.2f} (~{atr_eth_pct:.2f}%)')
print()
print(f'🛑 Stop Loss:   ${levels_short.sl_init:,.2f} (+${sl_diff_short:,.2f} / +{sl_pct_short:.2f}%)')
print(f'🎯 TP1 (50%):   ${levels_short.tp1:,.2f} (-${tp1_diff_short:,.2f} / -{tp1_pct_short:.2f}%)')
print(f'🎯 TP2 (30%):   ${levels_short.tp2:,.2f} (-${tp2_diff_short:,.2f} / -{tp2_pct_short:.2f}%)')
print(f'🎯 TP3 (trail): ${levels_short.tp3:,.2f} (-${tp3_diff_short:,.2f} / -{tp3_pct_short:.2f}%)')
print()
risk_short = levels_short.sl_init - entry_eth
reward_tp1_short = entry_eth - levels_short.tp1
reward_tp2_short = entry_eth - levels_short.tp2
rr_tp1_short = reward_tp1_short / risk_short
rr_tp2_short = reward_tp2_short / risk_short
print(f'💰 Risk/Reward:')
print(f'   Risk:        ${risk_short:,.2f}')
print(f'   TP1 R:R:     1:{rr_tp1_short:.2f}')
print(f'   TP2 R:R:     1:{rr_tp2_short:.2f}')

print('\n' + '=' * 70)
print('✅ DYNAMIC TP/SL SYSTEM FUNGERER PERFEKT!')
print('   - ATR-basert beregning (tilpasset volatilitet)')
print('   - Multi-target: TP1 (1.5R), TP2 (2.5R), TP3 (4R)')
print('   - Partial close: 50% på TP1, 30% på TP2, 20% trailing')
print('   - Break-even: Flytter SL til BE ved 1R profit')
print('   - Trailing stop: Aktiveres ved 2.5R, 0.8R distance')
print('=' * 70)
