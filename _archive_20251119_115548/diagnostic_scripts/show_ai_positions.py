"""Show AI TP/SL levels for active positions"""
import json
from pathlib import Path

# Read trade state
state_path = Path('/app/backend/data/trade_state.json')
if state_path.exists():
    with open(state_path) as f:
        state = json.load(f)
    
    print('=' * 80)
    print('[TARGET] AI-DRIVEN POSITION STATUS')
    print('=' * 80)
    
    ai_managed = 0
    static_managed = 0
    
    for symbol, data in state.items():
        has_ai = 'ai_tp_pct' in data or 'ai_sl_pct' in data
        if has_ai:
            ai_managed += 1
            print(f'\n[CHART] {symbol} (AI-MANAGED):')
            entry = float(data["avg_entry"])
            peak = float(data.get("peak", entry))
            ai_tp = float(data.get("ai_tp_pct", 0))
            ai_sl = float(data.get("ai_sl_pct", 0))
            ai_trail = float(data.get("ai_trail_pct", 0))
            partial = float(data.get("ai_partial_tp", 1))
            
            print(f'  Entry Price:    ${entry:.4f}')
            print(f'  Peak Price:     ${peak:.4f} ({(peak/entry-1)*100:+.2f}%)')
            print(f'')
            print(f'  [TARGET] AI TP Target:   ${entry * (1 + ai_tp):.4f} (+{ai_tp*100:.2f}%)')
            print(f'  [SHIELD]  AI SL Stop:     ${entry * (1 - ai_sl):.4f} (-{ai_sl*100:.2f}%)')
            print(f'  📉 AI Trail Stop:  ${peak * (1 - ai_trail):.4f} (-{ai_trail*100:.2f}% from peak)')
            print(f'  [MONEY] Partial Exit:   {partial*100:.0f}%')
        else:
            static_managed += 1
    
    print('\n' + '=' * 80)
    print(f'Summary: {ai_managed} AI-managed | {static_managed} static-managed positions')
    print('=' * 80)
