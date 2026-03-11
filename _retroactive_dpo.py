#!/usr/bin/env python3
"""Retroactive DPO export: re-runs RewardEngine over baked replay records
so the new PATCH-10A counterfactual rules are applied to existing data."""
import json, sys, pathlib
sys.path.insert(0, '/home/qt/quantum_trader')
from microservices.exit_management_agent.reward_engine import RewardEngine

engine = RewardEngine()
src = pathlib.Path('/home/qt/quantum_trader/logs/replay/postfix/replay_postfix_2026.jsonl')
out = pathlib.Path('/tmp/dpo_pairs_patch10a.jsonl')

total = changed = pairs = 0
with src.open() as f, out.open('w') as g:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        total += 1
        snapshot = {k: rec.get(k, '') for k in
                    ('live_action', 'exit_score', 'formula_action', 'qwen3_action',
                     'diverged', 'side', 'entry_price', 'quantity')}
        outcome = {k: rec.get(k, '') for k in
                   ('hold_duration_sec', 'close_price', 'closed_by', 'outcome_action')}
        new_result = engine.compute(snapshot, outcome)
        old_pa = rec.get('preferred_action', '')
        new_pa = new_result.preferred_action
        live = rec.get('live_action', '')
        reward = float(rec.get('reward', 0))
        if old_pa != new_pa:
            changed += 1
        if new_pa != live and abs(reward) >= 0.05:
            pairs += 1
            g.write(json.dumps({
                'decision_id':    rec.get('decision_id'),
                'symbol':         rec.get('symbol'),
                'live_action':    live,
                'preferred_action': new_pa,
                'old_preferred':  old_pa,
                'reward':         reward,
                'regret_label':   new_result.regret_label,
                'diverged':       rec.get('diverged'),
                'formula_action': rec.get('formula_action'),
            }) + '\n')

print(f'Total records             : {total}')
print(f'preferred_action changed  : {changed}')
print(f'DPO pairs (|reward|>=0.05): {pairs}')
