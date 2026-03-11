#!/usr/bin/env python3
"""Inspect the 4 diverged=true records and explain the 96-mismatch gap."""
import redis

r = redis.Redis()
raw = r.xrange("quantum:stream:exit.replay", "1773017591000-0", "+", count=10000)

print(f"Total post-fix records: {len(raw)}")
print()

diverged_records = []
move_to_be_null = []

for sid, fields in raw:
    rec = {k.decode(): v.decode() for k, v in fields.items()}
    fa = rec.get("formula_action", "")
    qa = rec.get("qwen3_action", "")
    div = rec.get("diverged", "false")
    if div == "true":
        diverged_records.append(rec)
    if fa == "MOVE_TO_BREAKEVEN" and qa in ("null", "", "MOVE_TO_BREAKEVEN"):
        move_to_be_null.append(rec)

print(f"=== {len(diverged_records)} DIVERGED=TRUE RECORDS ===")
for rec in diverged_records:
    print(f"  symbol={rec.get('symbol')}  side={rec.get('side')}")
    print(f"  formula={rec.get('formula_action')}  qwen3={rec.get('qwen3_action')}  live={rec.get('live_action')}")
    print(f"  reward={rec.get('reward')}  regret={rec.get('regret_label')}  preferred={rec.get('preferred_action')}")
    print()

print(f"=== MOVE_TO_BREAKEVEN with null qwen3 (explains 9.7% mismatch gap) ===")
print(f"  Count: {len(move_to_be_null)}")
if move_to_be_null:
    sample = move_to_be_null[0]
    print(f"  Sample symbol={sample.get('symbol')}  formula={sample.get('formula_action')}  qwen3={sample.get('qwen3_action')}  live={sample.get('live_action')}")
    print(f"  reward={sample.get('reward')}  regret={sample.get('regret_label')}  preferred={sample.get('preferred_action')}")

print()
# Check premature_close records
premature = [
    {k.decode(): v.decode() for k, v in fields.items()}
    for sid, fields in raw
    if (lambda f: f.get(b"regret_label", b"").decode() == "premature_close")(fields)
]
print(f"=== {len(premature)} PREMATURE_CLOSE RECORDS ===")
action_counts = {}
for rec in premature:
    fa = rec.get("formula_action", "?")
    action_counts[fa] = action_counts.get(fa, 0) + 1
print(f"  By formula_action: {action_counts}")
if premature:
    s = premature[0]
    print(f"  Sample: symbol={s.get('symbol')} formula={s.get('formula_action')} qwen3={s.get('qwen3_action')} reward={s.get('reward')} preferred={s.get('preferred_action')}")
