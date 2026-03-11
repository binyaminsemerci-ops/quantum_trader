#!/usr/bin/env python3
"""Quick stats on the exit.replay stream."""
import redis
import collections

r = redis.Redis()
entries = r.xrange("quantum:stream:exit.replay", count=5000)
print(f"Total replay records: {len(entries)}")

symbols = collections.Counter()
actions = collections.Counter()
sides = collections.Counter()
divergences = 0
qwen3_fallbacks = 0
non_hold = 0

for eid, fields in entries:
    sym  = (fields.get(b"symbol") or b"").decode()
    act  = (fields.get(b"formula_action") or b"").decode()
    qa   = (fields.get(b"qwen3_action") or b"").decode()
    side = (fields.get(b"side") or b"").decode()
    fb   = (fields.get(b"qwen3_fallback") or b"false").decode()

    symbols[sym] += 1
    actions[act] += 1
    sides[side] += 1

    if fb == "true":
        qwen3_fallbacks += 1
    if fb != "true" and qa and qa != act:
        divergences += 1
    if act != "HOLD":
        non_hold += 1

print("\n--- By Symbol ---")
for s, c in symbols.most_common(20):
    print(f"  {s}: {c}")
print("\n--- By Formula Action ---")
for a, c in actions.most_common():
    print(f"  {a}: {c}")
print("\n--- By Side ---")
for s, c in sides.most_common():
    print(f"  {s}: {c}")
print(f"\nDivergences (Qwen3 vs formula, non-fallback): {divergences}")
print(f"Non-HOLD actions: {non_hold}")
print(f"Distinct symbols: {len(symbols)}")
print(f"Qwen3 fallback records: {qwen3_fallbacks} / {len(entries)}")
