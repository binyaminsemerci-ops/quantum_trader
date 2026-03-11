"""DPO dataset quality report — PATCH-10A.

Reads /tmp/dpo_pairs_patch10a.jsonl (51 pairs), copies to a stable path,
prints counts, 10 representative examples, and flags quality concerns.
"""
import json, pathlib, collections, shutil, statistics

SRC  = pathlib.Path('/tmp/dpo_pairs_patch10a.jsonl')
DEST = pathlib.Path('/home/qt/quantum_trader/logs/dpo/dpo_patch10a_v1.jsonl')
DEST.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(SRC, DEST)

data = [json.loads(l) for l in SRC.open()]

# ── counts ────────────────────────────────────────────────────────────────────
def tbl(counter, label):
    print(f'\n{label}:')
    for k, v in sorted(counter.items(), key=lambda x: -x[1]):
        bar = '█' * int(v / max(counter.values()) * 20)
        print(f'  {k:<30} {v:>3}  {bar}')

print('=' * 60)
print(f'PATCH-10A DPO DATASET — {len(data)} pairs')
print(f'Stable path: {DEST}')
print('=' * 60)

tbl(collections.Counter(r['regret_label']   for r in data), 'By regret_label')
tbl(collections.Counter(r['symbol']         for r in data), 'By symbol')
tbl(collections.Counter(r['live_action']    for r in data), 'By live_action')
tbl(collections.Counter(r['preferred_action'] for r in data), 'By preferred_action')

rewards = [r['reward'] for r in data]
print(f'\nReward distribution:')
print(f'  min    : {min(rewards):.4f}')
print(f'  max    : {max(rewards):.4f}')
print(f'  mean   : {statistics.mean(rewards):.4f}')
print(f'  median : {statistics.median(rewards):.4f}')
print(f'  stdev  : {statistics.stdev(rewards):.4f}')

# ── 10 representative examples ─────────────────────────────────────────────────
print('\n' + '=' * 60)
print('10 REPRESENTATIVE EXAMPLES')
print('=' * 60)
# Pick: first 3, middle 4, last 3
picks = data[:3] + data[len(data)//2-2 : len(data)//2+2] + data[-3:]
for i, r in enumerate(picks, 1):
    print(f'\n[{i:02d}] {r["symbol"]:12s}  regret={r["regret_label"]}')
    print(f'     live={r["live_action"]}  →  preferred={r["preferred_action"]}')
    print(f'     reward={r["reward"]:.4f}  diverged={r["diverged"]}  formula={r["formula_action"]}')

# ── quality flags ─────────────────────────────────────────────────────────────
print('\n' + '=' * 60)
print('QUALITY FLAGS')
print('=' * 60)

flags = []

# Flag 1: live==preferred (should be impossible but sanity check)
same = [r for r in data if r['live_action'] == r['preferred_action']]
if same:
    flags.append(f'CRITICAL: {len(same)} pairs have live_action == preferred_action (no training signal)')

# Flag 2: very shallow rewards (|reward| < 0.06) — noisy signal
shallow = [r for r in data if abs(r['reward']) < 0.06]
if shallow:
    flags.append(f'WARN: {len(shallow)} pairs have |reward| < 0.06 (weak signal)')

# Flag 3: only one regret_label type dominates (>90%)
top_rl = max(collections.Counter(r['regret_label'] for r in data).values())
if top_rl / len(data) > 0.90:
    flags.append(f'WARN: one regret_label covers >{top_rl/len(data)*100:.0f}% of pairs (diversity risk)')

# Flag 4: only one preferred_action type
pa_counts = collections.Counter(r['preferred_action'] for r in data)
if len(pa_counts) == 1:
    flags.append(f'WARN: all {len(data)} preferred_actions are the same value ({list(pa_counts)[0]}) — low DPO contrast')

# Flag 5: only one live_action type
la_counts = collections.Counter(r['live_action'] for r in data)
if len(la_counts) == 1:
    flags.append(f'WARN: all {len(data)} live_actions are the same value ({list(la_counts)[0]}) — low contrast')

# Flag 6: total pair count
if len(data) < 100:
    flags.append(f'SIZE: {len(data)} pairs is below the ~100 minimum recommended for stable DPO fine-tuning')

# Flag 7: symbol diversity
n_sym = len(set(r['symbol'] for r in data))
if n_sym < 3:
    flags.append(f'WARN: only {n_sym} distinct symbols — generalisation risk')

# Flag 8: all rewards negative (expected here, but verify no sign errors)
pos = [r for r in data if r['reward'] > 0]
if pos:
    flags.append(f'WARN: {len(pos)} pairs have positive reward but preferred != live — review logic')

if not flags:
    flags.append('No quality issues detected.')

for f in flags:
    print(f'  • {f}')

# ── recommendation ─────────────────────────────────────────────────────────────
print('\n' + '=' * 60)
print('RECOMMENDATION')
print('=' * 60)

n = len(data)
pa_div = len(pa_counts)
la_div = len(la_counts)
sym_div = n_sym

if n < 50:
    verdict = 'NOT USABLE — too few pairs.'
elif pa_div == 1 and la_div == 1:
    verdict = ('PROMPT-TUNING ONLY — all pairs encode a single (live→preferred) '
               'transition; DPO will see no cross-action contrast and may overfit.')
elif n < 200:
    verdict = ('FIRST EXPERIMENTAL DPO — small but structurally valid. '
               'Suitable for a low-rank LoRA experiment with heavy regularisation '
               '(β≥0.5). Do NOT use for production fine-tuning until ≥200 pairs '
               'across ≥3 distinct preferred_actions are available.')
else:
    verdict = 'USABLE FOR DPO — sufficient size and diversity.'

print(f'  {verdict}')
print(f'\n  n={n}  symbols={sym_div}  preferred_action_types={pa_div}  live_action_types={la_div}')
print()
