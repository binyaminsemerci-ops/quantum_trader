"""One-shot deep analysis of the extracted exit.audit replay data."""
import json, statistics, collections
from datetime import datetime, timezone

lines = []
with open('logs/replay/replay_2026-03-09.jsonl') as f:
    for l in f:
        if l.strip():
            lines.append(json.loads(l))

print(f'=== DATASET: {len(lines)} records ===')

# Date range from record_time_epoch
epochs = [r['record_time_epoch'] for r in lines if r.get('record_time_epoch')]
if epochs:
    t0 = datetime.fromtimestamp(min(epochs), tz=timezone.utc)
    t1 = datetime.fromtimestamp(max(epochs), tz=timezone.utc)
    fmt = "%Y-%m-%d %H:%M:%S"
    print(f'Date range : {t0.strftime(fmt)} UTC  -->  {t1.strftime(fmt)} UTC')
    print(f'Duration   : {(max(epochs)-min(epochs))/3600:.2f} hours')
    print(f'Scan rate  : {len(lines)/(max(epochs)-min(epochs)):.1f} decisions/sec')

# Action distribution
act_cnt = collections.Counter(r['live_action'] for r in lines)
print('\n--- ACTION DISTRIBUTION ---')
for a, n in act_cnt.most_common():
    bar = '█' * int(30 * n / len(lines))
    print(f'  {a:<25} {bar:<30} {n:>6}  ({100*n/len(lines):.2f}%)')

# Divergence (live != formula)
diverged = [r for r in lines if r.get('diverged') == 'true']
print(f'\n--- FORMULA DIVERGENCE (live_action vs formula_action) ---')
print(f'  live != formula : {len(diverged)}/{len(lines)}  ({100*len(diverged)/len(lines):.3f}%)')
for r in diverged[:10]:
    print(f'  EX: live={r["live_action"]!r:30} formula={r["formula_action"]!r:30} sym={r["symbol"]} side={r["side"]}')

# Side distribution
side_cnt = collections.Counter(r['side'] for r in lines)
print('\n--- SIDE DISTRIBUTION ---')
for s, n in side_cnt.most_common():
    print(f'  {s:<10}  {n:>6}  ({100*n/len(lines):.2f}%)')

# Symbol distribution
sym_cnt = collections.Counter(r['symbol'] for r in lines)
print('\n--- SYMBOL DISTRIBUTION ---')
for s, n in sym_cnt.most_common():
    bar = '█' * int(20 * n / max(sym_cnt.values()))
    print(f'  {s:<12}  {bar:<20} {n:>6}  ({100*n/len(lines):.2f}%)')

# Exit score by action
from_es = collections.defaultdict(list)
for r in lines:
    es = r.get('exit_score', '')
    if es:
        try:
            from_es[r['live_action']].append(float(es))
        except Exception:
            pass
print('\n--- EXIT SCORE STATS BY ACTION ---')
for a, vals in sorted(from_es.items(), key=lambda x: -statistics.mean(x[1]) if x[1] else 0):
    if vals:
        med = statistics.median(vals)
        mn  = statistics.mean(vals)
        hi  = max(vals)
        lo  = min(vals)
        print(f'  {a:<25} n={len(vals):>5}  mean={mn:.4f}  median={med:.4f}  max={hi:.4f}  min={lo:.4f}')

# Exit score distribution for HOLD: how many have high score but HOLD anyway?
hold_es = [float(r['exit_score']) for r in lines if r['live_action']=='HOLD' and r.get('exit_score')]
if hold_es:
    high_score_hold = [v for v in hold_es if v > 0.5]
    mid_score_hold  = [v for v in hold_es if 0.3 <= v <= 0.5]
    print(f'\n--- HOLD DECISIONS WITH HIGH EXIT SCORE ---')
    print(f'  HOLD with exit_score > 0.5  : {len(high_score_hold)} ({100*len(high_score_hold)/len(hold_es):.2f}%)')
    print(f'  HOLD with exit_score 0.3-0.5: {len(mid_score_hold)} ({100*len(mid_score_hold)/len(hold_es):.2f}%)')

# Non-HOLD breakdown by symbol
print('\n--- NON-HOLD ACTIONS BY SYMBOL ---')
sym_action = collections.defaultdict(collections.Counter)
sym_total  = collections.Counter(r['symbol'] for r in lines)
for r in lines:
    if r['live_action'] != 'HOLD':
        sym_action[r['symbol']][r['live_action']] += 1
for sym, cnts in sorted(sym_action.items(), key=lambda x: -sum(x[1].values())):
    total_nh = sum(cnts.values())
    st = sym_total[sym]
    nh_rate = 100 * total_nh / st if st else 0
    print(f'  {sym:<12}  non-hold={total_nh:>4}  ({nh_rate:.1f}% of {st})  {dict(cnts.most_common())}')

# Non-HOLD breakdown by SIDE
print('\n--- NON-HOLD ACTIONS BY SIDE ---')
side_action = collections.defaultdict(collections.Counter)
side_total  = collections.Counter(r['side'] for r in lines)
for r in lines:
    if r['live_action'] != 'HOLD':
        side_action[r['side']][r['live_action']] += 1
for side, cnts in sorted(side_action.items()):
    total_nh = sum(cnts.values())
    st = side_total[side]
    nh_rate = 100 * total_nh / st if st else 0
    print(f'  {side:<10}  non-hold={total_nh:>4}  ({nh_rate:.1f}% of {st})  {dict(cnts.most_common())}')

# Exit score thresholds that caused action
print('\n--- EXIT SCORE AT TRIGGER (when non-HOLD action was taken) ---')
non_hold_scores = [float(r['exit_score']) for r in lines if r['live_action'] != 'HOLD' and r.get('exit_score')]
if non_hold_scores:
    buckets = {'<0.3': 0, '0.3-0.5': 0, '0.5-0.7': 0, '0.7-0.9': 0, '>0.9': 0}
    for s in non_hold_scores:
        if s < 0.3:   buckets['<0.3'] += 1
        elif s < 0.5: buckets['0.3-0.5'] += 1
        elif s < 0.7: buckets['0.5-0.7'] += 1
        elif s < 0.9: buckets['0.7-0.9'] += 1
        else:         buckets['>0.9'] += 1
    for b, n in buckets.items():
        print(f'  exit_score {b:<10}: {n:>4} ({100*n/len(non_hold_scores):.1f}%)')

# Summary of the 3 FULL_CLOSE events
full_close = [r for r in lines if r['live_action'] == 'FULL_CLOSE']
print(f'\n--- FULL_CLOSE DETAILS (n={len(full_close)}) ---')
for r in full_close:
    print(f'  sym={r["symbol"]} side={r["side"]} exit_score={r.get("exit_score","n/a")} entry={r.get("entry_price","n/a")} close={r.get("close_price","n/a")}')

# Preferred action frequency
pref_cnt = collections.Counter(r['preferred_action'] for r in lines)
print(f'\n--- PREFERRED ACTION FREQUENCY ---')
for a, n in pref_cnt.most_common():
    bar = '█' * int(30 * n / len(lines))
    print(f'  {a:<25} {bar:<30} {n:>6}  ({100*n/len(lines):.2f}%)')

print(f'\n  preferred == live: {sum(1 for r in lines if r["preferred_action"] == r["live_action"])} ({100*sum(1 for r in lines if r["preferred_action"]==r["live_action"])/len(lines):.2f}%)')

# Patch distribution
patch_cnt = collections.Counter(r.get('patch','') for r in lines)
print(f'\n--- PATCH VERSION ---')
for p, n in patch_cnt.most_common():
    print(f'  {p}  : {n:>6}')

print('\n=== END ANALYSIS ===')
