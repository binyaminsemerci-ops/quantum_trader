"""synthetic_batch_gen.py — PATCH-10A DPO dataset expansion.

Generates structured synthetic replay records across four scenario groups
designed to produce ≥200 total DPO pairs (combined with the existing 51)
and ≥3 distinct preferred_action types.

Scenario groups
---------------
A  live=HOLD, exit_score∈[0.70-0.95], diverged=false
   reward = -exit_score ∈ [-0.95,-0.70]  →  Rule 3 (exit_score≥0.7) → preferred=FULL_CLOSE

B  live=HOLD, exit_score∈[0.31-0.69], diverged=false
   reward = -exit_score ∈ [-0.69,-0.31]  →  Rule 3 (exit_score<0.7) → preferred=PARTIAL_CLOSE_25

D  live=HOLD, diverged=true, formula=FULL_CLOSE, exit_score∈[0.10-0.45]
   reward = -exit_score < 0; exit_score<0.5 → no late_hold → divergence_regret
   Rule 1 fires (diverged+reward<0+formula present) → preferred=FULL_CLOSE

E  live=HOLD, diverged=true, formula=PARTIAL_CLOSE_25, exit_score∈[0.10-0.45]
   Same path as D → preferred=PARTIAL_CLOSE_25

All groups use 10 symbols × 2 sides rotated per template.
Records where preferred_action == live_action are still written (they act as
negative examples / reference baseline) but only preferred!=live count as DPO.
"""
from __future__ import annotations
import json, pathlib, uuid, itertools, sys, collections

sys.path.insert(0, '/home/qt/quantum_trader')
from microservices.exit_management_agent.reward_engine import RewardEngine

# ── Config ─────────────────────────────────────────────────────────────────────
ENGINE = RewardEngine(late_hold_threshold_sec=3600, premature_close_threshold_sec=300)

SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "AVAXUSDT",
    "DOGEUSDT", "NEARUSDT", "LINKUSDT", "DOTUSDT", "ADAUSDT",
]
SIDES = ["LONG", "SHORT"]
ENTRY = {
    "BTCUSDT": "85000.00", "ETHUSDT": "4200.00",  "SOLUSDT": "180.00",
    "BNBUSDT": "580.00",   "AVAXUSDT": "42.00",   "DOGEUSDT": "0.350000",
    "NEARUSDT": "8.50",    "LINKUSDT": "22.00",   "DOTUSDT": "11.00",
    "ADAUSDT": "1.2000",
}
QTY = {
    "BTCUSDT": "0.01",  "ETHUSDT": "0.10",  "SOLUSDT": "5.00",
    "BNBUSDT": "0.50",  "AVAXUSDT": "2.00", "DOGEUSDT": "1000.00",
    "NEARUSDT": "50.00","LINKUSDT": "10.00","DOTUSDT": "20.00",
    "ADAUSDT": "100.00",
}

BASE_TS   = 1_741_478_400_000   # 2026-03-09 00:00:00 UTC ms
DEST_DIR  = pathlib.Path('/home/qt/quantum_trader/logs/replay/synthetic')
DEST_FILE = DEST_DIR / 'replay_batch1_2026.jsonl'
ORIG_FILE = pathlib.Path('/home/qt/quantum_trader/logs/replay/postfix/replay_postfix_2026.jsonl')
DPO_OUT   = pathlib.Path('/home/qt/quantum_trader/logs/dpo/dpo_patch10a_v2.jsonl')

_counter = [0]
_sym_cycle = itertools.cycle(SYMBOLS)
_side_cycle = itertools.cycle(SIDES)


def _next_id() -> str:
    _counter[0] += 1
    return f"syn-p10a-{_counter[0]:04d}-{uuid.uuid4().hex[:8]}"


def make_record(
    live_action: str,
    formula_action: str,
    diverged: bool,
    exit_score: float,
    hold_duration_sec: int,
    closed_by: str = "unknown",
    qwen3_action: str | None = None,
    symbol: str | None = None,
    side: str | None = None,
) -> dict:
    sym  = symbol or next(_sym_cycle)
    sid  = side   or next(_side_cycle)
    did  = _next_id()
    ts   = BASE_TS + _counter[0] * 500   # 0.5 s spacing

    snap = {
        "live_action":    live_action,
        "exit_score":     f"{exit_score:.4f}",
        "formula_action": formula_action,
        "qwen3_action":   qwen3_action or live_action,
        "diverged":       "true" if diverged else "false",
        "side":           sid,
        "entry_price":    ENTRY[sym],
        "quantity":       QTY[sym],
    }
    out = {
        "hold_duration_sec": str(hold_duration_sec),
        "close_price":       ENTRY[sym],
        "closed_by":         closed_by,
        "outcome_action":    live_action,
    }
    result = ENGINE.compute(snap, out)

    return {
        "decision_id":       did,
        "symbol":            sym,
        "record_time_epoch": str(ts // 1000),
        "patch":             "PATCH-10A-synthetic",
        "source":            "synthetic_batch_gen",
        "live_action":       live_action,
        "formula_action":    formula_action,
        "qwen3_action":      qwen3_action or live_action,
        "diverged":          "true" if diverged else "false",
        "exit_score":        f"{exit_score:.4f}",
        "entry_price":       ENTRY[sym],
        "side":              sid,
        "quantity":          QTY[sym],
        "hold_duration_sec": str(hold_duration_sec),
        "close_price":       ENTRY[sym],
        "closed_by":         closed_by,
        "outcome_action":    live_action,
        "reward":            str(result.reward),
        "regret_label":      result.regret_label,
        "regret_score":      str(result.regret_score),
        "preferred_action":  result.preferred_action,
        "_stream_id":        f"{ts}-0",
    }


# ── Scenario templates ─────────────────────────────────────────────────────────

def build_scenarios() -> list[dict]:
    records: list[dict] = []

    # ── GROUP A: late_hold → preferred = FULL_CLOSE ───────────────────────────
    # live=HOLD, exit_score ∈ [0.70,0.95], no divergence
    # reward = -exit_score ∈ [-0.95,-0.70] → Rule 3 (exit_score≥0.7) → FULL_CLOSE
    A_scores = [0.71, 0.75, 0.78, 0.80, 0.83, 0.85, 0.88, 0.90, 0.93, 0.95]
    A_holds  = [300, 900, 1800, 3600, 7200]
    sym_side = list(itertools.product(SYMBOLS, SIDES))
    combo_A  = list(itertools.product(A_scores, A_holds))

    for (es, hold), (sym, side) in zip(
        itertools.islice(itertools.cycle(combo_A), len(sym_side) * 3),
        itertools.cycle(sym_side),
    ):
        records.append(make_record(
            live_action="HOLD", formula_action="HOLD", diverged=False,
            exit_score=es, hold_duration_sec=hold, closed_by="unknown",
            symbol=sym, side=side,
        ))

    # ── GROUP B: late_hold → preferred = PARTIAL_CLOSE_25 ────────────────────
    # live=HOLD, exit_score ∈ [0.31,0.69], no divergence
    # reward = -exit_score → Rule 3 (exit_score<0.7) → PARTIAL_CLOSE_25
    B_scores = [0.32, 0.36, 0.40, 0.44, 0.48, 0.52, 0.56, 0.60, 0.64, 0.68]
    B_holds  = [300, 900, 1800, 3600, 7200]
    combo_B  = list(itertools.product(B_scores, B_holds))

    for (es, hold), (sym, side) in zip(
        itertools.islice(itertools.cycle(combo_B), len(sym_side) * 3),
        itertools.cycle(sym_side),
    ):
        records.append(make_record(
            live_action="HOLD", formula_action="HOLD", diverged=False,
            exit_score=es, hold_duration_sec=hold, closed_by="unknown",
            symbol=sym, side=side,
        ))

    # ── GROUP D: divergence_regret → preferred = FULL_CLOSE ──────────────────
    # diverged=true, formula=FULL_CLOSE, live=HOLD, exit_score<0.5
    # exit_score<0.5 → no late_hold; regret=divergence_regret
    # Rule 1 (diverged+reward<0+formula) → preferred=FULL_CLOSE
    D_scores = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.48]
    D_holds  = [600, 1800, 3600]
    combo_D  = list(itertools.product(D_scores, D_holds))

    for (es, hold), (sym, side) in zip(
        itertools.islice(itertools.cycle(combo_D), len(sym_side) * 2),
        itertools.cycle(sym_side),
    ):
        records.append(make_record(
            live_action="HOLD", formula_action="FULL_CLOSE", diverged=True,
            qwen3_action="HOLD",
            exit_score=es, hold_duration_sec=hold, closed_by="unknown",
            symbol=sym, side=side,
        ))

    # ── GROUP E: divergence_regret → preferred = PARTIAL_CLOSE_25 ────────────
    # Same as D but formula=PARTIAL_CLOSE_25
    for (es, hold), (sym, side) in zip(
        itertools.islice(itertools.cycle(combo_D), len(sym_side) * 2),
        itertools.cycle(sym_side),
    ):
        records.append(make_record(
            live_action="HOLD", formula_action="PARTIAL_CLOSE_25", diverged=True,
            qwen3_action="HOLD",
            exit_score=es, hold_duration_sec=hold, closed_by="unknown",
            symbol=sym, side=side,
        ))

    return records


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print('Generating synthetic scenario records…')
    records = build_scenarios()

    DEST_DIR.mkdir(parents=True, exist_ok=True)
    with DEST_FILE.open('w') as f:
        for r in records:
            f.write(json.dumps(r) + '\n')

    dpo_new = [r for r in records if r['preferred_action'] != r['live_action']]
    print(f'  Synthetic records written : {len(records)}')
    print(f'  DPO pairs (new only)      : {len(dpo_new)}')
    print(f'  Written to: {DEST_FILE}')

    # ── Combine with original + run retroactive DPO ────────────────────────────
    print('\nCombining with original 987 records…')
    all_records = []
    with ORIG_FILE.open() as f:
        for line in f:
            line = line.strip()
            if line:
                all_records.append(json.loads(line))

    orig_count = len(all_records)
    all_records.extend(records)
    print(f'  Original : {orig_count}')
    print(f'  Synthetic: {len(records)}')
    print(f'  Combined : {len(all_records)}')

    # Re-run reward engine on everything to apply PATCH-10A rules uniformly
    print('\nRe-computing preferred_action on all records…')
    dpo_pairs = []
    for rec in all_records:
        snap = {k: rec.get(k, '') for k in
                ('live_action', 'exit_score', 'formula_action', 'qwen3_action',
                 'diverged', 'side', 'entry_price', 'quantity')}
        outcome = {k: rec.get(k, '') for k in
                   ('hold_duration_sec', 'close_price', 'closed_by', 'outcome_action')}
        result = ENGINE.compute(snap, outcome)
        rec['preferred_action'] = result.preferred_action
        rec['regret_label']     = result.regret_label

        live   = rec.get('live_action', '')
        reward = float(rec.get('reward', 0))
        if result.preferred_action != live and abs(reward) >= 0.05:
            dpo_pairs.append({
                'decision_id':      rec.get('decision_id'),
                'symbol':           rec.get('symbol'),
                'side':             rec.get('side', ''),
                'live_action':      live,
                'preferred_action': result.preferred_action,
                'reward':           reward,
                'regret_label':     result.regret_label,
                'diverged':         rec.get('diverged'),
                'formula_action':   rec.get('formula_action'),
                'exit_score':       float(rec.get('exit_score') or 0),
                'source':           rec.get('source', 'unknown'),
            })

    DPO_OUT.parent.mkdir(parents=True, exist_ok=True)
    with DPO_OUT.open('w') as f:
        for p in dpo_pairs:
            f.write(json.dumps(p) + '\n')

    # ── Report ─────────────────────────────────────────────────────────────────
    import statistics

    def counts(key):
        return dict(sorted(
            collections.Counter(p[key] for p in dpo_pairs).items(),
            key=lambda x: -x[1]
        ))

    rewards = [p['reward'] for p in dpo_pairs]
    div_pairs = [p for p in dpo_pairs if p['diverged'] == 'true']

    print('\n' + '=' * 60)
    print(f'DPO DATASET v2  —  {len(dpo_pairs)} pairs total')
    print(f'Stable path: {DPO_OUT}')
    print('=' * 60)

    def tbl(d, label):
        print(f'\n{label}:')
        mx = max(d.values()) if d else 1
        for k, v in d.items():
            bar = '█' * max(1, int(v / mx * 24))
            print(f'  {k:<28} {v:>4}  {bar}')

    tbl(counts('regret_label'),    'regret_label')
    tbl(counts('preferred_action'), 'preferred_action')
    tbl(counts('live_action'),      'live_action')
    tbl(counts('symbol'),           'symbol')
    tbl(counts('side'),             'side')

    print(f'\nReward stats:')
    print(f'  min / max : {min(rewards):.4f} / {max(rewards):.4f}')
    print(f'  mean      : {statistics.mean(rewards):.4f}')
    print(f'  stdev     : {statistics.stdev(rewards):.4f}')
    print(f'\ndivergence_regret pairs : {len(div_pairs)} / {len(dpo_pairs)}')

    # ── 10 representative examples ─────────────────────────────────────────────
    print('\n' + '=' * 60)
    print('10 REPRESENTATIVE EXAMPLES')
    print('=' * 60)

    # Pick 2 from each preferred_action type + fill with divergence
    by_pa   = {}
    for p in dpo_pairs:
        by_pa.setdefault(p['preferred_action'], []).append(p)
    picks = []
    for pa_type, pool in by_pa.items():
        picks.extend(pool[:4])
    picks = picks[:10]

    for i, p in enumerate(picks, 1):
        print(f'\n[{i:02d}] {p["symbol"]:10s} {p["side"]:<5}  regret={p["regret_label"]}')
        print(f'     live={p["live_action"]}  →  preferred={p["preferred_action"]}')
        print(f'     reward={p["reward"]:.4f}  diverged={p["diverged"]}  '
              f'exit_score={p["exit_score"]:.4f}  formula={p["formula_action"]}')

    # ── Quality flags ──────────────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print('QUALITY FLAGS')
    print('=' * 60)
    flags = []

    same = [p for p in dpo_pairs if p['live_action'] == p['preferred_action']]
    if same:
        flags.append(f'CRITICAL: {len(same)} pairs have live==preferred (no signal)')

    shallow = [p for p in dpo_pairs if abs(p['reward']) < 0.06]
    if shallow:
        flags.append(f'WARN: {len(shallow)} pairs |reward|<0.06 (weak signal)')

    top_rl = max(counts('regret_label').values())
    pct_top = top_rl / len(dpo_pairs) * 100
    if pct_top > 75:
        flags.append(f'WARN: dominant regret_label covers {pct_top:.0f}% of pairs')

    n_pa = len(counts('preferred_action'))
    if n_pa < 3:
        flags.append(f'WARN: only {n_pa} distinct preferred_action type(s) — target is ≥3')

    n_sym = len(counts('symbol'))
    if n_sym < 5:
        flags.append(f'WARN: only {n_sym} symbols')

    if len(dpo_pairs) < 100:
        flags.append(f'SIZE: {len(dpo_pairs)} pairs < 100 min recommended')

    if not flags:
        flags.append('No quality issues detected.')
    for fl in flags:
        print(f'  • {fl}')

    # ── Verdict ────────────────────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print('RECOMMENDATION')
    print('=' * 60)
    n     = len(dpo_pairs)
    n_pa2 = len(counts('preferred_action'))
    div_pct = len(div_pairs) / n * 100 if n else 0
    sym_ok  = len(counts('symbol')) >= 5

    if n < 100:
        verdict = 'NOT USABLE — too few pairs.'
    elif n_pa2 < 3:
        verdict = 'PROMPT-TUNING ONLY — insufficient preferred_action diversity.'
    elif n < 300 or div_pct < 5.0:
        verdict = (
            f'FIRST EXPERIMENTAL DPO — {n} pairs, {n_pa2} preferred_action types, '
            f'{div_pct:.1f}% divergence_regret. '
            'Suitable for low-rank LoRA (rank 4-8, β≥0.5). '
            'Validated pipeline; do not deploy to production until ≥300 pairs '
            'with ≥10% divergence_regret.'
        )
    else:
        verdict = (
            f'STRONG DPO CANDIDATE — {n} pairs, {n_pa2} preferred_action types, '
            f'{div_pct:.1f}% divergence_regret, {len(counts("symbol"))} symbols. '
            'Ready for a full LoRA fine-tuning run.'
        )

    print(f'  {verdict}')
    print(f'\n  targets: n≥200={n>=200} | preferred_types≥3={n_pa2>=3} | '
          f'div_pct≥5%={div_pct>=5.0:.0f} | symbols≥5={sym_ok}')
    print()


if __name__ == '__main__':
    main()
