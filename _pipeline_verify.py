"""
END-TO-END PIPELINE VERIFICATION
=================================
Tests every step of the signal pipeline:

  Binance API
      ↓ [1] Data fetch
  market.tick (Redis stream)
      ↓ [2] Stream health
  quantum:stream:features.* (Redis stream)
      ↓ [3] Feature completeness (49 features)
  EnsemblePredictorService
      ↓ [4a-e] Each agent: data → model → prediction
  quantum:stream:signal.score (Redis stream)
      ↓ [5] Signal output freshness
  Logging
      ↓ [6] Log integrity (no ERROR/WARN per agent)
"""
import sys, subprocess, json, re
from datetime import datetime, timezone
from pathlib import Path

# ── helpers ──────────────────────────────────────────────────────────────────
OK   = "  ✓"
WARN = "  ⚠"
FAIL = "  ✗"

def ssh(cmd: str, timeout: int = 20) -> str:
    r = subprocess.run(
        ["wsl", "bash", "-c",
         f"ssh -o StrictHostKeyChecking=no -i ~/.ssh/hetzner_fresh "
         f"root@46.224.116.254 '{cmd}'"],
        capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=timeout
    )
    return (r.stdout + r.stderr).strip()

def section(title: str):
    print(f"\n{'═'*66}")
    print(f"  {title}")
    print('═'*66)

results: dict[str, str] = {}   # step → OK/WARN/FAIL

# ═════════════════════════════════════════════════════════════════════════════
# [0] Service liveness
# ═════════════════════════════════════════════════════════════════════════════
section("[0] Service liveness")

svc_status = ssh("systemctl is-active quantum-ensemble-predictor")
pid_out    = ssh("systemctl show quantum-ensemble-predictor --property=MainPID --value")
print(f"  quantum-ensemble-predictor : {svc_status}  (PID {pid_out.strip()})")

if "active" in svc_status:
    print(f"{OK} Service is running")
    results["service"] = "OK"
else:
    print(f"{FAIL} Service NOT running")
    results["service"] = "FAIL"

# ═════════════════════════════════════════════════════════════════════════════
# [1] Redis connectivity + stream existence
# ═════════════════════════════════════════════════════════════════════════════
section("[1] Redis streams")

streams = [
    "quantum:stream:features",
    "quantum:stream:signal.score",
    "quantum:rl:experience",
    "market.tick",
]

for stream in streams:
    out = ssh(f"redis-cli XLEN '{stream}' 2>&1")
    try:
        length = int(out.strip())
        age_out = ssh(
            f"redis-cli XREVRANGE '{stream}' + - COUNT 1 2>&1 | head -3"
        )
        age_ts_match = re.search(r'^(\d{13})', age_out, re.M)
        if age_ts_match:
            ts_ms  = int(age_ts_match.group(1))
            age_s  = (datetime.now(tz=timezone.utc).timestamp() * 1000 - ts_ms) / 1000
            age_str = f"last msg {age_s:.0f}s ago"
        else:
            age_str = "age unknown"
        icon = OK if length > 0 else WARN
        print(f"{icon} {stream:<42}  len={length:>8}  {age_str}")
        results[f"stream:{stream}"] = "OK" if length > 0 else "WARN"
    except ValueError:
        print(f"{FAIL} {stream:<42}  ERROR: {out[:60]}")
        results[f"stream:{stream}"] = "FAIL"

# ═════════════════════════════════════════════════════════════════════════════
# [2] Feature stream: 49-feature completeness (last message)
# ═════════════════════════════════════════════════════════════════════════════
section("[2] Feature stream completeness (49 features)")

FEATURES_V6 = [
    'returns','log_returns','price_range','body_size','upper_wick','lower_wick',
    'is_doji','is_hammer','is_engulfing','gap_up','gap_down','rsi','macd',
    'macd_signal','macd_hist','stoch_k','stoch_d','roc','ema_9','ema_9_dist',
    'ema_21','ema_21_dist','ema_50','ema_50_dist','ema_200','ema_200_dist',
    'sma_20','sma_50','adx','plus_di','minus_di','bb_middle','bb_upper',
    'bb_lower','bb_width','bb_position','atr','atr_pct','volatility',
    'volume_sma','volume_ratio','obv','obv_ema','vpt','momentum_5',
    'momentum_10','momentum_20','acceleration','relative_spread',
]

feat_raw = ssh("redis-cli XREVRANGE 'quantum:stream:features' + - COUNT 1 2>&1")
# Parse field-value pairs from Redis XREVRANGE output
feat_fields: set[str] = set()
lines = feat_raw.splitlines()
# XREVRANGE output: line 0 = message-ID, then field/value pairs on alternating lines.
# Field names are at indices 1, 3, 5 … (odd); values at indices 2, 4, 6 … (even).
for i in range(1, len(lines), 2):
    name = lines[i].strip()
    if name:  # skip blank lines
        feat_fields.add(name)

if feat_fields:
    missing = [f for f in FEATURES_V6 if f not in feat_fields]
    extra   = [f for f in feat_fields if f not in FEATURES_V6 and not f.startswith(('1', '2'))]
    if not missing:
        print(f"{OK} All 49 V6 features present in stream")
        results["features_49"] = "OK"
    else:
        print(f"{WARN} Missing features ({len(missing)}): {missing[:10]}")
        results["features_49"] = "WARN"
    if extra:
        print(f"  Extra fields in stream: {extra[:5]}")
else:
    print(f"{WARN} Could not parse feature stream (stream may use JSON blob)")
    # Try JSON blob approach
    feat_json = ssh("redis-cli XREVRANGE 'quantum:stream:features' + - COUNT 1 2>&1")
    print(f"  Raw (first 200 chars): {feat_json[:200]}")
    results["features_49"] = "WARN"

# ═════════════════════════════════════════════════════════════════════════════
# [3] Model files on VPS
# ═════════════════════════════════════════════════════════════════════════════
section("[3] Model files on VPS (/app/models/)")

# Patterns match what each agent's _find_latest() actually picks up:
#   XGB:      xgb_v6_TIMESTAMP.pkl  (no scaler/meta/features suffix)
#   LGBM:     lightgbm_v*_v3.pkl    (explicit v3 suffix from train_lgbm_v2)
#   NHiTS:    nhits_v*_v3.pth       (explicit v3 suffix)
#   PatchTST: patchtst_v6_*.pth     (agent prefers v6 prefix; no _v3 suffix)
#   TFT:      tft_v*.pth            (agent picks newest mtime; old=no suffix, new=_v3)
model_checks = [
    ("XGB",      "xgb_v6_2*.pkl",                 "xgboost"),  # ls cmd excludes _scaler/_meta
    ("LGBM",     "lightgbm_v*_v3.pkl",            "lightgbm"),
    ("NHiTS",    "nhits_v*_v3.pth",               "nhits"),
    ("PatchTST", "patchtst_v6_2*.pth",            "patchtst"),
    ("TFT",      "tft_v6_2*_v3.pth",              "tft"),
]

for name, pattern, _ in model_checks:
    # Exclude auxiliary files (scaler, meta, features) — keep main model only
    out = ssh(
        f"ls -t /app/models/{pattern} 2>/dev/null"
        " | grep -v _scaler | grep -v _meta | grep -v _features | head -1"
    )
    if out.strip():
        fname = Path(out.strip()).name
        size  = ssh(f"du -h /app/models/{fname} 2>/dev/null | cut -f1")
        print(f"{OK} {name:<10} {fname}  ({size.strip()})")
        results[f"model:{name}"] = "OK"
    else:
        print(f"{FAIL} {name:<10} no file matching {pattern}")
        results[f"model:{name}"] = "FAIL"

# ═════════════════════════════════════════════════════════════════════════════
# [4] Live agent predictions (last 60s of logs)
# ═════════════════════════════════════════════════════════════════════════════
section("[4] Live agent predictions (last 2 min)")

agent_patterns = {
    "LGBM":     r"LGBM \w+: (BUY|SELL|HOLD) \(conf=([\d.]+), probs=\[([0-9., ]+)\]",
    "XGB":      r"XGB \w+: (BUY|SELL|HOLD) ([\d.]+)%",
    "NHiTS":    r"N-HiTS \w+: (BUY|SELL|HOLD) \(conf=([\d.]+)",
    "PatchTST": r"PatchTST[^\|]+: (BUY|SELL|HOLD) \(conf=([\d.]+)",
    "TFT":      r"TFT-Agent.*\| \w+.{1,5}(BUY|SELL|HOLD) \(TFT, conf=([\d.]+)\)",
}

log_lines = ssh("journalctl -u quantum-ensemble-predictor -n 1000 --no-pager 2>&1", timeout=30)

for agent, pat in agent_patterns.items():
    hits = re.findall(pat, log_lines)
    if not hits:
        print(f"{FAIL} {agent:<10}  NOT SEEN in recent logs")
        results[f"pred:{agent}"] = "FAIL"
        continue

    actions = [h[0] for h in hits]
    confs   = [float(h[1]) for h in hits]
    from collections import Counter
    cnt = Counter(actions)
    total = len(actions)
    sell_pct = cnt.get("SELL", 0) / total * 100
    hold_pct = cnt.get("HOLD", 0) / total * 100
    buy_pct  = cnt.get("BUY",  0) / total * 100
    avg_conf = sum(confs) / len(confs)
    dominant = max(cnt, key=cnt.get)
    dominant_pct = cnt[dominant] / total * 100

    # Flag if one class > 92% AND conf > 0.80 (definitive bias) or fallback pattern
    is_fallback = avg_conf > 0.85 and dominant_pct > 95
    is_biased   = dominant_pct > 92

    # Special case: LGBM check probs field for HOLD=0%
    lgbm_hold_ok = True
    if agent == "LGBM":
        hold_probs = []
        for h in hits[:20]:
            if len(h) > 2:
                try:
                    p = [float(x) for x in h[2].split(",")]
                    if len(p) == 3:
                        hold_probs.append(p[1])
                except Exception:
                    pass
        if hold_probs:
            avg_hold_prob = sum(hold_probs) / len(hold_probs)
            lgbm_hold_ok = avg_hold_prob > 0.08
            hold_note = f"  avg HOLD-prob={avg_hold_prob:.3f}"
        else:
            hold_note = ""
    else:
        hold_note = ""

    icon = FAIL if is_fallback else (WARN if is_biased else OK)
    tag  = "FALLBACK?" if is_fallback else ("BIASED" if is_biased else "OK")
    print(
        f"{icon} {agent:<10}  "
        f"S={sell_pct:5.1f}% H={hold_pct:5.1f}% B={buy_pct:5.1f}%  "
        f"n={total:4}  conf={avg_conf:.3f}  [{tag}]{hold_note}"
    )
    results[f"pred:{agent}"] = tag

# ═════════════════════════════════════════════════════════════════════════════
# [5] Output stream: signal.score freshness & schema
# ═════════════════════════════════════════════════════════════════════════════
section("[5] Output stream: quantum:stream:signal.score")

score_out = ssh("redis-cli XREVRANGE 'quantum:stream:signal.score' + - COUNT 3 2>&1")
if score_out and "1)" not in score_out and len(score_out) < 5:
    print(f"{WARN} signal.score stream empty or not found")
    results["signal_score"] = "WARN"
else:
    # Check freshness
    ts_match = re.search(r'(\d{13})', score_out)
    if ts_match:
        ts_ms  = int(ts_match.group(1))
        age_s  = (datetime.now(tz=timezone.utc).timestamp() * 1000 - ts_ms) / 1000
        print(f"  Last entry: {age_s:.0f}s ago")
        if age_s < 120:
            print(f"{OK} signal.score is fresh (< 2 min)")
            results["signal_score"] = "OK"
        else:
            print(f"{WARN} signal.score stale ({age_s:.0f}s old)")
            results["signal_score"] = "WARN"
    else:
        print(f"  Raw output (200 chars): {score_out[:200]}")
        results["signal_score"] = "WARN"

    # Schema check: should have suggested_action, confidence, horizon
    for field in ["suggested_action", "confidence", "horizon"]:
        if field in score_out:
            print(f"{OK} field '{field}' present in output schema")
        else:
            print(f"{WARN} field '{field}' NOT found in last 3 messages")

# ═════════════════════════════════════════════════════════════════════════════
# [6] Error/warning scan in service logs (last 500 lines)
# ═════════════════════════════════════════════════════════════════════════════
section("[6] Log error scan (last 500 lines)")

# Note: the SSH helper wraps the command in single quotes, so we must use
# double quotes for the grep pattern (single-inside-single breaks parsing).
error_out = ssh(
    'journalctl -u quantum-ensemble-predictor -n 500 --no-pager 2>&1'
    ' | grep -iE "ERROR|CRITICAL|Traceback|ValueError|KeyError|fallback_prediction"'
    ' | grep -v grep | head -20'
)
error_lines = [l for l in error_out.splitlines() if l.strip()]

fallback_count = sum(1 for l in error_lines if "fallback" in l.lower())
exception_count = sum(1 for l in error_lines if any(
    x in l for x in ["ERROR", "CRITICAL", "Traceback", "ValueError", "KeyError", "Exception"]))

if fallback_count == 0 and exception_count == 0:
    print(f"{OK} No errors, no fallbacks in recent 500 log lines")
    results["log_errors"] = "OK"
else:
    if fallback_count > 0:
        print(f"{WARN} {fallback_count} fallback occurrences detected:")
        for l in error_lines:
            if "fallback" in l.lower():
                print(f"     {l.strip()[-100:]}")
    if exception_count > 0:
        print(f"{FAIL} {exception_count} error/exception lines:")
        for l in error_lines:
            if any(x in l for x in ["ERROR", "CRITICAL", "Traceback", "ValueError"]):
                print(f"     {l.strip()[-100:]}")
    results["log_errors"] = "WARN" if exception_count == 0 else "FAIL"

# ═════════════════════════════════════════════════════════════════════════════
# [7] Local model sanity (random + zeros)
# ═════════════════════════════════════════════════════════════════════════════
section("[7] Local model calibration (random N(0,1) x500)")

try:
    import numpy as np, joblib, pickle
    import sys; sys.path.insert(0, ".")
    import xgboost as xgb

    LABELS = ["SELL", "HOLD", "BUY"]

    model_tests = [
        ("LGBM", lambda: (
            joblib.load(sorted(Path("ai_engine/models").glob("lightgbm_v*_v3.pkl"))[-1]),
            lambda m, X: m.predict_proba(X)
        )),
        ("XGB", lambda: (
            pickle.load(open(sorted(p for p in Path("ai_engine/models").glob("xgb_v6_*.pkl")
                              if p.stem.count("_") == 3)[-1], "rb")),
            lambda m, X: m.predict(xgb.DMatrix(X))
        )),
    ]

    np.random.seed(42)
    X = np.random.randn(500, 49).astype("f")

    for name, loader in model_tests:
        try:
            model, predict_fn = loader()
            proba = predict_fn(model, X)
            cnt = np.bincount(np.argmax(proba, axis=1), minlength=3)
            dominant_pct = cnt.max() / cnt.sum() * 100
            icon = OK if dominant_pct < 90 else WARN
            print(f"{icon} {name:<10}  SELL={cnt[0]:3d} HOLD={cnt[1]:3d} BUY={cnt[2]:3d}  "
                  f"dominant={dominant_pct:.0f}%  {'OK' if dominant_pct<90 else 'BIASED'}")
            results[f"local_cal:{name}"] = "OK" if dominant_pct < 90 else "WARN"
        except Exception as e:
            print(f"{FAIL} {name:<10}  {e}")
            results[f"local_cal:{name}"] = "FAIL"
except Exception as e:
    print(f"{WARN} Local model test skipped: {e}")

# ═════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═════════════════════════════════════════════════════════════════════════════
section("SUMMARY")

fail_count = sum(1 for v in results.values() if v == "FAIL")
warn_count = sum(1 for v in results.values() if v in ("WARN", "BIASED"))
ok_count   = sum(1 for v in results.values() if v == "OK")

print(f"\n  {'CHECK':<36}  STATUS")
print(f"  {'─'*36}  ──────")
for k, v in results.items():
    icon = OK if v == "OK" else (WARN if v in ("WARN","BIASED") else FAIL)
    print(f"{icon} {k:<36}  {v}")

print()
print(f"  Total: {ok_count} OK  |  {warn_count} WARN  |  {fail_count} FAIL")
if fail_count == 0 and warn_count == 0:
    print("\n  ✅ FULL PIPELINE VERIFIED — all checks passed")
elif fail_count == 0:
    print("\n  ⚠  PIPELINE OK WITH WARNINGS — review above")
else:
    print("\n  ❌ PIPELINE HAS FAILURES — fix required")
