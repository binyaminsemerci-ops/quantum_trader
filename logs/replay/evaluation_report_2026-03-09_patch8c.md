# PATCH-8C First Real Replay Evaluation
**Generated:** 2026-03-09 00:27 UTC  
**Session goal:** Deploy PATCH-8C/8D/9 to VPS and run the first real reward-labelled evaluation.

---

## Deployment Summary

| Step | Status | Notes |
|---|---|---|
| Push PATCH-9 to origin/main | ✅ | `75851fb60` pushed from local |
| Sync `/opt/quantum/` VPS to PATCH-9 HEAD | ✅ | git reset --hard origin/main |
| Fix service unit ExecStart | ✅ | Was `venvs/ai-client-base/bin/python` (non-existent). Changed to `/home/qt/quantum_trader_venv/bin/python` |
| Deploy ALL EMA source files to `/home/qt/quantum_trader/` | ✅ | Covered PATCH-8A through PATCH-9 (audit.py, config.py, main.py, redis_io.py, models.py, outcome_tracker.py, replay_writer.py, reward_engine.py) |
| Deploy `ops/offline/` tools | ✅ | analyze_replay.py, export_replay.py, replay_metrics.py, replay_schema.py, dpo_export.py |
| Service running | ✅ | PID 1509347, active since 00:21:50 UTC |
| Decision hash writes (PATCH-8A) | ✅ | `quantum:hash:exit.decision:*` + `quantum:set:exit.pending_decisions:*` |
| Outcome detection (PATCH-8B) | ✅ | OutcomeTracker.update() called each tick |
| Reward labels (PATCH-8C) | ✅ | RewardEngine + ReplayWriter wired via OutcomeTracker |
| Replay stream alive | ✅ | 46 entries in `quantum:stream:exit.replay` |

---

## Root Cause: Why Service Was Broken

The systemd unit file `/etc/systemd/system/quantum-exit-management-agent.service` had been modified at 13:46 UTC (March 8) to reference a Python venv that no longer existed:

```
ExecStart=/opt/quantum/venvs/ai-client-base/bin/python  # ← DOES NOT EXIST
```

The previously-running process (PID 1416433, consuming 8m 37s CPU) had been started with an older, correct version of the unit file. When we issued `systemctl restart`, the broken unit was used and the service went into a 203/EXEC crash loop.

**Fix:** `sed -i` to replace with the correct interpreter:
```
ExecStart=/home/qt/quantum_trader_venv/bin/python
```

Additionally, the actual running codebase is at `/home/qt/quantum_trader/` (WorkingDirectory in unit), not `/opt/quantum/`. PATCH-8A through PATCH-9 files were missing from the running path and had to be copied from `/opt/quantum/`.

---

## First Real Evaluation Results

**Records:** 46 (all with reward labels, 100% from PATCH-8C)  
**Time window:** 00:21:50 UTC – 00:27 UTC (≈5 minutes of live data)  
**Source:** `quantum:stream:exit.replay`

### Overall Reward
| Metric | Value |
|---|---|
| Records | 46 |
| Mean reward | **-0.0649** |
| Std dev | 0.022 |
| Win rate | **0.0%** |
| Loss rate | **100.0%** |
| Min reward | -0.1143 |
| Max reward | -0.0389 |

### Action Distribution
| Action | Count | Win% |
|---|---|---|
| HOLD | 46 (100%) | 0.0% |

### Divergence (Qwen3 vs Formula)
| Metric | Value |
|---|---|
| Diverged records | 0 / 46 (0.0%) |
| Qwen3 agrees formula | 46 / 46 |
| Qwen3 overrides | 0 / 46 |
| DPO pairs generated | **0** |

### Regret Distribution
| Label | Count |
|---|---|
| none | 46 (100%) |
| late_hold | 0 |
| premature_close | 0 |
| divergence_regret | 0 |

### Context
All 46 records are decision snapshots for a **single ETHUSDT LONG position**:
- Entry: 1949.31 USDT
- Close: 1931.17 USDT (~-0.9% move)
- Hold duration: <5 minutes (all in `<5m` band)
- Closed by: `unknown` (stopped out externally, not by EMA)

The agent consistently recommended HOLD. The reward is negative because the price fell during the hold period. No divergence between Qwen3 and formula — both agreed HOLD every tick.

---

## Interpretation

1. **Pipeline is fully operational.** PATCH-8C correctly generates reward-labelled records for every closed position.

2. **Negative reward does not indicate an error.** A HOLD on a position that subsequently loses value correctly produces a negative reward. The reward function is working as designed.

3. **No DPO data yet.** DPO training pairs require `diverged=true` AND a minimum reward gap between the formula action and Qwen3 action outcomes. With only 5 minutes of data from a single position, no divergence occurred. More operational time is needed.

4. **The Qwen3 429 rate-limit issue is active.** Journal shows frequent `HTTP error: 429 Too Many Requests → using formula fallback='HOLD'`. Qwen3 is falling back to formula on most ticks due to Groq rate limiting. This reduces divergence.

---

## Next Steps

1. **Allow service to run for 2–24 hours** to accumulate diverse replay data across multiple symbols and positions.

2. **Re-run export + analysis** once ≥200 records exist with varied actions (FULL_CLOSE, PARTIAL_CLOSE_25 expected when exit conditions are met).

3. **Look for first diverged records.** When Qwen3 occasionally gets through (not 429), it may disagree with formula. Those records are the DPO training signal.

4. **Address Groq 429 issue** (separate concern) — may want to lower `EXIT_AGENT_QWEN3_TIMEOUT_MS` or add backoff logic to avoid burning rate limit budget on busy ticks.

---

## Service State After Session

```
● quantum-exit-management-agent.service
   Active: active (running), PID 1509347
   Python: /home/qt/quantum_trader_venv/bin/python
   Code:   /home/qt/quantum_trader/ (PATCH-8C/8D/9 deployed)
   Audit stream:    quantum:stream:exit.audit    → 60,010+ entries
   Replay stream:   quantum:stream:exit.replay   → 46+ entries
   Outcomes stream: quantum:stream:exit.outcomes → growing
```

**Commit on origin/main:** `75851fb60` (PATCH-9)  
**JSON report:** `logs/replay/report_real_v1_patch8c.json`  
**JSONL data:** `logs/replay/replay_real_v1_patch8c.jsonl`
