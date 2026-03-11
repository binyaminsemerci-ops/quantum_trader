# Exit Agent Evaluation Report — PATCH-9 First Analysis
**Generated:** 2026-03-09 00:04 UTC  
**Report version:** v1 (pre–PATCH-8C baseline; no reward labels)  
**Analyst:** PATCH-9 replay consumer (`ops/offline/analyze_replay.py` + custom deep analysis)

---

## 1. Commands Run

```bash
# 1. Extract 10,000 most-recent decisions from VPS Redis (quantum:stream:exit.audit)
wsl scp _bootstrap_replay_export.py root@46.224.116.254:/tmp/
wsl ssh root@46.224.116.254 "python3 /tmp/_bootstrap_replay_export.py" \
    > logs/replay/replay_2026-03-09.jsonl

# 2. Run PATCH-9 evaluation pipeline
python -m ops.offline.analyze_replay \
    --replay-dir logs/replay \
    --out logs/replay/report_2026-03-09.json \
    --dpo-out logs/dpo/dpo_2026-03-09.jsonl \
    --min-reward-gap 0.1

# 3. Deep custom analysis (exit_score / symbol / side breakdown)
python _deep_analyze_exit.py
```

---

## 2. Dataset

| Attribute | Value |
|---|---|
| **Source stream** | `quantum:stream:exit.audit` |
| **Total stream depth** | 58,752 decisions |
| **Records analyzed** | 10,000 (most recent) |
| **Date range** | 2026-03-08 22:12 UTC → 2026-03-09 00:03 UTC |
| **Duration** | 1.85 hours |
| **Scan rate** | ~1.5 decisions / second (~1–2 open positions) |
| **Deployment version** | PATCH-7B (95.9%) + PATCH-7A (4.1%) |
| **Dry-run** | 100% of records (paper trading mode) |
| **Reward labels** | **NONE** — PATCH-8C not deployed to VPS yet |
| **Qwen3 decisions** | **NONE** — not deployed to VPS yet |

> **Data provenance note:** This report uses `exit.audit` as the data source because
> `quantum:stream:exit.replay` (written by PATCH-8C's ReplayWriter) does not yet exist
> on the VPS. Fields `reward`, `regret_label`, `hold_duration_sec`, and `qwen3_action`
> are all absent. Reward-based metrics are N/A until PATCH-8C is deployed.

---

## 3. Key Metrics Table

### 3a. Action Distribution

| Action | Count | Rate |
|---|---|---|
| HOLD | 9,407 | 94.07% |
| MOVE_TO_BREAKEVEN | 413 | 4.13% |
| PARTIAL_CLOSE_25 | 177 | 1.77% |
| FULL_CLOSE | 3 | 0.03% |

### 3b. Formula Divergence (live_action ≠ formula_action)

| Metric | Value |
|---|---|
| Diverged records | **3 / 10,000 (0.030%)** |
| Formula agreement | **99.97%** |
| Qwen3 available | 0 (not deployed) |
| Qwen3 divergence rate | N/A |

### 3c. Reward Statistics — ALL N/A

| Metric | Value |
|---|---|
| Mean reward overall | **N/A** — no PATCH-8C data |
| Mean reward by action | **N/A** |
| Mean reward by Qwen3 action | **N/A** — no Qwen3 data |
| Win rate / Loss rate | **N/A** |
| DPO pairs generated | **0** (no reward gap data) |

### 3d. Exit Score at Decision Time (proxy for urgency level)

| Action | n | Mean score | Median | Max | Min |
|---|---|---|---|---|---|
| PARTIAL_CLOSE_25 | 177 | **0.274** | 0.277 | 0.297 | 0.250 |
| MOVE_TO_BREAKEVEN | 413 | **0.120** | 0.113 | 0.249 | 0.036 |
| HOLD | 9,407 | **0.043** | 0.041 | 0.152 | 0.000 |
| FULL_CLOSE | 3 | **0.034** | 0.036 | 0.051 | 0.015 |

### 3e. Regret Distribution

| Label | Count | Rate |
|---|---|---|
| none | 9,997 | 99.97% |
| divergence_regret | 3 | 0.03% |
| late_hold | 0 | 0.00% |
| premature_close | 0 | 0.00% |
| Actionable rate | — | 0.03% |

> Regret labels are derived only from formula divergence at this stage.
> `late_hold` and `premature_close` require outcome reward data (PATCH-8C).

### 3f. Preferred Action Distribution (formula as baseline)

| Action | Count | Rate |
|---|---|---|
| HOLD | 9,410 | 94.10% |
| MOVE_TO_BREAKEVEN | 413 | 4.13% |
| PARTIAL_CLOSE_25 | 177 | 1.77% |
| Preferred == live | 9,997 | **99.97%** |
| Preferred ≠ live (missed signal) | 3 | 0.03% |

### 3g. Reward by Symbol — ALL N/A

| Symbol | n | Mean reward |
|---|---|---|
| LINKUSDT | 1,300 | N/A |
| BTCUSDT | 1,299 | N/A |
| BNBUSDT | 1,299 | N/A |
| XRPUSDT | 1,183 | N/A |
| ETHUSDT | 957 | N/A |
| AVAXUSDT | 883 | N/A |
| LTCUSDT | 720 | N/A |
| SOLUSDT | 647 | N/A |
| ADAUSDT | 646 | N/A |
| SUIUSDT | 610 | N/A |
| DOTUSDT | 364 | N/A |
| NEARUSDT | 92 | N/A |

### 3h. Non-HOLD Action Rate by Symbol

| Symbol | Total scans | Non-HOLD | Non-HOLD rate |
|---|---|---|---|
| **BTCUSDT** | 1,299 | **511** | **39.3%** ⚠️ |
| LINKUSDT | 1,300 | 75 | 5.8% |
| XRPUSDT | 1,183 | 4 | 0.3% |
| SOLUSDT | 647 | 3 | 0.5% |
| ETHUSDT | 957 | 0 | 0.0% |
| BNBUSDT | 1,299 | 0 | 0.0% |
| AVAXUSDT | 883 | 0 | 0.0% |
| (others) | — | 0 | 0.0% |

### 3i. Non-HOLD Action Rate by Side

| Side | Total scans | Non-HOLD | Non-HOLD rate |
|---|---|---|---|
| SHORT | 6,561 | 590 | **9.0%** |
| LONG | 3,439 | 3 | **0.1%** ⚠️ |

### 3j. Qwen3 Fallback Rate

| Metric | Value |
|---|---|
| Qwen3-available records | 0 |
| Qwen3 fallback rate | **N/A** — not deployed |

---

## 4. Top 5 Strongest Patterns

### P1 — Formula agreement is near-perfect (99.97%)
The exit agent follows the formula on 9,997 of 10,000 decisions. In the pre-Qwen3 era this confirms that the deterministic formula rules are the sole decision engine. There is no unexpected model drift, overrides, or logic surprises in normal operation.

### P2 — PARTIAL_CLOSE_25 fires in a tightly bounded exit-score band (0.25–0.30)
All 177 PARTIAL_CLOSE_25 triggers occurred with exit_score in the range [0.250, 0.297]. This is very consistent: the formula has a specific threshold that predictably maps to this action. The band is well-separated from HOLD (<0.15) and could serve as a reliable "partial profit lock" signal.

### P3 — MOVE_TO_BREAKEVEN is the most active non-HOLD action and fires at low urgency (score 0.04–0.25)
With 413 triggers, MOVE_TO_BREAKEVEN is the workhorse protective action and it fires at a much lower exit_score than PARTIAL_CLOSE_25. The formula is conservatively protecting positions before urgency climbs. This is the correct defensive behaviour against drawdowns.

### P4 — HOLD exit_score ceiling is 0.152 (HOLD never fires at high urgency)
No HOLD decision occurred when exit_score exceeded 0.152 in this window. The formula correctly uses the low-urgency zone for HOLDs. This means either: the scoring function works as designed, or the position set is currently in comfortable territory.

### P5 — System is operating in low-urgency steady state
All non-HOLD actions occur at exit_score < 0.30. The high urgency bands (0.5–1.0) produced zero decisions. The system is currently dealing with positions that are not in acute distress, consistent with normal market conditions and moderate R_net values.

---

## 5. Top 5 Weaknesses / Failure Patterns

### W1 — CRITICAL: FULL_CLOSE fires at the LOWEST exit score of all actions
The 3 FULL_CLOSE events occurred at scores **0.015, 0.036, 0.051** — lower than the 9,407 HOLD decisions (max HOLD score = 0.152) and far below PARTIAL_CLOSE_25 (min = 0.250). If exit_score represents urgency, FULL_CLOSE should be triggered by the *highest* scores. Firing at near-zero urgency suggests one of:
- FULL_CLOSE is triggered by a **separate non-score path** (e.g. stop-loss breach, time stop) that bypasses the urgency score
- Possible **formula logic inversion or edge case** for the specific SOLUSDT LONG position

**Risk:** If Qwen3 is later trained on this data, it could learn that low exit_score → FULL_CLOSE, which would be catastrophic.

### W2 — CRITICAL: LONG positions have a 0.1% action rate vs SHORT's 9.0%
LONG positions saw only 3 non-HOLD actions (all FULL_CLOSE on 1 position) in 10,000 decisions. SHORT positions triggered exits 590 times. This **asymmetric coverage** strongly suggests the formula's exit triggers are calibrated primarily or exclusively on SHORT-side conditions. LONG positions may be systematically under-monitored.

**Risk:** In a bull market, LONG positions could accumulate large drawdowns without the formula triggering protective actions.

### W3 — CRITICAL: BTCUSDT accounts for 86% of all non-HOLD exit actions
511 of 593 non-HOLD actions are on BTCUSDT (39.3% action rate), while all other symbols show ≤5.8%. This extreme concentration could mean:
- BTC had specific price action during this window that justified it
- The formula has a BTC-specific trigger or threshold mismatch
- Position sizes/leverage make BTC uniquely sensitive

**Risk:** The model cannot generalise across the trading universe if one symbol dominates the action space during training.

### W4 — No reward labels: model evaluation is blind
With 0 reward records, it is *impossible* to determine whether any decision was profitable or harmful. We do not know if:
- The 9,407 HOLDs were correct (positions recovered) or wrong (positions decayed)
- The MOVE_TO_BREAKEVEN triggers prevented losses or cut winners early
- The 3 FULL_CLOSE events (at negative R_net) were correct stop-outs or premature panic exits

**Risk:** Without deploying PATCH-8C's ReplayWriter + reward computation, the system is flying blind. There is no feedback loop.

### W5 — High exit_score zone (>0.3) has never been observed triggering any action
The formula has only been exercised in the 0.00–0.30 exit_score range. All exit decisions are low-urgency. We have no empirical data about how the agent behaves under high-urgency conditions (score 0.5–1.0). This could mean:
- All positions are in comfortable profit zones (score never climbs)
- The score calculation has a ceiling issue
- The high-urgency path is untested in dry-run conditions

---

## 6. Qwen3: Better, Worse, or Too Early to Tell?

> **Verdict: 🟡 TOO EARLY TO TELL**

Qwen3 has not been deployed to the VPS. The PATCH-8C/PATCH-9 codebase exists locally but has not been pushed to `origin/main` and is not running in the live service.

- Zero Qwen3 decisions available  
- Zero reward labels for any decision  
- No DPO pairs can be generated  
- No divergence analysis between Qwen3 and formula is possible

This evaluation covers the deterministic formula-only era (PATCH-7A/7B). Qwen3 evaluation requires deploying PATCH-8C to the VPS and running for a minimum of 24–48 hours to accumulate labelled replay records.

---

## 7. Recommended Next Action

**Priority order:**

### 🔴 1 — Deploy PATCH-8C to VPS immediately (blocker for everything else)

PATCH-8C's `ReplayWriter` writes to `quantum:stream:exit.replay` with reward, regret_label, preferred_action, hold_duration_sec, and Qwen3 fields. Without it, all reward-based metrics in PATCH-9 are permanently N/A. The local codebase is already at commit `75851fb60` (PATCH-9). The VPS is stuck at `37e5e9b8` (Feb 27 build). This gap must close before any further evaluation is meaningful.

```bash
# Push HEAD to VPS
wsl ssh root@46.224.116.254 "cd /opt/quantum && git pull origin main && systemctl restart quantum-exit-management-agent"
```

### 🔴 2 — Investigate the FULL_CLOSE / exit_score anomaly before Qwen3 trains on it

The 3 FULL_CLOSE events firing at exit_score < 0.05 while PARTIAL_CLOSE_25 requires score > 0.25 is a probable formula logic path issue. Before Qwen3 training data is collected:
- Trace the SOLUSDT LONG FULL_CLOSE path in the formula source to understand what bypassed the urgency score
- If this is a stop-loss path, add a `trigger_reason` field to the replay record to disambiguate score-driven exits from breach exits

### 🟡 3 — Investigate LONG/SHORT exit-rate asymmetry

Review formula conditions for LONG positions specifically. The 0.1% vs 9.0% action rate disparity is not explained by the current dataset. This needs a targeted formula audit.

### 🟡 4 — Run full history export (all 58,752 records) once PATCH-8C is deployed

After PATCH-8C produces a minimum 2,000 labelled replay records in `exit.replay`, run:
```bash
python -m ops.offline.export_replay --count 5000 --out logs/replay/
python -m ops.offline.analyze_replay --replay-dir logs/replay/ --dpo-out logs/dpo/dpo_v1.jsonl
```

### 🟢 5 — DPO dataset generation (after reward labels exist)

Once PATCH-8C generates ≥500 labelled records with non-null rewards, run `build_dpo_dataset()` with `min_reward_gap=0.2`. The DPO pairs will capture cases where the formula or Qwen3 disagreed with the live action AND the outcome validated the disagreement. This dataset should be used for targeted Qwen3 fine-tuning.

### 🟢 6 — Prompt tuning (only after DPO data exists)

No prompt changes are recommended at this stage. The agent is acting on formula-only signals and there is no reward feedback to validate any prompt direction. Prompt tuning without outcome data would be speculative.

---

*End of report. Next evaluation checkpoint: after PATCH-8C deployment + 24h of live replay data.*
