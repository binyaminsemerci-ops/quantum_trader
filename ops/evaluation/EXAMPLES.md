# Evaluation Framework Examples

## Example 1: Pre-Deployment Check

Before activating a new model, validate the workspace is healthy:

```bash
# Step 1: Run full evaluation
cd /home/qt/quantum_trader
make eval-workspace

# Step 2: Review report
cat reports/evaluation/workspace_eval_*.md | tail -100

# Step 3: Check status
# Look for "Overall Status" in output
# - PASS ✅ → Safe to proceed with activation
# - PASS_WITH_WARNINGS ⚠️ → Monitor before activation
# - FAIL_BLOCKERS ❌ → Fix issues first

# Step 4: If PASS, proceed with canary activation
# sudo ops/model_safety/canary_activate.sh <model_name> <env_key>
```

---

## Example 2: Post-Patch Validation (XGB Confidence Fix)

After deploying a patch to fix hardcoded confidence values:

```bash
# Patch deployed at 2026-01-10 05:43:15 UTC
CUTOVER_TS="2026-01-10T05:43:15Z"

# Step 1: Wait for sufficient events (≥200)
redis-cli XLEN quantum:stream:trade.intent
# Should return ≥200

# Step 2: Run post-cutover evaluation
make eval-cutover CUTOVER_TS=$CUTOVER_TS

# Step 3: Review comparison metrics
cat reports/evaluation/workspace_eval_*_post_cutover.md

# Look for improvements:
# - HOLD% decrease ✅
# - Confidence std increase ✅
# - P10-P90 range increase ✅

# Step 4: Verify no blockers
# If PASS, patch successfully fixed the issue
```

---

## Example 3: Degeneracy Detection

If models appear stuck or producing constant outputs:

```bash
# Step 1: Run evaluation with focus on degeneracy
make eval-workspace

# Step 2: Check degeneracy section in report
cat reports/evaluation/workspace_eval_*.md | grep -A 20 "Degeneracy Check"

# Example output showing degenerate model:
# ## Degeneracy Check ❌
# **Degenerate Models:** 1
# 
# ### xgb_model (DEGENERATE)
# - Constant confidence (std=0.0023)
# - HOLD collapse (94.2%)

# Step 3: Run detailed diagnostics
ops/run.sh ai-engine ops/model_safety/diagnose_collapse.py

# Step 4: Take action based on findings
# - If feature pipeline issue → Fix pipeline
# - If training data issue → Retrain model
# - If model architecture issue → Review model code
```

---

## Example 4: Ensemble Health Monitoring

Daily check to ensure models work together properly:

```bash
# Step 1: Run ensemble-focused evaluation
ops/run.sh ai-engine ops/evaluation/workspace_evaluator.py --mode ensemble-only

# Step 2: Check agreement metrics
# Look for "Ensemble Health" section
# - Agreement: 55-80% = HEALTHY ✅
# - Agreement: 40-90% = WARNING ⚠️
# - Agreement: <40% or >90% = UNHEALTHY ❌

# Step 3: If unhealthy, investigate
# High disagreement (>30%) → Models producing conflicting signals
# Low agreement (<40%) → Models too diverse, may need retraining
# Very high agreement (>90%) → Models too similar, reduce correlation

# Step 4: Adjust if needed
# - Disable conflicting models temporarily
# - Retrain models with different data windows
# - Review ensemble policy in config
```

---

## Example 5: Quick Model-Only Check

Fast check of individual model quality without ensemble analysis:

```bash
# Faster than full evaluation, focuses on model quality gates
ops/run.sh ai-engine ops/evaluation/workspace_evaluator.py --mode models-only

# Output shows each model's status:
# ✅ xgb_model: 345 predictions - PASS
# ✅ patchtst: 298 predictions - PASS
# ❌ catboost: 267 predictions - FAIL
#    - HOLD majority 87.3% (>85% threshold)

# If any models FAIL, run full diagnostics
```

---

## Example 6: Cutover Analysis with Custom Event Threshold

For testing environments with lower traffic:

```bash
# Use lower event threshold for testing
CUTOVER_TS="2026-01-10T05:43:15Z"

ops/run.sh ai-engine ops/evaluation/workspace_evaluator.py \
  --after $CUTOVER_TS \
  --min-events 50

# Useful for:
# - Testnet environments
# - Early validation with limited data
# - Quick smoke tests
```

---

## Example 7: Automated Pipeline Integration

Integrate evaluation into CI/CD pipeline:

```bash
#!/bin/bash
# deploy_and_validate.sh

set -e

# Step 1: Deploy code
./deploy.sh

# Step 2: Wait for AI engine restart
sleep 30
systemctl is-active quantum-ai-engine

# Step 3: Wait for events
echo "Waiting for events..."
EVENTS=0
while [ $EVENTS -lt 200 ]; do
    EVENTS=$(redis-cli XLEN quantum:stream:trade.intent)
    echo "Events: $EVENTS/200"
    sleep 10
done

# Step 4: Run evaluation
CUTOVER_TS=$(date -u -Iseconds)
make eval-cutover CUTOVER_TS=$CUTOVER_TS

# Step 5: Check exit code
if [ $? -eq 0 ]; then
    echo "✅ Evaluation PASSED"
    exit 0
else
    echo "❌ Evaluation FAILED"
    echo "Review report: reports/evaluation/workspace_eval_*_post_cutover.md"
    exit 1
fi
```

---

## Example 8: Investigating Insufficient Events

If evaluation fails due to insufficient data:

```bash
# Check current event count
redis-cli XLEN quantum:stream:trade.intent

# If count is low (<200):
# 1. Check AI engine is running
systemctl status quantum-ai-engine

# 2. Check AI engine is producing trade intents
journalctl -u quantum-ai-engine -n 50 | grep "trade.intent"

# 3. Check Redis stream directly
redis-cli XREVRANGE quantum:stream:trade.intent + - COUNT 5

# 4. If no events are being generated:
# - Check feature pipeline
# - Check market data ingestion
# - Review AI engine logs for errors

# 5. Once fixed, re-run evaluation
make eval-workspace
```

---

## Example 9: Report Analysis

Understanding the evaluation report:

```bash
# View latest report
cat reports/evaluation/workspace_eval_*.md

# Key sections to review:

# 1. Executive Summary
#    - Overall status (PASS/FAIL)
#    - Blockers (must fix)
#    - Warnings (should monitor)
#    - Recommendations (action items)

# 2. Per-Model Analysis
#    - Action distribution (BUY/SELL/HOLD %)
#    - Confidence stats (mean, std, P10-P90)
#    - Quality gate failures (if any)

# 3. Ensemble Health
#    - Active model count (≥3 recommended)
#    - Agreement % (55-80% healthy)
#    - Hard disagree % (<20% healthy)

# 4. Degeneracy Check
#    - Lists any stuck/degenerate models
#    - Shows reasons (constant output, collapse, etc.)
```

---

## Example 10: Comparison Workflow

Compare workspace health over time:

```bash
# Run evaluation daily and archive reports
DATE=$(date +%Y%m%d)
make eval-workspace

# Archive report
mkdir -p reports/evaluation/archive/
cp reports/evaluation/workspace_eval_*.md \
   reports/evaluation/archive/workspace_eval_$DATE.md

# Compare with previous day
diff -u \
  reports/evaluation/archive/workspace_eval_$(date -d '1 day ago' +%Y%m%d).md \
  reports/evaluation/archive/workspace_eval_$DATE.md \
  | grep -E "^\+|^-" | head -50

# Look for trends:
# - Increasing HOLD % → Models getting stuck
# - Decreasing confidence std → Models flattening
# - Decreasing agreement → Models diverging
```

---

## Exit Code Reference

```bash
# Use exit codes in scripts for automation
make eval-workspace
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "PASS - Safe to proceed"
elif [ $EXIT_CODE -eq 2 ]; then
    echo "FAIL - Blockers detected"
else
    echo "UNKNOWN - Unexpected error"
fi
```

---

## Quick Decision Guide

```
Need to activate a model?
    ↓
Run: make eval-workspace
    ↓
Check status in output
    ↓
┌─────────────────┬──────────────────┬───────────────┐
│ PASS ✅         │ PASS_WARNINGS ⚠️ │ FAIL ❌       │
│                 │                  │               │
│ → Activate      │ → Review         │ → Fix         │
│ → Monitor       │ → Monitor        │ → Re-evaluate │
│                 │ → Consider wait  │ → DO NOT      │
│                 │                  │   activate    │
└─────────────────┴──────────────────┴───────────────┘
```
