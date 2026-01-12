# Workspace Evaluation - Quick Reference

**One-page guide for operators**

## Common Commands

### Full Evaluation
```bash
# Standard evaluation
make eval-workspace

# View latest report
cat reports/evaluation/workspace_eval_*.md | tail -100
```

### Post-Cutover Check
```bash
# After deploying a patch
CUTOVER_TS="2026-01-10T05:43:15Z"
make eval-cutover CUTOVER_TS=$CUTOVER_TS

# Or direct call
ops/run.sh ai-engine ops/evaluation/workspace_evaluator.py --after $CUTOVER_TS
```

### Models-Only
```bash
# Quick model quality check
ops/run.sh ai-engine ops/evaluation/workspace_evaluator.py --mode models-only
```

---

## Status Meanings

| Status | Icon | Meaning | Action |
|--------|------|---------|--------|
| PASS | ✅ | All checks passed | Safe to activate |
| PASS_WITH_WARNINGS | ⚠️  | Minor issues | Monitor before activation |
| FAIL_BLOCKERS | ❌ | Critical issues | DO NOT activate |
| FAIL_INSUFFICIENT_DATA | ❌ | Not enough events | Wait for more data |

---

## Quick Checks

### 1. Are there enough events?
```bash
# Check event count
redis-cli XLEN quantum:stream:trade.intent
```
**Need:** ≥200 events

### 2. Are models passing?
```bash
# Run quality gate
make quality-gate
```
**Need:** All models PASS

### 3. Is ensemble healthy?
```bash
# Check ensemble
make eval-workspace | grep "Ensemble"
```
**Need:** Agreement 55-80%, Hard Disagree <20%

---

## Troubleshooting

### Insufficient Events
```bash
# Check if AI engine is running
systemctl status quantum-ai-engine

# Check Redis
redis-cli PING
```

### All Models Failing
```bash
# Check AI engine logs
journalctl -u quantum-ai-engine -n 100

# Run diagnostics
make diagnose
```

### Redis Connection Failed
```bash
# Start Redis
sudo systemctl start redis-server

# Verify
redis-cli PING
```

---

## Exit Codes

- `0` = PASS or PASS_WITH_WARNINGS
- `2` = FAIL (blockers or insufficient data)

---

## Report Location

`reports/evaluation/workspace_eval_<timestamp>[_post_cutover].md`

---

## When to Run

1. **Before Activation** - Always check before activating a new model
2. **After Patch** - Validate patch fixed issues
3. **Daily** - Regular health monitoring
4. **After Incident** - Post-mortem analysis

---

## Decision Tree

```
Run evaluation
    ↓
Status?
    ↓
┌───────────────┬─────────────────┬──────────────────┐
│ PASS          │ PASS_WARNINGS   │ FAIL_*           │
↓               ↓                 ↓                  │
Safe to         Monitor before    DO NOT activate   │
activate        activation        ↓                  │
                                 Fix blockers       │
                                 Re-evaluate        │
                                 ↓                  │
                                 PASS? ──────────────┘
```

---

## Integration Flow

```
┌──────────────────┐
│ Make code change │
└────────┬─────────┘
         │
         ↓
┌──────────────────┐
│ Deploy to VPS    │
└────────┬─────────┘
         │
         ↓
┌──────────────────┐
│ Wait for events  │
│ (≥200 events)    │
└────────┬─────────┘
         │
         ↓
┌──────────────────┐
│ Run evaluation   │
│ make eval-cutover│
└────────┬─────────┘
         │
         ↓
┌──────────────────┐
│ Check status     │
└────────┬─────────┘
         │
    ┌────┴────┐
    │         │
  PASS      FAIL
    │         │
    ↓         ↓
Activate   Fix &
           Retry
```

---

## Key Metrics

| Metric | Healthy Range | Warning | Blocker |
|--------|---------------|---------|---------|
| Events | ≥200 | 150-199 | <150 |
| Conf Std | ≥0.05 | 0.03-0.05 | <0.03 |
| HOLD % | ≤85% | 85-90% | >90% |
| Agreement | 55-80% | 40-90% | <40% or >90% |
| Hard Disagree | <20% | 20-30% | >30% |

---

## Contact

For issues:
1. Check report recommendations
2. Review AI engine logs
3. Run diagnostics: `make diagnose`
4. Escalate to model safety team
