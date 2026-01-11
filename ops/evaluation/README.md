# Evaluation Framework

**Comprehensive workspace evaluation for Quantum Trader AI models**

## Overview

The evaluation framework provides tools for assessing the quality and health of AI models in the current workspace. It combines quality gates, degeneracy detection, ensemble health metrics, and comprehensive reporting.

## Components

### 1. Workspace Evaluator (Main Tool)

**Script:** `ops/evaluation/workspace_evaluator.py`

**Purpose:** Comprehensive evaluation orchestrator that checks:
- Model quality (passes quality gates)
- Event count validation (minimum 200 events)
- Degeneracy detection (constant outputs, stuck models)
- Ensemble health (agreement metrics)
- Overall workspace status

**Usage:**

```bash
# Full evaluation (all checks)
ops/run.sh ai-engine ops/evaluation/workspace_evaluator.py --mode full

# Post-cutover evaluation (compare before/after patch)
ops/run.sh ai-engine ops/evaluation/workspace_evaluator.py --after 2026-01-10T05:43:15Z

# Models-only check
ops/run.sh ai-engine ops/evaluation/workspace_evaluator.py --mode models-only

# Custom minimum events threshold
ops/run.sh ai-engine ops/evaluation/workspace_evaluator.py --min-events 300
```

**Exit Codes:**
- `0` = PASS (all checks passed or warnings only)
- `2` = FAIL (blockers detected)

**Output:** `reports/evaluation/workspace_eval_<timestamp>.md`

---

## Evaluation Modes

### Full Mode (Default)
Runs all evaluation checks:
1. Event count validation
2. Per-model quality analysis
3. Ensemble health assessment
4. Degeneracy detection
5. Overall status determination

### Models-Only Mode
Focuses on individual model quality:
- Quality gate checks
- Action distribution analysis
- Confidence statistics
- Violation detection

### Ensemble-Only Mode
Focuses on ensemble behavior:
- Agreement metrics
- Hard disagree detection
- Active model count

---

## Quality Checks

### Event Count Validation
- **Requirement:** Minimum 200 events (configurable)
- **Reason:** Insufficient data = unreliable metrics
- **Status:** BLOCKER if below threshold

### Model Quality Gate
Each model must pass:
- No single action >70% (prevents bias)
- Confidence std ≥0.05 (prevents flatness)
- P10-P90 range ≥0.12 (prevents narrow distribution)
- HOLD ≤85% (prevents dead zone collapse)
- No confidence violations (invalid values)

### Degeneracy Detection
Identifies "stuck" or degenerate models:
- **Constant confidence:** std <0.01
- **Single-action dominance:** >95%
- **HOLD collapse:** >90%
- **Confidence violations:** Invalid/NaN values

### Ensemble Health
Checks model agreement and diversity:
- **HEALTHY:** 55-80% agreement, <20% hard disagree
- **WARNING:** 40-90% agreement, <30% hard disagree
- **UNHEALTHY:** Outside healthy ranges
- **DEGRADED:** <2 active models

---

## Cutover Analysis

When using `--after <timestamp>`, the evaluator:
1. Analyzes post-cutover events only
2. Validates minimum event count (default: 200)
3. Checks for improvements vs pre-cutover baseline
4. Reports whether patch fixed issues

**Use case:** Validate that a patch/deployment improved model quality

```bash
# Analyze events after AI engine restart
ops/run.sh ai-engine ops/evaluation/workspace_evaluator.py \
  --after 2026-01-10T05:43:15Z
```

---

## Status Determination

### PASS ✅
- All models pass quality gates
- No degeneracy detected
- Ensemble health is HEALTHY
- **Action:** Safe to proceed with activation

### PASS_WITH_WARNINGS ⚠️
- All models pass quality gates
- Ensemble health is WARNING (minor issues)
- **Action:** Monitor before activation

### FAIL_BLOCKERS ❌
- One or more models fail quality gates
- Degeneracy detected
- Ensemble health is UNHEALTHY
- **Action:** DO NOT activate - fix blockers first

### FAIL_INSUFFICIENT_DATA ❌
- Event count below minimum threshold
- **Action:** Wait for more data

---

## Report Format

Reports are generated in Markdown format at:
`reports/evaluation/workspace_eval_<timestamp>_[post_cutover].md`

**Sections:**
1. **Executive Summary** - Overall status and blockers/warnings
2. **Event Metrics** - Event count and coverage
3. **Per-Model Analysis** - Detailed breakdown per model
4. **Ensemble Health** - Agreement and diversity metrics
5. **Degeneracy Check** - Stuck model detection
6. **Recommendations** - Action items

---

## Integration with Quality Gate

The workspace evaluator extends the quality gate (`ops/model_safety/quality_gate.py`) with:
- Multi-model orchestration
- Ensemble health metrics
- Degeneracy detection
- Cutover-aware analysis
- Comprehensive reporting

**Relationship:**
- Quality Gate: Per-model PASS/FAIL checks
- Workspace Evaluator: Holistic workspace assessment

---

## Common Workflows

### 1. Pre-Deployment Check
```bash
# Before activating a new model
ops/run.sh ai-engine ops/evaluation/workspace_evaluator.py --mode full

# Check report
cat reports/evaluation/workspace_eval_*.md | tail -100

# If PASS, proceed with canary activation
# If FAIL, investigate blockers
```

### 2. Post-Patch Validation
```bash
# After deploying a patch (e.g., fixing hardcoded confidence)
CUTOVER_TS="2026-01-10T05:43:15Z"

# Run evaluation on post-cutover events
ops/run.sh ai-engine ops/evaluation/workspace_evaluator.py --after $CUTOVER_TS

# Compare pre/post metrics in report
cat reports/evaluation/workspace_eval_*_post_cutover.md
```

### 3. Model Health Monitoring
```bash
# Regular health check (daily)
ops/run.sh ai-engine ops/evaluation/workspace_evaluator.py --mode models-only

# Archive reports for trending
mv reports/evaluation/workspace_eval_*.md reports/evaluation/archive/
```

### 4. Degeneracy Investigation
```bash
# If models appear stuck or producing constant outputs
ops/run.sh ai-engine ops/evaluation/workspace_evaluator.py --mode full

# Check degeneracy section in report
# Run diagnostics on flagged models
```

---

## Troubleshooting

### "Insufficient events" Error
**Cause:** Less than 200 events in Redis stream

**Solutions:**
1. Wait for more trading activity
2. Lower threshold: `--min-events 100`
3. Check AI engine is running and generating trade intents

### "Redis connection failed" Error
**Cause:** Redis not running or not accessible

**Solutions:**
1. Check Redis: `redis-cli PING`
2. Start Redis: `sudo systemctl start redis-server`
3. Verify port 6379 is accessible

### All Models Failing
**Cause:** Possible system-wide issue

**Solutions:**
1. Check AI engine logs: `journalctl -u quantum-ai-engine -n 100`
2. Verify feature pipeline is working
3. Check for recent configuration changes
4. Run quality gate diagnostics: `ops/model_safety/diagnose_collapse.py`

### High Hard Disagree
**Cause:** Models producing conflicting signals (BUY vs SELL)

**Solutions:**
1. Review model training data
2. Check for market regime changes
3. Consider retraining models
4. Temporarily disable conflicting models

---

## Makefile Integration

Add to `Makefile`:

```makefile
.PHONY: eval-workspace eval-cutover

eval-workspace:
	ops/run.sh ai-engine ops/evaluation/workspace_evaluator.py --mode full

eval-cutover:
	@echo "Usage: make eval-cutover CUTOVER_TS=2026-01-10T05:43:15Z"
	@if [ -z "$(CUTOVER_TS)" ]; then \
		echo "Error: CUTOVER_TS not set"; \
		exit 1; \
	fi
	ops/run.sh ai-engine ops/evaluation/workspace_evaluator.py --after $(CUTOVER_TS)
```

**Usage:**
```bash
make eval-workspace              # Full evaluation
make eval-cutover CUTOVER_TS=... # Post-cutover analysis
```

---

## Future Enhancements

Planned features:
1. **Historical trending** - Track metrics over time
2. **Alert thresholds** - Automated notifications
3. **Model comparison** - Side-by-side analysis
4. **Confidence calibration** - Validate probability accuracy
5. **Performance attribution** - Link to trading outcomes

---

## Related Tools

- **Quality Gate:** `ops/model_safety/quality_gate.py` - Per-model validation
- **Scoreboard:** `ops/model_safety/scoreboard.py` - Status overview
- **Diagnostics:** `ops/model_safety/diagnose_collapse.py` - Debug model issues

---

## Support

For issues or questions:
1. Review report recommendations section
2. Check AI engine logs: `journalctl -u quantum-ai-engine`
3. Run quality gate diagnostics
4. Consult model safety team
