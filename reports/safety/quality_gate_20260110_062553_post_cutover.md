# Quality Gate Report (Telemetry-Only)

**Timestamp:** 2026-01-10 06:25:53 UTC

## Cutover Analysis

**Cutover Timestamp:** 2026-01-10T05:43:15Z
**Mode:** Post-cutover analysis (events after patch deployment)

## Confidence Normalization Audit

**Total predictions:** 280
**Normalized (logit → prob):** 50 (17.9%)
**Violations (BLOCKER):** 0

**Normalization applied:**
- Values >1.0 treated as logits
- Sigmoid applied: prob = 1 / (1 + exp(-logit))
- Quality gate uses normalized [0, 1] range only

## Telemetry Info
- Redis stream: quantum:stream:trade.intent
- Events analyzed: 70
- Events requested: 2000

## ⚠️ INSUFFICIENT DATA (FAIL-CLOSED)

Minimum required: 200 events
Found: 70

**BLOCKER:** Cannot validate model safety without sufficient telemetry.

## Model Breakdown

**No models found in telemetry**
