# Quality Gate Report (Telemetry-Only)

**Timestamp:** 2026-01-11 00:04:54 UTC

## Cutover Analysis

**Cutover Timestamp:** 2026-01-11T00:01:44Z
**Mode:** Post-cutover analysis (events after patch deployment)

## Confidence Normalization Audit

**Total predictions:** 282
**Normalized (logit → prob):** 0 (0.0%)
**Violations (BLOCKER):** 0

## Telemetry Info
- Redis stream: quantum:stream:trade.intent
- Events analyzed: 94
- Events requested: 2000

## ⚠️ INSUFFICIENT DATA (FAIL-CLOSED)

Minimum required: 100 events
Found: 94

**BLOCKER:** Cannot validate model safety without sufficient telemetry.

## Model Breakdown

**No models found in telemetry**
