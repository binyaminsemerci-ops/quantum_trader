# Quality Gate Report (Telemetry-Only)

**Timestamp:** 2026-01-10 23:58:45 UTC

## Cutover Analysis

**Cutover Timestamp:** 2026-01-10T23:37:27Z
**Mode:** Post-cutover analysis (events after patch deployment)

## Confidence Normalization Audit

**Total predictions:** 288
**Normalized (logit → prob):** 0 (0.0%)
**Violations (BLOCKER):** 0

## Telemetry Info
- Redis stream: quantum:stream:trade.intent
- Events analyzed: 96
- Events requested: 2000

## ⚠️ INSUFFICIENT DATA (FAIL-CLOSED)

Minimum required: 200 events
Found: 96

**BLOCKER:** Cannot validate model safety without sufficient telemetry.

## Model Breakdown

**No models found in telemetry**
