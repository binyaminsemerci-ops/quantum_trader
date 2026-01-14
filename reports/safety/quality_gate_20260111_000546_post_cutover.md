# Quality Gate Report (Telemetry-Only)

**Timestamp:** 2026-01-11 00:05:46 UTC

## Cutover Analysis

**Cutover Timestamp:** 2026-01-11T00:01:44Z
**Mode:** Post-cutover analysis (events after patch deployment)

## Confidence Normalization Audit

**Total predictions:** 354
**Normalized (logit → prob):** 0 (0.0%)
**Violations (BLOCKER):** 0

## Telemetry Info
- Redis stream: quantum:stream:trade.intent
- Events analyzed: 118
- Events requested: 2000

## Model Breakdown

### Pre/Post Cutover Comparison

| Model | Metric | Before Patch | After Patch | Delta |
|-------|--------|--------------|-------------|-------|
| nhits | HOLD% | 6.7% | 31.4% | +24.7% |
| nhits | Conf Std | 0.0386 | 0.0698 | +0.0312 |
| nhits | P10-P90 | 0.1136 | 0.1500 | +0.0364 |
| patchtst | HOLD% | 0.0% | 0.0% | +0.0% |
| patchtst | Conf Std | 0.0000 | 0.0000 | +0.0000 |
| patchtst | P10-P90 | 0.0000 | 0.0000 | +0.0000 |
| xgb | HOLD% | 0.0% | 0.0% | +0.0% |
| xgb | Conf Std | 0.0068 | 0.0062 | -0.0007 |
| xgb | P10-P90 | 0.0162 | 0.0160 | -0.0002 |

**Improvement indicators:**
- HOLD% decrease = Less dead zone trap ✅
- Conf Std increase = More variance ✅
- P10-P90 increase = Wider distribution ✅

### nhits - ✅ PASS

**Action Distribution:**
- BUY: 39.0% (46/118)
- SELL: 29.7% (35/118)
- HOLD: 31.4% (37/118)

**Confidence Stats:**
- Mean: 0.5878
- Std: 0.0698
- P10: 0.5000
- P90: 0.6500
- P10-P90 Range: 0.1500

**Quality Checks:**
- ✅ All checks passed

### patchtst - ❌ FAIL

**Action Distribution:**
- BUY: 100.0% (118/118)
- SELL: 0.0% (0/118)
- HOLD: 0.0% (0/118)

**Confidence Stats:**
- Mean: 0.6150
- Std: 0.0000
- P10: 0.6150
- P90: 0.6150
- P10-P90 Range: 0.0000

**Quality Checks:**
- ❌ BUY majority 100.0% (>70% threshold)
- ❌ Confidence std 0.0000 (<0.05 threshold - flat)
- ❌ P10-P90 range 0.0000 (<0.12 threshold - narrow)
- ❌ Constant output detected (std=0.0000)
- ❌ Constant output detected (P10=0.6150, P90=0.6150)

### xgb - ❌ FAIL

**Action Distribution:**
- BUY: 100.0% (118/118)
- SELL: 0.0% (0/118)
- HOLD: 0.0% (0/118)

**Confidence Stats:**
- Mean: 0.9827
- Std: 0.0062
- P10: 0.9772
- P90: 0.9932
- P10-P90 Range: 0.0160

**Quality Checks:**
- ❌ BUY majority 100.0% (>70% threshold)
- ❌ Confidence std 0.0062 (<0.05 threshold - flat)
- ❌ P10-P90 range 0.0160 (<0.12 threshold - narrow)
- ❌ Constant output detected (std=0.0062)
