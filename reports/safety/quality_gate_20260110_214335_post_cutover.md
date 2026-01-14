# Quality Gate Report (Telemetry-Only)

**Timestamp:** 2026-01-10 21:43:35 UTC

## Cutover Analysis

**Cutover Timestamp:** 2026-01-10T05:43:15Z
**Mode:** Post-cutover analysis (events after patch deployment)

## Confidence Normalization Audit

**Total predictions:** 5660
**Normalized (logit → prob):** 1068 (18.9%)
**Violations (BLOCKER):** 0

**Normalization applied:**
- Values >1.0 treated as logits
- Sigmoid applied: prob = 1 / (1 + exp(-logit))
- Quality gate uses normalized [0, 1] range only

## Telemetry Info
- Redis stream: quantum:stream:trade.intent
- Events analyzed: 1415
- Events requested: 2000

## Model Breakdown

### Pre/Post Cutover Comparison

| Model | Metric | Before Patch | After Patch | Delta |
|-------|--------|--------------|-------------|-------|
| lgbm | HOLD% | 0.4% | 0.4% | +0.0% |
| lgbm | Conf Std | 0.0568 | 0.0568 | +0.0000 |
| lgbm | P10-P90 | 0.1074 | 0.1074 | +0.0000 |
| nhits | HOLD% | 0.0% | 0.0% | +0.0% |
| nhits | Conf Std | 0.0086 | 0.0086 | +0.0000 |
| nhits | P10-P90 | 0.0000 | 0.0000 | +0.0000 |
| patchtst | HOLD% | 0.0% | 0.0% | +0.0% |
| patchtst | Conf Std | 0.0000 | 0.0000 | +0.0000 |
| patchtst | P10-P90 | 0.0000 | 0.0000 | +0.0000 |
| xgb | HOLD% | 0.0% | 0.0% | +0.0% |
| xgb | Conf Std | 0.0069 | 0.0069 | +0.0000 |
| xgb | P10-P90 | 0.0162 | 0.0162 | +0.0000 |

**Improvement indicators:**
- HOLD% decrease = Less dead zone trap ✅
- Conf Std increase = More variance ✅
- P10-P90 increase = Wider distribution ✅

### lgbm - ❌ FAIL

**Normalization:** 1068/1415 predictions (75.5%) normalized from logits

**Action Distribution:**
- BUY: 49.4% (699/1415)
- SELL: 50.2% (710/1415)
- HOLD: 0.4% (6/1415)

**Confidence Stats:**
- Mean: 0.7320
- Std: 0.0568
- P10: 0.6333
- P90: 0.7408
- P10-P90 Range: 0.1074

**Quality Checks:**
- ❌ P10-P90 range 0.1074 (<0.12 threshold - narrow)

### nhits - ❌ FAIL

**Action Distribution:**
- BUY: 0.4% (6/1415)
- SELL: 99.6% (1409/1415)
- HOLD: 0.0% (0/1415)

**Confidence Stats:**
- Mean: 0.5370
- Std: 0.0086
- P10: 0.5364
- P90: 0.5364
- P10-P90 Range: 0.0000

**Quality Checks:**
- ❌ SELL majority 99.6% (>70% threshold)
- ❌ Confidence std 0.0086 (<0.05 threshold - flat)
- ❌ P10-P90 range 0.0000 (<0.12 threshold - narrow)
- ❌ Constant output detected (std=0.0086)
- ❌ Constant output detected (P10=0.5364, P90=0.5364)

### patchtst - ❌ FAIL

**Action Distribution:**
- BUY: 100.0% (1415/1415)
- SELL: 0.0% (0/1415)
- HOLD: 0.0% (0/1415)

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
- BUY: 100.0% (1415/1415)
- SELL: 0.0% (0/1415)
- HOLD: 0.0% (0/1415)

**Confidence Stats:**
- Mean: 0.9855
- Std: 0.0069
- P10: 0.9772
- P90: 0.9934
- P10-P90 Range: 0.0162

**Quality Checks:**
- ❌ BUY majority 100.0% (>70% threshold)
- ❌ Confidence std 0.0069 (<0.05 threshold - flat)
- ❌ P10-P90 range 0.0162 (<0.12 threshold - narrow)
- ❌ Constant output detected (std=0.0069)
