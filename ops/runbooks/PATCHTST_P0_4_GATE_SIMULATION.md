# PATCHTST P0.4 — GATE DECISION SIMULATION

**Model**: patchtst_v20260109_233444.pth (612K params, retrained)  
**Known Issues**: 100% WIN bias in training, low confidence spread expected  
**Environment**: VPS systemd-only (NO DOCKER)  
**Simulation Date**: 2026-01-10

---

## SNAPSHOT A — 2 HOURS (~35 predictions)

### Metrics

| Gate | Metric | Value | Threshold | Status |
|------|--------|-------|-----------|--------|
| **1: Action Diversity** | | | | |
| | BUY | 31 (89%) | <70% | ❌ FAIL |
| | SELL | 2 (6%) | | |
| | HOLD | 2 (6%) | | |
| | Classes >10% | 1 | ≥2 | ❌ FAIL |
| **2: Confidence Spread** | | | | |
| | Mean | 0.6148 | - | - |
| | Stddev | 0.0023 | ≥0.05 | ❌ FAIL |
| | P10 | 0.6145 | - | - |
| | P50 | 0.6148 | - | - |
| | P90 | 0.6151 | - | - |
| | P10-P90 Range | 0.0006 | ≥0.12 | ❌ FAIL |
| **3: Agreement** | | | | |
| | Agreement % | 68% | 55-75% | ✅ PASS |
| | Hard Disagree % | 12% | <25% | ✅ PASS |
| **4: Calibration** | | | | |
| | Status | N/A | - | ⏳ DEFER |

### Gate Results
- **Gate 1**: ❌ FAIL (89% BUY, only 1 class >10%)
- **Gate 2**: ❌ FAIL (stddev=0.0023, range=0.0006)
- **Gate 3**: ✅ PASS (68% agreement, within healthy range)
- **Gate 4**: ⏳ DEFER (insufficient data)

**Gates Passed**: 1/3 (Gate 4 deferred)

### Technical Reason
BUY bias from training persists in production. Confidence flatlined (range 0.6145-0.6151). Agreement acceptable but diversity/spread critical failures.

### Decision
**⏳ CONTINUE SHADOW** — Too early for decision, but red flags evident. Need 6h+ data to confirm if bias stabilizes or worsens.

---

## SNAPSHOT B — 6 HOURS (~95 predictions)

### Metrics

| Gate | Metric | Value | Threshold | Status |
|------|--------|-------|-----------|--------|
| **1: Action Diversity** | | | | |
| | BUY | 83 (87%) | <70% | ❌ FAIL |
| | SELL | 6 (6%) | | |
| | HOLD | 6 (6%) | | |
| | Classes >10% | 1 | ≥2 | ❌ FAIL |
| **2: Confidence Spread** | | | | |
| | Mean | 0.6150 | - | - |
| | Stddev | 0.0028 | ≥0.05 | ❌ FAIL |
| | P10 | 0.6144 | - | - |
| | P50 | 0.6150 | - | - |
| | P90 | 0.6156 | - | - |
| | P10-P90 Range | 0.0012 | ≥0.12 | ❌ FAIL |
| **3: Agreement** | | | | |
| | Agreement % | 64% | 55-75% | ✅ PASS |
| | Hard Disagree % | 15% | <25% | ✅ PASS |
| **4: Calibration** | | | | |
| | Status | N/A | - | ⏳ DEFER |

### Gate Results
- **Gate 1**: ❌ FAIL (87% BUY, persistent bias)
- **Gate 2**: ❌ FAIL (stddev=0.0028, minimal improvement)
- **Gate 3**: ✅ PASS (64% agreement, stable)
- **Gate 4**: ⏳ DEFER (need outcomes)

**Gates Passed**: 1/3 (Gate 4 deferred)

### Technical Reason
BUY bias persistent across 6h (87%). Confidence spread marginally better but still critically low (stddev 0.0028 vs threshold 0.05). Agreement stable at 64% suggests model not contrarian but lacks diversity.

### Decision
**⏳ CONTINUE SHADOW** — Gate 1+2 failures confirmed. Model lacks action diversity and confidence calibration. Extend to 24h to collect calibration data before re-training decision.

---

## SNAPSHOT C — 24 HOURS (~380 predictions)

### Metrics

| Gate | Metric | Value | Threshold | Status |
|------|--------|-------|-----------|--------|
| **1: Action Diversity** | | | | |
| | BUY | 332 (87%) | <70% | ❌ FAIL |
| | SELL | 24 (6%) | | |
| | HOLD | 24 (6%) | | |
| | Classes >10% | 1 | ≥2 | ❌ FAIL |
| **2: Confidence Spread** | | | | |
| | Mean | 0.6151 | - | - |
| | Stddev | 0.0031 | ≥0.05 | ❌ FAIL |
| | P10 | 0.6142 | - | - |
| | P50 | 0.6151 | - | - |
| | P90 | 0.6160 | - | - |
| | P10-P90 Range | 0.0018 | ≥0.12 | ❌ FAIL |
| **3: Agreement** | | | | |
| | Agreement % | 62% | 55-75% | ✅ PASS |
| | Hard Disagree % | 16% | <25% | ✅ PASS |
| **4: Calibration** | | | | |
| | Bucket 0.50-0.60 | N/A (0 samples) | - | - |
| | Bucket 0.60-0.65 | 48% hit rate (185 samples) | - | - |
| | Bucket 0.65-0.70 | 51% hit rate (195 samples) | - | - |
| | Bucket 0.70-0.80 | N/A (0 samples) | - | - |
| | Monotonic Trend | NO (48% → 51%, too flat) | YES | ❌ FAIL |

### Gate Results
- **Gate 1**: ❌ FAIL (87% BUY, no improvement over 24h)
- **Gate 2**: ❌ FAIL (stddev=0.0031, far below 0.05 threshold)
- **Gate 3**: ✅ PASS (62% agreement, healthy)
- **Gate 4**: ❌ FAIL (non-monotonic, 48%→51% minimal lift)

**Gates Passed**: 1/4

### Technical Reason
24h data confirms persistent BUY bias (87%). Confidence flatlined within 0.6142-0.6160 range (0.0018 span vs 0.12 threshold). Calibration shows model predicts ~0.615 regardless of signal strength (hit rate 48-51%, barely above random). Agreement stable but not sufficient to overcome diversity/spread failures.

### Decision
**❌ NO-GO (RE-TRAINING REQUIRED)** — Only Gate 3 passed. Model exhibits:
1. Severe BUY bias (87% vs 33% expected)
2. Confidence collapse (stddev 0.0031 vs 0.05 threshold)
3. Poor calibration (48-51% hit rate, non-monotonic)

**Action**: Keep shadow mode, initiate P0.6 re-training with:
- Balanced class sampling (50/50 WIN/LOSS per batch)
- Confidence regularization
- Extended feature set (20+ indicators)

---

## SIMULATION SUMMARY

| Window | Gates Passed | Decision | Reason |
|--------|--------------|----------|--------|
| 2h | 1/3 (Gate 4 defer) | ⏳ Continue | Too early, red flags visible |
| 6h | 1/3 (Gate 4 defer) | ⏳ Continue | Bias confirmed, need calibration data |
| 24h | 1/4 | ❌ NO-GO | Persistent bias, confidence collapse, poor calibration |

---

## FINAL RECOMMENDATION

**❌ NO-GO FOR P0.5 ACTIVATION**

### Evidence
1. **Gate 1 (Action Diversity)**: FAIL across all windows (87-89% BUY)
2. **Gate 2 (Confidence Spread)**: FAIL across all windows (stddev 0.0023-0.0031 vs 0.05 threshold)
3. **Gate 3 (Agreement)**: PASS (62-68%, stable)
4. **Gate 4 (Calibration)**: FAIL (48-51% hit rate, non-monotonic)

### Root Cause
P0.4 model trained on imbalanced dataset (60% WIN / 40% LOSS) → learned to predict WIN (BUY) with flat confidence ~0.615 regardless of features.

### Path Forward
**P0.6 Re-training** with:
1. **Balanced sampling**: Force 50/50 WIN/LOSS per batch
2. **Confidence regularization**: Penalize flat predictions
3. **Feature engineering**: Add regime-specific indicators (volatility, volume, trend strength)
4. **Validation split**: Stratified by class and regime
5. **Early stopping**: Monitor calibration ECE (Expected Calibration Error)

### Shadow Mode Status
**✅ KEEP ACTIVE** — Continue collecting data for:
- Regime-specific analysis (trending vs ranging markets)
- Symbol-specific bias patterns
- Feature importance validation
- Calibration dataset for P0.6

### Timeline
- **Now → T+7d**: Collect 500+ shadow predictions with outcomes
- **T+7d → T+10d**: P0.6 re-training with improvements
- **T+10d → T+12d**: P0.6 shadow observation
- **T+12d**: Re-evaluate gates for P0.6 activation decision

---

## DECISION LOG

| Field | Value |
|-------|-------|
| **Simulation Date** | 2026-01-10T02:50 UTC |
| **Model Version** | patchtst_v20260109_233444.pth (P0.4) |
| **Shadow Mode Duration** | Simulated 24h |
| **Gates Passed (24h)** | 1/4 (Gate 3 only) |
| **Critical Failures** | Gate 1 (BUY bias), Gate 2 (confidence collapse), Gate 4 (poor calibration) |
| **Decision** | ❌ NO-GO — Re-training required (P0.6) |
| **Approved By** | [Pending OPS approval] |
| **Next Action** | Continue shadow, prepare P0.6 training plan |
| **Rollback Needed** | NO (shadow mode safe) |

---

**Simulation Status**: ✅ COMPLETE  
**Activation Status**: ❌ BLOCKED (Gate failures)  
**Next Review**: After P0.6 re-training
