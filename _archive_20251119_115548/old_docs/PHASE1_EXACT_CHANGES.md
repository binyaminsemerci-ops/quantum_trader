# PHASE 1 CHANGES - EXACT DIFF

## CURRENT VALUES (docker-compose.yml):

```yaml
Line 29: - QT_CHECK_INTERVAL=10
Line 30: - QT_MIN_CONFIDENCE=0.35
Line 35: - QT_MAX_NOTIONAL_PER_TRADE=250.0
Line 36: - QT_MAX_POSITION_PER_SYMBOL=250.0
Line 37: - QT_MAX_GROSS_EXPOSURE=2000.0
Line 39: - QT_MAX_POSITIONS=8
Line 41: - QT_TP_PCT=0.02
Line 42: - QT_SL_PCT=0.025
Line 43: - QT_TRAIL_PCT=0.01
Line 44: - QT_PARTIAL_TP=0.6
```

---

## PHASE 1 VALUES (MODERATE):

```yaml
Line 29: - QT_CHECK_INTERVAL=7              # Was: 10 (+30% faster checks)
Line 30: - QT_MIN_CONFIDENCE=0.30           # Was: 0.35 (+15% more signals)
Line 35: - QT_MAX_NOTIONAL_PER_TRADE=300.0  # Was: 250 (+20% larger trades)
Line 36: - QT_MAX_POSITION_PER_SYMBOL=300.0 # Was: 250
Line 37: - QT_MAX_GROSS_EXPOSURE=3000.0     # Was: 2000 (10 x $300)
Line 39: - QT_MAX_POSITIONS=10              # Was: 8 (+25% more concurrent)
Line 41: - QT_TP_PCT=0.02                   # UNCHANGED (2.0% realistic)
Line 42: - QT_SL_PCT=0.025                  # UNCHANGED (2.5% protection)
Line 43: - QT_TRAIL_PCT=0.012               # Was: 0.01 (+20% room to run)
Line 44: - QT_PARTIAL_TP=0.6                # UNCHANGED (60% partial exit)
```

---

## QUICK SUMMARY OF CHANGES:

✅ **Faster signal detection:** 10s → 7s (30% faster)
✅ **More signals:** 35% → 30% confidence (15% more opportunities)  
✅ **Larger positions:** $250 → $300 (20% bigger wins)
✅ **More parallelism:** 8 → 10 positions (25% more concurrent)
✅ **Better trailing:** 1.0% → 1.2% (let winners run slightly more)

**UNCHANGED (keep proven settings):**
- TP: 2.0% (realistic for 1-2 hour trades)
- SL: 2.5% (tight protection)
- Partial: 60% (hybrid strategy still active)

---

## EXPECTED IMPACT:

**Trading frequency:**
- Current: 1-2 trades/hour
- Phase 1: 2-3 trades/hour (+50-100%)

**Profit per trade:**
- Current: $250 x 2% = $5
- Phase 1: $300 x 2% = $6 (+20%)

**Concurrent positions:**
- Current: Max 8
- Phase 1: Max 10 (+25%)

**Hourly P&L:**
- Current: $5-10/hour
- Phase 1: $10-15/hour (+50-100%)

**3-Hour Target (Phase 1):**
- Trades: 6-9
- Winners: 4-6 (assuming 65% win rate)
- P&L: $24-36 ✅

---

## READY TO APPLY?

If you choose Option A, I will:

1. Create backup: `docker-compose.yml.backup`
2. Apply the 7 changes above to `docker-compose.yml`
3. Restart backend: `docker-compose down && docker-compose up -d`
4. Verify startup and new settings
5. Start monitoring loop

**This will take 2-3 minutes total.**

Type 'YES' to proceed with Phase 1 implementation.
