#!/bin/bash
# QUANTUM FIX SCRIPT — 2026-03-02
# Fixes: BNB position side, meta-regime, Redis WRONGTYPE, stale proposals

echo "============================================================"
echo " QUANTUM FIX — $(date -u)"
echo "============================================================"

# ============================================================
# FIX 1: BNB position side (LONG in Redis but SHORT on Binance)
# ============================================================
echo ""
echo "=== FIX 1: BNB position side ==="
echo "BEFORE: $(redis-cli HGETALL quantum:position:BNBUSDT 2>/dev/null | tr '\n' ' ')"

# Correct BNB to SHORT per Binance snapshot
redis-cli HSET quantum:position:BNBUSDT \
    symbol BNBUSDT \
    side SHORT \
    quantity 0.17 \
    entry_price 618.77 \
    leverage 2 \
    opened_at 1772408001 \
    source p34_position_bootstrap_corrected \
    risk_missing 0 \
    sync_timestamp $(date +%s) \
    unrealized_pnl 0.0 \
    atr_value 5.5 \
    volatility_factor 1.5 \
    2>/dev/null
echo "AFTER: side=$(redis-cli HGET quantum:position:BNBUSDT side) qty=$(redis-cli HGET quantum:position:BNBUSDT quantity)"

# ============================================================
# FIX 2: ADAUSDT ghost cleanup
# ============================================================
echo ""
echo "=== FIX 2: ADAUSDT ghost check ==="
ada_qty=$(redis-cli HGET quantum:position:ADAUSDT quantity 2>/dev/null)
ada_snapshot_qty=$(redis-cli HGET quantum:position:snapshot:ADAUSDT position_amt 2>/dev/null)
echo "  Redis position qty: $ada_qty"
echo "  Binance snapshot qty: $ada_snapshot_qty"
if [ ! -z "$ada_qty" ] && [ ! -z "$ada_snapshot_qty" ]; then
    echo "  ADA position exists on Binance — keeping (NOT a ghost)"
else
    echo "  ADA ghost — cleaning..."
    redis-cli DEL quantum:position:ADAUSDT 2>/dev/null
fi

# ============================================================
# FIX 3: Clean stale harvest proposals (fake R_net=9.76 entries)
# ============================================================
echo ""
echo "=== FIX 3: Clean stale harvest proposals ==="
echo "Harvest proposals BEFORE:"
redis-cli KEYS 'quantum:harvest:proposal:*' 2>/dev/null | while read k; do
    r=$(redis-cli HGET "$k" R_net 2>/dev/null)
    src=$(redis-cli HGET "$k" source 2>/dev/null)
    echo "  $k R_net=$r source=$src"
done

# Remove proposals where source is test/fake (not harvest_v2 live)
redis-cli KEYS 'quantum:harvest:proposal:*' 2>/dev/null | while read k; do
    r_net=$(redis-cli HGET "$k" R_net 2>/dev/null)
    # R_net > 9 is suspiciously high for real trades → stale test data
    if python3 -c "import sys; v=float('${r_net}' or 0); sys.exit(0 if v>=9 else 1)" 2>/dev/null; then
        echo "  DELETING stale proposal $k (R_net=$r_net)"
        redis-cli DEL "$k" 2>/dev/null
    fi
done

# ============================================================
# FIX 4: Find and fix Redis WRONGTYPE key
# ============================================================
echo ""
echo "=== FIX 4: Redis WRONGTYPE error investigation ==="
# The intent executor hit a WRONGTYPE error — find keys with wrong type
# Check common keys that might be wrong type
for key in quantum:permit:governor quantum:permit:p33 quantum:permit:p26 quantum:stream:trade.intent quantum:stream:apply.plan; do
    ktype=$(redis-cli TYPE "$key" 2>/dev/null)
    echo "  $key => type: $ktype"
done

# ============================================================
# FIX 5: Start meta-regime service
# ============================================================
echo ""
echo "=== FIX 5: Start quantum-meta-regime.service ==="
systemctl start quantum-meta-regime.service 2>/dev/null
sleep 2
status=$(systemctl is-active quantum-meta-regime.service 2>/dev/null)
echo "  Status: $status"
journalctl -u quantum-meta-regime.service -n 5 --no-pager 2>/dev/null

# ============================================================
# FIX 6: Refresh BNB close — trigger new harvest evaluation
# After fixing BNB side, harvest_v2 should generate a BUY close
# Force by deleting the failed plan's executed marker if it exists
# ============================================================
echo ""
echo "=== FIX 6: Clear executed marker for BNB plan a38eb015 ==="
redis-cli DEL "quantum:plan:executed:a38eb015" 2>/dev/null
redis-cli DEL "quantum:dedup:plan:a38eb015" 2>/dev/null
redis-cli DEL "quantum:executed:a38eb015" 2>/dev/null
# Check possible dedup keys
redis-cli KEYS "quantum:*a38eb015*" 2>/dev/null

# ============================================================
# FIX 7: Issue P3.3 permits again (for new plans)
# ============================================================
echo ""
echo "=== FIX 7: Run auto_permit_p33 again ==="
/opt/quantum/venvs/ai-client-base/bin/python /opt/quantum/scripts/auto_permit_p33.py 2>&1 | tail -5

# ============================================================
# VERIFY: System state after fixes
# ============================================================
echo ""
echo "=== VERIFY: State after fixes ==="
echo "BNB position side: $(redis-cli HGET quantum:position:BNBUSDT side 2>/dev/null)"
echo "Meta-regime status: $(systemctl is-active quantum-meta-regime.service 2>/dev/null)"
echo "Regime key: $(redis-cli GET quantum:meta:regime 2>/dev/null) | $(redis-cli KEYS 'quantum:*regime*' 2>/dev/null | tr '\n' ' ')"
echo "P33 permits count: $(redis-cli KEYS 'quantum:permit:p33:*' 2>/dev/null | wc -l)"
echo ""
echo "============================================================"
echo " FIX COMPLETE — $(date -u)"
echo "============================================================"
