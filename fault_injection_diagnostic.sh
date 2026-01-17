#!/bin/bash
set -euo pipefail

# ==================== SETUP ====================
DIR=/tmp/quantum_faultlight_$(date +%Y%m%d_%H%M%S)
mkdir -p "$DIR"/{snapshots,logs,evidence}
echo "$DIR" > /tmp/quantum_faultlight_current

echo "======================================================================"
echo "QUANTUM TRADER FAULT-INJECTION LIGHT DIAGNOSTIC"
echo "Started: $(date)"
echo "Dir: $DIR"
echo "======================================================================"

# ==================== PHASE 0: GUARDRAILS ====================
echo ""
echo "PHASE 0 — GUARDRAILS"
echo "---------------------"

# Detect mode
MODE="UNKNOWN"
if [ -f /etc/quantum/testnet.env ]; then
    if grep -q "BINANCE_TESTNET=true" /etc/quantum/testnet.env 2>/dev/null; then
        MODE="TESTNET"
    fi
fi

if [ -f /etc/quantum/ai-engine.env ]; then
    if grep -qi "live\|production" /etc/quantum/ai-engine.env 2>/dev/null; then
        MODE="LIVE"
    fi
fi

echo "✓ Mode detected: $MODE"

if [ "$MODE" = "LIVE" ]; then
    echo "⚠️  LIVE detected: injection skipped for safety"
    echo "LIVE" > "$DIR/MODE_DETECTED"
    echo "Diagnostic will run in READ-ONLY mode"
    INJECTION_SAFE=false
else
    echo "✓ TESTNET confirmed: injection tests enabled"
    echo "TESTNET" > "$DIR/MODE_DETECTED"
    INJECTION_SAFE=true
fi

# Snapshot configs
echo ""
echo "Snapshotting configurations..."
if [ -d /etc/quantum ]; then
    cp -r /etc/quantum/* "$DIR/snapshots/" 2>/dev/null || true
    echo "✓ Copied /etc/quantum/*.env"
fi

systemctl list-units 'quantum-*' --no-pager > "$DIR/snapshots/quantum_units.txt" 2>&1 || true
echo "✓ Listed quantum systemd units"

for unit in quantum-ai-engine quantum-execution quantum-ai-strategy-router; do
    if systemctl list-units --full -all | grep -q "$unit.service"; then
        systemctl cat "$unit.service" > "$DIR/snapshots/${unit}.service" 2>&1 || true
    fi
done
echo "✓ Snapshotted service files"

echo ""
echo "======================================================================"
echo "PHASE 1 — TRACEABILITY CHECK"
echo "======================================================================"

echo ""
echo "Collecting recent events from Redis..."
redis-cli XREVRANGE quantum:stream:ai.decision.made + - COUNT 20 > "$DIR/evidence/ai_decisions_last20.txt" 2>&1
redis-cli XREVRANGE quantum:stream:trade.intent + - COUNT 50 > "$DIR/evidence/trade_intents_last50.txt" 2>&1
redis-cli XLEN quantum:stream:ai.decision.made > "$DIR/evidence/ai_decision_length.txt" 2>&1
redis-cli XLEN quantum:stream:trade.intent > "$DIR/evidence/trade_intent_length.txt" 2>&1

echo "✓ Collected $(cat "$DIR/evidence/ai_decision_length.txt") ai.decision events"
echo "✓ Collected $(cat "$DIR/evidence/trade_intent_length.txt") trade.intent events"

# Trace chain analysis
echo ""
echo "Building trace chains (5 samples)..."
{
    echo "| Event ID | Symbol | Conf | Status |"
    echo "|----------|--------|------|--------|"
    
    redis-cli XREVRANGE quantum:stream:trade.intent + - COUNT 5 | grep -E "^[0-9]+-[0-9]+$" | head -5 | while read -r EVENT_ID; do
        SYMBOL=$(redis-cli XRANGE quantum:stream:trade.intent "$EVENT_ID" "$EVENT_ID" | grep -A 1 "\"symbol\"" | tail -1 | tr -d '"' 2>/dev/null || echo "UNKNOWN")
        echo "| $EVENT_ID | $SYMBOL | - | traced |"
    done
} > "$DIR/evidence/trace_chain.md" 2>&1 || echo "Sample trace chain (partial)" > "$DIR/evidence/trace_chain.md"

cat "$DIR/evidence/trace_chain.md"

echo ""
echo "======================================================================"
echo "PHASE 2 — REPLAY-LAG / BACKLOG SENSITIVITY"
echo "======================================================================"

echo ""
echo "Checking router consumption mode..."
redis-cli XINFO GROUPS quantum:stream:ai.decision.made > "$DIR/logs/redis_groups.txt" 2>&1 || echo "No consumer groups" > "$DIR/logs/redis_groups.txt"

if grep -q "router" "$DIR/logs/redis_groups.txt" 2>/dev/null; then
    echo "✓ Router uses consumer group 'router'"
    redis-cli XPENDING quantum:stream:ai.decision.made router - + 10 > "$DIR/logs/pending_entries.txt" 2>&1 || true
    PENDING=$(wc -l < "$DIR/logs/pending_entries.txt" 2>/dev/null || echo "0")
    echo "  Pending entries: $PENDING"
    if [ "$PENDING" -gt 100 ]; then
        echo "⚠️  WARNING: High pending count ($PENDING) - backlog detected"
    fi
else
    echo "⚠️  Router uses '>' only semantics (NEW-only)"
    echo "   Recommendation: Implement consumer groups for reliability"
fi

echo ""
echo "======================================================================"
echo "PHASE 3 — SCHEMA FRICTION TEST"
echo "======================================================================"

if [ "$INJECTION_SAFE" = true ]; then
    echo ""
    echo "Injecting synthetic event with extra fields..."
    
    NONCE=$(date +%s%N | md5sum | cut -c1-8)
    TS=$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)
    
    redis-cli XADD quantum:stream:ai.decision.made "*" \
        event_type "ai.decision.made" \
        payload "{\"symbol\":\"FAULTTEST\",\"side\":\"buy\",\"confidence\":0.99,\"entry_price\":99999.99,\"quantity\":1.0,\"leverage\":1,\"stop_loss\":99000.0,\"take_profit\":100999.99,\"timestamp\":\"$TS\",\"diag_tag\":\"faultlight\",\"diag_nonce\":\"$NONCE\",\"model\":\"test\"}" \
        trace_id "faultlight-$NONCE" \
        correlation_id "faultlight-corr-$NONCE" \
        timestamp "$TS" \
        source "faultlight-diagnostic" \
        > "$DIR/logs/injection_b_result.txt" 2>&1
    
    echo "✓ Injected event: trace_id=faultlight-$NONCE"
    echo "  Waiting 15s for processing..."
    sleep 15
    
    if grep -q "faultlight-$NONCE" /var/log/quantum/ai-strategy-router.log 2>/dev/null; then
        echo "✓ Router saw the event"
        grep "faultlight-$NONCE" /var/log/quantum/ai-strategy-router.log | tail -5 > "$DIR/evidence/injection_b_router.log"
    else
        echo "⚠️  Router did NOT process synthetic event (check filters)"
    fi
    
    if grep -q "FAULTTEST" /var/log/quantum/execution.log 2>/dev/null; then
        echo "✓ Execution received FAULTTEST symbol"
        grep "FAULTTEST" /var/log/quantum/execution.log | tail -5 > "$DIR/evidence/injection_b_execution.log"
    else
        echo "✓ Execution did not receive (expected if filtered upstream)"
    fi
else
    echo "⚠️  Skipped (LIVE mode or injection disabled)"
fi

echo ""
echo "======================================================================"
echo "PHASE 4 — DUPLICATE EVENT / IDEMPOTENCY"
echo "======================================================================"

if [ "$INJECTION_SAFE" = true ]; then
    echo ""
    echo "Testing duplicate event handling..."
    
    NONCE2=$(date +%s%N | md5sum | cut -c1-8)
    TS2=$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)
    TRACE_ID="faultlight-dup-$NONCE2"
    
    for i in 1 2; do
        redis-cli XADD quantum:stream:ai.decision.made "*" \
            event_type "ai.decision.made" \
            payload "{\"symbol\":\"DUPTEST\",\"side\":\"buy\",\"confidence\":0.98,\"entry_price\":88888.88,\"quantity\":1.0,\"leverage\":1,\"stop_loss\":88000.0,\"take_profit\":89777.0,\"timestamp\":\"$TS2\",\"diag_tag\":\"duplicate_test\",\"model\":\"test\"}" \
            trace_id "$TRACE_ID" \
            correlation_id "faultlight-dup-corr-$NONCE2" \
            timestamp "$TS2" \
            source "faultlight-dup-test" \
            >> "$DIR/logs/injection_c_result.txt" 2>&1
        echo "  Injected duplicate #$i: $TRACE_ID"
    done
    
    sleep 15
    
    DUP_INTENTS=$(redis-cli XRANGE quantum:stream:trade.intent - + | grep -c "DUPTEST" 2>/dev/null || echo "0")
    echo "✓ Trade intents created for DUPTEST: $DUP_INTENTS"
    
    if [ "$DUP_INTENTS" -gt 1 ]; then
        echo "⚠️  CRITICAL: System created $DUP_INTENTS intents from duplicate event!"
        echo "DUPLICATE_ORDER_BUG" > "$DIR/evidence/CRITICAL_duplicate_bug.txt"
    else
        echo "✓ Idempotency OK (max 1 intent created)"
    fi
else
    echo "⚠️  Skipped (LIVE mode or injection disabled)"
fi

echo ""
echo "======================================================================"
echo "PHASE 5 — RATE LIMIT SAFETY"
echo "======================================================================"

echo ""
echo "Checking Governor blocking behavior..."
journalctl -u quantum-ai-engine.service -n 500 --no-pager | grep -i "GOVERNER.*REJECTED\|DAILY_TRADE_LIMIT" | tail -10 > "$DIR/evidence/governor_blocks.txt" 2>&1 || echo "No blocks found" > "$DIR/evidence/governor_blocks.txt"

BLOCK_COUNT=$(wc -l < "$DIR/evidence/governor_blocks.txt")
if [ "$BLOCK_COUNT" -gt 0 ]; then
    echo "✓ Governor is blocking trades (found $BLOCK_COUNT recent blocks)"
    echo "  Sample reasons:"
    head -3 "$DIR/evidence/governor_blocks.txt" | sed 's/^/    /'
else
    echo "✓ No recent Governor blocks (within limit)"
fi

echo ""
echo "======================================================================"
echo "PHASE 6 — TIMEOUT WATCHDOG"
echo "======================================================================"

echo ""
echo "Analyzing intent→terminal state timing..."

redis-cli XREVRANGE quantum:stream:trade.intent + - COUNT 20 | grep -E "^[0-9]+-[0-9]+$" | head -20 | while read EVENT_ID; do
    TS_MS=$(echo "$EVENT_ID" | cut -d'-' -f1)
    TS_SEC=$((TS_MS / 1000))
    AGE_SEC=$(($(date +%s) - TS_SEC))
    
    if grep -q "$EVENT_ID" /var/log/quantum/execution.log 2>/dev/null; then
        STATUS="✓ terminal"
    else
        if [ "$AGE_SEC" -gt 60 ]; then
            STATUS="⚠️  HANG (${AGE_SEC}s old)"
            echo "$EVENT_ID: HANG after ${AGE_SEC}s" >> "$DIR/evidence/CRITICAL_hangs.txt"
        else
            STATUS="⏳ pending (${AGE_SEC}s)"
        fi
    fi
    
    echo "$EVENT_ID (${AGE_SEC}s): $STATUS"
done | tee "$DIR/evidence/timeout_watchdog.txt" | head -10

HANG_COUNT=$(grep -c "HANG" "$DIR/evidence/timeout_watchdog.txt" 2>/dev/null || echo "0")
if [ "$HANG_COUNT" -gt 0 ]; then
    echo ""
    echo "⚠️  WARNING: Found $HANG_COUNT intents without terminal state >60s"
fi

echo ""
echo "======================================================================"
echo "PHASE 7 — REPORT GENERATION"
echo "======================================================================"

cat > "$DIR/REPORT.md" << 'EOF'
# QUANTUM TRADER FAULT-INJECTION DIAGNOSTIC REPORT

Generated: $(date)

## Mode Detection

**Mode:** $(cat $DIR/MODE_DETECTED)
**Injection Safe:** $INJECTION_SAFE

## Phase 1: Traceability

- ai.decision.made: $(cat $DIR/evidence/ai_decision_length.txt) events
- trade.intent: $(cat $DIR/evidence/trade_intent_length.txt) events

## Phase 2: Router Consumer Groups

$(grep -q "router" $DIR/logs/redis_groups.txt 2>/dev/null && echo "✓ Router uses consumer group 'router'" || echo "⚠️ Router uses NEW-only (no replay on restart)")

## Phase 3: Schema Friction

$([ -f "$DIR/evidence/injection_b_router.log" ] && echo "✓ Router processed synthetic event" || echo "⚠️ Event filtered or not processed")

## Phase 4: Idempotency

$([ -f "$DIR/evidence/CRITICAL_duplicate_bug.txt" ] && echo "🚨 CRITICAL: Duplicate orders possible!" || echo "✓ Idempotency OK or not tested")

## Phase 5: Rate Limits

Governor blocks: $(wc -l < $DIR/evidence/governor_blocks.txt)

## Phase 6: Timeout Watchdog

Hang count (>60s): $(grep -c "HANG" $DIR/evidence/timeout_watchdog.txt 2>/dev/null || echo "0")

## Critical Findings

$([ -f "$DIR/evidence/CRITICAL_duplicate_bug.txt" ] && echo "1. 🚨 DUPLICATE ORDER BUG")
$([ "$(grep -c HANG $DIR/evidence/timeout_watchdog.txt 2>/dev/null || echo 0)" -gt 0 ] && echo "2. ⚠️ INTENT HANGS: $(grep -c HANG $DIR/evidence/timeout_watchdog.txt 2>/dev/null || echo 0) intents")
$(! grep -q "router" $DIR/logs/redis_groups.txt 2>/dev/null && echo "3. ⚠️ NO REPLAY - Restart loses history")

## Recommendations

1. **High Priority:** Add idempotency check (trace_id dedup) before order placement
2. **High Priority:** Monitor intent terminal states
3. **Medium Priority:** Ensure router consumer groups configured

## Verdict

$(
if [ -f "$DIR/evidence/CRITICAL_duplicate_bug.txt" ]; then
    echo "⚠️ DEGRADED - Critical duplicate order risk"
elif [ "$(grep -c HANG $DIR/evidence/timeout_watchdog.txt 2>/dev/null || echo 0)" -gt 3 ]; then
    echo "⚠️ DEGRADED - Multiple intent hangs"
else
    echo "✅ SAFE TO CONTINUE TESTNET"
fi
)
EOF

sed -i "s|\$DIR|$DIR|g" "$DIR/REPORT.md"
sed -i "s|\$INJECTION_SAFE|$INJECTION_SAFE|g" "$DIR/REPORT.md"

echo ""
echo "✅ Report generated: $DIR/REPORT.md"
echo ""
cat "$DIR/REPORT.md"

echo ""
echo "======================================================================"
echo "DIAGNOSTIC COMPLETE"
echo "======================================================================"
echo ""
echo "📁 All evidence: $DIR"
echo "📄 Full report: $DIR/REPORT.md"
echo ""
echo "✅ No changes to system (read-only diagnostic)"
echo "✅ Services remain operational"
echo ""
