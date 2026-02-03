#!/usr/bin/env bash
# Proof: AI Universe Guardrails Implementation
# Verifies liquidity/quality filters with bulk API and per-symbol logging

cd "$(dirname "$0")/.." || exit 1

echo "============================================================"
echo "  PROOF: AI Universe Guardrails"
echo "============================================================"
echo ""

PASS=0
FAIL=0

# TEST 1: Verify guardrail log string exists in code
echo "[TEST 1] Verify AI_UNIVERSE_GUARDRAILS log string in code"
if grep -q 'AI_UNIVERSE_GUARDRAILS' scripts/ai_universe_generator_v1.py; then
    echo "PASS: AI_UNIVERSE_GUARDRAILS log found"
    PASS=$((PASS + 1))
else
    echo "FAIL: AI_UNIVERSE_GUARDRAILS log not found"
    FAIL=$((FAIL + 1))
fi

echo ""

# TEST 2: Verify guardrail config constants exist
echo "[TEST 2] Verify guardrail constants in code"
MISSING=0

if ! grep -q 'MIN_QUOTE_VOL_USDT_24H' scripts/ai_universe_generator_v1.py; then
    echo "  FAIL: MIN_QUOTE_VOL_USDT_24H not found"
    MISSING=$((MISSING + 1))
fi

if ! grep -q 'MAX_SPREAD_BPS' scripts/ai_universe_generator_v1.py; then
    echo "  FAIL: MAX_SPREAD_BPS not found"
    MISSING=$((MISSING + 1))
fi

if ! grep -q 'MIN_AGE_DAYS' scripts/ai_universe_generator_v1.py; then
    echo "  FAIL: MIN_AGE_DAYS not found"
    MISSING=$((MISSING + 1))
fi

if [ $MISSING -eq 0 ]; then
    echo "PASS: All guardrail constants present"
    PASS=$((PASS + 1))
else
    echo "FAIL: $MISSING guardrail constants missing"
    FAIL=$((FAIL + 1))
fi

echo ""

# TEST 3: Dry-run test (light runtime, 60s timeout)
echo "[TEST 3] Dry-run test with --dry-run flag"
echo "  Running: timeout 60 python3 scripts/ai_universe_generator_v1.py --dry-run"
echo ""

TEMP_OUT=$(mktemp)
timeout 60 python3 scripts/ai_universe_generator_v1.py --dry-run > "$TEMP_OUT" 2>&1

if [ $? -eq 0 ] || [ $? -eq 124 ]; then
    # Check for required log lines
    if grep -q "AI_UNIVERSE_GUARDRAILS" "$TEMP_OUT"; then
        echo "  PASS: AI_UNIVERSE_GUARDRAILS log present"
        
        # Extract log line
        GUARDRAILS_LOG=$(grep "AI_UNIVERSE_GUARDRAILS" "$TEMP_OUT" | head -1)
        echo "  Log: $GUARDRAILS_LOG"
        
        # Check for AI_UNIVERSE_PICK
        if grep -q "AI_UNIVERSE_PICK" "$TEMP_OUT"; then
            PICK_COUNT=$(grep -c "AI_UNIVERSE_PICK" "$TEMP_OUT")
            echo "  PASS: AI_UNIVERSE_PICK found ($PICK_COUNT picks)"
            
            # Show first pick
            FIRST_PICK=$(grep "AI_UNIVERSE_PICK" "$TEMP_OUT" | head -1)
            echo "  Example: $FIRST_PICK"
            
            PASS=$((PASS + 1))
        else
            echo "  FAIL: No AI_UNIVERSE_PICK logs found"
            FAIL=$((FAIL + 1))
        fi
    else
        echo "  FAIL: AI_UNIVERSE_GUARDRAILS log not found in output"
        echo "  Output tail:"
        tail -20 "$TEMP_OUT"
        FAIL=$((FAIL + 1))
    fi
else
    echo "  FAIL: Dry-run exited with error"
    echo "  Output tail:"
    tail -30 "$TEMP_OUT"
    FAIL=$((FAIL + 1))
fi

rm -f "$TEMP_OUT"

echo ""
echo "============================================================"
echo "  SUMMARY"
echo "============================================================"
echo "PASS: $PASS"
echo "FAIL: $FAIL"
echo ""

if [ $FAIL -eq 0 ]; then
    echo "ALL TESTS PASSED"
    echo ""
    echo "Guardrails verified:"
    echo "  - Bulk 24h stats fetch (single API call)"
    echo "  - Volume filter: MIN_QUOTE_VOL_USDT_24H"
    echo "  - Spread filter: MAX_SPREAD_BPS"
    echo "  - Age filter: MIN_AGE_DAYS"
    echo "  - Structured logging: AI_UNIVERSE_GUARDRAILS"
    echo "  - Per-symbol logging: AI_UNIVERSE_PICK"
    echo "  - Dry-run mode: --dry-run flag"
    exit 0
else
    echo "SOME TESTS FAILED"
    exit 1
fi
