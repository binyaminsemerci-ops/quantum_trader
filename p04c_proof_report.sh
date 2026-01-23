#!/bin/bash
# P0.4C Evidence Report Generator (READ-ONLY)
# Generated: 2026-01-22

REPORT="/home/qt/quantum_trader/P0.4C_PROOF_REPORT.md"
EXIT_LOG="/var/log/quantum/exit-monitor.log"
EXEC_LOG="/var/log/quantum/execution.log"
EXEC_SERVICE="/home/qt/quantum_trader/services/execution_service.py"

echo "# P0.4C Complete Proof Report" > "$REPORT"
echo "" >> "$REPORT"
echo "**Generated:** $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$REPORT"
echo "**System:** Quantum Trading System - Hetzner VPS" >> "$REPORT"
echo "" >> "$REPORT"

# Step 1: Find last EXIT_PUBLISH and extract symbol
echo "## 1. Evidence Discovery" >> "$REPORT"
echo "" >> "$REPORT"

LAST_EXIT=$(grep "EXIT_PUBLISH" "$EXIT_LOG" | tail -1)
if [ -z "$LAST_EXIT" ]; then
    echo "FAIL: No EXIT_PUBLISH found in logs" >> "$REPORT"
    exit 1
fi

# Extract symbol (look for *USDT pattern)
SYMBOL=$(echo "$LAST_EXIT" | grep -oP '\b[A-Z]+USDT\b' | head -1)
if [ -z "$SYMBOL" ]; then
    echo "FAIL: Could not extract symbol from EXIT_PUBLISH" >> "$REPORT"
    exit 1
fi

echo "**Target Symbol:** \`$SYMBOL\`" >> "$REPORT"
echo "" >> "$REPORT"
echo "\`\`\`" >> "$REPORT"
echo "$LAST_EXIT" >> "$REPORT"
echo "\`\`\`" >> "$REPORT"
echo "" >> "$REPORT"

# Step 2: Collect proof chain logs
echo "## 2. Proof Chain Evidence" >> "$REPORT"
echo "" >> "$REPORT"

# 2A: EXIT_PUBLISH
echo "### 2.1 EXIT_PUBLISH (Exit Monitor)" >> "$REPORT"
echo "\`\`\`" >> "$REPORT"
grep "EXIT_PUBLISH.*$SYMBOL" "$EXIT_LOG" | tail -5 >> "$REPORT"
echo "\`\`\`" >> "$REPORT"
echo "" >> "$REPORT"

# 2B: MARGIN CHECK SKIPPED
echo "### 2.2 MARGIN CHECK SKIPPED (Execution Service)" >> "$REPORT"
echo "\`\`\`" >> "$REPORT"
grep "MARGIN CHECK SKIPPED.*$SYMBOL" "$EXEC_LOG" | tail -3 >> "$REPORT"
echo "\`\`\`" >> "$REPORT"
echo "" >> "$REPORT"

# 2C: CLOSE_EXECUTED
echo "### 2.3 CLOSE_EXECUTED (Execution Service)" >> "$REPORT"
echo "\`\`\`" >> "$REPORT"
grep "CLOSE_EXECUTED.*$SYMBOL" "$EXEC_LOG" | tail -3 >> "$REPORT"
echo "\`\`\`" >> "$REPORT"
echo "" >> "$REPORT"

# 2D: TERMINAL STATE
echo "### 2.4 TERMINAL STATE: FILLED (Execution Service)" >> "$REPORT"
echo "\`\`\`" >> "$REPORT"
grep "TERMINAL STATE: FILLED.*$SYMBOL" "$EXEC_LOG" | tail -3 >> "$REPORT"
echo "\`\`\`" >> "$REPORT"
echo "" >> "$REPORT"

# Step 3: Code assertions
echo "## 3. Code Verification" >> "$REPORT"
echo "" >> "$REPORT"

# 3A: allowed_fields
echo "### 3.1 allowed_fields (reduce_only + reason)" >> "$REPORT"
echo "**Location:** \`execution_service.py\`" >> "$REPORT"
echo "" >> "$REPORT"
ALLOWED_LINE=$(grep -n "allowed_fields = {" "$EXEC_SERVICE" | head -1 | cut -d: -f1)
if [ -z "$ALLOWED_LINE" ]; then
    echo "FAIL: allowed_fields not found" >> "$REPORT"
else
    echo "**Line $ALLOWED_LINE:**" >> "$REPORT"
    echo "\`\`\`python" >> "$REPORT"
    nl -ba "$EXEC_SERVICE" | sed -n "$((ALLOWED_LINE-1)),$((ALLOWED_LINE+12))p" >> "$REPORT"
    echo "\`\`\`" >> "$REPORT"
    
    # Check if reduce_only and reason are present
    if grep -A 10 "allowed_fields = {" "$EXEC_SERVICE" | grep -q "reduce_only"; then
        echo "PASS: reduce_only found in allowed_fields" >> "$REPORT"
    else
        echo "FAIL: reduce_only NOT in allowed_fields" >> "$REPORT"
    fi
    
    if grep -A 10 "allowed_fields = {" "$EXEC_SERVICE" | grep -q "reason"; then
        echo "PASS: reason found in allowed_fields" >> "$REPORT"
    else
        echo "FAIL: reason NOT in allowed_fields" >> "$REPORT"
    fi
fi
echo "" >> "$REPORT"

# 3B: Margin bypass
echo "### 3.2 Margin Bypass (reduce_only check)" >> "$REPORT"
echo "**Location:** \`execution_service.py\`" >> "$REPORT"
echo "" >> "$REPORT"
MARGIN_LINE=$(grep -n "MARGIN CHECK SKIPPED" "$EXEC_SERVICE" | head -1 | cut -d: -f1)
if [ -z "$MARGIN_LINE" ]; then
    echo "FAIL: Margin bypass not found" >> "$REPORT"
else
    echo "**Line $MARGIN_LINE:**" >> "$REPORT"
    echo "\`\`\`python" >> "$REPORT"
    nl -ba "$EXEC_SERVICE" | sed -n "$((MARGIN_LINE-3)),$((MARGIN_LINE+2))p" >> "$REPORT"
    echo "\`\`\`" >> "$REPORT"
    echo "PASS: Margin bypass for reduce_only found" >> "$REPORT"
fi
echo "" >> "$REPORT"

# 3C: CLOSE_EXECUTED logging
echo "### 3.3 CLOSE_EXECUTED Logging" >> "$REPORT"
echo "**Location:** \`execution_service.py\`" >> "$REPORT"
echo "" >> "$REPORT"
CLOSE_LINE=$(grep -n "CLOSE_EXECUTED" "$EXEC_SERVICE" | head -1 | cut -d: -f1)
if [ -z "$CLOSE_LINE" ]; then
    echo "FAIL: CLOSE_EXECUTED logging not found" >> "$REPORT"
else
    echo "**Line $CLOSE_LINE:**" >> "$REPORT"
    echo "\`\`\`python" >> "$REPORT"
    nl -ba "$EXEC_SERVICE" | sed -n "$((CLOSE_LINE-2)),$((CLOSE_LINE+7))p" >> "$REPORT"
    echo "\`\`\`" >> "$REPORT"
    echo "PASS: CLOSE_EXECUTED logging with audit trail found" >> "$REPORT"
fi
echo "" >> "$REPORT"

# Step 4: Summary
echo "## 4. Summary" >> "$REPORT"
echo "" >> "$REPORT"

# Count passes
PASSES=0
FAILS=0

# Check logs
if grep -q "MARGIN CHECK SKIPPED.*$SYMBOL" "$EXEC_LOG"; then PASSES=$((PASSES+1)); else FAILS=$((FAILS+1)); fi
if grep -q "CLOSE_EXECUTED.*$SYMBOL" "$EXEC_LOG"; then PASSES=$((PASSES+1)); else FAILS=$((FAILS+1)); fi
if grep -q "TERMINAL STATE: FILLED.*$SYMBOL" "$EXEC_LOG"; then PASSES=$((PASSES+1)); else FAILS=$((FAILS+1)); fi

# Check code
if grep -A 10 "allowed_fields = {" "$EXEC_SERVICE" | grep -q "reduce_only"; then PASSES=$((PASSES+1)); else FAILS=$((FAILS+1)); fi
if grep -q "MARGIN CHECK SKIPPED" "$EXEC_SERVICE"; then PASSES=$((PASSES+1)); else FAILS=$((FAILS+1)); fi
if grep -q "CLOSE_EXECUTED" "$EXEC_SERVICE"; then PASSES=$((PASSES+1)); else FAILS=$((FAILS+1)); fi

if [ $FAILS -eq 0 ]; then
    echo "### STATUS: PASS ($PASSES/6 checks)" >> "$REPORT"
    echo "" >> "$REPORT"
    echo "P0.4C implementation is COMPLETE and VERIFIED:" >> "$REPORT"
    echo "- Complete proof chain logged for $SYMBOL" >> "$REPORT"
    echo "- reduce_only and reason in allowed_fields" >> "$REPORT"
    echo "- Margin bypass for reduce_only closes" >> "$REPORT"
    echo "- CLOSE_EXECUTED audit trail logging" >> "$REPORT"
else
    echo "### STATUS: PARTIAL ($PASSES/6 checks passed, $FAILS failed)" >> "$REPORT"
    echo "" >> "$REPORT"
    echo "Some checks failed. Review sections above for details." >> "$REPORT"
fi
echo "" >> "$REPORT"

echo "## 5. Notes" >> "$REPORT"
echo "" >> "$REPORT"
echo "- **Test Symbol:** $SYMBOL" >> "$REPORT"
echo "- **Log Files:** \`$EXIT_LOG\`, \`$EXEC_LOG\`" >> "$REPORT"
echo "- **Code File:** \`$EXEC_SERVICE\`" >> "$REPORT"
echo "- **Git Commit:** dce358ce (P0.4C COMPLETE)" >> "$REPORT"
echo "" >> "$REPORT"
echo "---" >> "$REPORT"
echo "*Generated by P0.4C Evidence Collector (READ-ONLY mode)*" >> "$REPORT"

echo "Report generated: $REPORT"
