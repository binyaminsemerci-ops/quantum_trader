#!/bin/bash
# P2.6 Portfolio Heat Gate - VPS Proof Script
# Generates comprehensive deployment verification report

set -euo pipefail

OUTPUT_FILE="docs/P2_6_VPS_PROOF.txt"
mkdir -p docs

echo "Generating P2.6 VPS Proof Report..."

{
    echo "================================================================================"
    echo "P2.6 PORTFOLIO HEAT GATE - VPS PROOF REPORT"
    echo "================================================================================"
    echo "Generated: $(date -u +"%Y-%m-%d %H:%M:%S UTC")"
    echo "Hostname: $(hostname)"
    echo ""
    
    echo "───────────────────────────────────────────────────────────────────────────────"
    echo "1. SERVICE STATUS"
    echo "───────────────────────────────────────────────────────────────────────────────"
    systemctl status quantum-portfolio-heat-gate --no-pager | head -20 || echo "Service not found"
    echo ""
    
    echo "───────────────────────────────────────────────────────────────────────────────"
    echo "2. SERVICE LOGS (Last 30 lines)"
    echo "───────────────────────────────────────────────────────────────────────────────"
    journalctl -u quantum-portfolio-heat-gate -n 30 --no-pager || echo "No logs available"
    echo ""
    
    echo "───────────────────────────────────────────────────────────────────────────────"
    echo "3. PROMETHEUS METRICS"
    echo "───────────────────────────────────────────────────────────────────────────────"
    timeout 5 curl -s http://localhost:8056/metrics | grep -E "^p26_" || echo "Metrics not available"
    echo ""
    
    echo "───────────────────────────────────────────────────────────────────────────────"
    echo "4. PORTFOLIO HEAT CALCULATION"
    echo "───────────────────────────────────────────────────────────────────────────────"
    echo "Portfolio State:"
    redis-cli HGETALL quantum:state:portfolio 2>/dev/null || echo "Not available"
    echo ""
    echo "Recent Position Snapshots (last 5):"
    redis-cli XREVRANGE quantum:stream:position.snapshot + - COUNT 5 2>/dev/null | head -50 || echo "Not available"
    echo ""
    
    echo "───────────────────────────────────────────────────────────────────────────────"
    echo "5. STREAM EVIDENCE"
    echo "───────────────────────────────────────────────────────────────────────────────"
    echo "Harvest Proposals (last 3):"
    redis-cli XREVRANGE quantum:stream:harvest.proposal + - COUNT 3 2>/dev/null | head -30 || echo "No proposals"
    echo ""
    echo "Calibrated Proposals (last 3):"
    redis-cli XREVRANGE quantum:stream:harvest.calibrated + - COUNT 3 2>/dev/null | head -30 || echo "No calibrated proposals"
    echo ""
    
    echo "───────────────────────────────────────────────────────────────────────────────"
    echo "6. CONSUMER GROUP STATUS"
    echo "───────────────────────────────────────────────────────────────────────────────"
    redis-cli XINFO GROUPS quantum:stream:harvest.proposal 2>/dev/null | grep -A5 "p26_heat_gate" || echo "Consumer group not found"
    echo ""
    
    echo "───────────────────────────────────────────────────────────────────────────────"
    echo "7. CONFIGURATION"
    echo "───────────────────────────────────────────────────────────────────────────────"
    echo "Environment file:"
    cat /etc/quantum/portfolio-heat-gate.env 2>/dev/null || echo "Not found"
    echo ""
    
    echo "───────────────────────────────────────────────────────────────────────────────"
    echo "8. DOWNGRADE EXAMPLES"
    echo "───────────────────────────────────────────────────────────────────────────────"
    echo "Recent downgrades from logs:"
    journalctl -u quantum-portfolio-heat-gate --since "1 hour ago" | grep -i "DOWNGRADE" | tail -10 || echo "No downgrades found"
    echo ""
    
    echo "───────────────────────────────────────────────────────────────────────────────"
    echo "9. HEALTH CHECK SUMMARY"
    echo "───────────────────────────────────────────────────────────────────────────────"
    SERVICE_ACTIVE=$(systemctl is-active quantum-portfolio-heat-gate 2>/dev/null || echo "inactive")
    METRICS_OK=$(timeout 2 curl -s http://localhost:8056/metrics > /dev/null 2>&1 && echo "OK" || echo "FAIL")
    
    echo "Service Active: ${SERVICE_ACTIVE}"
    echo "Metrics Responding: ${METRICS_OK}"
    echo "Redis Connected: $(redis-cli PING 2>/dev/null || echo 'FAIL')"
    echo ""
    
    if [ "$SERVICE_ACTIVE" = "active" ] && [ "$METRICS_OK" = "OK" ]; then
        echo "✅ P2.6 PORTFOLIO HEAT GATE OPERATIONAL"
    else
        echo "❌ P2.6 PORTFOLIO HEAT GATE ISSUES DETECTED"
    fi
    echo ""
    
    echo "================================================================================"
    echo "END OF REPORT"
    echo "================================================================================"
    
} > ${OUTPUT_FILE}

echo "✅ Proof report generated: ${OUTPUT_FILE}"
cat ${OUTPUT_FILE}
