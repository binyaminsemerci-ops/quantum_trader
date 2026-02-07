#!/bin/bash
#
# CALIBRATION-ONLY SAFE RUNNER
# Følger CALIBRATION_PRODUCTION_SAFETY_CHECKLIST.md
#
# Usage: ./run_calibration_safe.sh
#

set -e

BOLD='\033[1m'
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BOLD}🔒 CALIBRATION-ONLY SAFE RUNNER${NC}"
echo "=================================================="
echo ""

# FASE 0.1: Systemhelse
echo -e "${BOLD}FASE 0.1: Systemhelse${NC}"
echo "Sjekker services..."

services=(
    "quantum-ai-engine"
    "quantum-execution"
    "quantum-learning-monitor"
    "quantum-learning-api"
)

all_active=true
for service in "${services[@]}"; do
    if systemctl is-active --quiet "$service"; then
        echo -e "  ✅ $service: ${GREEN}ACTIVE${NC}"
    else
        echo -e "  ❌ $service: ${RED}INACTIVE${NC}"
        all_active=false
    fi
done

if [ "$all_active" = false ]; then
    echo -e "\n${RED}❌ STOPP: Ikke alle services er active${NC}"
    exit 1
fi

echo ""
echo "Sjekker for errors siste 30 min..."
error_count=$(journalctl -u quantum-ai-engine --since "30 minutes ago" | grep -i "error\|traceback" | wc -l || echo "0")
if [ "$error_count" -gt 0 ]; then
    echo -e "  ${YELLOW}⚠️  Fant $error_count errors/tracebacks siste 30 min${NC}"
    read -p "Fortsett likevel? (y/N): " continue
    if [ "$continue" != "y" ]; then
        echo -e "${RED}❌ STOPP: Error i logs${NC}"
        exit 1
    fi
else
    echo -e "  ✅ Ingen errors siste 30 min"
fi

echo ""
echo -e "${GREEN}✅ FASE 0.1: PASS${NC}"
echo ""
sleep 1

# FASE 0.2: Data-integritet
echo -e "${BOLD}FASE 0.2: Data-integritet (CLM)${NC}"

CLM_FILE="/home/qt/quantum_trader/data/clm_trades.jsonl"

if [ ! -f "$CLM_FILE" ]; then
    echo -e "${RED}❌ STOPP: CLM file ikke funnet: $CLM_FILE${NC}"
    exit 1
fi

trade_count=$(wc -l < "$CLM_FILE")
echo "  📊 Total trades: $trade_count"

if [ "$trade_count" -lt 50 ]; then
    echo -e "${RED}❌ STOPP: Ikke nok trades ($trade_count < 50)${NC}"
    exit 1
fi

# Kjør Python validering
python3 <<EOF
import json
from pathlib import Path
import sys

file = Path('$CLM_FILE')
trades = [json.loads(line) for line in file.read_text().splitlines()]

symbols = len(set(t["symbol"] for t in trades))
wins = sum(1 for t in trades if t.get('actual_pnl_pct', 0) > 0)
losses = sum(1 for t in trades if t.get('actual_pnl_pct', 0) < 0)

print(f"  📈 Symbols: {symbols}")
print(f"  📊 Wins: {wins}, Losses: {losses}")

# Validering
issues = []
if symbols < 2:
    issues.append("Mindre enn 2 symbols")
if wins == 0:
    issues.append("Ingen wins")
if losses == 0:
    issues.append("Ingen losses")

# Sjekk for bad data
bad_count = 0
for t in trades:
    if not t.get('entry_price') or not t.get('exit_price'):
        bad_count += 1

if bad_count > 0:
    issues.append(f"{bad_count} trades med missing prices")

if issues:
    print()
    for issue in issues:
        print(f"❌ {issue}")
    sys.exit(1)

print("  ✅ Data valid")
EOF

if [ $? -ne 0 ]; then
    echo -e "\n${RED}❌ STOPP: Data-validering feilet${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✅ FASE 0.2: PASS${NC}"
echo ""
sleep 1

# FASE 0.3: Learning Cadence Gate
echo -e "${BOLD}FASE 0.3: Learning Cadence Gate${NC}"

readiness=$(curl -s http://localhost:8003/api/learning/readiness/simple)
ready=$(echo "$readiness" | jq -r '.ready')
actions=$(echo "$readiness" | jq -r '.actions[]' 2>/dev/null | tr '\n' ' ')

echo "  Response: $readiness"

if [ "$ready" != "true" ]; then
    echo -e "${RED}❌ STOPP: Learning Cadence ikke ready${NC}"
    exit 1
fi

if ! echo "$actions" | grep -q "calibration"; then
    echo -e "${RED}❌ STOPP: 'calibration' ikke i tillatte actions${NC}"
    exit 1
fi

echo -e "  ✅ Ready: true"
echo -e "  ✅ Actions: $actions"

echo ""
echo -e "${GREEN}✅ FASE 0.3: PASS${NC}"
echo ""
sleep 1

# FASE 1: Pre-flight lockdown
echo -e "${BOLD}FASE 1: Pre-flight Lockdown${NC}"
echo ""
echo "Bekreft at du forstår:"
echo "  • Calibration vil IKKE endre noe automatisk"
echo "  • Kun analyse og rapport genereres"
echo "  • Du må manuelt approve deployment"
echo ""
read -p "Forstått? (y/N): " understood
if [ "$understood" != "y" ]; then
    echo -e "${RED}❌ STOPP: Du må forstå isolasjon${NC}"
    exit 1
fi

echo -e "${GREEN}✅ FASE 1: PASS${NC}"
echo ""
sleep 1

# FASE 2: Kjør calibration
echo -e "${BOLD}FASE 2: Kjør Calibration${NC}"
echo ""
echo "Starter calibration analyse..."
echo ""

cd /home/qt/quantum_trader
python microservices/learning/calibration_cli.py run

if [ $? -ne 0 ]; then
    echo -e "\n${RED}❌ STOPP: Calibration feilet${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✅ FASE 2: CALIBRATION KJØRT${NC}"
echo ""

# FASE 3: Resultatvalg
echo -e "${BOLD}FASE 3: Resultatvalg${NC}"
echo ""

# Finn nyeste rapport
latest_report=$(ls -t /tmp/calibration_cal_*.md 2>/dev/null | head -1)
latest_json=$(ls -t /tmp/calibration_cal_*.json 2>/dev/null | head -1)

if [ -z "$latest_report" ] || [ -z "$latest_json" ]; then
    echo -e "${RED}❌ STOPP: Finner ikke rapport${NC}"
    exit 1
fi

echo "📄 Rapport: $latest_report"
echo "📄 JSON: $latest_json"
echo ""

# Ekstraher job_id fra JSON
job_id=$(jq -r '.version' "$latest_json")
echo "🆔 Job ID: $job_id"
echo ""

# Hent metrics
mse_improvement=$(jq -r '.confidence_calibration.mse_improvement_pct // "N/A"' "$latest_json")
validation_passed=$(jq -r '.validation.passed // 0' "$latest_json")
validation_total=$(jq -r '.validation.total // 0' "$latest_json")

echo "📊 MSE Improvement: $mse_improvement%"
echo "✅ Validation: $validation_passed/$validation_total passed"
echo ""

# Sjekk safety bounds
if [ "$mse_improvement" != "N/A" ]; then
    # Bash kan ikke sammenligne floats direkte, bruk bc
    if (( $(echo "$mse_improvement < 5.0" | bc -l) )); then
        echo -e "${YELLOW}⚠️  WARNING: MSE improvement < 5% (har $mse_improvement%)${NC}"
        echo "   Safety bound: MSE forbedring må være ≥5%"
        echo ""
        read -p "Fortsett likevel? (IKKE anbefalt) (y/N): " continue
        if [ "$continue" != "y" ]; then
            echo -e "${RED}❌ ABORT: MSE improvement for lav${NC}"
            exit 1
        fi
    fi
fi

echo "Les rapporten:"
echo "  cat $latest_report"
echo ""
echo "Eller:"
echo "  less $latest_report"
echo ""

read -p "Har du lest og forstått rapporten? (y/N): " read_report
if [ "$read_report" != "y" ]; then
    echo -e "${YELLOW}⚠️  STOPP: Les rapporten først${NC}"
    echo ""
    echo "Når du er klar, kjør:"
    echo "  python microservices/learning/calibration_cli.py approve $job_id"
    exit 0
fi

echo ""
echo -e "${BOLD}🚨 KRITISK BESLUTNING${NC}"
echo ""
echo "Du står nå foran deployment av calibration config."
echo "Dette vil påvirke AI Engine predictions umiddelbart."
echo ""
echo "Etter deployment MÅ du:"
echo "  • Overvåke systemet i 24 timer"
echo "  • Være klar til å rollback"
echo ""
read -p "Er du klar til å approve og monitor i 24t? (y/N): " approve
if [ "$approve" != "y" ]; then
    echo -e "${YELLOW}⚠️  ABORT: Ikke approved${NC}"
    echo ""
    echo "Når du er klar, kjør:"
    echo "  python microservices/learning/calibration_cli.py approve $job_id"
    exit 0
fi

echo ""
echo "Deployer calibration config..."
python microservices/learning/calibration_cli.py approve "$job_id"

if [ $? -ne 0 ]; then
    echo -e "\n${RED}❌ STOPP: Deployment feilet${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✅ FASE 4: DEPLOYMENT COMPLETE${NC}"
echo ""

# FASE 4.2: Post-Apply Watch
echo -e "${BOLD}FASE 4.2: Post-Apply Watch (24t)${NC}"
echo ""
echo "Verifiser at config er lastet:"
journalctl -u quantum-ai-engine -n 50 | grep -i calibration | tail -5
echo ""
echo -e "${GREEN}Calibration deployed!${NC}"
echo ""
echo "📊 MONITORING PÅKREVD I 24 TIMER:"
echo ""
echo "1. PnL drift:"
echo "   tail -f /home/qt/quantum_trader/data/clm_trades.jsonl | jq '.actual_pnl_pct'"
echo ""
echo "2. Win rate (last 20):"
echo "   python microservices/learning/calibration_cli.py status"
echo ""
echo "3. AI Engine logs:"
echo "   journalctl -u quantum-ai-engine -f | grep -i confidence"
echo ""
echo "🔄 ROLLBACK hvis nødvendig:"
echo "   python microservices/learning/calibration_cli.py rollback"
echo ""
echo -e "${BOLD}⏰ Sett reminder om 24 timer for å verifisere resultater!${NC}"
