#!/usr/bin/env bash
# Universe Service Deployment Checklist
# Run this on VPS after git pull

set -euo pipefail

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║   UNIVERSE SERVICE - DEPLOYMENT CHECKLIST                      ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

CHECKS_PASSED=0
CHECKS_TOTAL=0

check() {
    CHECKS_TOTAL=$((CHECKS_TOTAL + 1))
    if eval "$2" > /dev/null 2>&1; then
        echo "✅ $1"
        CHECKS_PASSED=$((CHECKS_PASSED + 1))
        return 0
    else
        echo "❌ $1"
        return 1
    fi
}

check_file() {
    CHECKS_TOTAL=$((CHECKS_TOTAL + 1))
    if [ -f "$2" ]; then
        echo "✅ $1"
        CHECKS_PASSED=$((CHECKS_PASSED + 1))
        return 0
    else
        echo "❌ $1 (file not found: $2)"
        return 1
    fi
}

echo "Pre-deployment checks:"
echo ""

# File existence
check_file "Service code exists" "microservices/universe_service/main.py"
check_file "Config example exists" "microservices/universe_service/universe-service.env.example"
check_file "Systemd unit exists" "ops/systemd/quantum-universe-service.service"
check_file "Proof script exists" "ops/proof_universe.sh"

echo ""

# Python syntax
check "Python syntax valid" "python3 -m py_compile microservices/universe_service/main.py"

# Redis connectivity
check "Redis reachable" "redis-cli ping"

# User exists
check "User 'qt' exists" "id -u qt"

echo ""
echo "Deployment status: $CHECKS_PASSED/$CHECKS_TOTAL checks passed"
echo ""

if [ $CHECKS_PASSED -eq $CHECKS_TOTAL ]; then
    echo "✅ All pre-deployment checks passed"
    echo ""
    echo "Ready to deploy. Run these commands:"
    echo ""
    echo "  sudo cp microservices/universe_service/universe-service.env.example /etc/quantum/universe-service.env"
    echo "  sudo chown qt:qt /etc/quantum/universe-service.env"
    echo "  sudo cp ops/systemd/quantum-universe-service.service /etc/systemd/system/"
    echo "  sudo systemctl daemon-reload"
    echo "  sudo systemctl enable quantum-universe-service"
    echo "  sudo systemctl start quantum-universe-service"
    echo "  sleep 3"
    echo "  sudo systemctl status quantum-universe-service"
    echo "  bash ops/proof_universe.sh"
    echo ""
else
    echo "❌ Some checks failed. Please fix issues before deploying."
    exit 1
fi
