#!/usr/bin/env bash
# P2.7 Portfolio Clusters VPS Deployment + Proof Pack
set -e

echo "=============================="
echo "P2.7 DEPLOYMENT STARTING"
echo "=============================="
echo ""

# Verify we're in repo root
if [ ! -f "microservices/portfolio_clusters/main.py" ]; then
    echo "ERROR: Not in quantum_trader root or portfolio_clusters/main.py not found"
    exit 1
fi

PROOF_OUTPUT="/home/qt/P2_7_PROOF_$(date +%Y%m%d_%H%M%S).txt"

echo "[1/10] Syncing P2.7 and P2.6 patch to /home/qt/quantum_trader..."
# Sync P2.7 microservice
rsync -av \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='venv' \
    --exclude='node_modules' \
    --exclude='.env' \
    microservices/portfolio_clusters/ \
    /home/qt/quantum_trader/microservices/portfolio_clusters/

# Sync P2.6 patch (cluster stress integration)
rsync -av \
    microservices/portfolio_gate/main.py \
    /home/qt/quantum_trader/microservices/portfolio_gate/main.py

echo "✓ Files synced"
echo ""

echo "[1.5/10] RSYNC PROOF (self-verification)..."
echo "  Checking P2.6 cluster integration patch..."
grep -n "p26_cluster_stress_used" /home/qt/quantum_trader/microservices/portfolio_gate/main.py | head -1 || (echo "✗ P26 PATCH MISSING" && exit 1)
echo "  ✓ P2.6 patch verified"

echo "  Checking P2.7 main service..."
test -f /home/qt/quantum_trader/microservices/portfolio_clusters/main.py || (echo "✗ P27 CODE MISSING" && exit 1)
echo "  ✓ P2.7 code verified"
echo ""

echo "[2/10] Installing /etc/quantum/portfolio-clusters.env..."
mkdir -p /etc/quantum
cp deployment/config/portfolio-clusters.env /etc/quantum/portfolio-clusters.env
chown root:root /etc/quantum/portfolio-clusters.env
chmod 644 /etc/quantum/portfolio-clusters.env
echo "✓ P2.7 config installed"
echo ""

echo "[3/10] Installing systemd service..."
cp deployment/systemd/quantum-portfolio-clusters.service /etc/systemd/system/
systemctl daemon-reload
echo "✓ Systemd service installed"
echo ""

echo "[4/10] Enabling and starting quantum-portfolio-clusters..."
systemctl enable quantum-portfolio-clusters
systemctl restart quantum-portfolio-clusters
sleep 3

if systemctl is-active --quiet quantum-portfolio-clusters; then
    echo "✓ quantum-portfolio-clusters ACTIVE"
else
    echo "✗ quantum-portfolio-clusters FAILED TO START"
    journalctl -u quantum-portfolio-clusters -n 50 --no-pager
    exit 1
fi
echo ""

echo "[5/10] Checking P2.7 metrics (port 8048)..."
sleep 2
if curl -sf http://127.0.0.1:8048/metrics > /dev/null; then
    echo "✓ P2.7 metrics responding"
else
    echo "✗ P2.7 metrics not responding"
    exit 1
fi
echo ""

echo "[6/10] Restarting quantum-portfolio-gate (P2.6 to pick up cluster stress)..."
systemctl restart quantum-portfolio-gate
sleep 3

if systemctl is-active --quiet quantum-portfolio-gate; then
    echo "✓ Portfolio Gate restarted successfully"
else
    echo "✗ Portfolio Gate failed to restart"
    exit 1
fi
echo ""

echo "[7/10] Checking Redis cluster state keys..."
echo "  Cluster state key:"
redis-cli EXISTS quantum:portfolio:cluster_state
echo "  Clusters key:"
redis-cli EXISTS quantum:portfolio:clusters
echo ""

echo "[8/10] Checking P2.7 metrics snapshot..."
curl -s http://127.0.0.1:8048/metrics | grep -E "p27_(corr_ready|symbols_in_matrix|clusters_count|cluster_stress)" | head -20
echo ""

echo "[9/10] Checking P2.6 integration (cluster fallback metric)..."
curl -s http://127.0.0.1:8047/metrics | grep -E "p26_cluster" | head -10
echo ""

echo "[10/10] Generating proof document..."

{
    echo "===== P2.7 PORTFOLIO CLUSTERS DEPLOYMENT PROOF ====="
    echo "Generated: $(date)"
    echo "Commit: $(git rev-parse --short HEAD)"
    echo ""
    
    echo "===== SERVICE STATUS ====="
    systemctl status quantum-portfolio-clusters --no-pager | head -15
    echo ""
    
    echo "===== P2.7 METRICS (port 8048) ====="
    curl -s http://127.0.0.1:8048/metrics | grep p27_ | head -80
    echo ""
    
    echo "===== CLUSTER STATE (Redis) ====="
    echo "--- Global cluster state ---"
    redis-cli HGETALL quantum:portfolio:cluster_state | head -20
    echo ""
    echo "--- Clusters mapping ---"
    redis-cli HGETALL quantum:portfolio:clusters | head -30
    echo ""
    echo "--- Cluster stream (last 2 entries) ---"
    redis-cli XREVRANGE quantum:stream:portfolio.cluster_state + - COUNT 2 | head -40
    echo ""
    
    echo "===== P2.6 INTEGRATION CHECK ====="
    echo "--- P2.6 cluster metrics ---"
    curl -s http://127.0.0.1:8047/metrics | grep -E "p26_cluster|p26_corr_proxy|p26_stress" | head -20
    echo ""
    echo "--- P2.6 logs (cluster stress usage) ---"
    journalctl -u quantum-portfolio-gate --since "2 minutes ago" -n 40 --no-pager | grep -i "cluster\|corr" || echo "No cluster mentions yet (OK if just started)"
    echo ""
    
    echo "===== P2.7 LOGS (last 60 lines) ====="
    journalctl -u quantum-portfolio-clusters --since "2 minutes ago" -n 60 --no-pager
    echo ""
    
    echo "===== PROOF COMPLETE ====="
    echo "Review: cat ${PROOF_OUTPUT}"
    
} > "${PROOF_OUTPUT}"

chown qt:qt "${PROOF_OUTPUT}"

echo "✓ Proof document saved: ${PROOF_OUTPUT}"
echo ""
echo "=============================="
echo "P2.7 DEPLOYMENT COMPLETE ✅"
echo "=============================="
echo ""
echo "Next steps:"
echo "1. Review proof: cat ${PROOF_OUTPUT}"
echo "2. Monitor P2.7: journalctl -u quantum-portfolio-clusters -f"
echo "3. Monitor P2.6 integration: journalctl -u quantum-portfolio-gate -f | grep cluster"
echo "4. Check metrics: curl http://localhost:8048/metrics | grep p27_"
echo "5. Verify cluster stress: redis-cli HGET quantum:portfolio:cluster_state cluster_stress"
echo ""
echo "Rollback (if needed):"
echo "  systemctl stop quantum-portfolio-clusters"
echo "  systemctl disable quantum-portfolio-clusters"
echo "  systemctl restart quantum-portfolio-gate  # P2.6 will fallback to proxy"
echo "  git revert <commit>"
echo ""

# Exit 0 on success (CI/ops best practice)
if systemctl is-active --quiet quantum-portfolio-clusters && \
   systemctl is-active --quiet quantum-portfolio-gate && \
   curl -sf http://127.0.0.1:8048/metrics > /dev/null && \
   [ -f "${PROOF_OUTPUT}" ]; then
    echo "✅ Deployment verified - exit 0"
    
    # P5+ Auto-ledger
    echo ""
    echo "=============================="
    echo "P5+ AUTO-LEDGER"
    echo "=============================="
    
    STRICT_LEDGER=${STRICT_LEDGER:-false}
    
    python3 ops/ops_ledger_append.py \
        --operation "P2.7 Deploy — Portfolio Clusters (atomic) + P2.6 sync" \
        --objective "Deploy P2.7 correlation matrix + capital clustering and verify P2.6 cluster K integration" \
        --risk_class SERVICE_RESTART \
        --blast_radius "Portfolio gating + clusters only; no execution impact" \
        --allowed_paths microservices/portfolio_clusters/ \
        --allowed_paths microservices/portfolio_gate/main.py \
        --allowed_paths deployment/systemd/quantum-portfolio-clusters.service \
        --allowed_paths deployment/config/portfolio-clusters.env \
        --allowed_paths ops/p27_deploy_and_proof.sh \
        --allowed_services quantum-portfolio-clusters \
        --allowed_services quantum-portfolio-gate \
        --changes_summary "Deployed P2.7 + synced P2.6 patch; atomic rsync with proof" \
        --proof_path "${PROOF_OUTPUT}" \
        --metrics_urls "http://127.0.0.1:8048/metrics" \
        --metrics_urls "http://127.0.0.1:8047/metrics" \
        --metrics_grep "p27_(corr_ready|clusters_count|cluster_stress_sum|updates_total|min_points|points_per_symbol)" \
        --metrics_grep "p26_cluster_(stress_used|fallback)" \
        --redis_cmds "HGETALL quantum:portfolio:cluster_state" \
        --notes "Auto-ledger via P5+ ops_ledger_append.py" \
        ${STRICT_LEDGER:+--strict}
    
    LEDGER_RC=$?
    if [ $LEDGER_RC -ne 0 ]; then
        echo "⚠️  Auto-ledger failed (exit $LEDGER_RC)"
        if [ "$STRICT_LEDGER" = "true" ]; then
            echo "❌ STRICT_LEDGER=true - deployment fails"
            exit 1
        else
            echo "ℹ️  STRICT_LEDGER=false - deployment succeeds despite ledger failure"
        fi
    else
        echo "✓ Auto-ledger entry created"
    fi
    
    exit 0
else
    echo "❌ Deployment verification failed - exit 1"
    exit 1
fi
