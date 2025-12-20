#!/bin/bash
# Quantum Trader V3 - Boot-time Auto-Repair Hook
# Creates systemd service and cron job for automatic repairs on boot

cat > /tmp/quantum-autofix.service <<'EOF'
[Unit]
Description=Quantum Trader V3 Auto-Repair Service
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
ExecStart=/bin/bash -c '
# Wait for Docker to be fully ready
sleep 10

# Phase 1: Database Auto-Creation
echo "[$(date)] Checking PostgreSQL databases..."
docker start quantum_postgres 2>/dev/null || true
sleep 3
docker exec quantum_postgres psql -U quantum -tc "SELECT 1 FROM pg_database WHERE datname='\''quantum'\''" | grep -q 1 || {
    echo "[$(date)] Creating quantum database..."
    docker exec quantum_postgres psql -U quantum -d quantum_trader -c "CREATE DATABASE quantum;"
}

# Phase 2: XGBoost Model Validation
echo "[$(date)] Validating XGBoost models..."
docker exec quantum_ai_engine python3 -c "
import joblib, os
try:
    model = joblib.load('\''/app/models/xgb_futures_model.joblib'\'')
    if model.n_features_in_ != 22:
        print(f'\''Warning: Model expects {model.n_features_in_} features, expected 22'\'')
        exit(1)
    print(f'\''Model OK: {model.n_features_in_} features'\'')
except Exception as e:
    print(f'\''Error validating model: {e}'\'')
    exit(1)
" 2>&1 | tee -a /var/log/quantum_autofix.log

# Phase 3: Container Health Check
echo "[$(date)] Checking container health..."
UNHEALTHY=$(docker ps --filter name=quantum --filter health=unhealthy --format "{{.Names}}")
if [ -n "$UNHEALTHY" ]; then
    echo "[$(date)] Restarting unhealthy containers: $UNHEALTHY"
    echo "$UNHEALTHY" | xargs -r docker restart
fi

echo "[$(date)] Auto-repair boot check complete"
'
RemainAfter=yes

[Install]
WantedBy=multi-user.target
EOF

# Install systemd service
sudo mv /tmp/quantum-autofix.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable quantum-autofix.service

# Also create cron job as fallback
echo '@reboot root sleep 30 && /bin/systemctl start quantum-autofix.service' | sudo tee /etc/cron.d/quantum-autofix

echo "✅ Boot-time auto-repair hooks installed"
echo "   - Systemd service: quantum-autofix.service"
echo "   - Cron fallback: /etc/cron.d/quantum-autofix"
echo ""
echo "To test immediately: sudo systemctl start quantum-autofix.service"
echo "To view logs: journalctl -u quantum-autofix.service"
