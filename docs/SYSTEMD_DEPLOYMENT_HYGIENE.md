# Systemd Deployment Hygiene - Apply Layer & P3.4

**Problem:** WorkingDirectory was empty in systemd unit, causing deployment confusion (root vs qt)

**Solution:** Lock down systemd configuration to prevent drift

---

## Apply Layer Systemd Unit

**File:** `/etc/systemd/system/quantum-apply-layer.service`

```ini
[Unit]
Description=Quantum Trader - Apply Layer (P3)
After=network.target redis.service
Wants=redis.service

[Service]
Type=simple
User=qt
Group=qt
WorkingDirectory=/home/qt/quantum_trader
EnvironmentFile=/etc/quantum/apply-layer.env
Environment=PYTHONUNBUFFERED=1
ExecStart=/usr/bin/python3 -u microservices/apply_layer/main.py
Restart=always
RestartSec=10s
StandardOutput=journal
StandardError=journal
SyslogIdentifier=quantum-apply-layer

[Install]
WantedBy=multi-user.target
```

**Key Fields:**
- `WorkingDirectory=/home/qt/quantum_trader` - **EXPLICIT PATH** (not empty)
- `EnvironmentFile=/etc/quantum/apply-layer.env` - Config from file
- `User=qt` - Run as qt user (not root)

---

## P3.4 Reconcile Engine Systemd Unit

**File:** `/etc/systemd/system/quantum-reconcile-engine.service`

```ini
[Unit]
Description=Quantum Trader - P3.4 Reconcile Engine
After=network.target redis.service
Wants=redis.service

[Service]
Type=simple
User=qt
Group=qt
WorkingDirectory=/home/qt/quantum_trader
EnvironmentFile=/etc/quantum/reconcile-engine.env
Environment=PYTHONUNBUFFERED=1
ExecStart=/usr/bin/python3 -u microservices/reconcile_engine/main.py
Restart=always
RestartSec=10s
StandardOutput=journal
StandardError=journal
SyslogIdentifier=quantum-reconcile-engine

[Install]
WantedBy=multi-user.target
```

---

## Deployment Procedure (Lock Down)

### 1. Always Use Canonical Path

```bash
# Get WorkingDirectory from systemd (never guess)
WORKING_DIR=$(systemctl show -p WorkingDirectory --value quantum-apply-layer)

# If empty, fix systemd unit first (see below)
if [ -z "$WORKING_DIR" ]; then
    echo "ERROR: WorkingDirectory not set in systemd unit"
    exit 1
fi

# Deploy to canonical path
cd "$WORKING_DIR"
git pull origin main
systemctl restart quantum-apply-layer quantum-reconcile-engine
```

### 2. Fix Empty WorkingDirectory

If `systemctl show -p WorkingDirectory` returns empty:

```bash
# Edit unit file
sudo systemctl edit --full quantum-apply-layer

# Add/verify WorkingDirectory line:
WorkingDirectory=/home/qt/quantum_trader

# Reload and verify
sudo systemctl daemon-reload
systemctl show -p WorkingDirectory --value quantum-apply-layer
# Should output: /home/qt/quantum_trader
```

### 3. Verify Configuration

```bash
# Check all critical fields
systemctl cat quantum-apply-layer | grep -E "WorkingDirectory|User|ExecStart|EnvironmentFile"

# Expected output:
# WorkingDirectory=/home/qt/quantum_trader
# User=qt
# Group=qt
# EnvironmentFile=/etc/quantum/apply-layer.env
# ExecStart=/usr/bin/python3 -u microservices/apply_layer/main.py
```

### 4. Deployment Script

**File:** `ops/deploy_apply_layer.sh`

```bash
#!/bin/bash
set -e

SERVICE="quantum-apply-layer"
BRANCH="${1:-main}"

# Get canonical path from systemd
WORKING_DIR=$(systemctl show -p WorkingDirectory --value "$SERVICE")

if [ -z "$WORKING_DIR" ]; then
    echo "ERROR: WorkingDirectory not set in systemd unit"
    echo "Run: sudo systemctl edit --full $SERVICE"
    echo "Add: WorkingDirectory=/home/qt/quantum_trader"
    exit 1
fi

echo "=== Deploying $SERVICE from $BRANCH to $WORKING_DIR ==="

# Deploy
cd "$WORKING_DIR"
git fetch origin
git checkout "$BRANCH"
git pull origin "$BRANCH"

# Verify code deployed
echo "Verifying deployment..."
git log -1 --oneline

# Restart service
echo "Restarting service..."
sudo systemctl restart "$SERVICE"
sleep 3

# Check status
systemctl status "$SERVICE" --no-pager -l | head -20

echo "✓ Deployment complete"
```

---

## Environment Files

### Apply Layer Config

**File:** `/etc/quantum/apply-layer.env`

```bash
# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# Binance Testnet
BINANCE_TESTNET_API_KEY=your_api_key
BINANCE_TESTNET_API_SECRET=your_api_secret

# Apply Layer
APPLY_MODE=testnet
SYMBOLS=BTCUSDT,ETHUSDT,SOLUSDT
APPLY_ALLOWLIST=BTCUSDT
APPLY_POLL_SEC=5
APPLY_METRICS_PORT=8043

# Safety
K_BLOCK_CRITICAL=0.80
K_BLOCK_WARNING=0.60
APPLY_KILL_SWITCH=false
```

### P3.4 Config

**File:** `/etc/quantum/reconcile-engine.env`

```bash
# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# Binance Testnet (for position checks)
BINANCE_TESTNET_API_KEY=your_api_key
BINANCE_TESTNET_API_SECRET=your_api_secret

# P3.4 Settings
P34_SYMBOLS=BTCUSDT,ETHUSDT,SOLUSDT
P34_CHECK_INTERVAL_SEC=2
P34_METRICS_PORT=8046

# RECONCILE_CLOSE Lease
P34_RECONCILE_HOLD_LEASE_SEC=900
```

---

## Verification Checklist

After deployment, verify:

```bash
# 1. Service running from correct path
systemctl show -p WorkingDirectory quantum-apply-layer
# Output: /home/qt/quantum_trader

# 2. Process running as correct user
ps aux | grep "apply_layer" | grep -v grep
# Should show: qt (not root)

# 3. Code version matches
cd /home/qt/quantum_trader
git log -1 --oneline
# Should match latest commit

# 4. Metrics available
curl -s http://localhost:8043/metrics | grep reconcile_close | head -5
# Should show metrics

# 5. Logs flowing
journalctl -u quantum-apply-layer --since "1 minute ago" --no-pager | tail -10
```

---

## Common Issues

### Issue: "WorkingDirectory empty"

**Symptom:** `systemctl show -p WorkingDirectory` returns nothing

**Fix:**
```bash
sudo systemctl edit --full quantum-apply-layer
# Add: WorkingDirectory=/home/qt/quantum_trader
sudo systemctl daemon-reload
```

### Issue: "Code not found"

**Symptom:** `ExecStart` fails with "No such file or directory"

**Cause:** Service runs from root (`/`), not WorkingDirectory

**Fix:** Ensure WorkingDirectory is set in unit file

### Issue: "Different code versions in /root and /home/qt"

**Symptom:** Edits in /root don't affect running service

**Fix:** 
1. Delete `/root/quantum_trader` or mark clearly as dev-only
2. Always work in `/home/qt/quantum_trader`
3. Use deployment script that checks systemd path

---

## Production Hardening

### 1. Read-Only WorkingDirectory (Optional)

For maximum security, make code directory read-only:

```bash
# Service runs as qt, but code owned by root
sudo chown -R root:qt /home/qt/quantum_trader
sudo chmod -R 755 /home/qt/quantum_trader

# qt can read/execute but not modify
```

### 2. Separate Config Directory

Keep secrets in `/etc/quantum/`, owned by root:

```bash
sudo chown root:root /etc/quantum/*.env
sudo chmod 600 /etc/quantum/*.env
```

### 3. Audit Deployment

Log all deployments:

```bash
# In deploy script
echo "$(date) - $USER deployed $BRANCH to $SERVICE" >> /var/log/quantum/deployments.log
```

---

**Status:** Systemd hygiene locked down. No more root/qt confusion.
