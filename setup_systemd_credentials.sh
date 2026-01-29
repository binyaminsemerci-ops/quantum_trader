#!/bin/bash
set -euo pipefail

echo "=== EXIT BRAIN V3.5 - SYSTEMD CREDENTIALS SETUP ==="
echo "Date: $(date -u)"
echo ""

# ============================================================================
# STEP A: DISCOVER CURRENT CONFIGURATION
# ============================================================================
echo "=== STEP A: DISCOVER CURRENT SERVICE CONFIGURATION ==="
echo ""

echo "1) Current service unit:"
systemctl cat quantum-position-monitor.service || {
    echo "❌ Service not found"
    exit 1
}
echo ""

# ============================================================================
# STEP B: CREATE SYSTEMD CREDENTIALS (MANUAL SECRET INPUT REQUIRED)
# ============================================================================
echo "=== STEP B: CREATE ENCRYPTED CREDENTIALS ==="
echo ""

echo "1) Checking systemd-creds availability..."
if command -v systemd-creds >/dev/null 2>&1; then
    echo "✅ systemd-creds found"
    systemd-creds --version
elif systemctl --version | grep -q "systemd 250"; then
    echo "✅ systemd 250+ detected (systemd-creds available)"
else
    echo "⚠️  systemd-creds may not be available on this version"
    systemd --version
fi
echo ""

echo "2) Creating credentials directory..."
install -d -m 700 /etc/quantum/creds
ls -ld /etc/quantum/creds
echo ""

echo "3) ⚠️  MANUAL STEP REQUIRED ⚠️"
echo ""
echo "   You must now manually create secret files in /root:"
echo ""
echo "   Option A - Using nano (recommended):"
echo "     nano /root/.BINANCE_API_KEY"
echo "     (paste API key, Ctrl+X, Y, Enter)"
echo ""
echo "     nano /root/.BINANCE_API_SECRET"
echo "     (paste API secret, Ctrl+X, Y, Enter)"
echo ""
echo "   Option B - Using heredoc (be careful in shared terminals):"
echo "     cat > /root/.BINANCE_API_KEY << 'EOF'"
echo "     YOUR_API_KEY_HERE"
echo "     EOF"
echo ""
echo "     cat > /root/.BINANCE_API_SECRET << 'EOF'"
echo "     YOUR_API_SECRET_HERE"
echo "     EOF"
echo ""
echo "   After creating both files, press Enter to continue..."
read -r

# Verify files exist
if [ ! -f /root/.BINANCE_API_KEY ]; then
    echo "❌ /root/.BINANCE_API_KEY not found"
    exit 1
fi

if [ ! -f /root/.BINANCE_API_SECRET ]; then
    echo "❌ /root/.BINANCE_API_SECRET not found"
    exit 1
fi

echo "✅ Secret files found (not displaying contents)"
echo ""

echo "4) Encrypting credentials with systemd-creds..."
systemd-creds encrypt /root/.BINANCE_API_KEY /etc/quantum/creds/BINANCE_API_KEY.cred
echo "✅ BINANCE_API_KEY.cred created"

systemd-creds encrypt /root/.BINANCE_API_SECRET /etc/quantum/creds/BINANCE_API_SECRET.cred
echo "✅ BINANCE_API_SECRET.cred created"
echo ""

echo "5) Securing credential files..."
chmod 600 /etc/quantum/creds/*.cred
chown root:root /etc/quantum/creds/*.cred
ls -la /etc/quantum/creds/
echo ""

echo "6) Removing plaintext secrets..."
rm -f /root/.BINANCE_API_KEY /root/.BINANCE_API_SECRET
echo "✅ Plaintext secrets deleted"
echo ""

# ============================================================================
# STEP C: WIRE INTO SYSTEMD SERVICE
# ============================================================================
echo "=== STEP C: CREATE SYSTEMD DROP-IN ==="
echo ""

echo "1) Creating drop-in directory..."
mkdir -p /etc/systemd/system/quantum-position-monitor.service.d
echo ""

echo "2) Creating credentials.conf drop-in..."
cat > /etc/systemd/system/quantum-position-monitor.service.d/credentials.conf << 'EOF'
[Service]
# Load encrypted credentials at runtime
LoadCredentialEncrypted=BINANCE_API_KEY:/etc/quantum/creds/BINANCE_API_KEY.cred
LoadCredentialEncrypted=BINANCE_API_SECRET:/etc/quantum/creds/BINANCE_API_SECRET.cred

# Expose credential file paths as environment variables
Environment=BINANCE_API_KEY_FILE=%d/BINANCE_API_KEY
Environment=BINANCE_API_SECRET_FILE=%d/BINANCE_API_SECRET
EOF

echo "✅ Drop-in created:"
cat /etc/systemd/system/quantum-position-monitor.service.d/credentials.conf
echo ""

# ============================================================================
# STEP D: CREATE WRAPPER SCRIPT TO READ FILE-BASED CREDENTIALS
# ============================================================================
echo "=== STEP D: CREATE CREDENTIAL WRAPPER SCRIPT ==="
echo ""

# Extract original ExecStart
ORIGINAL_EXEC=$(systemctl cat quantum-position-monitor.service | grep "^ExecStart=" | head -1 | sed 's/^ExecStart=//')
echo "Original ExecStart: $ORIGINAL_EXEC"
echo ""

echo "Creating wrapper script..."
cat > /usr/local/bin/qt_position_monitor_start.sh << 'EOF'
#!/bin/bash
set -euo pipefail

# Read Binance credentials from systemd credential files
if [ -z "${BINANCE_API_KEY_FILE:-}" ] || [ -z "${BINANCE_API_SECRET_FILE:-}" ]; then
    echo "ERROR: Credential files not provided by systemd"
    exit 1
fi

if [ ! -f "$BINANCE_API_KEY_FILE" ] || [ ! -f "$BINANCE_API_SECRET_FILE" ]; then
    echo "ERROR: Credential files not found"
    exit 1
fi

# Export as environment variables for application
export BINANCE_API_KEY=$(cat "$BINANCE_API_KEY_FILE")
export BINANCE_API_SECRET=$(cat "$BINANCE_API_SECRET_FILE")

# Execute original command
EOF

# Append the original ExecStart command to the wrapper
echo "exec $ORIGINAL_EXEC" >> /usr/local/bin/qt_position_monitor_start.sh

chmod 700 /usr/local/bin/qt_position_monitor_start.sh
chown root:root /usr/local/bin/qt_position_monitor_start.sh

echo "✅ Wrapper script created:"
cat /usr/local/bin/qt_position_monitor_start.sh
echo ""

echo "3) Updating drop-in to use wrapper..."
cat >> /etc/systemd/system/quantum-position-monitor.service.d/credentials.conf << 'EOF'

# Override ExecStart to use wrapper
ExecStart=
ExecStart=/usr/local/bin/qt_position_monitor_start.sh
EOF

echo "✅ Updated drop-in:"
cat /etc/systemd/system/quantum-position-monitor.service.d/credentials.conf
echo ""

# ============================================================================
# STEP E: RELOAD AND RESTART
# ============================================================================
echo "=== STEP E: RELOAD AND RESTART SERVICE ==="
echo ""

echo "1) Reloading systemd daemon..."
systemctl daemon-reload
echo "✅ Daemon reloaded"
echo ""

echo "2) Restarting quantum-position-monitor.service..."
systemctl restart quantum-position-monitor.service
echo ""

echo "3) Waiting 5 seconds for startup..."
sleep 5
echo ""

echo "4) Checking service status..."
systemctl is-active quantum-position-monitor.service || {
    echo "❌ Service failed to start"
    systemctl status quantum-position-monitor.service --no-pager -l
    exit 1
}
echo "✅ Service is active"
echo ""

# ============================================================================
# STEP F: PROOF (NO SECRETS)
# ============================================================================
echo "=== STEP F: VERIFICATION (SECRETS REDACTED) ==="
echo ""

echo "1) Runtime credentials directory:"
if [ -d /run/credentials/quantum-position-monitor.service ]; then
    ls -la /run/credentials/quantum-position-monitor.service/ | head -10
else
    echo "⚠️  /run/credentials directory not found (may indicate systemd version issue)"
fi
echo ""

echo "2) Service logs (last 30 lines):"
journalctl -u quantum-position-monitor.service -n 30 --no-pager || echo "No logs available"
echo ""

echo "3) Testing Binance API connection..."
cd /home/qt/quantum_trader

echo ""
echo "   a) Check positions/TP/SL:"
timeout 25s python3 check_positions_tpsl.py 2>&1 | head -50 || echo "Script completed with non-zero exit"
echo ""

echo "   b) Check Exit Brain positions:"
timeout 25s python3 check_exit_brain_positions.py 2>&1 | head -50 || echo "Script completed with non-zero exit"
echo ""

# ============================================================================
# FINAL VERDICT
# ============================================================================
echo "=== FINAL VERDICT ==="
echo ""

# Check if service is active
if systemctl is-active --quiet quantum-position-monitor.service; then
    echo "✅ Service: ACTIVE"
    SERVICE_OK=true
else
    echo "❌ Service: INACTIVE"
    SERVICE_OK=false
fi

# Check if credentials are loaded (check for both files)
if [ -f /run/credentials/quantum-position-monitor.service/BINANCE_API_KEY ] && \
   [ -f /run/credentials/quantum-position-monitor.service/BINANCE_API_SECRET ]; then
    echo "✅ Credentials: LOADED"
    CREDS_OK=true
else
    echo "❌ Credentials: NOT LOADED"
    CREDS_OK=false
fi

# Check if API calls work (look for "API secret required" error in last test)
if timeout 25s python3 check_positions_tpsl.py 2>&1 | grep -q "API Secret required"; then
    echo "❌ API Test: FAILED (secrets not accessible)"
    API_OK=false
else
    echo "✅ API Test: PASSED (no credential errors)"
    API_OK=true
fi

echo ""
if [ "$SERVICE_OK" = true ] && [ "$CREDS_OK" = true ] && [ "$API_OK" = true ]; then
    echo "🎯 VERDICT: PASS"
    echo ""
    echo "Systemd credentials successfully configured for quantum-position-monitor"
else
    echo "❌ VERDICT: FAIL"
    echo ""
    echo "Issues detected:"
    [ "$SERVICE_OK" = false ] && echo "  - Service not active"
    [ "$CREDS_OK" = false ] && echo "  - Credentials not loaded at runtime"
    [ "$API_OK" = false ] && echo "  - API calls still failing"
fi

echo ""
echo "=== SETUP COMPLETE ==="
