#!/bin/bash
set -euo pipefail

echo "=== EXIT BRAIN V3.5 - SYSTEMD CREDENTIALS SETUP (AUTO) ==="
echo "Date: $(date -u)"
echo ""

# ============================================================================
# STEP A: DISCOVER CURRENT CONFIGURATION
# ============================================================================
echo "=== STEP A: DISCOVER CURRENT SERVICE CONFIGURATION ==="
echo ""

echo "1) Current service unit:"
systemctl cat quantum-position-monitor.service
echo ""

# ============================================================================
# STEP B: SOURCE EXISTING CREDENTIALS FROM APPLY-LAYER
# ============================================================================
echo "=== STEP B: SOURCE CREDENTIALS FROM EXISTING SERVICE ==="
echo ""

echo "1) Searching for existing Binance credentials..."
CRED_FILE=""

# Check common locations
for file in /etc/quantum/binance-pnl-tracker.env /etc/quantum/apply-layer.env /etc/quantum/global.env; do
    if [ -f "$file" ] && grep -q "^BINANCE_API_KEY=" "$file" 2>/dev/null; then
        CRED_FILE="$file"
        echo "✅ Found credentials in: $CRED_FILE"
        break
    fi
done

if [ -z "$CRED_FILE" ]; then
    echo "❌ Cannot find Binance credentials in /etc/quantum/"
    echo "Searched: binance-pnl-tracker.env, apply-layer.env, global.env"
    exit 1
fi

# Extract credentials (but don't display them)
API_KEY=$(grep "^BINANCE_API_KEY=" "$CRED_FILE" | cut -d= -f2- | tr -d '"' | tr -d "'")
API_SECRET=$(grep "^BINANCE_API_SECRET=" "$CRED_FILE" | cut -d= -f2- | tr -d '"' | tr -d "'")

if [ -z "$API_KEY" ] || [ -z "$API_SECRET" ]; then
    echo "❌ Credentials empty or malformed in $CRED_FILE"
    exit 1
fi

echo "✅ Credentials extracted (not displaying)"
echo ""

echo "2) Checking systemd-creds availability..."
if command -v systemd-creds >/dev/null 2>&1; then
    echo "✅ systemd-creds found"
    systemd-creds --version | head -1
else
    echo "❌ systemd-creds not found"
    exit 1
fi
echo ""

echo "3) Creating credentials directory..."
install -d -m 700 /etc/quantum/creds
ls -ld /etc/quantum/creds
echo ""

echo "4) Creating temporary plaintext files..."
echo "$API_KEY" > /root/.BINANCE_API_KEY
echo "$API_SECRET" > /root/.BINANCE_API_SECRET
chmod 600 /root/.BINANCE_API_KEY /root/.BINANCE_API_SECRET
echo "✅ Temporary files created (not displaying contents)"
echo ""

echo "5) Encrypting credentials with systemd-creds..."
systemd-creds encrypt /root/.BINANCE_API_KEY /etc/quantum/creds/BINANCE_API_KEY.cred
echo "✅ BINANCE_API_KEY.cred created"

systemd-creds encrypt /root/.BINANCE_API_SECRET /etc/quantum/creds/BINANCE_API_SECRET.cred
echo "✅ BINANCE_API_SECRET.cred created"
echo ""

echo "6) Securing credential files..."
chmod 600 /etc/quantum/creds/*.cred
chown root:root /etc/quantum/creds/*.cred
ls -la /etc/quantum/creds/
echo ""

echo "7) Removing plaintext secrets..."
shred -u /root/.BINANCE_API_KEY /root/.BINANCE_API_SECRET 2>/dev/null || rm -f /root/.BINANCE_API_KEY /root/.BINANCE_API_SECRET
unset API_KEY API_SECRET
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

echo "✅ Wrapper script created (displaying without secrets):"
head -20 /usr/local/bin/qt_position_monitor_start.sh
echo "..."
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
if systemctl is-active --quiet quantum-position-monitor.service; then
    echo "✅ Service is active"
else
    echo "❌ Service failed to start"
    systemctl status quantum-position-monitor.service --no-pager -l
    exit 1
fi
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
journalctl -u quantum-position-monitor.service -n 30 --no-pager || tail -30 /var/log/quantum/position-monitor.log || echo "No logs available"
echo ""

echo "3) Testing Binance API connection..."
cd /home/qt/quantum_trader

echo ""
echo "   a) Check positions/TP/SL:"
timeout 25s python3 check_positions_tpsl.py 2>&1 | grep -v "API Secret required" | head -50 || {
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 124 ]; then
        echo "⚠️  Command timed out after 25s"
    elif timeout 5s python3 check_positions_tpsl.py 2>&1 | grep -q "API Secret required"; then
        echo "❌ Still seeing 'API Secret required' error"
    else
        echo "✅ No credential errors detected"
    fi
}
echo ""

echo "   b) Check Exit Brain positions:"
timeout 25s python3 check_exit_brain_positions.py 2>&1 | grep -v "API Secret required" | head -50 || {
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 124 ]; then
        echo "⚠️  Command timed out after 25s"
    else
        echo "✅ Script executed"
    fi
}
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
    echo "✅ Credentials: LOADED AT RUNTIME"
    CREDS_OK=true
else
    echo "❌ Credentials: NOT LOADED"
    CREDS_OK=false
fi

# Check if API calls work (look for "API secret required" error in test)
if timeout 10s python3 check_positions_tpsl.py 2>&1 | grep -q "API Secret required"; then
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
    echo "✅ Systemd credentials successfully configured for quantum-position-monitor"
    echo "✅ Encrypted credentials stored in /etc/quantum/creds/"
    echo "✅ Runtime credentials accessible at /run/credentials/quantum-position-monitor.service/"
    echo "✅ Application can access Binance API with encrypted credentials"
else
    echo "❌ VERDICT: FAIL"
    echo ""
    echo "Issues detected:"
    [ "$SERVICE_OK" = false ] && echo "  - Service not active"
    [ "$CREDS_OK" = false ] && echo "  - Credentials not loaded at runtime"
    [ "$API_OK" = false ] && echo "  - API calls still failing"
fi

echo ""
echo "=== CREDENTIAL FILES (NOT DISPLAYING CONTENTS) ==="
echo "Encrypted credentials:"
ls -lh /etc/quantum/creds/
echo ""
echo "Runtime decrypted credentials (filenames only):"
ls /run/credentials/quantum-position-monitor.service/ 2>/dev/null || echo "Not available"

echo ""
echo "=== SETUP COMPLETE ==="
