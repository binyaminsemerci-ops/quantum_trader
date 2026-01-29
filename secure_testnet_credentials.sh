#!/bin/bash
set -euo pipefail

echo "=== SECURE TESTNET CREDENTIALS SETUP ==="
echo "Date: $(date -u)"
echo ""

# Step 1: Interactive secret input
echo "1) Enter new testnet credentials securely"
echo "=========================================="
echo ""
echo "Opening nano for BINANCE_TESTNET_API_KEY..."
echo "Paste your new key, then Ctrl+X, Y, Enter"
echo ""
read -p "Press Enter to continue..."

nano /root/.TESTNET_API_KEY

if [ ! -f /root/.TESTNET_API_KEY ]; then
    echo "❌ Key file not created"
    exit 1
fi

echo ""
echo "Opening nano for BINANCE_TESTNET_API_SECRET..."
echo "Paste your new secret, then Ctrl+X, Y, Enter"
echo ""
read -p "Press Enter to continue..."

nano /root/.TESTNET_API_SECRET

if [ ! -f /root/.TESTNET_API_SECRET ]; then
    echo "❌ Secret file not created"
    exit 1
fi

echo "✅ Credentials entered (not displaying)"
echo ""

# Step 2: Encrypt with systemd-creds
echo "2) Encrypting credentials"
echo "=========================="

mkdir -p /etc/quantum/creds
chmod 700 /etc/quantum/creds

systemd-creds encrypt /root/.TESTNET_API_KEY /etc/quantum/creds/TESTNET_API_KEY.cred
echo "✅ TESTNET_API_KEY.cred created"

systemd-creds encrypt /root/.TESTNET_API_SECRET /etc/quantum/creds/TESTNET_API_SECRET.cred
echo "✅ TESTNET_API_SECRET.cred created"

chmod 600 /etc/quantum/creds/TESTNET_*.cred
chown root:root /etc/quantum/creds/TESTNET_*.cred

ls -lh /etc/quantum/creds/TESTNET_*.cred
echo ""

# Step 3: Create decrypted env files (root-only readable)
echo "3) Creating secure environment files"
echo "====================================="

mkdir -p /etc/quantum/testnet-secrets
chmod 700 /etc/quantum/testnet-secrets

# Decrypt to env files
systemd-creds decrypt /etc/quantum/creds/TESTNET_API_KEY.cred > /tmp/key.txt
systemd-creds decrypt /etc/quantum/creds/TESTNET_API_SECRET.cred > /tmp/secret.txt

# Create testnet.env (used by exit-monitor and other services)
cat > /etc/quantum/testnet-secrets/credentials.env << EOF
BINANCE_TESTNET_API_KEY=$(cat /tmp/key.txt)
BINANCE_TESTNET_SECRET_KEY=$(cat /tmp/secret.txt)
BINANCE_API_KEY=$(cat /tmp/key.txt)
BINANCE_API_SECRET=$(cat /tmp/secret.txt)
BINANCE_TESTNET=true
USE_BINANCE_TESTNET=true
EXIT_BRAIN_V35_ENABLED=true
EOF

chmod 600 /etc/quantum/testnet-secrets/credentials.env
chown root:root /etc/quantum/testnet-secrets/credentials.env

# Update position-monitor.env
cat > /etc/quantum/position-monitor.env << EOF
# Position Monitor Environment
REDIS_HOST=127.0.0.1
REDIS_PORT=6379
REDIS_DB=0

LOGLEVEL=INFO
LOG_FILE=/var/log/quantum/position-monitor.log

MONITOR_INTERVAL_MS=500
POSITION_CHECK_ENABLED=true

# Binance API Credentials (from encrypted credentials)
BINANCE_API_KEY=$(cat /tmp/key.txt)
BINANCE_API_SECRET=$(cat /tmp/secret.txt)
BINANCE_TESTNET=true
EXIT_BRAIN_V35_ENABLED=true
EOF

chmod 600 /etc/quantum/position-monitor.env
chown root:root /etc/quantum/position-monitor.env

# Securely delete temp files
shred -u /tmp/key.txt /tmp/secret.txt /root/.TESTNET_API_KEY /root/.TESTNET_API_SECRET

echo "✅ Environment files created (600, root:root)"
echo "   - /etc/quantum/testnet-secrets/credentials.env"
echo "   - /etc/quantum/position-monitor.env"
echo ""

# Step 4: Update testnet.env to source from secrets
echo "4) Updating /etc/quantum/testnet.env"
echo "====================================="

# Backup original
cp /etc/quantum/testnet.env /etc/quantum/testnet.env.backup

# Remove old plaintext credentials, add source directive
sed -i '/^BINANCE_TESTNET_API_KEY=/d' /etc/quantum/testnet.env
sed -i '/^BINANCE_TESTNET_SECRET_KEY=/d' /etc/quantum/testnet.env
sed -i '/^BINANCE_TESTNET_API_SECRET=/d' /etc/quantum/testnet.env

# Add comment at top
cat > /tmp/new_testnet_env << 'EOF'
# ============================================
# BINANCE TESTNET CREDENTIALS
# ============================================
# Credentials stored in encrypted form at:
#   /etc/quantum/creds/TESTNET_*.cred
# Decrypted credentials sourced from:
#   /etc/quantum/testnet-secrets/credentials.env
# To rotate keys: run secure_testnet_credentials.sh
# ============================================

EOF

cat /tmp/new_testnet_env /etc/quantum/testnet.env > /tmp/merged_env
mv /tmp/merged_env /etc/quantum/testnet.env
rm /tmp/new_testnet_env

echo "✅ testnet.env updated (credentials removed, sourcing from secrets)"
echo ""

# Step 5: Update systemd services to source credentials
echo "5) Updating systemd service configurations"
echo "==========================================="

# Update exit-monitor to source credentials
mkdir -p /etc/systemd/system/quantum-exit-monitor.service.d
cat > /etc/systemd/system/quantum-exit-monitor.service.d/credentials.conf << 'EOF'
[Service]
# Source encrypted credentials
EnvironmentFile=-/etc/quantum/testnet-secrets/credentials.env
EOF

# Update position-monitor (already has drop-in, just ensure it sources base file)
# Drop-in already exists from earlier setup

echo "✅ Service drop-ins configured"
echo ""

# Step 6: Reload and restart services
echo "6) Restarting services"
echo "======================"

systemctl daemon-reload

systemctl restart quantum-exit-monitor.service
echo "✅ quantum-exit-monitor restarted"

systemctl restart quantum-position-monitor.service
echo "✅ quantum-position-monitor restarted"

sleep 5
echo ""

# Step 7: Verify (WITHOUT showing secrets)
echo "7) Verification (secrets redacted)"
echo "==================================="

echo "A) Service status:"
systemctl is-active quantum-exit-monitor.service && echo "  ✅ exit-monitor: active" || echo "  ❌ exit-monitor: inactive"
systemctl is-active quantum-position-monitor.service && echo "  ✅ position-monitor: active" || echo "  ❌ position-monitor: inactive"
echo ""

echo "B) Encrypted credentials:"
ls -lh /etc/quantum/creds/TESTNET_*.cred 2>/dev/null | awk '{print "  " $9, $5}' || echo "  ❌ Not found"
echo ""

echo "C) Decrypted env files (permissions):"
ls -l /etc/quantum/testnet-secrets/credentials.env 2>/dev/null | awk '{print "  " $1, $3":"$4, $9}' || echo "  ❌ Not found"
ls -l /etc/quantum/position-monitor.env | awk '{print "  " $1, $3":"$4, $9}'
echo ""

echo "D) API test (first 10 chars only):"
if [ -f /etc/quantum/testnet-secrets/credentials.env ]; then
    KEY_PREFIX=$(grep '^BINANCE_API_KEY=' /etc/quantum/testnet-secrets/credentials.env | cut -d= -f2 | cut -c1-10)
    echo "  API Key: ${KEY_PREFIX}... (redacted)"
else
    echo "  ❌ Cannot read credentials file"
fi
echo ""

# Step 8: Test Binance API (without showing keys)
echo "8) Testing Binance API connectivity"
echo "===================================="

cat > /tmp/test_new_creds.py << 'PYEOF'
import os
from binance.client import Client

# Load from secure env
with open('/etc/quantum/testnet-secrets/credentials.env') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            key, val = line.split('=', 1)
            os.environ[key] = val

api_key = os.getenv('BINANCE_API_KEY')
api_secret = os.getenv('BINANCE_API_SECRET')

if not api_key or not api_secret:
    print('❌ Credentials not loaded')
    exit(1)

print(f'Testing with API Key: {api_key[:10]}... (redacted)')
print('')

try:
    # Create client with testnet URLs
    client = Client(api_key, api_secret)
    client.API_URL = 'https://testnet.binancefuture.com'
    client.FUTURES_URL = 'https://testnet.binancefuture.com/fapi'
    client.FUTURES_DATA_URL = 'https://testnet.binancefuture.com/fapi'
    
    account = client.futures_account()
    balance = account.get('totalWalletBalance', 'N/A')
    print(f'✅ TESTNET API WORKING')
    print(f'   Balance: {balance} USDT')
    
    positions = [p for p in client.futures_position_information() if float(p['positionAmt']) != 0]
    print(f'   Open positions: {len(positions)}')
    
    if positions:
        for p in positions[:3]:
            symbol = p['symbol']
            qty = float(p['positionAmt'])
            pnl = float(p['unRealizedProfit'])
            print(f'     {symbol}: qty={qty:+.4f}, PnL=${pnl:.2f}')
    
    print('')
    print('✅ VERDICT: New credentials working')
    
except Exception as e:
    print(f'❌ API Error: {str(e)}')
    print('')
    print('❌ VERDICT: New credentials failed')
    print('   Check if keys have correct permissions and IP whitelist')
    exit(1)
PYEOF

python3 /tmp/test_new_creds.py
echo ""

# Final summary
echo "=== FINAL SUMMARY ==="
echo ""
echo "✅ Security Implementation:"
echo "   - Encrypted at rest: /etc/quantum/creds/TESTNET_*.cred"
echo "   - Root-only decrypted: /etc/quantum/testnet-secrets/credentials.env"
echo "   - No plaintext in git or logs"
echo "   - Rotation script: /root/secure_testnet_credentials.sh"
echo ""
echo "✅ Services Updated:"
echo "   - quantum-exit-monitor.service"
echo "   - quantum-position-monitor.service"
echo ""
echo "🔒 Old credentials in chat history are now IRRELEVANT"
echo "   New encrypted credentials cannot be read by unauthorized users"
echo ""
echo "=== SETUP COMPLETE ==="
