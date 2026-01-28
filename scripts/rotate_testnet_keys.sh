#!/bin/bash
# Safe API Key Rotation Script
# Usage: ./rotate_testnet_keys.sh NEW_API_KEY NEW_API_SECRET

set -e

if [ "$#" -ne 2 ]; then
    echo "❌ Usage: $0 NEW_API_KEY NEW_API_SECRET"
    echo ""
    echo "Example:"
    echo "  $0 'AbCd1234...' 'XyZ9876...'"
    echo ""
    echo "⚠️  Use quotes around keys to prevent shell expansion"
    exit 1
fi

NEW_KEY="$1"
NEW_SECRET="$2"

# Validation
if [ ${#NEW_KEY} -lt 40 ]; then
    echo "❌ ERROR: API Key seems too short (${#NEW_KEY} chars)"
    exit 1
fi

if [ ${#NEW_SECRET} -lt 40 ]; then
    echo "❌ ERROR: API Secret seems too short (${#NEW_SECRET} chars)"
    exit 1
fi

echo "🔐 Starting testnet API key rotation..."
echo ""

# Files to update
FILES=(
    "/etc/quantum/apply-layer.env"
    "/etc/quantum/governor.env"
    "/etc/quantum/intent-executor.env"
    "/etc/quantum/testnet.env"
)

# Backup first
BACKUP_DIR="/etc/quantum/backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

echo "📦 Creating backups in $BACKUP_DIR..."
for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        cp "$file" "$BACKUP_DIR/$(basename $file)"
        echo "  ✅ Backed up: $file"
    else
        echo "  ⚠️  Not found: $file"
    fi
done
echo ""

# Update files
echo "🔄 Updating config files..."
for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  Updating: $file"
        
        # Update or add BINANCE_TESTNET_API_KEY
        if grep -q "^BINANCE_TESTNET_API_KEY=" "$file"; then
            sed -i "s|^BINANCE_TESTNET_API_KEY=.*|BINANCE_TESTNET_API_KEY=$NEW_KEY|" "$file"
        else
            echo "BINANCE_TESTNET_API_KEY=$NEW_KEY" >> "$file"
        fi
        
        # Update or add BINANCE_TESTNET_API_SECRET
        if grep -q "^BINANCE_TESTNET_API_SECRET=" "$file"; then
            sed -i "s|^BINANCE_TESTNET_API_SECRET=.*|BINANCE_TESTNET_API_SECRET=$NEW_SECRET|" "$file"
        else
            echo "BINANCE_TESTNET_API_SECRET=$NEW_SECRET" >> "$file"
        fi
        
        # Also handle old naming (testnet.env uses SECRET_KEY)
        if [ "$(basename $file)" = "testnet.env" ]; then
            if grep -q "^BINANCE_TESTNET_SECRET_KEY=" "$file"; then
                sed -i "s|^BINANCE_TESTNET_SECRET_KEY=.*|BINANCE_TESTNET_SECRET_KEY=$NEW_SECRET|" "$file"
            else
                echo "BINANCE_TESTNET_SECRET_KEY=$NEW_SECRET" >> "$file"
            fi
        fi
        
        echo "    ✅ Updated"
    fi
done
echo ""

# Verify (without printing secrets)
echo "✅ Verification (key lengths only)..."
for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        KEY_LEN=$(grep "^BINANCE_TESTNET_API_KEY=" "$file" | cut -d= -f2 | wc -c)
        SECRET_LEN=$(grep "^BINANCE_TESTNET_API_SECRET=" "$file" | cut -d= -f2 | wc -c)
        echo "  $file: KEY=$((KEY_LEN-1)) chars, SECRET=$((SECRET_LEN-1)) chars"
    fi
done
echo ""

echo "✅ Key rotation complete!"
echo ""
echo "Next steps:"
echo "  1. Restart services: systemctl restart quantum-governor quantum-apply-layer quantum-intent-executor"
echo "  2. Check logs: journalctl -u quantum-governor -n 20"
echo "  3. Test: python3 /home/qt/quantum_trader/scripts/dump_exchange_positions.py"
echo ""
echo "Backups saved in: $BACKUP_DIR"
