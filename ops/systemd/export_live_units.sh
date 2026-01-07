#!/bin/bash
# Export live systemd units to repository
# Usage: ./export_live_units.sh

set -e

REPO_ROOT="/home/qt/quantum_trader"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPORT_DIR="$REPO_ROOT/ops/systemd/live_units/$TIMESTAMP"

echo "========================================="
echo "EXPORT LIVE SYSTEMD UNITS"
echo "========================================="
echo "Export to: $EXPORT_DIR"
echo

# Create export directory
mkdir -p "$EXPORT_DIR"

# Copy unit files
echo "Copying quantum-*.service files..."
cp /etc/systemd/system/quantum-*.service "$EXPORT_DIR/" 2>/dev/null || true

echo "Copying quantum-trader.target..."
cp /etc/systemd/system/quantum-trader.target "$EXPORT_DIR/" 2>/dev/null || true

# Count exported files
EXPORTED_COUNT=$(ls -1 "$EXPORT_DIR" | wc -l)
echo "✅ Exported $EXPORTED_COUNT files"
echo

# Create metadata file
cat > "$EXPORT_DIR/metadata.txt" << EOF
Export Date: $(date '+%Y-%m-%d %H:%M:%S')
Hostname: $(hostname)
Systemd Version: $(systemctl --version | head -1)
Files Exported: $EXPORTED_COUNT

Running Services:
$(systemctl list-units "quantum*.service" --state=running --no-legend | wc -l)/32

Failed Units:
$(systemctl --failed --no-legend | wc -l)

Export Command: $0
EOF

echo "Metadata saved to: $EXPORT_DIR/metadata.txt"
echo

# List exported files
echo "Exported files:"
ls -lh "$EXPORT_DIR" | tail -n +2 | awk '{print "  " $9 " (" $5 ")"}'
echo

echo "✅ Export complete!"
echo "Location: $EXPORT_DIR"
echo
echo "To commit to git:"
echo "  cd $REPO_ROOT"
echo "  git add ops/systemd/live_units/$TIMESTAMP"
echo "  git commit -m \"ops(systemd): export live units $TIMESTAMP\""
echo
echo "========================================="
