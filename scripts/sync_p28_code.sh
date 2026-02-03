#!/bin/bash
# Sync P2.8 Portfolio Risk Governor code between repo and deployment path
# Run after: git pull

set -e

echo "Syncing P2.8 Portfolio Risk Governor..."

# Source: canonical repo
SRC="/home/qt/quantum_trader/microservices/portfolio_risk_governor"

# Destination: deployment path
DEST="/opt/quantum/microservices/portfolio_risk_governor"

if [ ! -d "$SRC" ]; then
    echo "ERROR: Source directory not found: $SRC"
    exit 1
fi

# Sync code
rsync -av --delete "$SRC/" "$DEST/"

# Fix ownership
chown -R qt:qt "$DEST"

echo "✅ P2.8 code synced from $SRC to $DEST"
