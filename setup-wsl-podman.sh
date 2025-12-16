#!/bin/bash
# Quantum Trader WSL Podman-Compose Setup
set -e

echo "?? Quantum Trader WSL Setup"
echo "==========================="
echo ""

# Sjekk WSL
if ! grep -qi microsoft /proc/version 2>/dev/null; then
    echo "? M? kj?res i WSL"
    exit 1
fi
echo "? Kj?rer i WSL"

# Installer podman
echo ""
echo "?? Installerer podman..."
if ! command -v podman &> /dev/null; then
    sudo apt-get update
    sudo apt-get install -y podman
fi

# Installer podman-compose (via pip hvis ikke tilgjengelig i apt)
if ! command -v podman-compose &> /dev/null; then
    sudo apt-get install -y python3-pip
    pip3 install podman-compose
fi

echo "? Podman: $(podman --version)"
echo "? Podman-compose: $(podman-compose --version)"

# Naviger til prosjekt
cd /mnt/c/quantum_trader
echo "? I: $(pwd)"

# Bygg
echo ""
echo "???  Bygger containere..."
podman-compose build

# Start
echo ""
echo "?? Starter Quantum Trader (dev)..."
podman-compose --profile dev up -d

# Status
echo ""
echo "?? Status:"
podman-compose ps

echo ""
echo "? Quantum Trader kj?rer!"
echo ""
echo "?? Nyttige kommandoer:"
echo "  podman-compose logs -f backend    # Se logger"
echo "  podman-compose ps                 # Se status"
echo "  podman-compose down               # Stopp alt"
echo ""
echo "?? Backend: http://localhost:8000"
echo "?? Docs: http://localhost:8000/docs"