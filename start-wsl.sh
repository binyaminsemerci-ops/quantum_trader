#!/bin/bash
# =============================================================================
# QUANTUM TRADER WSL STARTUP SCRIPT
# =============================================================================
# Kjører Redis + AI-Engine i containere med podman-compose
# Backend kjører i host venv for rask utvikling
# =============================================================================

set -e

echo ""
echo "🚀 QUANTUM TRADER - WSL STARTUP"
echo "================================"
echo ""

# Sjekk at vi er i riktig directory
if [ ! -f "docker-compose.wsl.yml" ]; then
    echo "❌ Feil: Må kjøres fra ~/quantum_trader"
    echo "   cd ~/quantum_trader && ./start-wsl.sh"
    exit 1
fi

echo "✅ I riktig directory: $(pwd)"
echo ""

# Sjekk at podman er installert
if ! command -v podman &> /dev/null; then
    echo "❌ podman ikke funnet. Installer med:"
    echo "   sudo apt-get update && sudo apt-get install -y podman"
    exit 1
fi

if ! command -v podman-compose &> /dev/null; then
    echo "❌ podman-compose ikke funnet. Installer med:"
    echo "   pip3 install podman-compose"
    exit 1
fi

echo "✅ Podman: $(podman --version)"
echo "✅ Podman-compose: $(podman-compose --version)"
echo ""

# Stopp eksisterende containere
echo "🛑 Stopper eksisterende containere..."
podman-compose -f docker-compose.wsl.yml down 2>/dev/null || true
echo ""

# Bygg AI Engine image
echo "🏗️  Bygger AI Engine image..."
podman-compose -f docker-compose.wsl.yml build ai-engine
echo ""

# Start Redis + AI-Engine
echo "🚀 Starter Redis + AI-Engine..."
podman-compose -f docker-compose.wsl.yml up -d redis ai-engine
echo ""

# Vent på health checks
echo "⏳ Venter på health checks..."
sleep 5

# Vis status
echo ""
echo "📊 CONTAINER STATUS:"
echo "==================="
podman-compose -f docker-compose.wsl.yml ps
echo ""

# Test Redis
echo "🔍 Testing Redis..."
if podman exec quantum_redis redis-cli ping &>/dev/null; then
    echo "✅ Redis: PONG"
else
    echo "❌ Redis: IKKE TILGJENGELIG"
fi

# Test AI Engine
echo "🔍 Testing AI Engine..."
sleep 3
if curl -s http://localhost:8001/health &>/dev/null; then
    echo "✅ AI Engine: OK"
else
    echo "⚠️  AI Engine: Starter opp..."
fi

echo ""
echo "📜 Se logger med:"
echo "   podman-compose -f docker-compose.wsl.yml logs -f ai-engine"
echo ""
echo "🌐 AI Engine: http://localhost:8001"
echo "🌐 Redis: localhost:6379"
echo ""
echo "✅ Quantum Trader containere kjører!"
echo ""
