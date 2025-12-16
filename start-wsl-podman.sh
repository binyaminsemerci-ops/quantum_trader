#!/bin/bash
# =============================================================================
# QUANTUM TRADER - WSL PODMAN STARTUP SCRIPT
# =============================================================================
# Dette scriptet starter Quantum Trader i WSL med podman-compose
# 
# Kjør fra WSL:
#   cd ~/quantum_trader
#   bash start-wsl-podman.sh
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# =============================================================================
# 1. VALIDERING
# =============================================================================
log_info "Validerer miljø..."

# Sjekk at vi er i WSL
if ! grep -qi microsoft /proc/version; then
    log_error "Dette scriptet må kjøres fra WSL"
    exit 1
fi
log_success "Kjører i WSL ✓"

# Sjekk current directory
CURRENT_DIR=$(pwd)
if [[ "$CURRENT_DIR" == "/mnt/c"* ]]; then
    log_error "Du er i /mnt/c path: $CURRENT_DIR"
    log_error "Naviger til ~/quantum_trader først:"
    echo ""
    echo "  cd ~/quantum_trader"
    echo "  bash start-wsl-podman.sh"
    exit 1
fi
log_success "Path OK: $CURRENT_DIR ✓"

# Sjekk at podman er installert
if ! command -v podman &> /dev/null; then
    log_error "Podman er ikke installert"
    echo ""
    echo "Installer med:"
    echo "  sudo apt-get update"
    echo "  sudo apt-get install -y podman"
    exit 1
fi
log_success "Podman installert: $(podman --version) ✓"

# Sjekk at podman-compose er installert
if ! command -v podman-compose &> /dev/null; then
    log_error "podman-compose er ikke installert"
    echo ""
    echo "Installer med:"
    echo "  pip3 install podman-compose"
    exit 1
fi
log_success "podman-compose installert ✓"

# Sjekk at docker-compose.wsl.yml finnes
if [ ! -f "docker-compose.wsl.yml" ]; then
    log_error "docker-compose.wsl.yml ikke funnet"
    echo ""
    echo "Kjør dette scriptet fra quantum_trader root directory:"
    echo "  cd ~/quantum_trader"
    exit 1
fi
log_success "docker-compose.wsl.yml funnet ✓"

# Sjekk at .env finnes
if [ ! -f ".env" ]; then
    log_warn ".env ikke funnet - bruker defaults fra docker-compose.wsl.yml"
fi

# =============================================================================
# 2. CLEANUP EXISTING CONTAINERS
# =============================================================================
log_info "Stopper eksisterende containers (hvis noen kjører)..."
podman-compose -f docker-compose.wsl.yml down 2>/dev/null || true
log_success "Cleanup OK ✓"

# =============================================================================
# 3. START SERVICES
# =============================================================================
log_info "Starter Redis + AI-Engine..."
podman-compose -f docker-compose.wsl.yml up -d redis ai-engine

# Wait for containers to start
sleep 3

# =============================================================================
# 4. VERIFY
# =============================================================================
log_info "Verifiserer services..."

# Check container status
CONTAINERS=$(podman ps --format "{{.Names}}" | grep -E "quantum_redis|quantum_ai_engine" || true)
if [ -z "$CONTAINERS" ]; then
    log_error "Ingen containers kjører!"
    exit 1
fi
log_success "Containers kjører:"
podman ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep quantum

# Wait for Redis to be ready
log_info "Venter på Redis..."
for i in {1..10}; do
    if podman exec quantum_redis redis-cli ping &>/dev/null; then
        log_success "Redis ready ✓"
        break
    fi
    if [ $i -eq 10 ]; then
        log_error "Redis startet ikke"
        podman logs quantum_redis
        exit 1
    fi
    sleep 1
done

# Wait for AI-Engine to be ready
log_info "Venter på AI-Engine..."
for i in {1..30}; do
    if curl -sf http://localhost:8001/health >/dev/null 2>&1; then
        log_success "AI-Engine ready ✓"
        break
    fi
    if [ $i -eq 30 ]; then
        log_error "AI-Engine startet ikke"
        podman logs quantum_ai_engine | tail -50
        exit 1
    fi
    sleep 2
done

# =============================================================================
# 5. HEALTH CHECK
# =============================================================================
log_info "Kjører health checks..."

# Redis health
REDIS_PING=$(podman exec quantum_redis redis-cli ping)
if [ "$REDIS_PING" = "PONG" ]; then
    log_success "Redis health: OK ✓"
else
    log_error "Redis health: FAILED"
    exit 1
fi

# AI-Engine health
AI_HEALTH=$(curl -s http://localhost:8001/health | grep -o '"status":"[^"]*"' | cut -d'"' -f4 || echo "error")
if [ "$AI_HEALTH" = "healthy" ]; then
    log_success "AI-Engine health: OK ✓"
else
    log_error "AI-Engine health: $AI_HEALTH"
    exit 1
fi

# Check for /mnt/c in sys.path
log_info "Sjekker at ingen /mnt/c paths..."
SYSPATH=$(podman exec quantum_ai_engine python -c "import sys; print('\\n'.join(sys.path))" | grep -c "/mnt/c" || echo "0")
if [ "$SYSPATH" -eq 0 ]; then
    log_success "Ingen /mnt/c paths i sys.path ✓"
else
    log_error "FEIL: Fant /mnt/c paths i sys.path!"
    podman exec quantum_ai_engine python -c "import sys; print('\\n'.join(sys.path))"
    exit 1
fi

# =============================================================================
# 6. SUCCESS
# =============================================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log_success "🚀 QUANTUM TRADER STARTET VELLYKKET!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📊 Services kjører:"
echo "   - Redis:      http://localhost:6379"
echo "   - AI-Engine:  http://localhost:8001"
echo ""
echo "🔍 Neste steg:"
echo ""
echo "   1. Følg AI-Engine logs:"
echo "      podman logs -f quantum_ai_engine"
echo ""
echo "   2. Start Backend (i ny terminal):"
echo "      source .venv/bin/activate"
echo "      python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000"
echo ""
echo "   3. Test full system:"
echo "      curl http://localhost:8000/health"
echo "      curl http://localhost:8001/health"
echo ""
echo "🛑 For å stoppe:"
echo "   podman-compose -f docker-compose.wsl.yml down"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
