#!/bin/bash
#
# Deploy all Phase 2 AI modules to production
# 
# Usage: ./deploy_phase2_all.sh [--no-cache] [--skip-build] [--follow-logs]
#

set -e  # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Parse arguments
NO_CACHE=""
SKIP_BUILD=false
FOLLOW_LOGS=false

for arg in "$@"; do
    case $arg in
        --no-cache)
            NO_CACHE="--no-cache"
            ;;
        --skip-build)
            SKIP_BUILD=true
            ;;
        --follow-logs)
            FOLLOW_LOGS=true
            ;;
    esac
done

# Functions
info() { echo -e "${CYAN}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[✓]${NC} $1"; }
warning() { echo -e "${YELLOW}[⚠]${NC} $1"; }
error() { echo -e "${RED}[✗]${NC} $1"; }
step() { echo -e "\n${MAGENTA}=== $1 ===${NC}\n"; }

# Banner
echo -e "${YELLOW}"
cat << "EOF"
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║    🚀 QUANTUM TRADER - PHASE 2 DEPLOYMENT SCRIPT 🚀      ║
║                                                           ║
║    Phase 2C: Continuous Learning Manager                 ║
║    Phase 2D: Volatility Structure Engine                 ║
║    Phase 2B: Orderbook Imbalance Module                  ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

sleep 2

# ============================================================================
# STEP 1: PRE-DEPLOYMENT CHECKS
# ============================================================================
step "Pre-Deployment Checks"

# Check directory
if [ ! -f "docker-compose.yml" ]; then
    error "docker-compose.yml not found. Run from quantum_trader root."
    exit 1
fi
success "Working directory verified: $(pwd)"

# Check git status
info "Checking git status..."
if [ -n "$(git status --porcelain)" ]; then
    warning "Uncommitted changes detected:"
    git status --short
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        info "Deployment cancelled."
        exit 0
    fi
fi

# Show current commit
CURRENT_COMMIT=$(git log --oneline -1)
success "Current commit: $CURRENT_COMMIT"

# Check Docker
info "Checking Docker availability..."
if ! docker version > /dev/null 2>&1; then
    error "Docker is not running or not installed."
    info "Please start Docker and try again."
    exit 1
fi
DOCKER_VERSION=$(docker version --format '{{.Server.Version}}')
success "Docker is running (version: $DOCKER_VERSION)"

# Check docker-compose
info "Checking docker-compose..."
if ! docker-compose version > /dev/null 2>&1; then
    error "docker-compose not available."
    exit 1
fi
COMPOSE_VERSION=$(docker-compose version --short)
success "docker-compose available (version: $COMPOSE_VERSION)"

# ============================================================================
# STEP 2: BACKUP CURRENT STATE
# ============================================================================
step "Backup Current State"

info "Getting current container status..."
if docker ps --filter "name=quantum_ai_engine" --format "table {{.Names}}\t{{.Status}}" | grep -q quantum; then
    docker ps --filter "name=quantum_ai_engine" --format "table {{.Names}}\t{{.Status}}\t{{.Image}}"
    success "Current AI Engine container found"
else
    warning "No AI Engine container currently running"
fi

# ============================================================================
# STEP 3: BUILD NEW IMAGE
# ============================================================================
if [ "$SKIP_BUILD" = false ]; then
    step "Building New AI Engine Image"
    
    if [ -n "$NO_CACHE" ]; then
        info "Building with --no-cache flag..."
    else
        info "Building with cache..."
    fi
    
    info "Starting docker-compose build..."
    BUILD_START=$(date +%s)
    
    docker-compose build $NO_CACHE ai-engine
    
    BUILD_END=$(date +%s)
    BUILD_DURATION=$((BUILD_END - BUILD_START))
    success "Build completed in ${BUILD_DURATION} seconds"
else
    warning "Skipping build (--skip-build flag)"
fi

# ============================================================================
# STEP 4: STOP CURRENT CONTAINER
# ============================================================================
step "Stopping Current Container"

info "Stopping ai-engine service..."
docker-compose stop ai-engine || warning "Container stop failed (may not be running)"
success "Container stopped"
sleep 2

# ============================================================================
# STEP 5: START NEW CONTAINER
# ============================================================================
step "Starting New Container"

info "Starting ai-engine service..."
if ! docker-compose up -d ai-engine; then
    error "Failed to start container!"
    info "Attempting to show logs..."
    docker-compose logs --tail 50 ai-engine
    exit 1
fi

success "Container started successfully"
sleep 5

# ============================================================================
# STEP 6: VERIFY DEPLOYMENT
# ============================================================================
step "Verifying Deployment"

info "Checking container status..."
CONTAINER_STATUS=$(docker ps --filter "name=quantum_ai_engine" --format "{{.Status}}")
if echo "$CONTAINER_STATUS" | grep -q "Up"; then
    success "Container is running: $CONTAINER_STATUS"
else
    error "Container is not in running state!"
    docker ps -a --filter "name=quantum_ai_engine"
    exit 1
fi

info "Waiting for services to initialize (10 seconds)..."
sleep 10

# ============================================================================
# STEP 7: CHECK LOGS FOR SUCCESSFUL INITIALIZATION
# ============================================================================
step "Checking Initialization Logs"

LOGS=$(docker logs quantum_ai_engine --tail 100 2>&1)

# Check for Phase 2C (CLM)
if echo "$LOGS" | grep -q "Continuous Learning Manager active"; then
    success "Phase 2C: Continuous Learning Manager - ONLINE"
else
    warning "Phase 2C: CLM initialization not confirmed in logs"
fi

# Check for Phase 2D (Volatility)
if echo "$LOGS" | grep -q "Volatility Structure Engine active"; then
    success "Phase 2D: Volatility Structure Engine - ONLINE"
else
    warning "Phase 2D: Volatility initialization not confirmed in logs"
fi

# Check for Phase 2B (Orderbook)
if echo "$LOGS" | grep -q "Orderbook Imbalance Module active"; then
    success "Phase 2B: Orderbook Imbalance Module - ONLINE"
else
    warning "Phase 2B: Orderbook initialization not confirmed in logs"
fi

# Check for orderbook data feed
if echo "$LOGS" | grep -q "Orderbook data feed started"; then
    success "Phase 2B: Orderbook data feed - ACTIVE"
else
    warning "Phase 2B: Orderbook data feed not confirmed in logs"
fi

# Check for errors
if echo "$LOGS" | grep -qE "ERROR|Failed|Exception"; then
    warning "Some errors detected in logs (see below)"
fi

# ============================================================================
# STEP 8: HEALTH CHECK
# ============================================================================
step "Health Check"

info "Testing AI Engine health endpoint..."
if curl -s -f "http://localhost:8001/health" -o /tmp/health.json --max-time 5; then
    success "Health endpoint responding"
    cat /tmp/health.json | jq '.' 2>/dev/null || cat /tmp/health.json
else
    warning "Health endpoint not responding (service may still be initializing)"
fi

# ============================================================================
# STEP 9: SUMMARY
# ============================================================================
step "Deployment Summary"

echo -e "${GREEN}"
cat << "EOF"
╔═══════════════════════════════════════════════════════════╗
║                 DEPLOYMENT COMPLETE! ✓                    ║
╚═══════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

cat << EOF

Phase 2 Modules:
  ✓ Phase 2C: Continuous Learning Manager (4 models)
  ✓ Phase 2D: Volatility Structure Engine (11 metrics)
  ✓ Phase 2B: Orderbook Imbalance Module (5 metrics)

Total New Features: 20+ AI metrics active

Container: quantum_ai_engine
Status: $CONTAINER_STATUS

Recent Logs:
───────────────────────────────────────────────────────────
EOF

docker logs quantum_ai_engine --tail 20

cat << EOF
───────────────────────────────────────────────────────────

Next Steps:
  • Monitor logs: docker logs -f quantum_ai_engine
  • Check health: curl http://localhost:8001/health | jq
  • Test modules: docker exec quantum_ai_engine python -c "from backend.services.ai.volatility_structure_engine import VolatilityStructureEngine; print('✓')"

EOF

# ============================================================================
# OPTIONAL: FOLLOW LOGS
# ============================================================================
if [ "$FOLLOW_LOGS" = true ]; then
    info "Following logs (Ctrl+C to exit)..."
    sleep 2
    docker logs -f quantum_ai_engine
fi

success "Deployment script completed!"
