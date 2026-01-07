#!/bin/bash
# MASTER MIGRATION SCRIPT - COMPLETE 32-SERVICE SYSTEMD MIGRATION
# Migrates Quantum Trader from Docker to systemd with full preflight checks
# Run as root on target Linux system

set -euo pipefail

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║  QUANTUM TRADER - DOCKER → SYSTEMD MIGRATION (32 SERVICES)   ║"
echo "║  Complete autonomous deployment with validation               ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Check root
if [ "$EUID" -ne 0 ]; then 
    echo "❌ This script must be run as root"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "📂 Project root: $PROJECT_ROOT"
echo ""

# ============================================================================
# PREFLIGHT CHECKS
# ============================================================================
echo "✈️  [PREFLIGHT] VALIDATION CHECKS"
echo "====================================="
echo ""

PREFLIGHT_FAILED=0

echo "🔍 [1/5] Checking systemd unit files..."
EXPECTED_UNITS=(
    "quantum-redis" "quantum-cross-exchange" "quantum-ai-engine" "quantum-market-publisher"
    "quantum-exposure-balancer" "quantum-portfolio-governance" "quantum-meta-regime"
    "quantum-portfolio-intelligence" "quantum-strategic-memory" "quantum-strategic-evolution"
    "quantum-position-monitor" "quantum-trade-intent-consumer" "quantum-ceo-brain"
    "quantum-strategy-brain" "quantum-risk-brain" "quantum-model-federation" "quantum-frontend"
    "quantum-retraining-worker" "quantum-universe-os" "quantum-pil" "quantum-model-supervisor"
    "quantum-rl-sizer" "quantum-strategy-ops" "quantum-rl-feedback-v2" "quantum-rl-monitor"
    "quantum-binance-pnl-tracker" "quantum-nginx-proxy" "quantum-rl-dashboard"
    "quantum-quantumfond-frontend" "quantum-risk-safety" "quantum-execution" "quantum-clm"
)

MISSING_UNITS=()
for unit in "${EXPECTED_UNITS[@]}"; do
    if [ ! -f "$SCRIPT_DIR/../units/${unit}.service" ]; then
        MISSING_UNITS+=("$unit")
        PREFLIGHT_FAILED=1
    fi
done

if [ ${#MISSING_UNITS[@]} -gt 0 ]; then
    echo "   ❌ Missing unit files: ${MISSING_UNITS[*]}"
else
    echo "   ✅ All 32 unit files present"
fi

echo ""
echo "🔍 [2/5] Checking env template files..."
EXPECTED_ENVS=(
    "ai-engine" "execution" "cross-exchange" "market-publisher" "exposure-balancer"
    "portfolio-governance" "meta-regime" "portfolio-intelligence" "strategic-memory"
    "strategic-evolution" "position-monitor" "trade-intent-consumer" "ceo-brain"
    "strategy-brain" "risk-brain" "model-federation" "retraining-worker" "universe-os"
    "pil" "model-supervisor" "rl-sizer" "strategy-ops" "rl-feedback-v2" "rl-monitor"
    "binance-pnl-tracker" "risk-safety" "clm" "frontend" "quantumfond-frontend" "rl-dashboard"
)

MISSING_ENVS=()
for env in "${EXPECTED_ENVS[@]}"; do
    if [ ! -f "$SCRIPT_DIR/../env-templates/${env}.env" ]; then
        MISSING_ENVS+=("$env")
        PREFLIGHT_FAILED=1
    fi
done

if [ ${#MISSING_ENVS[@]} -gt 0 ]; then
    echo "   ❌ Missing env files: ${MISSING_ENVS[*]}"
else
    echo "   ✅ All 30 env templates present (Redis/nginx use system configs)"
fi

echo ""
echo "🔍 [3/5] Checking required scripts..."
REQUIRED_SCRIPTS=("create_users.sh" "setup_venvs.sh" "deploy_systemd.sh" "start_all.sh" "stop_all.sh" "verify_health.sh")
MISSING_SCRIPTS=()
for script in "${REQUIRED_SCRIPTS[@]}"; do
    if [ ! -f "$SCRIPT_DIR/$script" ]; then
        MISSING_SCRIPTS+=("$script")
        PREFLIGHT_FAILED=1
    fi
done

if [ ${#MISSING_SCRIPTS[@]} -gt 0 ]; then
    echo "   ❌ Missing scripts: ${MISSING_SCRIPTS[*]}"
else
    echo "   ✅ All required scripts present"
fi

echo ""
echo "🔍 [4/5] Checking system requirements..."
PYTHON_OK=1
command -v python3.11 &>/dev/null || PYTHON_OK=0
NODE_OK=1
command -v node &>/dev/null || NODE_OK=0
REDIS_OK=1
command -v redis-server &>/dev/null || REDIS_OK=0
NGINX_OK=1
command -v nginx &>/dev/null || NGINX_OK=0

if [ $PYTHON_OK -eq 0 ]; then
    echo "   ⚠️  Python 3.11 not found (will be installed during setup_venvs.sh)"
fi
if [ $NODE_OK -eq 0 ]; then
    echo "   ❌ Node.js not found - required for frontends"
    PREFLIGHT_FAILED=1
fi
if [ $REDIS_OK -eq 0 ]; then
    echo "   ❌ redis-server not found - required for quantum-redis.service"
    PREFLIGHT_FAILED=1
fi
if [ $NGINX_OK -eq 0 ]; then
    echo "   ❌ nginx not found - required for quantum-nginx-proxy.service"
    PREFLIGHT_FAILED=1
fi

if [ $PYTHON_OK -eq 1 ] && [ $NODE_OK -eq 1 ] && [ $REDIS_OK -eq 1 ] && [ $NGINX_OK -eq 1 ]; then
    echo "   ✅ All system dependencies available"
fi

echo ""
echo "🔍 [5/5] Checking disk space..."
AVAILABLE_GB=$(df -BG / | tail -1 | awk '{print $4}' | sed 's/G//')
if [ "$AVAILABLE_GB" -lt 20 ]; then
    echo "   ❌ Insufficient disk space: ${AVAILABLE_GB}GB available (need 20GB)"
    PREFLIGHT_FAILED=1
else
    echo "   ✅ Disk space OK: ${AVAILABLE_GB}GB available"
fi

echo ""
if [ $PREFLIGHT_FAILED -eq 1 ]; then
    echo "❌ PREFLIGHT FAILED - Fix errors before proceeding"
    exit 1
else
    echo "✅ PREFLIGHT PASSED - All validation checks successful"
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo ""

# Confirmation
read -p "⚠️  This will migrate from Docker to systemd. Continue? (yes/no): " -r
if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
    echo "❌ Migration cancelled"
    exit 1
fi

echo ""
echo "🔄 Starting migration..."
echo ""

# ============================================================================
# PHASE 1: PRE-MIGRATION BACKUP
# ============================================================================
echo "📦 [PHASE 1/7] PRE-MIGRATION BACKUP"
echo "-----------------------------------"

BACKUP_DIR="/backup/quantum_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Backup Docker data
if command -v docker &> /dev/null; then
    echo "   💾 Backing up Docker data..."
    
    # Save Redis data
    if docker ps | grep -q quantum_redis; then
        docker exec quantum_redis redis-cli SAVE || true
        docker cp quantum_redis:/data "$BACKUP_DIR/redis_data" || true
    fi
    
    # Save models
    if docker ps | grep -q quantum_ai_engine; then
        docker cp quantum_ai_engine:/app/models "$BACKUP_DIR/models" || true
    fi
    
    # Export Docker config
    docker-compose -f "$PROJECT_ROOT/docker-compose.vps.yml" config > "$BACKUP_DIR/docker-config.yml" 2>/dev/null || true
    docker ps > "$BACKUP_DIR/docker-ps.txt" || true
    
    echo "   ✅ Backup saved to: $BACKUP_DIR"
else
    echo "   ⚠️  Docker not found - skipping backup"
fi

# ============================================================================
# PHASE 2: DEPLOY SYSTEMD INFRASTRUCTURE
# ============================================================================
echo ""
echo "🏗️  [PHASE 2/7] DEPLOY SYSTEMD INFRASTRUCTURE"
echo "--------------------------------------------"

bash "$SCRIPT_DIR/deploy_systemd.sh"

# ============================================================================
# PHASE 3: MIGRATE DATA
# ============================================================================
echo ""
echo "📊 [PHASE 3/7] MIGRATE DATA FROM DOCKER"
echo "---------------------------------------"

if command -v docker &> /dev/null; then
    echo "   🛑 Stopping Docker containers..."
    docker-compose -f "$PROJECT_ROOT/docker-compose.vps.yml" stop || true
    
    echo "   📦 Copying Redis data..."
    if [ -d "$BACKUP_DIR/redis_data" ]; then
        rsync -av "$BACKUP_DIR/redis_data/" /var/lib/quantum/redis/
        chown -R quantum-redis:quantum-redis /var/lib/quantum/redis
    fi
    
    echo "   🧠 Copying ML models..."
    if [ -d "$BACKUP_DIR/models" ]; then
        rsync -av "$BACKUP_DIR/models/" /data/quantum/models/
    elif [ -d "$PROJECT_ROOT/models" ]; then
        rsync -av "$PROJECT_ROOT/models/" /data/quantum/models/
    fi
    
    echo "   ✅ Data migrated"
else
    echo "   ⚠️  Docker not found - using local data"
    if [ -d "$PROJECT_ROOT/models" ]; then
        rsync -av "$PROJECT_ROOT/models/" /data/quantum/models/
    fi
fi

# ============================================================================
# PHASE 4: CONFIGURE ENVIRONMENT
# ============================================================================
echo ""
echo "⚙️  [PHASE 4/7] CONFIGURE ENVIRONMENT"
echo "------------------------------------"

echo "   📝 Please edit the following files with your API keys:"
echo "      /etc/quantum/ai-engine.env"
echo "      /etc/quantum/execution.env"
echo "      /etc/quantum/binance-pnl-tracker.env"
echo ""

if [ -f "$PROJECT_ROOT/.env" ]; then
    echo "   🔑 Found .env file - extracting credentials..."
    
    # Extract Binance keys if present
    if grep -q "BINANCE_API_KEY" "$PROJECT_ROOT/.env"; then
        BINANCE_KEY=$(grep "BINANCE_API_KEY" "$PROJECT_ROOT/.env" | cut -d'=' -f2)
        BINANCE_SECRET=$(grep "BINANCE_API_SECRET" "$PROJECT_ROOT/.env" | cut -d'=' -f2)
        
        # Update env files
        for env_file in ai-engine execution binance-pnl-tracker; do
            sed -i "s/BINANCE_API_KEY=.*/BINANCE_API_KEY=$BINANCE_KEY/" /etc/quantum/${env_file}.env || true
            sed -i "s/BINANCE_API_SECRET=.*/BINANCE_API_SECRET=$BINANCE_SECRET/" /etc/quantum/${env_file}.env || true
        done
        
        echo "   ✅ Credentials auto-configured"
    fi
fi

read -p "   Press ENTER when configuration is complete..." -r

# ============================================================================
# PHASE 5: 5-STAGE STARTUP SEQUENCE
# ============================================================================
echo ""
echo "🚀 [PHASE 5/7] 5-STAGE STARTUP SEQUENCE"
echo "---------------------------------------"
echo ""

echo "Stage 1: Redis (Infrastructure)"
echo "--------------------------------"
systemctl start quantum-redis.service
sleep 3
systemctl is-active --quiet quantum-redis.service && echo "   ✅ Redis started" || (echo "   ❌ Redis failed" && exit 1)

echo ""
echo "Stage 2: Brain Triumvirate (Decision Makers)"
echo "---------------------------------------------"
systemctl start quantum-ceo-brain.service
systemctl start quantum-strategy-brain.service
systemctl start quantum-risk-brain.service
sleep 5
systemctl is-active --quiet quantum-ceo-brain.service && echo "   ✅ CEO Brain started" || echo "   ⚠️  CEO Brain failed"
systemctl is-active --quiet quantum-strategy-brain.service && echo "   ✅ Strategy Brain started" || echo "   ⚠️  Strategy Brain failed"
systemctl is-active --quiet quantum-risk-brain.service && echo "   ✅ Risk Brain started" || echo "   ⚠️  Risk Brain failed"

echo ""
echo "Stage 3: Model Servers (Heavy ML)"
echo "----------------------------------"
systemctl start quantum-ai-engine.service
systemctl start quantum-rl-sizer.service
systemctl start quantum-strategy-ops.service
sleep 15
systemctl is-active --quiet quantum-ai-engine.service && echo "   ✅ AI Engine started" || echo "   ⚠️  AI Engine failed"
systemctl is-active --quiet quantum-rl-sizer.service && echo "   ✅ RL Sizer started" || echo "   ⚠️  RL Sizer failed"
systemctl is-active --quiet quantum-strategy-ops.service && echo "   ✅ Strategy Ops started" || echo "   ⚠️  Strategy Ops failed"

echo ""
echo "Stage 4: Core AI Clients (Critical Path)"
echo "------------------------------------------"
CORE_SERVICES=(
    "quantum-cross-exchange"
    "quantum-market-publisher"
    "quantum-exposure-balancer"
    "quantum-execution"
    "quantum-portfolio-governance"
    "quantum-meta-regime"
    "quantum-portfolio-intelligence"
    "quantum-strategic-memory"
    "quantum-strategic-evolution"
    "quantum-position-monitor"
    "quantum-trade-intent-consumer"
)

for service in "${CORE_SERVICES[@]}"; do
    systemctl start ${service}.service
done
sleep 10
for service in "${CORE_SERVICES[@]}"; do
    systemctl is-active --quiet ${service}.service && echo "   ✅ ${service#quantum-} started" || echo "   ⚠️  ${service#quantum-} failed"
done

echo ""
echo "Stage 5: Remaining Services (Monitoring, Dashboards, Frontends)"
echo "----------------------------------------------------------------"
REMAINING_SERVICES=(
    "quantum-model-federation"
    "quantum-retraining-worker"
    "quantum-universe-os"
    "quantum-pil"
    "quantum-model-supervisor"
    "quantum-rl-feedback-v2"
    "quantum-rl-monitor"
    "quantum-binance-pnl-tracker"
    "quantum-risk-safety"
    "quantum-clm"
    "quantum-frontend"
    "quantum-quantumfond-frontend"
    "quantum-rl-dashboard"
    "quantum-nginx-proxy"
)

for service in "${REMAINING_SERVICES[@]}"; do
    systemctl start ${service}.service
done
sleep 10
for service in "${REMAINING_SERVICES[@]}"; do
    systemctl is-active --quiet ${service}.service && echo "   ✅ ${service#quantum-} started" || echo "   ⚠️  ${service#quantum-} failed"
done

echo ""
echo "✅ All services started (5-stage sequence complete)"

# ============================================================================
# PHASE 6: HEALTH VERIFICATION
# ============================================================================
echo ""
echo "🏥 [PHASE 6/7] HEALTH VERIFICATION"
echo "----------------------------------"

sleep 10

echo "   🔍 Running health checks..."
bash "$SCRIPT_DIR/verify_health.sh"

HEALTH_CHECK_RESULT=$?

# ============================================================================
# PHASE 7: CLEANUP & ROLLBACK OPTIONS
# ============================================================================
echo ""
echo "🧹 [PHASE 7/7] CLEANUP & FINALIZATION"
echo "-------------------------------------"

if [ $HEALTH_CHECK_RESULT -eq 0 ]; then
    echo ""
    echo "   ✅ All services healthy!"
    echo ""
    
    if command -v docker &> /dev/null; then
        read -p "   🗑️  Remove Docker containers? (yes/no): " -r
        if [[ $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
            docker-compose -f "$PROJECT_ROOT/docker-compose.vps.yml" down || true
            echo "   ✅ Docker containers removed"
            echo "   ℹ️  Docker images preserved for emergency rollback"
        fi
    fi
else
    echo ""
    echo "   ⚠️  Some services failed health check - migration INCOMPLETE"
    echo ""
    echo "   🔙 ROLLBACK OPTIONS:"
    echo "      Option 1: Fix failing services and restart:"
    echo "         systemctl restart quantum-<failed-service>"
    echo ""
    echo "      Option 2: Full rollback to Docker (MANUAL):"
    echo "         systemctl stop quantum-trader.target"
    echo "         # Ensure Docker daemon is running"
    echo "         systemctl start docker"
    echo "         # Restore from backup if needed"
    echo "         rsync -av $BACKUP_DIR/redis_data/ /path/to/docker/redis/volume/"
    echo "         rsync -av $BACKUP_DIR/models/ $PROJECT_ROOT/models/"
    echo "         # Start Docker containers"
    echo "         cd $PROJECT_ROOT"
    echo "         docker-compose -f docker-compose.vps.yml up -d"
    echo ""
    echo "   ⚠️  NOTE: This rollback does NOT use docker-compose commands directly"
    echo "   You must manually ensure Docker daemon is running and volumes are restored"
    exit 1
fi

# ============================================================================
# FINAL REPORT
# ============================================================================
echo ""
echo "╔════════════════════════════════════════════════════════╗"
echo "║  MIGRATION COMPLETE                                    ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""
echo "📊 SYSTEM STATUS:"
echo ""
systemctl status quantum-trader.target --no-pager | head -15
echo ""
echo "💾 MEMORY USAGE:"
free -h | grep -E 'Mem:|Swap:'
echo ""
echo "📋 USEFUL COMMANDS:"
echo "   View all services:     systemctl list-units 'quantum-*'"
echo "   Stop all:              systemctl stop quantum-trader.target"
echo "   Start all:             systemctl start quantum-trader.target"
echo "   View logs:             journalctl -u quantum-<service> -f"
echo "   Health check:          $SCRIPT_DIR/verify_health.sh"
echo ""
echo "🎯 NEXT STEPS:"
echo "   1. Monitor logs for 1 hour"
echo "   2. Test trading functionality"
echo "   3. If stable, remove Docker images"
echo ""
echo "🔙 ROLLBACK (if needed):"
echo "   systemctl stop quantum-trader.target"
echo "   docker-compose -f $PROJECT_ROOT/docker-compose.vps.yml up -d"
echo ""
echo "✅ Migration completed successfully!"
