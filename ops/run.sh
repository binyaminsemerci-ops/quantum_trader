#!/bin/bash
# Quantum Ops Wrapper - GOLDEN CONTRACT ENFORCER
#
# Purpose: Ensure ALL ops jobs run with correct environment
# Usage: ops/run.sh <service> <script> [args...]
#
# Example:
#   ops/run.sh ai-engine ops/model_safety/quality_gate.py
#   ops/run.sh ai-engine ops/training/train_patchtst.py --epochs 100
#
# NEVER run Python scripts directly - ALWAYS use this wrapper.

set -euo pipefail  # Exit on error, undefined var, or pipe failure

# ============================================================================
# CONFIGURATION (DO NOT CHANGE)
# ============================================================================

REPO_ROOT="/home/qt/quantum_trader"
ENV_DIR="/etc/quantum"
VENV_ROOT="/opt/quantum/venvs"
DATA_DIR="/opt/quantum/data"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# ARGUMENT PARSING
# ============================================================================

if [ $# -lt 2 ]; then
    echo -e "${RED}❌ ERROR: Insufficient arguments${NC}"
    echo ""
    echo "Usage: $0 <service> <script> [args...]"
    echo ""
    echo "Services:"
    echo "  ai-engine   - AI Engine + training + quality gates"
    echo "  backend     - FastAPI backend"
    echo "  execution   - Position execution"
    echo "  rl-agent    - RL sizing + monitor"
    echo ""
    echo "Example:"
    echo "  $0 ai-engine ops/model_safety/quality_gate.py"
    echo "  $0 ai-engine ops/training/train_patchtst.py --epochs 100"
    exit 1
fi

SERVICE_NAME="$1"
SCRIPT_PATH="$2"
shift 2
SCRIPT_ARGS="$@"

# ============================================================================
# PATH VALIDATION
# ============================================================================

echo -e "${BLUE}🔍 Validating Golden Contract...${NC}"
echo ""

# Check we're in repo root
if [ ! -d "$REPO_ROOT" ]; then
    echo -e "${RED}❌ FAIL: Repo root not found: $REPO_ROOT${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Repo root: $REPO_ROOT${NC}"

# Check venv exists
VENV_PATH="$VENV_ROOT/$SERVICE_NAME"
if [ ! -d "$VENV_PATH" ]; then
    echo -e "${RED}❌ FAIL: Venv not found: $VENV_PATH${NC}"
    echo -e "${YELLOW}Available venvs:${NC}"
    ls -1 "$VENV_ROOT" 2>/dev/null || echo "  (none)"
    exit 1
fi
echo -e "${GREEN}✅ Venv: $VENV_PATH${NC}"

# Check Python executable
PYTHON_BIN="$VENV_PATH/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
    echo -e "${RED}❌ FAIL: Python not found: $PYTHON_BIN${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Python: $PYTHON_BIN${NC}"

# Check env file exists
ENV_FILE="$ENV_DIR/$SERVICE_NAME.env"
if [ ! -f "$ENV_FILE" ]; then
    echo -e "${RED}❌ FAIL: Environment file not found: $ENV_FILE${NC}"
    echo -e "${YELLOW}Available env files:${NC}"
    ls -1 "$ENV_DIR"/*.env 2>/dev/null || echo "  (none)"
    exit 1
fi
echo -e "${GREEN}✅ Env file: $ENV_FILE${NC}"

# Check script exists
FULL_SCRIPT_PATH="$REPO_ROOT/$SCRIPT_PATH"
if [ ! -f "$FULL_SCRIPT_PATH" ]; then
    echo -e "${RED}❌ FAIL: Script not found: $FULL_SCRIPT_PATH${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Script: $FULL_SCRIPT_PATH${NC}"

# ============================================================================
# DEPENDENCY VALIDATION (FAIL-CLOSED)
# ============================================================================

echo ""
echo -e "${BLUE}🔍 Validating dependencies...${NC}"
echo ""

# Check Redis
if ! redis-cli PING >/dev/null 2>&1; then
    echo -e "${RED}❌ FAIL: Redis not responding${NC}"
    echo -e "${YELLOW}Fix: sudo systemctl start redis-server${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Redis: localhost:6379${NC}"

# Check Database
DB_PATH="$DATA_DIR/quantum_trader.db"
if [ ! -f "$DB_PATH" ]; then
    echo -e "${YELLOW}⚠️  WARNING: Database not found: $DB_PATH${NC}"
    echo -e "${YELLOW}   (may be created on first run)${NC}"
else
    echo -e "${GREEN}✅ Database: $DB_PATH${NC}"
fi

# Check data directory
if [ ! -d "$DATA_DIR" ]; then
    echo -e "${RED}❌ FAIL: Data directory not found: $DATA_DIR${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Data dir: $DATA_DIR${NC}"

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

echo ""
echo -e "${BLUE}🔧 Setting up environment...${NC}"
echo ""

# Source env file
# shellcheck disable=SC1090
source "$ENV_FILE"
echo -e "${GREEN}✅ Loaded: $ENV_FILE${NC}"

# Export critical paths
export REPO_ROOT="$REPO_ROOT"
export DATA_DIR="$DATA_DIR"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
echo -e "${GREEN}✅ PYTHONPATH: $PYTHONPATH${NC}"

# Update PATH to include venv bin
export PATH="$VENV_PATH/bin:$PATH"
echo -e "${GREEN}✅ PATH: $PATH${NC}"

# Verify Python version
PYTHON_VERSION=$("$PYTHON_BIN" --version 2>&1)
echo -e "${GREEN}✅ Python version: $PYTHON_VERSION${NC}"

# ============================================================================
# EXECUTION
# ============================================================================

echo ""
echo -e "${BLUE}🚀 Executing script...${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Change to repo root
cd "$REPO_ROOT"

# Execute with full args
exec "$PYTHON_BIN" "$FULL_SCRIPT_PATH" $SCRIPT_ARGS

# ============================================================================
# NOTES
# ============================================================================
#
# This script is the ONLY way to run ops jobs safely.
#
# What it does:
# 1. Validates all paths exist (repo, venv, env file, script)
# 2. Checks dependencies (Redis, database)
# 3. Sources environment file from /etc/quantum
# 4. Sets PYTHONPATH and PATH correctly
# 5. Executes script with correct Python interpreter
#
# What it prevents:
# - Running with wrong Python (system python, wrong venv)
# - Missing environment variables
# - Missing dependencies (Redis down, db missing)
# - Wrong working directory
# - Implicit defaults
#
# FAIL-CLOSED: If ANY check fails, script exits immediately.
#
# ============================================================================
