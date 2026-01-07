#!/bin/bash
# Setup Python virtual environments for ALL 32 services (one venv per service)
# Run as root

set -euo pipefail

echo "🐍 Setting up Python Virtual Environments"
echo "============================================"
echo "Creating 31 dedicated venvs (Redis doesn't need Python)"

BASE_VENV_DIR="/opt/quantum/venvs"
PYTHON_BIN="/usr/bin/python3.11"

# Ensure Python 3.11 is installed
if ! command -v $PYTHON_BIN &> /dev/null; then
    echo "❌ Python 3.11 not found. Installing..."
    apt-get update
    apt-get install -y python3.11 python3.11-venv python3.11-dev
fi

mkdir -p "$BASE_VENV_DIR"

# ==========================================
# STAGE 1: Model Servers (3 venvs - HEAVY)
# ==========================================

echo "🧠 [1/31] Creating venv: ai-engine (Model Server - FULL ML STACK)"
$PYTHON_BIN -m venv "$BASE_VENV_DIR/ai-engine"
"$BASE_VENV_DIR/ai-engine/bin/pip" install --upgrade pip
"$BASE_VENV_DIR/ai-engine/bin/pip" install --no-cache-dir \
    fastapi==0.109.0 uvicorn[standard]==0.27.0 redis==5.0.1 structlog==24.1.0 \
    xgboost==2.0.3 lightgbm==4.3.0 torch==2.2.0 numpy==1.26.3 pandas==2.2.0 \
    scikit-learn==1.4.0 aiohttp==3.9.1 httpx==0.26.0 python-dotenv==1.0.1 \
    pydantic-settings==2.1.0 prometheus-client==0.19.0 psutil==5.9.8

echo "🎯 [2/31] Creating venv: rl-sizer (Model Server - TORCH ONLY)"
$PYTHON_BIN -m venv "$BASE_VENV_DIR/rl-sizer"
"$BASE_VENV_DIR/rl-sizer/bin/pip" install --upgrade pip
"$BASE_VENV_DIR/rl-sizer/bin/pip" install --no-cache-dir \
    redis==5.0.1 torch==2.2.0 numpy==1.26.3 structlog==24.1.0 python-dotenv==1.0.1

echo "📊 [3/31] Creating venv: strategy-ops (Model Server - TORCH ONLY)"
$PYTHON_BIN -m venv "$BASE_VENV_DIR/strategy-ops"
"$BASE_VENV_DIR/strategy-ops/bin/pip" install --upgrade pip
"$BASE_VENV_DIR/strategy-ops/bin/pip" install --no-cache-dir \
    redis==5.0.1 torch==2.2.0 numpy==1.26.3 structlog==24.1.0 python-dotenv==1.0.1

# ==========================================
# STAGE 2: AI Clients (24 venvs - LIGHT)
# ==========================================

AI_CLIENT_DEPS="redis==5.0.1 structlog==24.1.0 httpx==0.26.0 aiohttp==3.9.1 python-dotenv==1.0.1 pydantic-settings==2.1.0"

echo "🤖 [4/31] Creating venv: cross-exchange"
$PYTHON_BIN -m venv "$BASE_VENV_DIR/cross-exchange"
"$BASE_VENV_DIR/cross-exchange/bin/pip" install --upgrade pip
"$BASE_VENV_DIR/cross-exchange/bin/pip" install --no-cache-dir $AI_CLIENT_DEPS

echo "🤖 [5/31] Creating venv: market-publisher"
$PYTHON_BIN -m venv "$BASE_VENV_DIR/market-publisher"
"$BASE_VENV_DIR/market-publisher/bin/pip" install --upgrade pip
"$BASE_VENV_DIR/market-publisher/bin/pip" install --no-cache-dir $AI_CLIENT_DEPS

echo "🤖 [6/31] Creating venv: exposure-balancer"
$PYTHON_BIN -m venv "$BASE_VENV_DIR/exposure-balancer"
"$BASE_VENV_DIR/exposure-balancer/bin/pip" install --upgrade pip
"$BASE_VENV_DIR/exposure-balancer/bin/pip" install --no-cache-dir $AI_CLIENT_DEPS

echo "🤖 [7/31] Creating venv: portfolio-governance"
$PYTHON_BIN -m venv "$BASE_VENV_DIR/portfolio-governance"
"$BASE_VENV_DIR/portfolio-governance/bin/pip" install --upgrade pip
"$BASE_VENV_DIR/portfolio-governance/bin/pip" install --no-cache-dir $AI_CLIENT_DEPS

echo "🤖 [8/31] Creating venv: meta-regime"
$PYTHON_BIN -m venv "$BASE_VENV_DIR/meta-regime"
"$BASE_VENV_DIR/meta-regime/bin/pip" install --upgrade pip
"$BASE_VENV_DIR/meta-regime/bin/pip" install --no-cache-dir $AI_CLIENT_DEPS

echo "🤖 [9/31] Creating venv: portfolio-intelligence"
$PYTHON_BIN -m venv "$BASE_VENV_DIR/portfolio-intelligence"
"$BASE_VENV_DIR/portfolio-intelligence/bin/pip" install --upgrade pip
"$BASE_VENV_DIR/portfolio-intelligence/bin/pip" install --no-cache-dir $AI_CLIENT_DEPS fastapi==0.109.0 uvicorn[standard]==0.27.0

echo "🤖 [10/31] Creating venv: strategic-memory"
$PYTHON_BIN -m venv "$BASE_VENV_DIR/strategic-memory"
"$BASE_VENV_DIR/strategic-memory/bin/pip" install --upgrade pip
"$BASE_VENV_DIR/strategic-memory/bin/pip" install --no-cache-dir $AI_CLIENT_DEPS

echo "🤖 [11/31] Creating venv: strategic-evolution"
$PYTHON_BIN -m venv "$BASE_VENV_DIR/strategic-evolution"
"$BASE_VENV_DIR/strategic-evolution/bin/pip" install --upgrade pip
"$BASE_VENV_DIR/strategic-evolution/bin/pip" install --no-cache-dir $AI_CLIENT_DEPS

echo "🤖 [12/31] Creating venv: position-monitor"
$PYTHON_BIN -m venv "$BASE_VENV_DIR/position-monitor"
"$BASE_VENV_DIR/position-monitor/bin/pip" install --upgrade pip
"$BASE_VENV_DIR/position-monitor/bin/pip" install --no-cache-dir $AI_CLIENT_DEPS

echo "🤖 [13/31] Creating venv: trade-intent-consumer"
$PYTHON_BIN -m venv "$BASE_VENV_DIR/trade-intent-consumer"
"$BASE_VENV_DIR/trade-intent-consumer/bin/pip" install --upgrade pip
"$BASE_VENV_DIR/trade-intent-consumer/bin/pip" install --no-cache-dir $AI_CLIENT_DEPS

echo "🤖 [14/31] Creating venv: ceo-brain"
$PYTHON_BIN -m venv "$BASE_VENV_DIR/ceo-brain"
"$BASE_VENV_DIR/ceo-brain/bin/pip" install --upgrade pip
"$BASE_VENV_DIR/ceo-brain/bin/pip" install --no-cache-dir $AI_CLIENT_DEPS

echo "🤖 [15/31] Creating venv: strategy-brain"
$PYTHON_BIN -m venv "$BASE_VENV_DIR/strategy-brain"
"$BASE_VENV_DIR/strategy-brain/bin/pip" install --upgrade pip
"$BASE_VENV_DIR/strategy-brain/bin/pip" install --no-cache-dir $AI_CLIENT_DEPS

echo "🤖 [16/31] Creating venv: risk-brain"
$PYTHON_BIN -m venv "$BASE_VENV_DIR/risk-brain"
"$BASE_VENV_DIR/risk-brain/bin/pip" install --upgrade pip
"$BASE_VENV_DIR/risk-brain/bin/pip" install --no-cache-dir $AI_CLIENT_DEPS

echo "🤖 [17/31] Creating venv: model-federation"
$PYTHON_BIN -m venv "$BASE_VENV_DIR/model-federation"
"$BASE_VENV_DIR/model-federation/bin/pip" install --upgrade pip
"$BASE_VENV_DIR/model-federation/bin/pip" install --no-cache-dir $AI_CLIENT_DEPS

echo "🤖 [18/31] Creating venv: retraining-worker"
$PYTHON_BIN -m venv "$BASE_VENV_DIR/retraining-worker"
"$BASE_VENV_DIR/retraining-worker/bin/pip" install --upgrade pip
"$BASE_VENV_DIR/retraining-worker/bin/pip" install --no-cache-dir $AI_CLIENT_DEPS numpy==1.26.3

echo "🤖 [19/31] Creating venv: universe-os"
$PYTHON_BIN -m venv "$BASE_VENV_DIR/universe-os"
"$BASE_VENV_DIR/universe-os/bin/pip" install --upgrade pip
"$BASE_VENV_DIR/universe-os/bin/pip" install --no-cache-dir $AI_CLIENT_DEPS fastapi==0.109.0 uvicorn[standard]==0.27.0

echo "🤖 [20/31] Creating venv: pil"
$PYTHON_BIN -m venv "$BASE_VENV_DIR/pil"
"$BASE_VENV_DIR/pil/bin/pip" install --upgrade pip
"$BASE_VENV_DIR/pil/bin/pip" install --no-cache-dir $AI_CLIENT_DEPS

echo "🤖 [21/31] Creating venv: model-supervisor"
$PYTHON_BIN -m venv "$BASE_VENV_DIR/model-supervisor"
"$BASE_VENV_DIR/model-supervisor/bin/pip" install --upgrade pip
"$BASE_VENV_DIR/model-supervisor/bin/pip" install --no-cache-dir $AI_CLIENT_DEPS

echo "🤖 [22/31] Creating venv: rl-feedback-v2"
$PYTHON_BIN -m venv "$BASE_VENV_DIR/rl-feedback-v2"
"$BASE_VENV_DIR/rl-feedback-v2/bin/pip" install --upgrade pip
"$BASE_VENV_DIR/rl-feedback-v2/bin/pip" install --no-cache-dir $AI_CLIENT_DEPS

echo "🤖 [23/31] Creating venv: rl-monitor"
$PYTHON_BIN -m venv "$BASE_VENV_DIR/rl-monitor"
"$BASE_VENV_DIR/rl-monitor/bin/pip" install --upgrade pip
"$BASE_VENV_DIR/rl-monitor/bin/pip" install --no-cache-dir $AI_CLIENT_DEPS

echo "🤖 [24/31] Creating venv: binance-pnl-tracker"
$PYTHON_BIN -m venv "$BASE_VENV_DIR/binance-pnl-tracker"
"$BASE_VENV_DIR/binance-pnl-tracker/bin/pip" install --upgrade pip
"$BASE_VENV_DIR/binance-pnl-tracker/bin/pip" install --no-cache-dir $AI_CLIENT_DEPS python-binance==1.0.19

echo "🤖 [25/31] Creating venv: risk-safety"
$PYTHON_BIN -m venv "$BASE_VENV_DIR/risk-safety"
"$BASE_VENV_DIR/risk-safety/bin/pip" install --upgrade pip
"$BASE_VENV_DIR/risk-safety/bin/pip" install --no-cache-dir $AI_CLIENT_DEPS

echo "🤖 [26/31] Creating venv: execution"
$PYTHON_BIN -m venv "$BASE_VENV_DIR/execution"
"$BASE_VENV_DIR/execution/bin/pip" install --upgrade pip
"$BASE_VENV_DIR/execution/bin/pip" install --no-cache-dir $AI_CLIENT_DEPS python-binance==1.0.19

echo "🤖 [27/31] Creating venv: clm"
$PYTHON_BIN -m venv "$BASE_VENV_DIR/clm"
"$BASE_VENV_DIR/clm/bin/pip" install --upgrade pip
"$BASE_VENV_DIR/clm/bin/pip" install --no-cache-dir $AI_CLIENT_DEPS fastapi==0.109.0 uvicorn[standard]==0.27.0

# ==========================================
# STAGE 3: Infrastructure (4 Python venvs)
# ==========================================

echo "🖥️ [28/31] Creating venv: rl-dashboard"
$PYTHON_BIN -m venv "$BASE_VENV_DIR/rl-dashboard"
"$BASE_VENV_DIR/rl-dashboard/bin/pip" install --upgrade pip
"$BASE_VENV_DIR/rl-dashboard/bin/pip" install --no-cache-dir flask==3.0.0 redis==5.0.1 structlog==24.1.0

echo "🖥️ [29/31] Infrastructure venvs for frontend services (if needed)"
# Frontends use Node.js, no venv needed

echo ""
echo "✅ All 31 Python virtual environments created successfully"
echo "📦 Total disk usage estimate: ~3.5GB (vs Docker's ~10GB)"
echo ""
echo "Venv locations:"
echo "  - Model Servers: /opt/quantum/venvs/{ai-engine,rl-sizer,strategy-ops}"
echo "  - AI Clients:    /opt/quantum/venvs/{cross-exchange,market-publisher,...,clm}"
echo "  - Infrastructure: /opt/quantum/venvs/rl-dashboard"
    python-binance==1.0.19 \
    requests==2.31.0

# Execution service (SPECIAL - EXCHANGE LIBS)
echo "⚙️ Creating venv: execution"
$PYTHON_BIN -m venv "$BASE_VENV_DIR/execution"
"$BASE_VENV_DIR/execution/bin/pip" install --upgrade pip
"$BASE_VENV_DIR/execution/bin/pip" install --no-cache-dir \
    fastapi==0.109.0 \
    uvicorn[standard]==0.27.0 \
    redis==5.0.1 \
    structlog==24.1.0 \
    httpx==0.26.0 \
    python-binance==1.0.19 \
    ccxt==4.2.25 \
    python-dotenv==1.0.1 \
    pydantic-settings==2.1.0

echo ""
echo "✅ All Python venvs created successfully"
echo "📊 Disk usage:"
du -sh "$BASE_VENV_DIR"/*
