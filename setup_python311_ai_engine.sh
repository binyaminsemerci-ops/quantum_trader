#!/bin/bash
# ============================================================
# AI Engine Python 3.11 Setup Script
# Automatisk installasjon og konfigurasjon
# ============================================================

set -e  # Exit on error

echo "🚀 Starting Python 3.11 setup for ai_engine..."

# ============================================================
# 1. INSTALLER PYTHON 3.11 + BUILD TOOLS
# ============================================================
echo ""
echo "📦 Installing Python 3.11 and build tools..."

# Check if Python 3.11 is already available
if command -v python3.11 &> /dev/null; then
    echo "✓ Python 3.11 already installed"
    python3.11 --version
else
    echo "Installing Python 3.11 from deadsnakes PPA..."
    sudo apt update
    sudo apt install -y software-properties-common
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt update
    sudo apt install -y python3.11 python3.11-venv python3.11-dev
fi

# Install build tools
echo "Installing build tools..."
sudo apt install -y \
    build-essential gcc g++ make \
    libffi-dev libssl-dev \
    git curl

# Verify Python 3.11
echo ""
echo "✓ Python 3.11 version:"
python3.11 --version

# ============================================================
# 2. NAVIGER TIL PROSJEKTET
# ============================================================
cd ~/quantum_trader || {
    echo "❌ ERROR: ~/quantum_trader directory not found!"
    echo "Please ensure project is in ~/quantum_trader"
    exit 1
}

echo ""
echo "📁 Working directory: $(pwd)"

# ============================================================
# 3. DEAKTIVER OG SLETT GAMMEL VENV
# ============================================================
echo ""
echo "🗑️  Removing old .venv (Python 3.12)..."
deactivate 2>/dev/null || true
rm -rf .venv

# ============================================================
# 4. OPPRETT NY VENV MED PYTHON 3.11
# ============================================================
echo ""
echo "🔨 Creating new .venv with Python 3.11..."
python3.11 -m venv .venv

# Aktiver venv
source .venv/bin/activate

# Verify Python version
echo ""
echo "✓ Active Python version:"
python --version

# ============================================================
# 5. OPPGRADER PIP/SETUPTOOLS/WHEEL
# ============================================================
echo ""
echo "⬆️  Upgrading pip, setuptools, wheel..."
pip install --upgrade pip setuptools wheel

# ============================================================
# 6. INSTALLER AI_ENGINE REQUIREMENTS
# ============================================================
echo ""
echo "📚 Installing ai_engine dependencies..."

# A. Installer numpy FØRST (som wheel, ikke fra source)
echo ""
echo "  → Installing numpy==1.24.3 (wheel only)..."
pip install "numpy==1.24.3" --only-binary :all:

# Verify numpy
python -c "import numpy; print(f'  ✓ NumPy {numpy.__version__} installed successfully')"

# B. Installer PyTorch med CUDA 11.8 support (for RTX 3060)
echo ""
echo "  → Installing PyTorch 2.1.0 with CUDA 11.8 support..."
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# C. Installer resten av requirements
echo ""
echo "  → Installing remaining requirements..."
pip install -r microservices/ai_engine/requirements.txt

# ============================================================
# 7. VERIFISER TORCH CUDA SUPPORT
# ============================================================
echo ""
echo "🎮 Verifying PyTorch CUDA support..."
python -c "
import torch
print(f'  PyTorch version: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA version: {torch.version.cuda}')
    print(f'  GPU device: {torch.cuda.get_device_name(0)}')
    print(f'  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
else:
    print('  ⚠️  WARNING: CUDA not available - running on CPU only')
    print('  Check nvidia-smi and CUDA drivers')
"

# ============================================================
# 8. VERIFISER ALLE KRITISKE IMPORTS
# ============================================================
echo ""
echo "✅ Verifying all critical dependencies..."
python -c "
import numpy
import sklearn
import xgboost
import lightgbm
import torch
import pytorch_lightning
import fastapi
import uvicorn
import redis
print('  ✓ All critical dependencies imported successfully')
print(f'  ✓ numpy: {numpy.__version__}')
print(f'  ✓ scikit-learn: {sklearn.__version__}')
print(f'  ✓ torch: {torch.__version__}')
"

# ============================================================
# 9. OPPRETT LOGS DIRECTORY
# ============================================================
echo ""
echo "📁 Creating logs directory..."
mkdir -p logs

# ============================================================
# 10. GENERER STARTUP SCRIPT
# ============================================================
echo ""
echo "📝 Generating startup scripts..."

cat > start_ai_engine.sh << 'EOF'
#!/bin/bash
# AI Engine Startup Script

cd ~/quantum_trader
source .venv/bin/activate

# Export environment variables
export PYTHONPATH=$PWD
export REDIS_HOST=localhost
export REDIS_PORT=6379
export LOG_LEVEL=INFO

# Memory optimization for tight RAM
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

echo "🚀 Starting AI Engine on port 8001..."
echo "📊 Logs: logs/ai_engine.log"
echo "🛑 Stop with: kill \$(cat logs/ai_engine.pid)"
echo ""

# Start uvicorn in background
nohup uvicorn microservices.ai_engine.main:app \
    --host 0.0.0.0 \
    --port 8001 \
    --log-level info \
    > logs/ai_engine.log 2>&1 &

# Save PID
echo $! > logs/ai_engine.pid

sleep 2

# Check if running
if ps -p $(cat logs/ai_engine.pid) > /dev/null; then
    echo "✅ AI Engine started successfully (PID: $(cat logs/ai_engine.pid))"
    echo ""
    echo "Health check: curl http://localhost:8001/health"
    echo "Metrics: curl http://localhost:8001/metrics"
    echo "API Docs: http://localhost:8001/docs"
else
    echo "❌ Failed to start AI Engine - check logs/ai_engine.log"
    exit 1
fi
EOF

chmod +x start_ai_engine.sh

cat > stop_ai_engine.sh << 'EOF'
#!/bin/bash
# Stop AI Engine

if [ -f logs/ai_engine.pid ]; then
    PID=$(cat logs/ai_engine.pid)
    if ps -p $PID > /dev/null; then
        echo "🛑 Stopping AI Engine (PID: $PID)..."
        kill $PID
        sleep 2
        if ps -p $PID > /dev/null; then
            echo "Force killing..."
            kill -9 $PID
        fi
        rm logs/ai_engine.pid
        echo "✅ AI Engine stopped"
    else
        echo "AI Engine not running"
        rm logs/ai_engine.pid
    fi
else
    echo "No PID file found"
fi
EOF

chmod +x stop_ai_engine.sh

# ============================================================
# COMPLETION
# ============================================================
echo ""
echo "========================================="
echo "✅ INSTALLATION COMPLETE!"
echo "========================================="
echo ""
echo "Python 3.11 venv created at: .venv"
echo "All dependencies installed successfully"
echo ""
echo "📋 NEXT STEPS:"
echo ""
echo "1. Start Redis (required):"
echo "   podman-compose up -d redis"
echo ""
echo "2. Start AI Engine:"
echo "   ./start_ai_engine.sh"
echo ""
echo "3. Verify it's running:"
echo "   curl http://localhost:8001/health"
echo ""
echo "4. View logs:"
echo "   tail -f logs/ai_engine.log"
echo ""
echo "5. Stop AI Engine:"
echo "   ./stop_ai_engine.sh"
echo ""
echo "========================================="
