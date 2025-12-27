#!/bin/bash
# QuantumFond Backend Startup Script

echo "🚀 Starting QuantumFond Backend..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "⚠️  No .env file found. Copying from .env.example..."
    cp .env.example .env
    echo "⚠️  Please configure .env with your settings!"
fi

# Run backend
echo "✅ Starting FastAPI server..."
uvicorn main:app --reload --port 8000 --host 0.0.0.0
