# QuantumFond Backend Startup Script (PowerShell)

Write-Host "🚀 Starting QuantumFond Backend..." -ForegroundColor Green

# Check if virtual environment exists
if (-not (Test-Path "venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
}

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install dependencies
Write-Host "📦 Installing dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

# Check if .env exists
if (-not (Test-Path ".env")) {
    Write-Host "⚠️  No .env file found. Copying from .env.example..." -ForegroundColor Yellow
    Copy-Item .env.example .env
    Write-Host "⚠️  Please configure .env with your settings!" -ForegroundColor Red
}

# Run backend
Write-Host "✅ Starting FastAPI server..." -ForegroundColor Green
uvicorn main:app --reload --port 8000 --host 0.0.0.0
