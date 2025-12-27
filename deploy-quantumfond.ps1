# QuantumFond Deployment PowerShell Script
# Usage: .\deploy-quantumfond.ps1 [-Environment production]

param(
    [string]$Environment = "production",
    [switch]$Docker = $false
)

Write-Host "🚀 QuantumFond Deployment Script" -ForegroundColor Green
Write-Host "Environment: $Environment" -ForegroundColor Yellow
Write-Host "─────────────────────────────────" -ForegroundColor Gray

# Check if Docker is requested
if ($Docker) {
    Write-Host "`n🐳 Docker Deployment Mode" -ForegroundColor Cyan
    
    # Check if docker-compose exists
    if (-not (Get-Command docker-compose -ErrorAction SilentlyContinue)) {
        Write-Host "✗ Docker Compose not found. Please install Docker Desktop." -ForegroundColor Red
        exit 1
    }
    
    # Check if .env file exists
    if (-not (Test-Path ".env.quantumfond")) {
        Write-Host "⚠️  Creating .env file from template..." -ForegroundColor Yellow
        Copy-Item ".env.quantumfond" ".env"
        Write-Host "⚠️  Please configure .env with production settings!" -ForegroundColor Red
        Write-Host "Press any key to continue after configuring .env..."
        $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    }
    
    # Build and start containers
    Write-Host "`n→ Building Docker containers..." -ForegroundColor Cyan
    docker-compose -f docker-compose.quantumfond.yml build
    
    Write-Host "`n→ Starting containers..." -ForegroundColor Cyan
    docker-compose -f docker-compose.quantumfond.yml up -d
    
    # Wait for services to start
    Write-Host "`n→ Waiting for services to start..." -ForegroundColor Cyan
    Start-Sleep -Seconds 10
    
    # Check health
    Write-Host "`n✅ Checking service health..." -ForegroundColor Green
    docker-compose -f docker-compose.quantumfond.yml ps
    
    # Test endpoints
    Write-Host "`n→ Testing backend health..." -ForegroundColor Cyan
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing
        if ($response.StatusCode -eq 200) {
            Write-Host "✓ Backend is healthy" -ForegroundColor Green
        }
    } catch {
        Write-Host "✗ Backend health check failed" -ForegroundColor Red
    }
    
    Write-Host "`n→ Testing frontend health..." -ForegroundColor Cyan
    try {
        $response = Invoke-WebRequest -Uri "http://localhost/health" -UseBasicParsing
        if ($response.StatusCode -eq 200) {
            Write-Host "✓ Frontend is healthy" -ForegroundColor Green
        }
    } catch {
        Write-Host "✗ Frontend health check failed" -ForegroundColor Red
    }
    
    Write-Host "`n🎉 Docker Deployment Complete!" -ForegroundColor Green
    Write-Host "─────────────────────────────────" -ForegroundColor Gray
    Write-Host "Backend API: http://localhost:8000" -ForegroundColor Cyan
    Write-Host "Frontend App: http://localhost" -ForegroundColor Cyan
    Write-Host "API Docs: http://localhost:8000/docs" -ForegroundColor Cyan
    Write-Host "`nTo view logs: docker-compose -f docker-compose.quantumfond.yml logs -f" -ForegroundColor Yellow
    Write-Host "To stop: docker-compose -f docker-compose.quantumfond.yml down" -ForegroundColor Yellow
    
} else {
    Write-Host "`n⚠️  Manual Deployment Mode" -ForegroundColor Yellow
    Write-Host "This script provides deployment steps for Windows." -ForegroundColor Gray
    Write-Host "For production Linux deployment, use: ./deploy-quantumfond.sh" -ForegroundColor Gray
    
    Write-Host "`n📝 Deployment Steps:" -ForegroundColor Cyan
    Write-Host "1. Build Frontend:" -ForegroundColor White
    Write-Host "   cd quantumfond_frontend" -ForegroundColor Gray
    Write-Host "   npm install" -ForegroundColor Gray
    Write-Host "   npm run build" -ForegroundColor Gray
    
    Write-Host "`n2. Setup Backend:" -ForegroundColor White
    Write-Host "   cd quantumfond_backend" -ForegroundColor Gray
    Write-Host "   python -m venv venv" -ForegroundColor Gray
    Write-Host "   .\venv\Scripts\Activate.ps1" -ForegroundColor Gray
    Write-Host "   pip install -r requirements.txt" -ForegroundColor Gray
    
    Write-Host "`n3. Configure Environment:" -ForegroundColor White
    Write-Host "   Copy .env.quantumfond to .env" -ForegroundColor Gray
    Write-Host "   Update database credentials and JWT secret" -ForegroundColor Gray
    
    Write-Host "`n4. Start Services:" -ForegroundColor White
    Write-Host "   Backend: uvicorn main:app --host 0.0.0.0 --port 8000" -ForegroundColor Gray
    Write-Host "   Frontend: Use IIS or any static file server" -ForegroundColor Gray
    
    Write-Host "`n💡 Recommended: Use Docker deployment with -Docker flag" -ForegroundColor Yellow
}
