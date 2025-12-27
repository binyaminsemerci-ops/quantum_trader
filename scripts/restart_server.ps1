#!/usr/bin/env pwsh
# ============================================================================
# QUANTUM TRADER - SERVER RESTART SCRIPT
# ============================================================================
# This script safely restarts the backend server to activate new features:
# - JWT Authentication
# - Redis Caching
# - Security Headers
# ============================================================================

Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║        QUANTUM TRADER v2.0 - SERVER RESTART                ║" -ForegroundColor Cyan
Write-Host "║        Activating Authentication & Caching                  ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# Check if server is running
Write-Host "🔍 Checking for running server..." -ForegroundColor Yellow
$serverPort = 8000
$serverProcess = Get-NetTCPConnection -LocalPort $serverPort -ErrorAction SilentlyContinue 2>$null

if ($serverProcess) {
    $pid = $serverProcess.OwningProcess | Select-Object -First 1
    Write-Host "✅ Server found on port $serverPort (PID: $pid)" -ForegroundColor Green
    
    Write-Host ""
    Write-Host "⚠️  MANUAL RESTART REQUIRED" -ForegroundColor Yellow
    Write-Host "=" * 60
    Write-Host "The server is currently running in another terminal window."
    Write-Host "Please follow these steps:"
    Write-Host ""
    Write-Host "1. Go to the terminal running uvicorn"
    Write-Host "2. Press Ctrl+C to stop the server"
    Write-Host "3. Wait for it to fully shut down"
    Write-Host "4. Restart with: uvicorn backend.main:app --reload"
    Write-Host ""
    Write-Host "OR run this command to force restart:"
    Write-Host "   Stop-Process -Id $pid -Force" -ForegroundColor Cyan
    Write-Host "   Start-Sleep -Seconds 2"
    Write-Host "   uvicorn backend.main:app --reload" -ForegroundColor Cyan
    Write-Host ""
    
    $response = Read-Host "Do you want to force restart now? (y/N)"
    
    if ($response -eq 'y' -or $response -eq 'Y') {
        Write-Host ""
        Write-Host "🛑 Stopping server (PID: $pid)..." -ForegroundColor Red
        Stop-Process -Id $pid -Force
        Write-Host "✅ Server stopped" -ForegroundColor Green
        
        Write-Host ""
        Write-Host "⏳ Waiting 3 seconds for cleanup..." -ForegroundColor Yellow
        Start-Sleep -Seconds 3
        
        Write-Host ""
        Write-Host "🚀 Starting server with new features..." -ForegroundColor Green
        Write-Host ""
        Write-Host "=" * 60
        Write-Host "Starting: uvicorn backend.main:app --reload" -ForegroundColor Cyan
        Write-Host "=" * 60
        Write-Host ""
        
        # Start server in current terminal
        uvicorn backend.main:app --reload
        
    } else {
        Write-Host ""
        Write-Host "❌ Restart cancelled. Please restart manually." -ForegroundColor Yellow
        Write-Host ""
    }
    
} else {
    Write-Host "⚠️  No server found on port $serverPort" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "🚀 Starting server with new features..." -ForegroundColor Green
    Write-Host ""
    Write-Host "=" * 60
    Write-Host "Starting: uvicorn backend.main:app --reload" -ForegroundColor Cyan
    Write-Host "=" * 60
    Write-Host ""
    
    uvicorn backend.main:app --reload
}

Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║                   WHAT TO LOOK FOR                           ║" -ForegroundColor Green
Write-Host "╚══════════════════════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""
Write-Host "After server starts, check logs for:" -ForegroundColor Cyan
Write-Host "  ✅ 'Auth system initialized' or 'Authentication system initialized'" -ForegroundColor White
Write-Host "  ✅ 'Caching layer initialized' or 'Cache initialized'" -ForegroundColor White
Write-Host "  ✅ 'Security middleware added' or similar" -ForegroundColor White
Write-Host ""
Write-Host "Then verify new features:" -ForegroundColor Cyan
Write-Host "  1. Visit: http://localhost:8000/api/docs" -ForegroundColor White
Write-Host "  2. Look for: /api/auth/login, /api/auth/refresh, /api/auth/logout" -ForegroundColor White
Write-Host "  3. Test login with: username=admin, password=admin123" -ForegroundColor White
Write-Host ""
Write-Host "Run validation tests:" -ForegroundColor Cyan
Write-Host "  python scripts/test_integration.py" -ForegroundColor White
Write-Host "  python scripts/test_security.py" -ForegroundColor White
Write-Host "  python scripts/test_performance.py" -ForegroundColor White
Write-Host ""
