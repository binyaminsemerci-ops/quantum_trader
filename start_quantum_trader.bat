@echo off
REM Quantum Trader - System Startup Script
REM ========================================
REM Starts all components: Docker containers, backend, and monitoring

echo ================================================================================
echo   QUANTUM TRADER - SYSTEM STARTUP
echo ================================================================================
echo.

REM Check if Docker is running
echo [1/6] Checking Docker...
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker is not running!
    echo Please start Docker Desktop first.
    pause
    exit /b 1
)
echo OK: Docker is running

echo.
echo [2/6] Starting Docker containers...
docker-compose --profile dev up -d
if %errorlevel% neq 0 (
    echo ERROR: Failed to start containers
    echo Try: docker-compose down -v
    echo Then: docker-compose --profile dev up -d
    pause
    exit /b 1
)
echo OK: Containers started

echo.
echo [3/6] Waiting for backend to be ready...
timeout /t 5 /nobreak >nul

REM Wait for backend health check
set /a count=0
:check_health
set /a count+=1
if %count% gtr 30 (
    echo TIMEOUT: Backend did not start in time
    echo Check logs with: docker logs quantum_backend
    pause
    exit /b 1
)

curl -s http://localhost:8000/health >nul 2>&1
if %errorlevel% neq 0 (
    echo Waiting for backend... (%count%/30^)
    timeout /t 2 /nobreak >nul
    goto check_health
)
echo OK: Backend is healthy

echo.
echo [4/6] Waiting for frontend to be ready...
timeout /t 5 /nobreak >nul

REM Wait for frontend
set /a fcount=0
:check_frontend
set /a fcount+=1
if %fcount% gtr 20 (
    echo WARNING: Frontend not responding
    goto skip_frontend
)

curl -s http://localhost:3000 >nul 2>&1
if %errorlevel% neq 0 (
    echo Waiting for frontend... (%fcount%/20^)
    timeout /t 2 /nobreak >nul
    goto check_frontend
)
echo OK: Frontend is ready

:skip_frontend
echo.
echo [5/6] Verifying Trading Profile...
curl -s http://localhost:8000/trading-profile/config | findstr "enabled" >nul
if %errorlevel% equ 0 (
    echo OK: Trading Profile is active
) else (
    echo WARNING: Could not verify Trading Profile
)

echo.
echo [6/6] Opening monitoring dashboard...
start python quick_status.py

echo.
echo ================================================================================
echo   QUANTUM TRADER IS NOW RUNNING!
echo ================================================================================
echo.
echo Backend:  http://localhost:8000
echo Frontend: http://localhost:3000
echo.
echo Monitoring commands:
echo   - Quick Status:   python quick_status.py
echo   - Live Monitor:   python monitor_production.py
echo   - Backend Logs:   docker logs -f quantum_backend
echo.
echo Press any key to close this window...
pause >nul
