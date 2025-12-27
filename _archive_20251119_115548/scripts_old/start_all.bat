@echo off
echo Starting autonomous trading system...

REM Start backend in background
start "Backend" cmd /c "restart_backend.bat"

REM Wait a bit for backend to initialize
timeout /t 5 /nobreak > nul

REM Start training in background
start "Training" cmd /c "restart_training.bat"

echo Both backend and training are now running continuously.
echo Press Ctrl+C in each window to stop them individually.