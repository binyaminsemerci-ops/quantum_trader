@echo off
REM auto_training_scheduler.bat - Automatic training scheduler for Quantum Trader AI
REM This runs training at startup and every hour to keep the model fresh

setlocal

echo ================================================
echo 🤖 QUANTUM TRADER AUTO TRAINING SCHEDULER
echo ================================================
echo Starting continuous AI model training...
echo - Initial training at PC startup
echo - Hourly model updates while PC is active
echo ================================================
echo.

REM Set training parameters
set "TRAINING_LIMIT=1200"
set "LOG_DIR=C:\quantum_trader\logs\auto_training"
set "TRAINING_SCRIPT=C:\quantum_trader\start_training_optimized.bat"

REM Create log directory if it doesn't exist
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

REM Initial training at startup
echo [%DATE% %TIME%] 🚀 Starting initial training at PC startup...
cd /d "C:\quantum_trader"
call "%TRAINING_SCRIPT%" %TRAINING_LIMIT% > "%LOG_DIR%\startup_%DATE:~-4,4%%DATE:~-10,2%%DATE:~-7,2%_%TIME:~0,2%%TIME:~3,2%.log" 2>&1

if %ERRORLEVEL% EQU 0 (
    echo [%DATE% %TIME%] ✅ Initial training completed successfully
) else (
    echo [%DATE% %TIME%] ❌ Initial training failed
)

REM Start hourly training loop
echo [%DATE% %TIME%] ⏰ Starting hourly training schedule...
echo.

:HOURLY_LOOP
    echo [%DATE% %TIME%] 💤 Waiting 1 hour for next training cycle...
    timeout /t 3600 /nobreak > nul
    
    echo [%DATE% %TIME%] 🔄 Starting hourly model update...
    call "%TRAINING_SCRIPT%" %TRAINING_LIMIT% > "%LOG_DIR%\hourly_%DATE:~-4,4%%DATE:~-10,2%%DATE:~-7,2%_%TIME:~0,2%%TIME:~3,2%.log" 2>&1
    
    if %ERRORLEVEL% EQU 0 (
        echo [%DATE% %TIME%] ✅ Hourly training completed successfully
    ) else (
        echo [%DATE% %TIME%] ❌ Hourly training failed
    )
    
    REM Clean up old log files (keep last 24 hours only)
    forfiles /p "%LOG_DIR%" /m "*.log" /d -1 /c "cmd /c del @path" 2>nul
    
    goto HOURLY_LOOP

endlocal