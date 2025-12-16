@echo off
:loop
echo Starting backend...
python .\backend\main.py
echo Backend stopped, restarting in 5 seconds...
timeout /t 5 /nobreak > nul
goto loop