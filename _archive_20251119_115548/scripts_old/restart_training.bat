@echo off
:loop
echo Starting continuous training...
python .\scripts\train_continuous.py
echo Training stopped, restarting in 10 seconds...
timeout /t 10 /nobreak > nul
goto loop