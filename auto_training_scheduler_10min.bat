@echo off
:: Auto Training Scheduler - Runs every 10 minutes while window is open
SETLOCAL ENABLEDELAYEDEXPANSION

SET LIMIT=1200
IF NOT "%1"=="" SET LIMIT=%1

ECHO [AUTO-TRAIN-10MIN] Starting 10-min training loop with LIMIT=%LIMIT%

:loop
  ECHO [AUTO-TRAIN-10MIN] %DATE% %TIME% -> training start (LIMIT=%LIMIT%)
  CALL start_training_optimized.bat %LIMIT%
  ECHO [AUTO-TRAIN-10MIN] %DATE% %TIME% -> sleeping 600s
  TIMEOUT /T 600 /NOBREAK >NUL
GOTO loop
