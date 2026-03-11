@echo off
title DPO Training Pipeline
color 0A

echo.
echo  =============================================
echo   QUANTUM TRADER - DPO Training Pipeline
echo  =============================================
echo.
echo  Dette starter hele pipelinen automatisk:
echo.
echo    1. Bygg DPO datasett
echo    2. DPO fine-tuning   (~60-90 min)
echo    3. Shadow Eval        (~10-15 min)
echo    4. Go/No-Go gate
echo    5. Deploy til VPS     (kun ved GO)
echo.
echo  Loggfiler skrives til:
echo    ops\offline\train_dpo_stdout.log
echo    ops\offline\shadow_eval_stdout.log
echo    ops\offline\pipeline_report.txt
echo.

set /p CONFIRM="  Trykk ENTER for aa starte, eller Ctrl+C for aa avbryte... "

echo.
echo  Starter pipeline...
echo.

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0dpo_pipeline.ps1"

set EXIT_CODE=%ERRORLEVEL%

echo.
if %EXIT_CODE% == 0 (
    color 0A
    echo  =============================================
    echo   RESULTAT: GO — adapter deployed til VPS!
    echo  =============================================
) else (
    color 0E
    echo  =============================================
    echo   RESULTAT: NO-GO — se pipeline_report.txt
    echo  =============================================
)

echo.
echo  Se full rapport: ops\offline\pipeline_report.txt
echo.
pause
