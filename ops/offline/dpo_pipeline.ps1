##############################################################################
#  dpo_pipeline.ps1 -- Full DPO Pipeline (Train -> Eval -> Go/No-Go -> Deploy)
#
#  BRUK: Dobbeltklikk paa START_DPO_PIPELINE.bat
#        eller: powershell -ExecutionPolicy Bypass -File dpo_pipeline.ps1
#
#  Parametere:
#    -SkipFormat   Hopp over format_dpo_dataset.py
#    -SkipTrain    Hopp over trening (test resten av pipelinen)
#    -SkipDeploy   Ikke deploy til VPS selv ved GO
#    -Version v3   Adapter-versjon-tag (standard: v2)
##############################################################################

param(
    [switch]$SkipFormat,
    [switch]$SkipTrain,
    [switch]$SkipDeploy,
    [string]$Version = "v2"
)

Set-StrictMode -Version Latest

# Paths
$OFFLINE_DIR   = $PSScriptRoot
$PYTHON        = "c:\quantum_trader\.venv\Scripts\python.exe"
$VPS_HOST      = "root@46.224.116.254"
$VPS_ADAPTER   = "/opt/quantum/models/dpo_adapters/qwen2.5-3b-dpo-$Version"
$TRAIN_OUT     = Join-Path $OFFLINE_DIR "dpo_run_output"
$TRAINER_STATE = Join-Path $TRAIN_OUT "trainer_state.json"
$SHADOW_RESULT = Join-Path $OFFLINE_DIR "shadow_eval_results.json"
$REPORT_FILE   = Join-Path $OFFLINE_DIR "pipeline_report.txt"
$TRAIN_LOG     = Join-Path $OFFLINE_DIR "train_dpo_stdout.log"
$TRAIN_ERR     = Join-Path $OFFLINE_DIR "train_dpo_stderr.log"
$EVAL_LOG      = Join-Path $OFFLINE_DIR "shadow_eval_stdout.log"
$EVAL_ERR      = Join-Path $OFFLINE_DIR "shadow_eval_stderr.log"

# Helpers
function Write-Step { param($m) Write-Host ""; Write-Host "[STEP] $m" -ForegroundColor Cyan }
function Write-OK   { param($m) Write-Host "  [OK] $m"  -ForegroundColor Green  }
function Write-Warn { param($m) Write-Host "  [!!] $m"  -ForegroundColor Yellow }
function Write-Fail { param($m) Write-Host " [ERR] $m"  -ForegroundColor Red    }
function Get-Elapsed {
    param($start)
    $ts = [datetime]::Now - $start
    return "{0:mm}:{1:ss}" -f $ts, $ts
}

$pipeStart = [datetime]::Now

##############################################################################
# STEG 0 -- Rydd gamle kjoeringer
##############################################################################
Write-Step "STEG 0 -- Rydder gamle checkpoints"

Push-Location $OFFLINE_DIR

$oldFiles = @(
    "shadow_eval_base_checkpoint.json",
    "shadow_eval_tuned_checkpoint.json",
    "shadow_eval_results.json",
    "pipeline_report.txt"
)
foreach ($f in $oldFiles) {
    if (Test-Path $f) { Remove-Item $f -Force }
}

if (-not $SkipTrain) {
    if (Test-Path $TRAIN_OUT) { Remove-Item $TRAIN_OUT -Recurse -Force }
}

Pop-Location
Write-OK "Ryddet"

##############################################################################
# STEG 1 -- Format datasett
##############################################################################
if ($SkipFormat) {
    Write-Warn "STEG 1 hoppet over (SkipFormat er satt)"
} else {
    Write-Step "STEG 1 -- Bygger DPO datasett (format_dpo_dataset.py)"
    Push-Location $OFFLINE_DIR
    $t1 = [datetime]::Now
    & $PYTHON format_dpo_dataset.py
    $rc = $LASTEXITCODE
    Pop-Location
    if ($rc -ne 0) {
        Write-Fail "format_dpo_dataset.py feilet med kode $rc"
        exit 1
    }
    Write-OK "Datasett bygget ($(Get-Elapsed $t1))"
}

##############################################################################
# STEG 2 -- DPO Trening
##############################################################################
if ($SkipTrain) {
    Write-Warn "STEG 2 hoppet over (SkipTrain er satt)"
    if (-not (Test-Path $TRAINER_STATE)) {
        Write-Fail "trainer_state.json finnes ikke og SkipTrain er satt -- avbryter"
        exit 1
    }
} else {
    Write-Step "STEG 2 -- Starter DPO trening (train_dpo.py)"
    Write-Host "  Stdout: $TRAIN_LOG" -ForegroundColor DarkGray
    Write-Host "  Stderr: $TRAIN_ERR" -ForegroundColor DarkGray

    $trainProc = Start-Process `
        -FilePath $PYTHON `
        -ArgumentList @("-u", "train_dpo.py") `
        -WorkingDirectory $OFFLINE_DIR `
        -RedirectStandardOutput $TRAIN_LOG `
        -RedirectStandardError  $TRAIN_ERR `
        -WindowStyle Hidden `
        -PassThru

    Write-OK "Trening startet (PID $($trainProc.Id))"

    $trainStart = [datetime]::Now
    $maxWaitSec = 7200

    :trainLoop while ($true) {
        Start-Sleep -Seconds 20

        $alive = Get-Process -Id $trainProc.Id -ErrorAction SilentlyContinue

        if (Test-Path $TRAIN_LOG) {
            $ll = (Get-Content $TRAIN_LOG -Tail 1 2>$null)
            if ($ll) { Write-Host "  [$(Get-Elapsed $trainStart)] $ll" -ForegroundColor DarkGray }
        }

        if (Test-Path $TRAINER_STATE) {
            Write-OK "Trening ferdig ($(Get-Elapsed $trainStart))"
            break trainLoop
        }

        if ($null -eq $alive) {
            if (Test-Path $TRAINER_STATE) {
                Write-OK "Trening ferdig ($(Get-Elapsed $trainStart))"
                break trainLoop
            }
            Write-Fail "Treningsprosess krasjet -- sjekk $TRAIN_ERR"
            if (Test-Path $TRAIN_ERR) { Get-Content $TRAIN_ERR -Tail 20 }
            exit 1
        }

        $sec = ([datetime]::Now - $trainStart).TotalSeconds
        if ($sec -gt $maxWaitSec) {
            Write-Fail "Trening timeout etter 2 timer"
            exit 1
        }
    }
}

##############################################################################
# STEG 3 -- Shadow Eval
##############################################################################
Write-Step "STEG 3 -- Shadow Eval (shadow_eval.py)"
Write-Host "  Stdout: $EVAL_LOG" -ForegroundColor DarkGray

Push-Location $OFFLINE_DIR

$evalProc = Start-Process `
    -FilePath $PYTHON `
    -ArgumentList @("-u", "shadow_eval.py") `
    -WorkingDirectory $OFFLINE_DIR `
    -RedirectStandardOutput $EVAL_LOG `
    -RedirectStandardError  $EVAL_ERR `
    -WindowStyle Hidden `
    -PassThru

$evalStart  = [datetime]::Now
$maxEvalSec = 1800

:evalLoop while ($true) {
    Start-Sleep -Seconds 15

    $alive = Get-Process -Id $evalProc.Id -ErrorAction SilentlyContinue

    if (Test-Path $EVAL_LOG) {
        $ll = (Get-Content $EVAL_LOG -Tail 1 2>$null)
        if ($ll) { Write-Host "  [$(Get-Elapsed $evalStart)] $ll" -ForegroundColor DarkGray }
    }

    if ($null -eq $alive) {
        if (Test-Path $SHADOW_RESULT) {
            Write-OK "Shadow eval ferdig ($(Get-Elapsed $evalStart))"
        } else {
            Write-Fail "shadow_eval.py krasjet -- sjekk $EVAL_ERR"
            if (Test-Path $EVAL_ERR) { Get-Content $EVAL_ERR -Tail 20 }
            Pop-Location
            exit 1
        }
        break evalLoop
    }

    $sec = ([datetime]::Now - $evalStart).TotalSeconds
    if ($sec -gt $maxEvalSec) {
        Write-Fail "Shadow eval timeout etter 30 min"
        Pop-Location
        exit 1
    }
}

Pop-Location

##############################################################################
# STEG 4 -- Go/No-Go gate
##############################################################################
Write-Step "STEG 4 -- Go/No-Go gate (go_no_go.py)"

Push-Location $OFFLINE_DIR
$goOut  = & $PYTHON go_no_go.py 2>&1
$goCode = $LASTEXITCODE
Pop-Location

Write-Host ""
$goOut | ForEach-Object { Write-Host "  $_" -ForegroundColor White }
Write-Host ""

if ($goCode -eq 0) {
    $VERDICT = "GO"
    Write-OK "VERDICT: GO -- alle kriterier bestatt!"
} else {
    $VERDICT = "NO-GO"
    Write-Warn "VERDICT: NO-GO -- ett eller flere kriterier feilet"
}

##############################################################################
# STEG 5 -- Deploy til VPS
##############################################################################
if ($VERDICT -ne "GO") {
    Write-Warn "STEG 5 hoppet over -- NO-GO"
    Write-Host "  Sjekk shadow_eval_results.json for detaljer" -ForegroundColor DarkGray
} elseif ($SkipDeploy) {
    Write-Warn "STEG 5 hoppet over (SkipDeploy er satt) -- GO men ingen deploy"
} else {
    Write-Step "STEG 5 -- Deploy adapter til VPS"

    # Windows-sti -> WSL-sti
    $wslBase    = $OFFLINE_DIR -replace "\\", "/" -replace "^C:", "/mnt/c"
    $wslTar     = "$wslBase/dpo_run_output_$Version.tar.gz"
    $wslSrc     = $TRAIN_OUT -replace "\\", "/" -replace "^C:", "/mnt/c"
    $wslParent  = ($wslSrc -replace "/[^/]+$", "")
    $wslLeaf    = ($wslSrc -replace ".*/", "")

    # 5a: Pakk
    Write-Host "  Pakker adapter..." -ForegroundColor DarkGray
    wsl bash -lc "tar -czf '$wslTar' -C '$wslParent' '$wslLeaf'"
    if ($LASTEXITCODE -ne 0) { Write-Fail "tar feilet"; exit 1 }
    Write-OK "Adapter pakket"

    # 5b: SCP
    Write-Host "  Overforer til VPS..." -ForegroundColor DarkGray
    wsl bash -lc "scp -i ~/.ssh/hetzner_fresh '$wslTar' ${VPS_HOST}:/tmp/"
    if ($LASTEXITCODE -ne 0) { Write-Fail "SCP feilet"; exit 1 }
    Write-OK "Overfort til VPS"

    # 5c: Pakk ut paa VPS
    $extractCmd = "mkdir -p $VPS_ADAPTER && tar xzf /tmp/dpo_run_output_${Version}.tar.gz -C $VPS_ADAPTER --strip-components=1"
    wsl bash -lc "ssh -i ~/.ssh/hetzner_fresh $VPS_HOST '$extractCmd'"
    if ($LASTEXITCODE -ne 0) { Write-Fail "Utpakking paa VPS feilet"; exit 1 }
    Write-OK "Adapter klar paa VPS: $VPS_ADAPTER"

    # 5d: Redis-registrering
    $now      = Get-Date -Format "yyyy-MM-ddTHH:mm:ss"
    $rSet     = "redis-cli HSET quantum:dpo_adapter:$Version status active go_no_go GO path $VPS_ADAPTER deployed_at $now"
    wsl bash -lc "ssh -i ~/.ssh/hetzner_fresh $VPS_HOST '$rSet'"
    Write-OK "Redis oppdatert: quantum:dpo_adapter:$Version"

    $rMark = "redis-cli SET quantum:pipeline:latest_deploy dpo_$Version EX 86400"
    wsl bash -lc "ssh -i ~/.ssh/hetzner_fresh $VPS_HOST '$rMark'"

    Write-Host ""
    Write-Host "  *** ADAPTER $Version ER KLAR PAA VPS ***" -ForegroundColor Green
    Write-Host "  Aktiver exit-agenten via ENV-vars naar du er klar" -ForegroundColor Yellow
}

##############################################################################
# SUMMARY RAPPORT
##############################################################################
$total = Get-Elapsed $pipeStart

$rLines = @(
    "==========================================="
    "  DPO PIPELINE RAPPORT"
    "  Dato    : $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    "  Versjon : $Version"
    "  Tid     : $total"
    "  Verdict : $VERDICT"
    "-------------------------------------------"
    "  trainer_state  : $(if (Test-Path $TRAINER_STATE) { 'OK' } else { 'MANGLER' })"
    "  shadow_eval    : $(if (Test-Path $SHADOW_RESULT)  { 'OK' } else { 'MANGLER' })"
)

if (Test-Path $SHADOW_RESULT) {
    try {
        $sr = Get-Content $SHADOW_RESULT -Raw | ConvertFrom-Json
        $rLines += "  base_accuracy  : $($sr.base_accuracy)"
        $rLines += "  tuned_accuracy : $($sr.tuned_accuracy)"
        $rLines += "  lift           : $($sr.accuracy_lift)"
        $rLines += "  parse_err      : $($sr.parse_error_rate_tuned)"
    } catch { }
}

$deployInfo = if ($VERDICT -eq "GO" -and -not $SkipDeploy) { "JA -- $VPS_ADAPTER" } else { "NEI" }
$rLines += "  Deploy VPS     : $deployInfo"
$rLines += "==========================================="

Write-Host ""
foreach ($l in $rLines) { Write-Host $l -ForegroundColor White }

$rLines | Set-Content $REPORT_FILE -Encoding UTF8
Write-Host ""
Write-OK "Rapport lagret: $REPORT_FILE"

if ($VERDICT -eq "GO") { exit 0 } else { exit 1 }
