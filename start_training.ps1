param(
	[int]$Limit = 120,
	[string]$ModelDir = "",
	[switch]$Full,
	[bool]$NoReport = $true,
	[switch]$UseLiveData
)

# start_training.ps1 - PowerShell wrapper to run training.
# Usage examples:
#   .\start_training.ps1                    -> quick run (limit=120, no report)
#   .\start_training.ps1 300                -> run with limit=300 (positional)
#   .\start_training.ps1 300 models\\run1   -> run with limit=300 and model dir
#   .\start_training.ps1 -Full              -> full run (limit=600)
#   .\start_training.ps1 -Limit 120 -NoReport $false -ModelDir models\\run1

# Optional: Activate virtualenv (uncomment and edit path)
# & .\venv\Scripts\Activate.ps1

# Environment overrides (uncomment to set)
# $env:USE_SKLEARN = "1"
# $env:USE_XGB = "1"
# $env:ENABLE_LIVE_MARKET_DATA = "1"

if ($Full) { $Limit = 600 }

$noReportFlag = "--no-report"
if ($NoReport -eq $false) { $noReportFlag = "" }

$modelDirFlag = ""
if ($ModelDir -ne "") {
	# Use single quotes around the path to preserve backslashes/spaces when passed to Python
	$escaped = $ModelDir -replace "'","''"
	$modelDirFlag = "--model-dir '" + $escaped + "'"
}

Write-Host "Starting training (limit=$Limit, no-report=$NoReport, model-dir=$ModelDir, use-live-data=$UseLiveData)"

$argsList = @('main_train_and_backtest.py', 'train', '--limit', [string]$Limit)
if ($NoReport) { $argsList += '--no-report' }
if ($ModelDir -ne '') { $argsList += '--model-dir'; $argsList += $ModelDir }
if ($UseLiveData) { $argsList += '--use-live-data'; $env:ENABLE_LIVE_MARKET_DATA = '1' }

& python $argsList | Tee-Object -FilePath ai_train_${Limit}.log
Write-Host "Training finished. Log: ai_train_${Limit}.log"
