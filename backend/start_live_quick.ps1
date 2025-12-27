<#
 Start backend in live mode for quick verification
 - Disables the liquidity job to reduce background load
 - Shortens execution cadence for faster feedback
 - Uses python -m uvicorn to avoid module resolution quirks

 Note: Secrets are injected via environment for convenience here.
       Prefer posting keys via /settings in production.
#>

# Always run from this script's directory so imports resolve
Set-Location -Path $PSScriptRoot
$env:PYTHONPATH = $PSScriptRoot

# Core live settings
$env:STAGING_MODE = "false"
$env:QT_ADMIN_TOKEN = "live-admin-token"
$env:QT_EXECUTION_EXCHANGE = "binance-futures"
$env:QT_MARKET_TYPE = "usdm_perp"
$env:QT_MARGIN_MODE = "cross"
$env:QT_DEFAULT_LEVERAGE = "5"
$env:QT_LIQUIDITY_STABLE_QUOTES = "USDT,USDC"

# Risk limits
$env:QT_MAX_NOTIONAL_PER_TRADE = "150"
$env:QT_MAX_DAILY_LOSS = "300"
$env:QT_MAX_POSITION_PER_SYMBOL = "400"
$env:QT_MAX_GROSS_EXPOSURE = "2500"
$env:QT_KILL_SWITCH = "false"
$env:QT_FAILSAFE_RESET_MINUTES = "30"

# Scheduler tweaks for quick validation
$env:QUANTUM_TRADER_DISABLE_LIQUIDITY = "1"
$env:QUANTUM_TRADER_EXECUTION_SECONDS = "120"

# Binance credentials (for quick run). Prefer /settings in production.
$env:BINANCE_API_KEY = "5IvIbzNr5L3iYLjQK5l0vONkatU7jWunOnZqG52vMjMzPT43YyIrrJKiHrsUZz6p"
$env:BINANCE_API_SECRET = "9kDY8TQdaCZn0sunDJVOr2FyU4glVdcPP4nNUh6ltVNnsPyqP1jeFxFAQu6673ZC"

Write-Host "Launching backend (live quick mode) ..." -ForegroundColor Cyan
python -m uvicorn main:app --host 0.0.0.0 --port 8000
