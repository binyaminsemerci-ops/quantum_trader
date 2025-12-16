<#
 Start backend in live (non-staging) mode with Binance USDM futures

 Notes
 - This script does NOT persist API keys. You can either:
	 a) Post keys at runtime via /settings (recommended; see set_keys.ps1)
	 b) OR uncomment BINANCE_API_KEY/SECRET below to inject via environment
			(be careful: this stores secrets locally in plain text)
#>

# Ensure we run from the script directory so 'main:app' can be imported
Set-Location -Path $PSScriptRoot
# Ensure Python can import local packages when launched from elsewhere
$env:PYTHONPATH = $PSScriptRoot

$env:STAGING_MODE = "false"
$env:QT_ADMIN_TOKEN = "live-admin-token"
$env:QT_EXECUTION_EXCHANGE = "binance-futures"
$env:QT_MARKET_TYPE = "usdm_perp"
$env:QT_MARGIN_MODE = "cross"

# Risk limits for live trading
$env:QT_MAX_NOTIONAL_PER_TRADE = "2000"
$env:QT_MAX_DAILY_LOSS = "500"
$env:QT_MAX_POSITION_PER_SYMBOL = "2000"
$env:QT_MAX_GROSS_EXPOSURE = "10000"
$env:QT_KILL_SWITCH = "false"
$env:QT_FAILSAFE_RESET_MINUTES = "30"

# Optional: leverage & stable quotes if needed
$env:QT_DEFAULT_LEVERAGE = "5"
$env:QT_LIQUIDITY_STABLE_QUOTES = "USDC"
$env:QT_EXECUTION_QUOTE_ASSET = "USDC"

# Binance API credentials (loaded from environment)
$env:BINANCE_API_KEY = "5IvIbzNr5L3iYLjQK5l0vONkatU7jWunOnZqG52vMjMzPT43YyIrrJKiHrsUZz6p"
$env:BINANCE_API_SECRET = "9kDY8TQdaCZn0sunDJVOr2FyU4glVdcPP4nNUh6ltVNnsPyqP1jeFxFAQu6673ZC"

Write-Host "Launching backend (live mode) ..."
uvicorn main:app --host 0.0.0.0 --port 8000