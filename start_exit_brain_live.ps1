#!/usr/bin/env pwsh
# Start backend with Exit Brain V3 LIVE mode and Binance Futures testnet

Write-Host "🚀 Starting Quantum Trader Backend with Exit Brain V3 LIVE MODE" -ForegroundColor Green
Write-Host "=" * 70 -ForegroundColor DarkGray

# Set Exit Brain V3 LIVE mode
$env:EXIT_MODE="EXIT_BRAIN_V3"
$env:EXIT_EXECUTOR_MODE="LIVE"
$env:EXIT_BRAIN_V3_LIVE_ROLLOUT="ENABLED"
$env:EXIT_BRAIN_V3_ENABLED="true"

# Set Binance Futures testnet execution
$env:QT_EXECUTION_EXCHANGE="binance-futures"
$env:QT_EXECUTION_QUOTE_ASSET="USDT"
$env:QT_EXECUTION_BINANCE_TESTNET="true"

# Set Binance testnet mode
$env:BINANCE_TESTNET="true"
$env:BINANCE_USE_TESTNET="true"
$env:STAGING_MODE="false"
$env:QT_PAPER_TRADING="true"  # TRUE for testnet!

Write-Host "✅ Exit Brain V3 LIVE mode enabled" -ForegroundColor Green
Write-Host "✅ Binance Futures testnet execution enabled" -ForegroundColor Green
Write-Host "" -ForegroundColor DarkGray

# Start backend
Write-Host "🔄 Starting uvicorn backend..." -ForegroundColor Yellow
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
