param(
    [string]$Symbol
)

Write-Host "`n🔍 Checking $Symbol..." -ForegroundColor Cyan

# 1. Check position
Write-Host "`n📊 Position:" -ForegroundColor Yellow
docker exec quantum_backend python /app/quick_positions.py 2>&1 | Select-String $Symbol

# 2. Check logs for TP/SL attempt
Write-Host "`n📋 Logs (last 2 min):" -ForegroundColor Yellow
docker logs quantum_backend --since 2m 2>&1 | Select-String "$Symbol.*(Attempting|actual entry|precision|TP=|SL=|TP order placed|SL order placed|Failed)" | Select-Object -Last 15

# 3. Check if SL missing
$output = docker exec quantum_backend python /app/quick_positions.py 2>&1 | Out-String
if ($output -match "$Symbol.*SL: 0 orders") {
    Write-Host "`n⚠️  WARNING: $Symbol has NO SL protection!" -ForegroundColor Red
    Write-Host "Creating fix script..." -ForegroundColor Yellow
    
    # Get position details
    $posInfo = docker exec quantum_backend python -c "from binance.client import Client; import os; c=Client(os.getenv('BINANCE_API_KEY'),os.getenv('BINANCE_API_SECRET')); p=c.futures_position_information(symbol='$Symbol')[0]; print(f'{float(p[""positionAmt""])},{float(p[""entryPrice""])}')" 2>&1
    $size,$entry = $posInfo -split ','
    
    $fixScript = @"
#!/usr/bin/env python3
import os
from binance.client import Client

client = Client(os.getenv('BINANCE_API_KEY'), os.getenv('BINANCE_API_SECRET'))
symbol = '$Symbol'
size = $size
entry = $entry

# Get precision
exchange_info = client.futures_exchange_info()
symbol_info = next((s for s in exchange_info['symbols'] if s['symbol'] == symbol), None)
filters = {f['filterType']: f for f in symbol_info['filters']}
tick_size = float(filters['PRICE_FILTER']['tickSize'])
precision = len(str(tick_size).rstrip('0').split('.')[-1]) if '.' in str(tick_size) else 0

# Calculate SL (0.75% from entry)
sl_pct = 0.0075
if size > 0:  # LONG
    sl_price = round(entry * (1 - sl_pct), precision)
    sl_side = 'SELL'
else:  # SHORT
    sl_price = round(entry * (1 + sl_pct), precision)
    sl_side = 'BUY'

print(f'$Symbol {("LONG" if size > 0 else "SHORT")}: Entry={entry}, SL={sl_price}')

# Place SL
sl_order = client.futures_create_order(
    symbol=symbol,
    side=sl_side,
    type='STOP_MARKET',
    stopPrice=sl_price,
    closePosition=True,
    workingType='MARK_PRICE'
)
print(f'✅ SL order placed: {sl_order["orderId"]}')
"@
    
    $fixScript | Out-File -FilePath "fix_${Symbol}_sl.py" -Encoding UTF8
    Write-Host "✅ Created fix_${Symbol}_sl.py" -ForegroundColor Green
    Write-Host "Run: docker cp fix_${Symbol}_sl.py quantum_backend:/app/; docker exec quantum_backend python /app/fix_${Symbol}_sl.py" -ForegroundColor Yellow
}
