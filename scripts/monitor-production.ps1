# Production Monitoring Script for ExitBrain v3.5
# Run: .\scripts\monitor-production.ps1

$SSH_KEY = "$HOME\.ssh\hetzner_fresh"
$VPS = "qt@46.224.116.254"

Write-Host "==================================" -ForegroundColor Cyan
Write-Host "ExitBrain v3.5 Production Monitor" -ForegroundColor Cyan
Write-Host "==================================" -ForegroundColor Cyan
Write-Host ""

# Function to run SSH command and display output
function Invoke-VPSCommand {
    param($Command, $Title)
    Write-Host "[$Title]" -ForegroundColor Yellow
    ssh -i $SSH_KEY $VPS $Command
    Write-Host ""
}

# 1. Container Health
Invoke-VPSCommand @"
docker ps --filter name=quantum --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}'
"@ "Container Health"

# 2. Consumer Logs (Last 10 lines)
Invoke-VPSCommand @"
docker logs --since 5m quantum_trade_intent_consumer 2>&1 | tail -10
"@ "Consumer Logs (Last 5 min)"

# 3. ExitBrain v3.5 Activity
Invoke-VPSCommand @"
docker logs --since 10m quantum_trade_intent_consumer 2>&1 | grep -E '🎯 ExitBrain|Adaptive Levels Calculated' | tail -5
"@ "ExitBrain v3.5 Activity"

# 4. Adaptive Levels Stream Stats
Invoke-VPSCommand @"
echo "Stream Length: \$(docker exec quantum_redis redis-cli XLEN quantum:stream:exitbrain.adaptive_levels)"
echo ""
echo "Latest Event:"
docker exec quantum_redis redis-cli XREVRANGE quantum:stream:exitbrain.adaptive_levels + - COUNT 1 | head -20
"@ "Adaptive Levels Stream"

# 5. Error Check
Invoke-VPSCommand @"
echo "=== Recent Errors (Last 10 min) ==="
docker logs --since 10m quantum_trade_intent_consumer 2>&1 | grep -E 'ERROR|Exception|Traceback' | tail -5
echo ""
echo "=== -4164 MIN_NOTIONAL Errors ==="
docker logs --since 10m quantum_backend 2>&1 | grep -E '4164|MIN_NOTIONAL' | tail -3
if [ \$? -ne 0 ]; then echo "✅ No MIN_NOTIONAL errors"; fi
"@ "Error Check"

# 6. Trade Intent Stream Stats
Invoke-VPSCommand @"
echo "Trade Intent Stream Length: \$(docker exec quantum_redis redis-cli XLEN quantum:stream:trade.intent)"
echo "Latest Trade Intent:"
docker exec quantum_redis redis-cli XREVRANGE quantum:stream:trade.intent + - COUNT 1 | grep -E 'symbol|side|leverage|volatility' | head -5
"@ "Trade Intent Stream"

# 7. API Status
Invoke-VPSCommand @"
docker logs --since 5m quantum_trade_intent_consumer 2>&1 | grep -E 'API error|401|403|429' | tail -3
if [ \$? -ne 0 ]; then 
    echo "✅ No API errors detected"
else
    echo "⚠️ API errors found above"
fi
"@ "Binance API Status"

Write-Host "==================================" -ForegroundColor Cyan
Write-Host "Monitoring Complete" -ForegroundColor Green
Write-Host "==================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "💡 TIP: Run with -Continuous to loop every 30s:" -ForegroundColor Yellow
Write-Host "   while (\$true) { .\scripts\monitor-production.ps1; Start-Sleep 30; Clear-Host }" -ForegroundColor Gray
