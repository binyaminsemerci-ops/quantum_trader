# Quick System Health Check - Fixed
# Tests all critical endpoints and reports status

Write-Host "=== QUANTUM TRADER HEALTH CHECK ===" -ForegroundColor Cyan

$baseUrl = "http://localhost:8000"
$endpoints = @(
    "/",
    "/api/v1/system/status",
    "/api/v1/watchlist/full", 
    "/api/v1/portfolio",
    "/api/v1/ai-trading/status",
    "/api/v1/continuous-learning/status"
)

foreach ($endpoint in $endpoints) {
    $url = $baseUrl + $endpoint
    try {
        $null = Invoke-RestMethod -Uri $url -Method Get -TimeoutSec 2
        Write-Host "checkmark $endpoint" -ForegroundColor Green
    } catch {
        Write-Host "X $endpoint - $($_.Exception.Message)" -ForegroundColor Red
    }
}

Write-Host "=== WATCHLIST SAMPLE ===" -ForegroundColor Cyan
try {
    $watchlist = Invoke-RestMethod -Uri "$baseUrl/api/v1/watchlist/full" -Method Get -TimeoutSec 2
    $count = $watchlist.Count
    Write-Host "Watchlist contains $count coins" -ForegroundColor Green
    if ($count -gt 0) {
        $first = $watchlist[0]
        Write-Host "Sample: $($first.symbol) ($($first.name)) - $($first.price)" -ForegroundColor Yellow
    }
} catch {
    Write-Host "Failed to fetch watchlist sample: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host "=== HEALTH CHECK COMPLETE ===" -ForegroundColor Cyan