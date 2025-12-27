Param(
    [string]$BaseUrl = "http://localhost:8000",
    [string]$AdminToken = "live-admin-token"
)

$Headers = @{ 'X-Admin-Token' = $AdminToken }

Write-Host "=== Triggering Liquidity Refresh (manual) ===" -ForegroundColor Cyan
try {
    $Response = Invoke-RestMethod -Method POST -Uri "$BaseUrl/scheduler/liquidity" -Headers $Headers -TimeoutSec 60
    $Response | ConvertTo-Json -Depth 6
    exit 0
}
catch {
    Write-Host "Liquidity trigger failed:" -ForegroundColor Red
    Write-Output $_.Exception.Message
    if ($_.ErrorDetails -and $_.ErrorDetails.Message) { Write-Output $_.ErrorDetails.Message }
    exit 1
}
