Param(
    [string]$BaseUrl = "http://localhost:8000",
    [string]$AdminToken = "live-admin-token"
)

$Headers = @{ 'X-Admin-Token' = $AdminToken }

try {
    Invoke-RestMethod -Method GET -Uri "$BaseUrl/risk" -Headers $Headers -TimeoutSec 30 | ConvertTo-Json -Depth 6
} catch {
    Write-Host "Risk fetch failed:" -ForegroundColor Red
    Write-Output $_.Exception.Message
    if ($_.ErrorDetails -and $_.ErrorDetails.Message) { Write-Output $_.ErrorDetails.Message }
    exit 1
}
