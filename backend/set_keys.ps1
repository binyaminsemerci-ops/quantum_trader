# Quick script to set Binance API keys via dashboard settings
# Usage: Edit the keys below, then run: .\set_keys.ps1

$Headers = @{ 'X-Admin-Token' = 'live-admin-token' }

# PUT YOUR REAL KEYS HERE:
$ApiKey = '5IvIbzNr5L3iYLjQK5l0vONkatU7jWunOnZqG52vMjMzPT43YyIrrJKiHrsUZz6p'
$ApiSecret = '9kDY8TQdaCZn0sunDJVOr2FyU4glVdcPP4nNUh6ltVNnsPyqP1jeFxFAQu6673ZC'

$Body = @{
    api_key   = $ApiKey
    api_secret= $ApiSecret
} | ConvertTo-Json

Write-Host "Posting API keys to backend..." -ForegroundColor Cyan

try {
    $Response = Invoke-RestMethod -Method POST -Uri http://localhost:8000/settings -Headers $Headers -ContentType 'application/json' -Body $Body
    Write-Host "✓ Success: $($Response.message)" -ForegroundColor Green
    Write-Host "  Updated fields: $($Response.updated_fields -join ', ')" -ForegroundColor Green
}
catch {
    Write-Host "✗ Failed to set keys: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

# Verify keys are configured
Write-Host "`nVerifying configuration..." -ForegroundColor Cyan
try {
    $Status = Invoke-RestMethod -Method GET -Uri 'http://localhost:8000/settings?secure=true' -Headers $Headers
    Write-Host "  API Key configured:    $($Status.api_key_configured)" -ForegroundColor $(if ($Status.api_key_configured) { 'Green' } else { 'Red' })
    Write-Host "  API Secret configured: $($Status.api_secret_configured)" -ForegroundColor $(if ($Status.api_secret_configured) { 'Green' } else { 'Red' })
}
catch {
    Write-Host "✗ Failed to verify: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

Write-Host "`n✓ Keys are ready. You can now trigger execution cycles." -ForegroundColor Green
