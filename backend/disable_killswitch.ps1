<#
 Disable the kill-switch via /risk/kill-switch endpoint
#>
$body = '{"enabled":false}'
$headers = @{
    "X-Admin-Token" = "live-admin-token"
    "Content-Type"  = "application/json"
}
Invoke-RestMethod -Uri 'http://localhost:8000/risk/kill-switch' -Method Post -Headers $headers -Body $body | ConvertTo-Json -Depth 5
