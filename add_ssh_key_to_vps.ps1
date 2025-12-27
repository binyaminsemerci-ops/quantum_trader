# Add SSH Key to VPS
# Run this once with password to enable passwordless SSH

Write-Host "=== Adding SSH Key to VPS ===" -ForegroundColor Cyan
Write-Host ""

$sshKey = "$env:USERPROFILE\.ssh\hetzner_fresh"
$sshPubKey = "$env:USERPROFILE\.ssh\hetzner_fresh.pub"
$vpsUser = "benyamin"
$vpsHost = "46.224.116.254"

# Check if public key exists
if (-not (Test-Path $sshPubKey)) {
    Write-Host "❌ Public key not found: $sshPubKey" -ForegroundColor Red
    exit 1
}

# Read public key
$pubKeyContent = Get-Content $sshPubKey -Raw
Write-Host "Public key found: $sshPubKey" -ForegroundColor Green
Write-Host ""
Write-Host "This will add your SSH key to the VPS."
Write-Host "You will be prompted for your password ONE TIME." -ForegroundColor Yellow
Write-Host ""

# Add key to server
Write-Host "Connecting to VPS..." -ForegroundColor Yellow
ssh "$vpsUser@$vpsHost" @"
mkdir -p ~/.ssh
chmod 700 ~/.ssh
echo '$pubKeyContent' >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
echo '✅ SSH key added successfully!'
"@

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "=== Setup Complete ===" -ForegroundColor Green
    Write-Host ""
    Write-Host "Testing passwordless SSH..." -ForegroundColor Yellow
    ssh -i "$sshKey" "$vpsUser@$vpsHost" "echo '✅ Passwordless SSH works!'"
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "🎉 Success! You can now deploy without password:" -ForegroundColor Green
        Write-Host "   .\deploy_adaptive_to_vps.ps1" -ForegroundColor White
    }
} else {
    Write-Host ""
    Write-Host "❌ Failed to add SSH key" -ForegroundColor Red
}
