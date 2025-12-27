# Simple SSH Setup - Direct commands
Write-Host "=== Setting up SSH access ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "Enter root password when prompted: tUdjwqF3PuVH" -ForegroundColor Yellow
Write-Host ""

$vpsHost = "46.224.116.254"
$pubKey = Get-Content "$env:USERPROFILE\.ssh\hetzner_fresh.pub" -Raw

# Step 1: Create user
Write-Host "[1/6] Creating benyamin user..." -ForegroundColor Yellow
ssh root@$vpsHost "useradd -m -s /bin/bash benyamin 2>/dev/null || echo 'User exists'"

# Step 2: Create SSH directory
Write-Host "[2/6] Creating SSH directory..." -ForegroundColor Yellow
ssh root@$vpsHost "mkdir -p /home/benyamin/.ssh; chmod 700 /home/benyamin/.ssh"

# Step 3: Add SSH key
Write-Host "[3/6] Adding SSH key..." -ForegroundColor Yellow
echo $pubKey | ssh root@$vpsHost "cat > /home/benyamin/.ssh/authorized_keys"

# Step 4: Fix permissions
Write-Host "[4/6] Setting permissions..." -ForegroundColor Yellow
ssh root@$vpsHost "chmod 600 /home/benyamin/.ssh/authorized_keys; chown -R benyamin:benyamin /home/benyamin/.ssh"

# Step 5: Add sudo access
Write-Host "[5/6] Enabling sudo..." -ForegroundColor Yellow
ssh root@$vpsHost "usermod -aG sudo benyamin; echo 'benyamin ALL=(ALL) NOPASSWD: ALL' > /etc/sudoers.d/benyamin; chmod 0440 /etc/sudoers.d/benyamin"

# Step 6: Test
Write-Host "[6/6] Testing..." -ForegroundColor Yellow
ssh -i "$env:USERPROFILE\.ssh\hetzner_fresh" benyamin@$vpsHost "echo '✅ SSH works!' && sudo -n whoami && echo '✅ Sudo works!'"

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "🎉 SUCCESS! Everything is configured!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Now deploy AdaptiveLeverageEngine:" -ForegroundColor Cyan
    Write-Host "   .\deploy_adaptive_to_vps.ps1" -ForegroundColor White
} else {
    Write-Host "❌ Test failed" -ForegroundColor Red
}
