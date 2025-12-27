# Setup SSH Key using root access
# This will properly configure SSH access for benyamin user

Write-Host "=== Setting up SSH Key with root access ===" -ForegroundColor Cyan
Write-Host ""

$sshPubKey = "$env:USERPROFILE\.ssh\hetzner_fresh.pub"
$vpsHost = "46.224.116.254"

if (-not (Test-Path $sshPubKey)) {
    Write-Host "❌ Public key not found: $sshPubKey" -ForegroundColor Red
    exit 1
}

$pubKeyContent = Get-Content $sshPubKey -Raw
Write-Host "Public key: $sshPubKey" -ForegroundColor Green
Write-Host ""

# Create setup script on server first (avoids line ending issues)
Write-Host "Creating setup script on server..." -ForegroundColor Yellow
$setupScript = @'
#!/bin/bash
set -e
id -u benyamin &>/dev/null || useradd -m -s /bin/bash benyamin
mkdir -p /home/benyamin/.ssh
chmod 700 /home/benyamin/.ssh
chown benyamin:benyamin /home/benyamin/.ssh
usermod -aG sudo benyamin
echo 'benyamin ALL=(ALL) NOPASSWD: ALL' > /etc/sudoers.d/benyamin
chmod 0440 /etc/sudoers.d/benyamin
echo 'Setup complete!'
'@

$setupScript | ssh root@$vpsHost "cat > /tmp/setup_user.sh && chmod +x /tmp/setup_user.sh"

Write-Host "Running setup script..." -ForegroundColor Yellow
ssh root@$vpsHost "bash /tmp/setup_user.sh"

Write-Host "Adding SSH key..." -ForegroundColor Yellow
$pubKeyContent | ssh root@$vpsHost "cat > /home/benyamin/.ssh/authorized_keys && chmod 600 /home/benyamin/.ssh/authorized_keys && chown benyamin:benyamin /home/benyamin/.ssh/authorized_keys && echo '✅ SSH key added'"

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "=== Testing SSH ===" -ForegroundColor Yellow
    ssh -i "$env:USERPROFILE\.ssh\hetzner_fresh" benyamin@$vpsHost "echo '✅ Passwordless SSH works!' && sudo -n whoami && echo '✅ Passwordless sudo works!'"
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "🎉 Perfect! Everything is configured!" -ForegroundColor Green
        Write-Host ""
        Write-Host "Now you can deploy:" -ForegroundColor Cyan
        Write-Host "   .\deploy_adaptive_to_vps.ps1" -ForegroundColor White
    }
} else {
    Write-Host "❌ Setup failed" -ForegroundColor Red
}
