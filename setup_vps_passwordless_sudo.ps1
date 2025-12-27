# Setup Passwordless Sudo on VPS
# Run this script once to enable automated deployment

Write-Host "=== VPS Passwordless Sudo Setup ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "This will connect to your VPS and configure passwordless sudo."
Write-Host "You will need to enter your sudo password ONCE."
Write-Host ""

$sshKey = "$env:USERPROFILE\.ssh\hetzner_fresh"
$vpsUser = "benyamin"
$vpsHost = "46.224.116.254"

# Create the setup script on VPS
$setupScript = @'
#!/bin/bash
echo "=== Setting up passwordless sudo for benyamin user ==="
echo ""
echo "Creating sudoers configuration..."
echo 'benyamin ALL=(ALL) NOPASSWD: ALL' | sudo tee /etc/sudoers.d/benyamin > /dev/null
sudo chmod 0440 /etc/sudoers.d/benyamin

echo "Validating configuration..."
if sudo visudo -c -f /etc/sudoers.d/benyamin 2>&1 | grep -q "parsed OK"; then
    echo ""
    echo "✅ Passwordless sudo configured successfully!"
    echo ""
    echo "Testing..."
    if sudo -n whoami > /dev/null 2>&1; then
        echo "✅ Test passed - sudo works without password"
        echo ""
        echo "You can now run automated deployments!"
    else
        echo "⚠️ Test inconclusive - but configuration is valid"
    fi
else
    echo ""
    echo "❌ Error in sudoers file, removing..."
    sudo rm -f /etc/sudoers.d/benyamin
    exit 1
fi
'@

Write-Host "Step 1: Uploading setup script to VPS..." -ForegroundColor Yellow
$setupScript | ssh -i $sshKey "$vpsUser@$vpsHost" "cat > /tmp/setup_sudo.sh && chmod +x /tmp/setup_sudo.sh && echo '✅ Script uploaded'"

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to upload script" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Step 2: Running setup script on VPS..." -ForegroundColor Yellow
Write-Host "You will be prompted for your sudo password..." -ForegroundColor Green
Write-Host ""

ssh -i $sshKey -t "$vpsUser@$vpsHost" "bash /tmp/setup_sudo.sh"

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "=== Setup Complete ===" -ForegroundColor Green
    Write-Host ""
    Write-Host "✅ Passwordless sudo is now configured!"
    Write-Host "✅ Automated deployment can now proceed"
    Write-Host ""
    Write-Host "Next step: Run deployment with:" -ForegroundColor Cyan
    Write-Host "  .\deploy_adaptive_to_vps.ps1" -ForegroundColor White
} else {
    Write-Host ""
    Write-Host "❌ Setup failed" -ForegroundColor Red
}
