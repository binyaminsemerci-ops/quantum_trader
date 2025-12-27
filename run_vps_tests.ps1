#!/usr/bin/env pwsh
###############################################################################
# Kjør Comprehensive System Test på VPS
###############################################################################

$VPS_IP = "46.224.116.254"
$VPS_USER = "qt"
$SSH_KEY = "$HOME\.ssh\hetzner_fresh"

Write-Host "`n═══════════════════════════════════════════════════" -ForegroundColor Blue
Write-Host "  QUANTUM TRADER VPS TEST" -ForegroundColor Blue
Write-Host "═══════════════════════════════════════════════════`n" -ForegroundColor Blue

Write-Host "[INFO] VPS: $VPS_USER@$VPS_IP" -ForegroundColor Cyan
Write-Host "[INFO] SSH Key: $SSH_KEY`n" -ForegroundColor Cyan

# Test SSH connection via WSL
Write-Host "[1/4] Testing SSH connection..." -ForegroundColor Yellow
$sshTest = wsl bash -c "ssh -i ~/.ssh/hetzner_fresh -o ConnectTimeout=5 -o StrictHostKeyChecking=no qt@$VPS_IP 'echo OK' 2>&1"

if ($sshTest -match "OK") {
    Write-Host "  ✓ SSH connection successful`n" -ForegroundColor Green
} else {
    Write-Host "  ✗ SSH connection failed!" -ForegroundColor Red
    Write-Host "  Error: $sshTest`n" -ForegroundColor Red
    Write-Host "Troubleshooting steps:" -ForegroundColor Yellow
    Write-Host "1. Check if SSH key exists: ls ~/.ssh/hetzner_fresh" -ForegroundColor Yellow
    Write-Host "2. Check key permissions: chmod 600 ~/.ssh/hetzner_fresh" -ForegroundColor Yellow
    Write-Host "3. Test manual connection: ssh -i ~/.ssh/hetzner_fresh qt@$VPS_IP" -ForegroundColor Yellow
    exit 1
}

# Copy test script to VPS
Write-Host "[2/4] Copying test script to VPS..." -ForegroundColor Yellow
wsl bash -c "scp -i ~/.ssh/hetzner_fresh -o StrictHostKeyChecking=no ./comprehensive_system_test.sh qt@${VPS_IP}:/home/qt/comprehensive_system_test.sh 2>&1"

if ($LASTEXITCODE -eq 0) {
    Write-Host "  ✓ Script copied successfully`n" -ForegroundColor Green
} else {
    Write-Host "  ✗ Failed to copy script`n" -ForegroundColor Red
    exit 1
}

# Make script executable
Write-Host "[3/4] Making script executable..." -ForegroundColor Yellow
wsl bash -c "ssh -i ~/.ssh/hetzner_fresh qt@$VPS_IP 'chmod +x /home/qt/comprehensive_system_test.sh' 2>&1"

if ($LASTEXITCODE -eq 0) {
    Write-Host "  ✓ Script is executable`n" -ForegroundColor Green
} else {
    Write-Host "  ✗ Failed to set permissions`n" -ForegroundColor Red
    exit 1
}

# Run the test
Write-Host "[4/4] Running comprehensive system test on VPS..." -ForegroundColor Yellow
Write-Host "═══════════════════════════════════════════════════`n" -ForegroundColor Blue

wsl bash -c "ssh -i ~/.ssh/hetzner_fresh qt@$VPS_IP 'cd /home/qt/quantum_trader && bash /home/qt/comprehensive_system_test.sh' 2>&1"

Write-Host "`n═══════════════════════════════════════════════════" -ForegroundColor Blue
Write-Host "  TEST COMPLETE" -ForegroundColor Blue
Write-Host "═══════════════════════════════════════════════════`n" -ForegroundColor Blue
