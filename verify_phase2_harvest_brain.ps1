#!/usr/bin/env pwsh
# Phase 2 Harvest Brain Verification Script
# Verifies harvest-brain service status after deployment

Write-Host "=== PHASE 2: HARVEST BRAIN VERIFICATION ===" -ForegroundColor Cyan
Write-Host ""

$vpsHost = "root@46.224.116.254"
$sshKey = "~/.ssh/hetzner_fresh"

Write-Host "1. Checking harvest-brain service status..." -ForegroundColor Yellow
$status = wsl bash -c "ssh -i $sshKey $vpsHost 'systemctl status quantum-harvest-brain --no-pager -l' 2>&1"
Write-Host $status
Write-Host ""

Write-Host "2. Checking service is-active state..." -ForegroundColor Yellow  
$active = wsl bash -c "ssh -i $sshKey $vpsHost 'systemctl is-active quantum-harvest-brain' 2>&1"
Write-Host "Service state: $active"
Write-Host ""

Write-Host "3. Checking harvest-brain process..." -ForegroundColor Yellow
$process = wsl bash -c "ssh -i $sshKey $vpsHost 'ps aux | grep harvest_brain | grep -v grep' 2>&1"
Write-Host $process
Write-Host ""

Write-Host "4. Checking recent logs (last 30 lines)..." -ForegroundColor Yellow
$logs = wsl bash -c "ssh -i $sshKey $vpsHost 'journalctl -u quantum-harvest-brain --since \"5 minutes ago\" --no-pager -n 30' 2>&1"
Write-Host $logs
Write-Host ""

Write-Host "5. Checking harvest.intent stream..." -ForegroundColor Yellow
$stream = wsl bash -c "ssh -i $sshKey $vpsHost 'redis-cli XLEN quantum:stream:harvest.intent 2>&1; redis-cli XREVRANGE quantum:stream:harvest.intent + - COUNT 1 2>&1' 2>&1"
Write-Host $stream
Write-Host ""

Write-Host "6. Verifying service file fix..." -ForegroundColor Yellow
$serviceFile = wsl bash -c "ssh -i $sshKey $vpsHost 'grep ExecStart /etc/systemd/system/quantum-harvest-brain.service' 2>&1"
Write-Host $serviceFile
Write-Host ""

Write-Host "=== VERIFICATION COMPLETE ===" -ForegroundColor Green
