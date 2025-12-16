# Quantum Trader - Production Deployment Script (PowerShell)
# This script automates the deployment process to a VPS

param(
    [Parameter(Mandatory=$true)]
    [string]$VpsHost,
    
    [Parameter(Mandatory=$false)]
    [string]$VpsUser = "ubuntu",
    
    [Parameter(Mandatory=$false)]
    [switch]$SkipBackup
)

$ErrorActionPreference = "Stop"

Write-Host "🚀 Quantum Trader - Production Deployment" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

$deployDir = "/home/$VpsUser/quantum_trader"
$backupDir = "/home/$VpsUser/backups"

function Write-Info {
    param([string]$message)
    Write-Host "[INFO] $message" -ForegroundColor Green
}

function Write-Warn {
    param([string]$message)
    Write-Host "[WARN] $message" -ForegroundColor Yellow
}

function Write-Error-Custom {
    param([string]$message)
    Write-Host "[ERROR] $message" -ForegroundColor Red
}

# Check SSH connection
Write-Info "Testing SSH connection to $VpsUser@$VpsHost..."
try {
    $sshTest = ssh -o ConnectTimeout=5 "$VpsUser@$VpsHost" "echo 'SSH connection successful'" 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "SSH connection failed"
    }
    Write-Info "SSH connection successful"
} catch {
    Write-Error-Custom "Cannot connect to VPS via SSH"
    Write-Host "Make sure:"
    Write-Host "  1. SSH is configured with key-based auth"
    Write-Host "  2. VPS host is reachable: $VpsHost"
    Write-Host "  3. User exists on VPS: $VpsUser"
    exit 1
}

# Create backup
if (-not $SkipBackup) {
    Write-Info "Creating backup on VPS..."
    $backupScript = @"
if [ -d ~/quantum_trader ]; then
    BACKUP_NAME="quantum_trader_backup_`$(date +%Y%m%d_%H%M%S)"
    mkdir -p ~/backups
    cp -r ~/quantum_trader ~/backups/`$BACKUP_NAME
    echo "Backup created: `$BACKUP_NAME"
fi
"@
    ssh "$VpsUser@$VpsHost" $backupScript
}

# Upload code
Write-Info "Uploading code to VPS..."
$excludes = @(
    "node_modules",
    ".venv",
    "*.pyc",
    "__pycache__",
    ".git",
    "backend/trades.db",
    ".env"
)

# Create rsync exclude file
$excludeFile = New-TemporaryFile
$excludes | ForEach-Object { Add-Content -Path $excludeFile.FullName -Value $_ }

# Use rsync (requires WSL or rsync for Windows)
if (Get-Command rsync -ErrorAction SilentlyContinue) {
    rsync -avz --exclude-from=$excludeFile.FullName ./ "$VpsUser@${VpsHost}:$deployDir/"
} else {
    Write-Warn "rsync not found, using scp (slower)..."
    scp -r ./* "$VpsUser@${VpsHost}:$deployDir/"
}

Remove-Item $excludeFile.FullName

# Deploy
Write-Info "Running deployment on VPS..."
$deployScript = @"
set -e
cd ~/quantum_trader

echo "🔨 Building Docker images..."
docker-compose build --no-cache

echo "🛑 Stopping old containers..."
docker-compose down

echo "🚀 Starting new containers..."
docker-compose up -d

echo "⏳ Waiting for services to start..."
sleep 10

echo "✅ Checking health..."
HEALTH=`$(curl -s http://localhost:8000/health || echo "failed")
echo "Backend health: `$HEALTH"

echo "📊 Container status:"
docker-compose ps
"@

ssh "$VpsUser@$VpsHost" $deployScript

# Verify deployment
Write-Info "Verifying deployment..."
Start-Sleep -Seconds 5

try {
    $healthCheck = Invoke-RestMethod -Uri "http://$VpsHost/health" -TimeoutSec 10 -ErrorAction Stop
    
    if ($healthCheck.status -eq "healthy") {
        Write-Info "✅ Deployment successful!"
        Write-Host ""
        Write-Host "🌐 Dashboard: https://$VpsHost" -ForegroundColor Cyan
        Write-Host "📊 API: https://$VpsHost/api" -ForegroundColor Cyan
        Write-Host "❤️ Health: https://$VpsHost/health" -ForegroundColor Cyan
    } else {
        Write-Warn "⚠️ Service is running but health check returned: $($healthCheck.status)"
    }
} catch {
    Write-Error-Custom "❌ Deployment verification failed"
    Write-Warn "Check logs with: ssh $VpsUser@$VpsHost 'cd quantum_trader && docker-compose logs'"
    exit 1
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "🎉 Deployment Complete!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan
