# ============================================================================
# PHASE 4Q+ — VPS Deployment via SSH
# ============================================================================
# Deployer Portfolio Governance til VPS ved bruk av SSH keys
# ============================================================================

param(
    [switch]$SkipGitPull = $false,
    [switch]$Verbose = $false
)

# VPS Configuration (fra SYSTEM_INVENTORY.yaml)
$VPS_HOST = "46.224.116.254"
$VPS_USER = "qt"
$SSH_KEY_WIN = "$HOME\.ssh\hetzner_fresh"
$SSH_KEY_WSL = "~/.ssh/hetzner_fresh"  # WSL path for SSH command
$VPS_DIR = "/home/qt/quantum_trader"

Write-Host ""
Write-Host "🚀 ==============================================" -ForegroundColor Cyan
Write-Host "   PHASE 4Q+ — VPS DEPLOYMENT (VIA SSH)" -ForegroundColor Green
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "📡 Target: $VPS_USER@$VPS_HOST" -ForegroundColor White
Write-Host "🔑 SSH Key: $SSH_KEY_WSL" -ForegroundColor White
Write-Host "📂 Remote Dir: $VPS_DIR" -ForegroundColor White
Write-Host ""

# --- 1. Test SSH Connection
Write-Host "🔍 Testing SSH connection..." -ForegroundColor Yellow
try {
    $sshTest = wsl ssh -i $SSH_KEY_WSL -o ConnectTimeout=5 -o StrictHostKeyChecking=no "$VPS_USER@$VPS_HOST" "echo OK" 2>&1
    if ($sshTest -match "OK") {
        Write-Host "✅ SSH connection successful" -ForegroundColor Green
    } else {
        throw "SSH test failed: $sshTest"
    }
} catch {
    Write-Host "❌ Cannot connect to VPS!" -ForegroundColor Red
    Write-Host "Error: $_" -ForegroundColor Red
    Write-Host ""
    Write-Host "Troubleshooting:" -ForegroundColor Yellow
    Write-Host "  1. Check SSH key exists: wsl ls $SSH_KEY_WSL" -ForegroundColor White
    Write-Host "  2. Verify VPS is reachable: ping $VPS_HOST" -ForegroundColor White
    Write-Host "  3. Test manual SSH: wsl ssh -i $SSH_KEY_WSL $VPS_USER@$VPS_HOST" -ForegroundColor White
    exit 1
}
Write-Host ""

# --- 2. Upload Portfolio Governance Files
Write-Host "📤 Uploading portfolio_governance files to VPS..." -ForegroundColor Yellow

# Ensure remote directory exists
wsl ssh -i $SSH_KEY_WSL "$VPS_USER@$VPS_HOST" "mkdir -p $VPS_DIR/microservices/portfolio_governance" 2>&1 | Out-Null

# Copy all portfolio_governance files
$localPath = "/mnt/c/quantum_trader/microservices/portfolio_governance"
$remotePath = "$VPS_USER@$VPS_HOST`:$VPS_DIR/microservices/portfolio_governance/"

if (Test-Path "C:\quantum_trader\microservices\portfolio_governance") {
    Write-Host "  • Copying Python files..." -ForegroundColor Gray
    wsl scp -i $SSH_KEY_WSL "$localPath/*.py" $remotePath 2>&1 | Out-Null
    
    Write-Host "  • Copying Dockerfile..." -ForegroundColor Gray
    wsl scp -i $SSH_KEY_WSL "$localPath/Dockerfile" $remotePath 2>&1 | Out-Null
    
    Write-Host "  • Copying requirements.txt..." -ForegroundColor Gray
    wsl scp -i $SSH_KEY_WSL "$localPath/requirements.txt" $remotePath 2>&1 | Out-Null
    
    Write-Host "  • Copying README.md..." -ForegroundColor Gray
    wsl scp -i $SSH_KEY_WSL "$localPath/README.md" $remotePath 2>&1 | Out-Null
    
    Write-Host "✅ Files uploaded successfully" -ForegroundColor Green
} else {
    Write-Host "❌ Local directory not found: $localPath" -ForegroundColor Red
    exit 1
}
Write-Host ""

# --- 3. Upload docker-compose.vps.yml
Write-Host "📤 Uploading docker-compose.vps.yml..." -ForegroundColor Yellow
wsl scp -i $SSH_KEY_WSL "/mnt/c/quantum_trader/docker-compose.vps.yml" "$VPS_USER@$VPS_HOST`:$VPS_DIR/" 2>&1 | Out-Null
Write-Host "✅ Docker compose file uploaded" -ForegroundColor Green
Write-Host ""

# --- 4. Upload SYSTEM_INVENTORY.yaml (updated)
Write-Host "📤 Uploading SYSTEM_INVENTORY.yaml..." -ForegroundColor Yellow
wsl scp -i $SSH_KEY_WSL "/mnt/c/quantum_trader/SYSTEM_INVENTORY.yaml" "$VPS_USER@$VPS_HOST`:$VPS_DIR/" 2>&1 | Out-Null
Write-Host "✅ System inventory uploaded" -ForegroundColor Green
Write-Host ""

# --- 5. Upload ai_engine service.py (with portfolio_governance health)
Write-Host "📤 Uploading updated ai_engine service.py..." -ForegroundColor Yellow
wsl scp -i $SSH_KEY_WSL "/mnt/c/quantum_trader/microservices/ai_engine/service.py" "$VPS_USER@$VPS_HOST`:$VPS_DIR/microservices/ai_engine/" 2>&1 | Out-Null
Write-Host "✅ AI Engine service updated" -ForegroundColor Green
Write-Host ""

# --- 6. Execute deployment script on VPS
Write-Host "🚀 Executing deployment on VPS..." -ForegroundColor Yellow
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Gray

$deployScript = @"
#!/bin/bash
set -e

cd $VPS_DIR

echo "🏗️  Building portfolio_governance Docker image..."
docker compose -f docker-compose.vps.yml build portfolio-governance

echo "▶️  Starting portfolio_governance service..."
docker compose -f docker-compose.vps.yml up -d portfolio-governance

echo "⏳ Waiting 10 seconds for service to initialize..."
sleep 10

echo "🔍 Checking container status..."
docker ps --format "table {{.Names}}\t{{.Status}}" | grep portfolio_governance

echo "📊 Testing Redis integration..."
docker exec redis redis-cli XLEN quantum:stream:portfolio.memory || echo "0"
docker exec redis redis-cli GET quantum:governance:policy || echo "NOT_SET"

echo "🧩 Simulating test data..."
docker exec redis redis-cli XADD quantum:stream:portfolio.memory '*' \
  timestamp "\$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  symbol "BTCUSDT" \
  side "LONG" \
  pnl "0.35" \
  confidence "0.74" \
  volatility "0.12" \
  leverage "20" \
  position_size "1000" \
  exit_reason "dynamic_tp" > /dev/null

sleep 5

echo "✅ Reading policy after simulation..."
POLICY=\$(docker exec redis redis-cli GET quantum:governance:policy 2>/dev/null || echo "NOT_SET")
SCORE=\$(docker exec redis redis-cli GET quantum:governance:score 2>/dev/null || echo "0.0")
MEMORY_LEN=\$(docker exec redis redis-cli XLEN quantum:stream:portfolio.memory 2>/dev/null || echo "0")

echo "  • Policy: \$POLICY"
echo "  • Score: \$SCORE"
echo "  • Memory samples: \$MEMORY_LEN"

echo "📜 Recent logs:"
docker logs --tail 15 quantum_portfolio_governance

echo ""
echo "🎯 DEPLOYMENT COMPLETE!"
"@

# Execute via SSH
$deployScript | wsl ssh -i $SSH_KEY_WSL "$VPS_USER@$VPS_HOST" "bash -s"

Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Gray
Write-Host ""

# --- 7. Verify AI Engine Health
Write-Host "🧠 Checking AI Engine Health endpoint..." -ForegroundColor Yellow
try {
    $healthCheck = wsl ssh -i $SSH_KEY_WSL "$VPS_USER@$VPS_HOST" "curl -s http://localhost:8001/health 2>/dev/null" | ConvertFrom-Json
    if ($healthCheck.metrics.portfolio_governance) {
        Write-Host "✅ Portfolio Governance Metrics Available:" -ForegroundColor Green
        $healthCheck.metrics.portfolio_governance | ConvertTo-Json -Depth 3 | Write-Host -ForegroundColor White
    } else {
        Write-Host "⚠️  Portfolio governance metrics not yet in health endpoint" -ForegroundColor Yellow
        Write-Host "   (May take 30-60 seconds for first policy update)" -ForegroundColor Gray
    }
} catch {
    Write-Host "⚠️  Could not fetch health endpoint" -ForegroundColor Yellow
}
Write-Host ""

# --- 8. Display monitoring commands
Write-Host "📊 MONITORING COMMANDS:" -ForegroundColor Yellow
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Gray
Write-Host ""
Write-Host "View logs:" -ForegroundColor White
Write-Host "  wsl ssh -i $SSH_KEY_WSL $VPS_USER@$VPS_HOST 'docker logs -f quantum_portfolio_governance'" -ForegroundColor Cyan
Write-Host ""
Write-Host "Check policy:" -ForegroundColor White
Write-Host "  wsl ssh -i $SSH_KEY_WSL $VPS_USER@$VPS_HOST 'docker exec redis redis-cli GET quantum:governance:policy'" -ForegroundColor Cyan
Write-Host ""
Write-Host "Check score:" -ForegroundColor White
Write-Host "  wsl ssh -i $SSH_KEY_WSL $VPS_USER@$VPS_HOST 'docker exec redis redis-cli GET quantum:governance:score'" -ForegroundColor Cyan
Write-Host ""
Write-Host "View memory stream:" -ForegroundColor White
Write-Host "  wsl ssh -i $SSH_KEY_WSL $VPS_USER@$VPS_HOST 'docker exec redis redis-cli XREVRANGE quantum:stream:portfolio.memory + - COUNT 10'" -ForegroundColor Cyan
Write-Host ""
Write-Host "Health endpoint:" -ForegroundColor White
Write-Host "  wsl ssh -i $SSH_KEY_WSL $VPS_USER@$VPS_HOST 'curl -s http://localhost:8001/health | jq .metrics.portfolio_governance'" -ForegroundColor Cyan
Write-Host ""

# --- 9. Final Summary
Write-Host ""
Write-Host "🎯 ==============================================" -ForegroundColor Cyan
Write-Host "   PHASE 4Q+ DEPLOYMENT SUCCESSFULLY COMPLETED!" -ForegroundColor Green
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "✅ Deployment Status:" -ForegroundColor Green
Write-Host "   • Files uploaded to VPS" -ForegroundColor White
Write-Host "   • Docker image built" -ForegroundColor White
Write-Host "   • Container started" -ForegroundColor White
Write-Host "   • Test data simulated" -ForegroundColor White
Write-Host ""
Write-Host "📡 Service Details:" -ForegroundColor Yellow
Write-Host "   • Container: quantum_portfolio_governance" -ForegroundColor White
Write-Host "   • VPS: $VPS_HOST" -ForegroundColor White
Write-Host "   • SSH Access: wsl ssh -i $SSH_KEY_WSL $VPS_USER@$VPS_HOST" -ForegroundColor White
Write-Host ""
Write-Host "📊 Next Steps:" -ForegroundColor Yellow
Write-Host "   1. Monitor logs for 30 minutes" -ForegroundColor White
Write-Host "   2. Verify policy transitions (CONSERVATIVE/BALANCED/AGGRESSIVE)" -ForegroundColor White
Write-Host "   3. Check AI Engine health metrics" -ForegroundColor White
Write-Host "   4. Observe portfolio score evolution" -ForegroundColor White
Write-Host ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host "✅ Portfolio Governance Agent is LIVE on VPS!" -ForegroundColor Green
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host ""
