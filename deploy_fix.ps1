Write-Host "🚀 Deploying signal reading fix..." -ForegroundColor Green
Write-Host ""

# Upload file
Write-Host "📤 Uploading executor_service.py..." -ForegroundColor Cyan
scp -i ~/.ssh/hetzner_fresh c:\quantum_trader\backend\microservices\auto_executor\executor_service.py qt@46.224.116.254:/tmp/executor_service_fixed.py

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Upload successful" -ForegroundColor Green
    
    # Deploy
    Write-Host "🔧 Deploying and restarting..." -ForegroundColor Cyan
    scp -i ~/.ssh/hetzner_fresh c:\quantum_trader\deploy_signal_fix.sh qt@46.224.116.254:/tmp/
    ssh -i ~/.ssh/hetzner_fresh qt@46.224.116.254 "bash /tmp/deploy_signal_fix.sh"
    
    Write-Host ""
    Write-Host "✅ Deployment complete! Monitor with:" -ForegroundColor Green
    Write-Host "ssh -i ~/.ssh/hetzner_fresh qt@46.224.116.254 'docker logs quantum_auto_executor -f'" -ForegroundColor Yellow
} else {
    Write-Host "❌ Upload failed" -ForegroundColor Red
}
