Write-Host "=== Quantum Trader Container Status ===`n"

# Vis alle containere
docker ps -a

# Sjekk backend
$backendStatus = docker inspect -f '{{.State.Status}}' quantum_backend 2>$null
if ($backendStatus -eq "exited") {
    Write-Host "`n--- Backend logs (quantum_backend) ---" -ForegroundColor Yellow
    docker logs quantum_backend
}

# Sjekk frontend
$frontendStatus = docker inspect -f '{{.State.Status}}' quantum_frontend 2>$null
if ($frontendStatus -eq "exited") {
    Write-Host "`n--- Frontend logs (quantum_frontend) ---" -ForegroundColor Yellow
    docker logs quantum_frontend
}

Write-Host "`n=== Done ==="
