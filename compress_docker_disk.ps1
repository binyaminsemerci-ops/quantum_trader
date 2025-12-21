# Compress Docker Desktop VHDX Disk
# Run this script as Administrator

Write-Host "=== Docker Disk Compression Script ===" -ForegroundColor Cyan
Write-Host ""

$vhdxPath = "C:\Users\belen\AppData\Local\Docker\wsl\disk\docker_data.vhdx"

# Check if file exists
if (-not (Test-Path $vhdxPath)) {
    Write-Host "ERROR: Docker VHDX ikke funnet på: $vhdxPath" -ForegroundColor Red
    exit 1
}

# Show size before
$sizeBefore = (Get-Item $vhdxPath).Length / 1GB
Write-Host "Størrelse FØR komprimering: $([math]::Round($sizeBefore,2)) GB" -ForegroundColor Yellow
Write-Host ""

# Stop Docker and WSL
Write-Host "Stopper Docker Desktop og WSL..." -ForegroundColor Green
Stop-Process -Name "Docker Desktop" -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 3
wsl --shutdown
Start-Sleep -Seconds 2

Write-Host "Komprimerer disk... (dette kan ta 5-10 minutter)" -ForegroundColor Green
Write-Host ""

# Create diskpart script
$diskpartScript = @"
select vdisk file="$vhdxPath"
attach vdisk readonly
compact vdisk
detach vdisk
exit
"@

$tempScript = "$env:TEMP\compact_docker_vhdx.txt"
$diskpartScript | Out-File -FilePath $tempScript -Encoding ASCII

# Run diskpart
diskpart /s $tempScript

# Show size after
Start-Sleep -Seconds 2
$sizeAfter = (Get-Item $vhdxPath).Length / 1GB
$saved = $sizeBefore - $sizeAfter

Write-Host ""
Write-Host "=== RESULTAT ===" -ForegroundColor Cyan
Write-Host "Størrelse ETTER komprimering: $([math]::Round($sizeAfter,2)) GB" -ForegroundColor Green
Write-Host "Plass frigjort: $([math]::Round($saved,2)) GB" -ForegroundColor Green
Write-Host ""
Write-Host "Start Docker Desktop igjen for å fortsette." -ForegroundColor Yellow

# Cleanup
Remove-Item $tempScript -ErrorAction SilentlyContinue
