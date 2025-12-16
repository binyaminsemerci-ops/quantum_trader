# Run diagnostic in WSL and save to file
$scriptPath = "~/quantum_trader/diagnose.sh"
$outputPath = "/tmp/diagnose_output.txt"

# Copy script
wsl cp /mnt/c/quantum_trader/diagnose.sh ~/quantum_trader/
wsl chmod +x ~/quantum_trader/diagnose.sh

# Run and save to file
wsl bash -c "~/quantum_trader/diagnose.sh > $outputPath 2>&1"

# Read and display
Write-Host "`n╔═══════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║              DIAGNOSTIC RESULTS                           ║" -ForegroundColor Cyan
Write-Host "╚═══════════════════════════════════════════════════════════╝`n" -ForegroundColor Cyan

$output = wsl cat $outputPath
Write-Host $output

Write-Host "`n╚═══════════════════════════════════════════════════════════╝`n" -ForegroundColor Cyan
