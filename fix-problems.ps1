param(
    [string]$ProjectPath = "frontend",
    [string]$TypecheckScript = "typecheck",
    [string]$TestScript = "test:frontend"
)

Push-Location $ProjectPath

Write-Host "Installerer avhengigheter …"
npm install

Write-Host "Kjører lint med auto-fix …"
if (npm run lint -- --fix) {
    Write-Host "Lint OK."
} else {
    Write-Warning "Fant ikke lint-script. Kjører $TestScript i stedet."
    npm run $TestScript
}

Write-Host "Kjører typecheck …"
npm run $TypecheckScript

Pop-Location
