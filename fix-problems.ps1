param(
    [string]$ProjectPath = "frontend",
    [string]$TypecheckScript = "typecheck",
    [string]$FallbackTestScript = "test:frontend"
)

Push-Location $ProjectPath

Write-Host "Installerer avhengigheter …"
npm install

Write-Host "Prøver å kjøre lint …"
if (npm run lint -- --fix) {
    Write-Host "Lint OK."
} else {
    Write-Warning "Fant ikke lint-script. Kjør test-script i stedet."
    npm run $FallbackTestScript -- --run
}

Write-Host "Kjører typecheck …"
npm run $TypecheckScript

Pop-Location
