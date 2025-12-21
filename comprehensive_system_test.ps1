#Requires -Version 5.1

###############################################################################
# AI HEDGE FUND OS - HELHETLIG TESTPLAN (PowerShell Edition)
# Comprehensive System Validation Script
# Versjon: 1.0
# Dato: 2025-12-20
###############################################################################

$ErrorActionPreference = "Continue"

# Test results tracking
$script:TotalTests = 0
$script:PassedTests = 0
$script:FailedTests = 0

# Logging functions
function Write-TestInfo {
    param([string]$Message)
    Write-Host "[INFO] " -ForegroundColor Blue -NoNewline
    Write-Host $Message
}

function Write-TestSuccess {
    param([string]$Message)
    Write-Host "[✓] " -ForegroundColor Green -NoNewline
    Write-Host $Message
    $script:PassedTests++
    $script:TotalTests++
}

function Write-TestError {
    param([string]$Message)
    Write-Host "[✗] " -ForegroundColor Red -NoNewline
    Write-Host $Message
    $script:FailedTests++
    $script:TotalTests++
}

function Write-TestWarning {
    param([string]$Message)
    Write-Host "[!] " -ForegroundColor Yellow -NoNewline
    Write-Host $Message
}

function Write-TestSection {
    param([string]$Title)
    Write-Host ""
    Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Blue
    Write-Host "  $Title" -ForegroundColor Blue
    Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Blue
    Write-Host ""
}

# Summary
function Write-TestSummary {
    Write-Host ""
    Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Blue
    Write-Host "  TEST SAMMENDRAG" -ForegroundColor Blue
    Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Blue
    Write-Host "Total tester: $script:TotalTests"
    Write-Host "Bestått: $script:PassedTests" -ForegroundColor Green
    Write-Host "Feilet: $script:FailedTests" -ForegroundColor Red
    
    if ($script:FailedTests -eq 0) {
        Write-Host ""
        Write-Host "✅ ALLE TESTER BESTÅTT!" -ForegroundColor Green
        Write-Host "Systemet er verifisert funksjonelt og stabilt i dry-run mode." -ForegroundColor Green
        return $true
    } else {
        Write-Host ""
        Write-Host "❌ NOEN TESTER FEILET" -ForegroundColor Red
        Write-Host "Vennligst gå gjennom feilene ovenfor før du fortsetter." -ForegroundColor Red
        return $false
    }
}

###############################################################################
# TRINN 1 – KONTROLLER CONTAINER-HELSE
###############################################################################
function Test-ContainerHealth {
    Write-TestSection "TRINN 1 – CONTAINER HELSE"
    
    Write-TestInfo "Sjekker docker containers..."
    
    $requiredContainers = @(
        "backend",
        "ai_engine",
        "redis",
        "rl_optimizer",
        "strategy_evaluator",
        "strategy_evolution",
        "quantum_policy_memory",
        "global_policy_orchestrator",
        "federation_stub"
    )
    
    try {
        # Display container status
        Write-Host ""
        docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
        Write-Host ""
        
        # Check each required container
        $runningContainers = docker ps --format "{{.Names}}"
        
        foreach ($container in $requiredContainers) {
            if ($runningContainers -match "^$container$") {
                $status = docker ps --filter "name=^$container$" --format "{{.Status}}"
                if ($status -match "Up") {
                    Write-TestSuccess "Container '$container' er oppe og kjører"
                } else {
                    Write-TestError "Container '$container' kjører ikke (Status: $status)"
                }
            } else {
                Write-TestError "Container '$container' finnes ikke"
            }
        }
    } catch {
        Write-TestError "Docker er ikke tilgjengelig eller kjører ikke: $_"
    }
}

###############################################################################
# TRINN 2 – VALIDER INTERNE API-ER
###############################################################################
function Test-InternalAPIs {
    Write-TestSection "TRINN 2 – INTERNE API-ER"
    
    # Test Backend Health
    Write-TestInfo "Testing backend health endpoint..."
    try {
        $backendResponse = Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get -TimeoutSec 5
        $backendResponse | ConvertTo-Json -Depth 10
        
        if ($backendResponse.status -eq "ok") {
            Write-TestSuccess "Backend API returnerer status 'ok'"
        } else {
            Write-TestError "Backend API returnerer ikke 'ok' status"
        }
    } catch {
        Write-TestError "Kunne ikke nå backend API på http://localhost:8000/health: $_"
    }
    
    Write-Host ""
    
    # Test AI Engine Health
    Write-TestInfo "Testing AI Engine health endpoint..."
    try {
        $aiResponse = Invoke-RestMethod -Uri "http://localhost:8001/health" -Method Get -TimeoutSec 5
        $aiResponse | ConvertTo-Json -Depth 10
        
        if ($aiResponse.status -eq "ok") {
            Write-TestSuccess "AI Engine API returnerer status 'ok'"
        } else {
            Write-TestError "AI Engine API returnerer ikke 'ok' status"
        }
        
        if ($aiResponse.models) {
            Write-TestSuccess "AI Engine rapporterer lastede modeller"
        } else {
            Write-TestWarning "AI Engine rapporterer ingen modeller"
        }
    } catch {
        Write-TestError "Kunne ikke nå AI Engine API på http://localhost:8001/health: $_"
    }
}

###############################################################################
# TRINN 3 – REDIS DATAINTEGRITET
###############################################################################
function Test-RedisIntegrity {
    Write-TestSection "TRINN 3 – REDIS DATAINTEGRITET"
    
    # Check Redis memory usage
    Write-TestInfo "Sjekker Redis minnebruk..."
    try {
        $memoryInfo = docker exec redis redis-cli info memory 2>$null
        $memory = ($memoryInfo | Select-String "used_memory_human:").ToString().Split(':')[1].Trim()
        if ($memory) {
            Write-TestSuccess "Redis minnebruk: $memory"
        } else {
            Write-TestError "Kunne ikke hente Redis minnebruk"
        }
    } catch {
        Write-TestError "Feil ved henting av Redis minnebruk: $_"
    }
    
    Write-Host ""
    
    # Check for required keys
    Write-TestInfo "Sjekker tilstedeværelse av kritiske nøkler..."
    
    $requiredKeys = @(
        "governance_weights",
        "current_policy",
        "meta_best_strategy",
        "quantum_regime_forecast",
        "system_ssi"
    )
    
    try {
        $allKeys = docker exec redis redis-cli keys "*" 2>$null
        
        foreach ($key in $requiredKeys) {
            if ($allKeys -match $key) {
                Write-TestSuccess "Nøkkel '$key' finnes i Redis"
            } else {
                Write-TestWarning "Nøkkel '$key' mangler i Redis (kan være normalt ved oppstart)"
            }
        }
        
        Write-Host ""
        Write-TestInfo "Alle Redis nøkler:"
        $allKeys | ForEach-Object { Write-Host "  - $_" }
    } catch {
        Write-TestError "Feil ved henting av Redis nøkler: $_"
    }
}

###############################################################################
# TRINN 4 – AI-MODELL SANITY-CHECK
###############################################################################
function Test-AIModels {
    Write-TestSection "TRINN 4 – AI-MODELL SANITY-CHECK"
    
    Write-TestInfo "Tester AI-modeller (xgb, lgbm, nhits, patchtst)..."
    
    try {
        $modelTest = docker exec quantum_ai_engine python3 -c @"
try:
    from ai_engine.ensemble_manager import EnsembleManager
    e = EnsembleManager(enabled_models=['xgb','lgbm','nhits','patchtst'])
    result = {m: getattr(e, f'{m}_agent', None) is not None for m in e.enabled_models}
    print(result)
except Exception as ex:
    print(f'ERROR: {ex}')
"@ 2>&1
        
        Write-Host $modelTest
        
        if ($modelTest -match "ERROR") {
            Write-TestError "AI-modell test feilet"
        } else {
            $models = @('xgb', 'lgbm', 'nhits', 'patchtst')
            foreach ($model in $models) {
                if ($modelTest -match "'$model': True") {
                    Write-TestSuccess "Modell '$model' er lastet og tilgjengelig"
                } else {
                    Write-TestError "Modell '$model' er ikke lastet korrekt"
                }
            }
        }
    } catch {
        Write-TestError "Feil ved testing av AI-modeller: $_"
    }
}

###############################################################################
# TRINN 5 – REGIME-FORECAST VALIDERING
###############################################################################
function Test-RegimeForecast {
    Write-TestSection "TRINN 5 – REGIME-FORECAST VALIDERING"
    
    Write-TestInfo "Sjekker quantum_regime_forecast..."
    
    try {
        $forecast = docker exec redis redis-cli hgetall quantum_regime_forecast 2>$null
        
        if ($forecast) {
            $forecast | ForEach-Object { Write-Host $_ }
            
            if ($forecast -match "timestamp") {
                Write-TestSuccess "Regime forecast har tidsstempel"
                # Additional timestamp validation could be added here
            } else {
                Write-TestWarning "Ingen tidsstempel funnet i regime forecast"
            }
            
            if ($forecast -match "(bull|bear|neutral|volatile)") {
                Write-TestSuccess "Regime forecast inneholder regime-sannsynligheter"
            } else {
                Write-TestWarning "Ingen regime-sannsynligheter funnet"
            }
        } else {
            Write-TestWarning "quantum_regime_forecast finnes ikke eller er tom"
        }
    } catch {
        Write-TestError "Feil ved henting av regime forecast: $_"
    }
}

###############################################################################
# TRINN 6 – GOVERNANCE OG SSI
###############################################################################
function Test-GovernanceSSI {
    Write-TestSection "TRINN 6 – GOVERNANCE OG SSI"
    
    # Check System Stress Index
    Write-TestInfo "Sjekker System Stress Index (SSI)..."
    try {
        $ssi = docker exec redis redis-cli get system_ssi 2>$null
        
        if ($ssi) {
            Write-TestSuccess "SSI verdi: $ssi"
            
            $ssiValue = [double]$ssi
            if ($ssiValue -ge -2 -and $ssiValue -le 2) {
                Write-TestSuccess "SSI er innenfor gyldig område (-2 til 2)"
            } else {
                Write-TestWarning "SSI er utenfor normalt område (-2 til 2)"
            }
        } else {
            Write-TestWarning "system_ssi finnes ikke i Redis"
        }
    } catch {
        Write-TestWarning "Kunne ikke validere SSI: $_"
    }
    
    Write-Host ""
    
    # Check Governance Weights
    Write-TestInfo "Sjekker governance_weights..."
    try {
        $weights = docker exec redis redis-cli hgetall governance_weights 2>$null
        
        if ($weights) {
            $weights | ForEach-Object { Write-Host $_ }
            Write-TestSuccess "Governance weights finnes"
        } else {
            Write-TestWarning "governance_weights finnes ikke i Redis"
        }
    } catch {
        Write-TestError "Feil ved henting av governance weights: $_"
    }
}

###############################################################################
# TRINN 7 – FULL END-TO-END SIMULERING
###############################################################################
function Test-EndToEnd {
    Write-TestSection "TRINN 7 – END-TO-END SIMULERING"
    
    Write-TestInfo "Sender syntetiske signal-kall for BTC, ETH, SOL..."
    
    $symbols = @("BTCUSDT", "ETHUSDT", "SOLUSDT")
    
    foreach ($symbol in $symbols) {
        Write-Host ""
        Write-TestInfo "Testing signal for $symbol..."
        
        try {
            $body = @{ symbol = $symbol } | ConvertTo-Json
            $response = Invoke-RestMethod -Uri "http://localhost:8001/api/ai/signal" `
                -Method Post `
                -Body $body `
                -ContentType "application/json" `
                -TimeoutSec 10
            
            $response | ConvertTo-Json -Depth 5
            
            if ($response.action -and $response.action -match "^(BUY|SELL|HOLD)$") {
                Write-TestSuccess "$symbol`: Action=$($response.action), Confidence=$($response.confidence)"
                
                if ($response.confidence -gt 0.4) {
                    Write-TestSuccess "$symbol`: Confidence > 0.4 ✓"
                } else {
                    Write-TestWarning "$symbol`: Confidence < 0.4"
                }
            } else {
                Write-TestError "$symbol`: Ingen gyldig action i response"
            }
        } catch {
            Write-TestError "$symbol`: Feil ved API-kall: $_"
        }
    }
}

###############################################################################
# TRINN 8 – EVALUÉR LOGGENE
###############################################################################
function Test-Logs {
    Write-TestSection "TRINN 8 – LOGG-EVALUERING"
    
    Write-TestInfo "Søker etter kritiske feil i logger (siste 1000 linjer)..."
    
    try {
        $logs = docker compose logs --tail=1000 2>$null
        $errors = $logs | Select-String -Pattern "ERROR|CRITICAL|Exception"
        
        if ($errors.Count -eq 0) {
            Write-TestSuccess "Ingen kritiske feil funnet i logger"
        } else {
            Write-TestWarning "Fant $($errors.Count) linjer med ERROR/CRITICAL/Exception"
            Write-Host ""
            Write-TestInfo "Viser de siste 10 feilene:"
            $errors | Select-Object -Last 10 | ForEach-Object { Write-Host $_.Line }
        }
    } catch {
        Write-TestWarning "Kunne ikke lese logger: $_"
    }
}

###############################################################################
# MAIN EXECUTION
###############################################################################
function Main {
    Write-TestSection "AI HEDGE FUND OS - HELHETLIG TESTPLAN"
    Write-TestInfo "Starter omfattende systemvalidering..."
    Write-TestInfo "Dato: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    
    # Run all tests
    Test-ContainerHealth
    Test-InternalAPIs
    Test-RedisIntegrity
    Test-AIModels
    Test-RegimeForecast
    Test-GovernanceSSI
    Test-EndToEnd
    Test-Logs
    
    # Print summary
    $success = Write-TestSummary
    
    if ($success) {
        Write-TestSection "🏁 NESTE STEG: MANUELL TILKOBLING TIL BINANCE TESTNET"
        Write-Host ""
        Write-Host "For å koble til Binance Testnet:"
        Write-Host ""
        Write-Host "1. Gå til https://testnet.binance.vision og opprett testnet API-nøkler"
        Write-Host ""
        Write-Host "2. Oppdater .env-filen med dine testnet-credentials:"
        Write-Host "   BINANCE_API_KEY=din_testnet_key"
        Write-Host "   BINANCE_API_SECRET=din_testnet_secret"
        Write-Host "   BINANCE_BASE_URL=https://testnet.binance.vision/api"
        Write-Host "   MODE=testnet"
        Write-Host ""
        Write-Host "3. Start systemet på nytt:"
        Write-Host "   docker compose down && docker compose up -d"
        Write-Host ""
        Write-Host "⚠️  VIKTIG: Bruk kun små posisjoner på testnet!" -ForegroundColor Yellow
        Write-Host "⚠️  Aldri bruk live-nøkler før full bekreftet oppførsel!" -ForegroundColor Yellow
        Write-Host ""
    }
    
    return $success
}

# Run main function
$result = Main
if (-not $result) {
    exit 1
}
