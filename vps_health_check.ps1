# Quantum Trader V3 - Post-Migration Health Validation
# Date: 2025-12-17

$ErrorActionPreference = "Continue"

Write-Host "`n╔═══════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║  QUANTUM TRADER V3 - VPS RECOVERY HEALTH CHECK              ║" -ForegroundColor Yellow
Write-Host "╚═══════════════════════════════════════════════════════════════╝`n" -ForegroundColor Cyan

# Initialize report structure
$report = @{
    timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    activeModules = @()
    missingModules = @()
    brokenDependencies = @()
    suggestions = @()
    dockerStatus = "Unknown"
    containerStatus = @{}
    logAnalysis = @{}
}

# Step 1: Check Docker availability
Write-Host "[1/6] Checking Docker status..." -ForegroundColor Cyan
$dockerService = Get-Service "com.docker.service" -ErrorAction SilentlyContinue
if ($dockerService -and $dockerService.Status -eq "Running") {
    $report.dockerStatus = "Running"
    Write-Host "  ✓ Docker Desktop is running" -ForegroundColor Green
    
    # Give Docker daemon time to be ready
    Start-Sleep -Seconds 5
    
    # Check if docker command works
    $dockerTest = docker ps 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  ⚠ Docker daemon not ready yet, waiting..." -ForegroundColor Yellow
        Start-Sleep -Seconds 30
        $dockerTest = docker ps 2>&1
    }
} else {
    $report.dockerStatus = "Stopped"
    Write-Host "  ✗ Docker Desktop is not running" -ForegroundColor Red
    Write-Host "  → Starting Docker Desktop..." -ForegroundColor Yellow
    Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe" -ErrorAction SilentlyContinue
    Write-Host "  → Waiting 90 seconds for initialization..." -ForegroundColor Yellow
    Start-Sleep -Seconds 90
    
    $dockerTest = docker ps 2>&1
    if ($LASTEXITCODE -eq 0) {
        $report.dockerStatus = "Started"
        Write-Host "  ✓ Docker Desktop started successfully" -ForegroundColor Green
    } else {
        $report.dockerStatus = "Failed"
        Write-Host "  ✗ Docker Desktop failed to start" -ForegroundColor Red
    }
}

# Step 2: Check container status
Write-Host "`n[2/6] Checking container status..." -ForegroundColor Cyan
$containers = docker ps -a --format "{{.Names}}|{{.Status}}|{{.State}}" 2>&1 | Out-String

if ($LASTEXITCODE -eq 0 -and $containers) {
    $containerLines = $containers -split "`n" | Where-Object { $_ -match "quantum" }
    
    if ($containerLines.Count -eq 0) {
        Write-Host "  ⚠ No containers found - need to build and start" -ForegroundColor Yellow
        $report.containerStatus["quantum_backend"] = "Not Created"
    } else {
        foreach ($line in $containerLines) {
            if ($line -match "^([^|]+)\|([^|]+)\|([^|]+)") {
                $name = $matches[1]
                $status = $matches[2]
                $state = $matches[3]
                $report.containerStatus[$name] = "$state - $status"
                
                if ($state -eq "running") {
                    Write-Host "  ✓ $name is running" -ForegroundColor Green
                } elseif ($state -eq "exited") {
                    Write-Host "  ✗ $name has exited" -ForegroundColor Red
                } else {
                    Write-Host "  ⚠ $name is $state" -ForegroundColor Yellow
                }
            }
        }
    }
} else {
    Write-Host "  ✗ Cannot query containers (Docker not ready)" -ForegroundColor Red
    $report.containerStatus["error"] = "Docker daemon not accessible"
}

# Step 3: Analyze logs for critical modules
Write-Host "`n[3/6] Analyzing logs for AI module activation..." -ForegroundColor Cyan

$criticalModules = @(
    @{Name = "ExitBrainV3"; Pattern = "ExitBrainV3.*activated|exit_brain_v3.*initialized|EXIT_BRAIN_V3.*ready"},
    @{Name = "TPOptimizerV3"; Pattern = "TPOptimizerV3.*metrics.*synced|tp_optimizer_v3.*active|TP_OPTIMIZER.*initialized"},
    @{Name = "RLAgentV3"; Pattern = "RLAgentV3.*listening|rl_v3.*feedback|RL_AGENT.*ready"},
    @{Name = "RiskGateV3"; Pattern = "RiskGateV3.*active|risk_gate_v3.*initialized|RISK_GATE.*enabled"}
)

$backendRunning = $report.containerStatus.Keys | Where-Object { $_ -eq "quantum_backend" -and $report.containerStatus[$_] -match "running" }

if ($backendRunning) {
    Write-Host "  Fetching logs from quantum_backend..." -ForegroundColor Gray
    $logs = docker logs quantum_backend --tail 200 2>&1 | Out-String
    
    foreach ($module in $criticalModules) {
        $moduleName = $module.Name
        $pattern = $module.Pattern
        
        if ($logs -match $pattern) {
            Write-Host "  ✓ [$moduleName] activated" -ForegroundColor Green
            $report.activeModules += $moduleName
            $report.logAnalysis[$moduleName] = "Active"
        } else {
            Write-Host "  ⚠ [$moduleName] not found in logs" -ForegroundColor Yellow
            $report.missingModules += $moduleName
            $report.logAnalysis[$moduleName] = "Silent"
        }
    }
    
    # Check for errors
    if ($logs -match "ModuleNotFoundError|ImportError") {
        Write-Host "  ✗ Import errors detected in logs" -ForegroundColor Red
        $errorLines = $logs -split "`n" | Where-Object { $_ -match "ModuleNotFoundError|ImportError" }
        foreach ($err in $errorLines | Select-Object -First 5) {
            Write-Host "    • $($err.Trim())" -ForegroundColor Red
            $report.brokenDependencies += $err.Trim()
        }
    }
} else {
    Write-Host "  ⚠ quantum_backend not running - cannot analyze logs" -ForegroundColor Yellow
    foreach ($module in $criticalModules) {
        $report.missingModules += $module.Name
        $report.logAnalysis[$module.Name] = "Container not running"
    }
}

# Step 4: Check file structure
Write-Host "`n[4/6] Verifying module file structure..." -ForegroundColor Cyan

$expectedPaths = @{
    "ExitBrainV3" = "backend\domains\exits\exit_brain_v3\dynamic_executor.py"
    "RLAgentV3" = "backend\domains\learning\rl_v3\rl_manager_v3.py"
    "TPOptimizerV3" = "backend\services\monitoring\tp_optimizer_v3.py"
    "RiskGateV3" = "backend\risk\risk_gate_v3.py"
}

foreach ($moduleName in $expectedPaths.Keys) {
    $path = Join-Path "C:\quantum_trader" $expectedPaths[$moduleName]
    if (Test-Path $path) {
        Write-Host "  ✓ $moduleName files present" -ForegroundColor Green
    } else {
        Write-Host "  ✗ $moduleName files missing: $($expectedPaths[$moduleName])" -ForegroundColor Red
        if ($report.missingModules -notcontains $moduleName) {
            $report.missingModules += $moduleName
        }
        $report.suggestions += "Restore $($expectedPaths[$moduleName]) from local backup"
    }
}

# Step 5: Run smoke tests (if container is running)
Write-Host "`n[5/6] Running smoke tests..." -ForegroundColor Cyan

if ($backendRunning) {
    Write-Host "  Executing pytest smoke tests..." -ForegroundColor Gray
    $smokeTest = docker exec quantum_backend bash -c "pytest -q tests/smoke 2>&1 || echo 'TESTS_FAILED'" 2>&1 | Out-String
    
    if ($smokeTest -match "passed") {
        $passCount = [regex]::Match($smokeTest, "(\d+) passed").Groups[1].Value
        Write-Host "  ✓ Smoke tests passed ($passCount tests)" -ForegroundColor Green
        $report.smokeTests = "Passed ($passCount)"
    } elseif ($smokeTest -match "no tests ran|cannot find|not found") {
        Write-Host "  ⚠ No smoke tests found" -ForegroundColor Yellow
        $report.smokeTests = "Not Found"
    } else {
        Write-Host "  ✗ Smoke tests failed or errored" -ForegroundColor Red
        $report.smokeTests = "Failed"
        $errorLines = $smokeTest -split "`n" | Where-Object { $_ -match "FAILED|ERROR" }
        foreach ($err in $errorLines | Select-Object -First 3) {
            Write-Host "    • $($err.Trim())" -ForegroundColor Red
        }
    }
} else {
    Write-Host "  ⚠ Container not running - skipping smoke tests" -ForegroundColor Yellow
    $report.smokeTests = "Skipped (container not running)"
}

# Step 6: Check configuration files
Write-Host "`n[6/6] Verifying configuration files..." -ForegroundColor Cyan

$configFiles = @{
    ".env" = "C:\quantum_trader\.env"
    "docker-compose.yml" = "C:\quantum_trader\docker-compose.yml"
    "activation.yaml" = "C:\quantum_trader\activation.yaml"
    "go_live.yaml" = "C:\quantum_trader\config\go_live.yaml"
}

foreach ($configName in $configFiles.Keys) {
    $path = $configFiles[$configName]
    if (Test-Path $path) {
        Write-Host "  ✓ $configName present" -ForegroundColor Green
        
        # Check for GO_LIVE setting
        if ($configName -eq ".env") {
            $content = Get-Content $path -Raw
            if ($content -match "GO_LIVE\s*=\s*true") {
                Write-Host "    → GO_LIVE=true confirmed" -ForegroundColor Green
            } else {
                Write-Host "    ⚠ GO_LIVE not set to true" -ForegroundColor Yellow
                $report.suggestions += "Set GO_LIVE=true in .env file"
            }
        }
    } else {
        Write-Host "  ✗ $configName missing" -ForegroundColor Red
        $report.suggestions += "Restore $configName from local backup"
    }
}

# Generate markdown report
Write-Host "`n[*] Generating status report..." -ForegroundColor Cyan

$reportPath = "C:\quantum_trader\VPS_RECOVERY_REPORT.md"
$reportContent = @"
# Quantum Trader V3 - VPS Recovery Health Report

**Generated:** $($report.timestamp)  
**Docker Status:** $($report.dockerStatus)  
**Environment:** Production (VPS)

---

## Executive Summary

"@

# Determine overall status
$overallStatus = "Unknown"
$statusIcon = "⚠️"

if ($report.activeModules.Count -ge 3 -and $report.brokenDependencies.Count -eq 0) {
    $overallStatus = "Healthy"
    $statusIcon = "✅"
} elseif ($report.activeModules.Count -ge 1) {
    $overallStatus = "Partially Operational"
    $statusIcon = "⚠️"
} else {
    $overallStatus = "Needs Attention"
    $statusIcon = "❌"
}

$reportContent += @"
**Overall Status:** $statusIcon **$overallStatus**

- Active Modules: $($report.activeModules.Count)/4
- Missing Modules: $($report.missingModules.Count)
- Broken Dependencies: $($report.brokenDependencies.Count)
- Smoke Tests: $($report.smokeTests)

---

## ✅ Active Modules

"@

if ($report.activeModules.Count -gt 0) {
    foreach ($module in $report.activeModules) {
        $reportContent += "- ✅ **$module** - Activated and running`n"
    }
} else {
    $reportContent += "*No modules confirmed active (container may not be running)*`n"
}

$reportContent += @"

---

## ⚠️ Missing/Silent Modules

"@

if ($report.missingModules.Count -gt 0) {
    foreach ($module in $report.missingModules) {
        $status = $report.logAnalysis[$module]
        $reportContent += "- ⚠️ **$module** - $status`n"
    }
} else {
    $reportContent += "*All critical modules detected in logs*`n"
}

$reportContent += @"

---

## ❌ Broken Dependencies

"@

if ($report.brokenDependencies.Count -gt 0) {
    foreach ($dep in $report.brokenDependencies) {
        $reportContent += "- ❌ ``$dep```n"
    }
} else {
    $reportContent += "*No import errors detected*`n"
}

$reportContent += @"

---

## Container Status

"@

if ($report.containerStatus.Count -gt 0) {
    foreach ($container in $report.containerStatus.Keys) {
        $status = $report.containerStatus[$container]
        $icon = if ($status -match "running") { "🟢" } elseif ($status -match "exited") { "🔴" } else { "🟡" }
        $reportContent += "- $icon **$container**: $status`n"
    }
} else {
    $reportContent += "*No containers detected - need to build and start*`n"
}

$reportContent += @"

---

## 💡 Suggestions & Recovery Actions

"@

if ($report.suggestions.Count -gt 0) {
    foreach ($suggestion in $report.suggestions) {
        $reportContent += "1. $suggestion`n"
    }
} else {
    $reportContent += "*No immediate actions required*`n"
}

# Add standard recovery procedures
$reportContent += @"

### If Modules Are Missing:

1. **Check container logs:**
   ``````powershell
   docker logs quantum_backend --tail 100
   ``````

2. **Verify PYTHONPATH configuration:**
   ``````powershell
   docker exec quantum_backend env | Select-String "PYTHONPATH"
   ``````
   Should show: ``PYTHONPATH=/app/backend``

3. **Test imports manually:**
   ``````powershell
   docker exec quantum_backend python3 -c "from domains.exits.exit_brain_v3 import dynamic_executor; print('OK')"
   ``````

4. **Rebuild containers if needed:**
   ``````powershell
   docker compose build --no-cache backend
   docker compose up -d backend
   ``````

### If Container Is Not Running:

1. **Build and start:**
   ``````powershell
   cd C:\quantum_trader
   docker compose build backend
   docker compose up -d backend
   ``````

2. **Monitor startup:**
   ``````powershell
   docker logs quantum_backend --follow
   ``````

---

## Next Steps

"@

if ($overallStatus -eq "Healthy") {
    $reportContent += @"
✅ **System is healthy - ready for production operations**

- All critical modules are active
- No broken dependencies detected
- Ready to receive Sonnet TP optimizer prompts
- Monitor logs for 24 hours to ensure stability

**Start remaining services:**
``````powershell
docker compose up -d
``````
"@
} elseif ($overallStatus -eq "Partially Operational") {
    $reportContent += @"
⚠️ **System needs attention before full production**

- Some modules are silent or missing
- Review container logs for startup issues
- Verify all module files are present
- Run smoke tests to identify specific failures

**Recommended actions:**
1. Check logs: ``docker logs quantum_backend --tail 200``
2. Verify files: Check paths listed in "Missing/Silent Modules"
3. Rebuild if needed: ``docker compose build --no-cache backend``
4. Re-run this health check
"@
} else {
    $reportContent += @"
❌ **System requires immediate attention**

- Containers are not running or have critical errors
- Module files may be missing
- Configuration may be incorrect

**Critical actions:**
1. **Start Docker Desktop** (if not running)
2. **Build containers:** ``docker compose build backend``
3. **Start backend:** ``docker compose up -d backend``
4. **Check logs:** ``docker logs quantum_backend --tail 100``
5. **Restore missing files** from local backup (see suggestions above)
6. **Re-run health check:** ``.\vps_health_check.ps1``

**If containers fail to start:**
- Review ``DOCKER_BUILD_INSTRUCTIONS.md``
- Check ``build_error.log`` for build failures
- Verify all configuration files are present
"@
}

$reportContent += @"

---

## Documentation References

- [DOCKER_BUILD_INSTRUCTIONS.md](DOCKER_BUILD_INSTRUCTIONS.md) - Detailed build guide
- [RUNTIME_CONFIG_QUICKREF.md](RUNTIME_CONFIG_QUICKREF.md) - Configuration reference
- [VPS_MIGRATION_FOLDER_AUDIT.md](VPS_MIGRATION_FOLDER_AUDIT.md) - Folder structure audit

---

**Report generated by:** vps_health_check.ps1  
**Last updated:** $($report.timestamp)  
**Environment:** Windows + Docker Desktop  
**VPS Migration Status:** Post-migration validation
"@

# Save report
$reportContent | Out-File -FilePath $reportPath -Encoding UTF8
Write-Host "  ✓ Report saved to: $reportPath" -ForegroundColor Green

# Also try to save to /mnt/status if WSL is available
try {
    $wslTest = wsl --status 2>&1
    if ($LASTEXITCODE -eq 0) {
        wsl bash -c "mkdir -p /mnt/status 2>/dev/null; cat > /mnt/status/VPS_RECOVERY_REPORT.md" -InputObject $reportContent 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  ✓ Report also saved to: /mnt/status/VPS_RECOVERY_REPORT.md" -ForegroundColor Green
        }
    }
} catch {
    # WSL not available, skip
}

# Final summary
Write-Host "`n╔═══════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║  VALIDATION COMPLETE                                        ║" -ForegroundColor Yellow
Write-Host "╚═══════════════════════════════════════════════════════════════╝`n" -ForegroundColor Cyan

# One-line summary
$summaryLine = "Quantum Trader v3 environment validated — "
if ($overallStatus -eq "Healthy") {
    Write-Host "$summaryLine ready for Sonnet TP optimizer prompts. ✅" -ForegroundColor Green
} elseif ($overallStatus -eq "Partially Operational") {
    Write-Host "$summaryLine partially ready, some modules need attention. ⚠️" -ForegroundColor Yellow
} else {
    Write-Host "$summaryLine needs setup before production use. ❌" -ForegroundColor Red
}

Write-Host "`nDetailed report: $reportPath`n" -ForegroundColor Cyan
