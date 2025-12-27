# EPIC-OBS-001 Phase 2 Validation Script
# Quick test to verify observability module doesn't break service startup

Write-Host "=== EPIC-OBS-001 PHASE 2 VALIDATION ===" -ForegroundColor Cyan

# Check if observability module exists
$obsPath = "backend\infra\observability"
if (Test-Path $obsPath) {
    Write-Host "[OK] Observability module created at $obsPath" -ForegroundColor Green
    Get-ChildItem $obsPath | ForEach-Object { Write-Host "  - $($_.Name)" }
} else {
    Write-Host "[FAIL] Observability module NOT found" -ForegroundColor Red
    exit 1
}

# Check Python syntax
Write-Host "`n[CHECK] Validating Python syntax..." -ForegroundColor Yellow
$files = @(
    "backend\infra\observability\__init__.py",
    "backend\infra\observability\config.py",
    "backend\infra\observability\logging.py",
    "backend\infra\observability\tracing.py",
    "backend\infra\observability\metrics.py"
)

foreach ($file in $files) {
    python -m py_compile $file 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  [OK] $file" -ForegroundColor Green
    } else {
        Write-Host "  [FAIL] Syntax error in $file" -ForegroundColor Red
    }
}

# Test import
Write-Host "`n[CHECK] Testing module import..." -ForegroundColor Yellow
$importTest = @"
import sys
sys.path.insert(0, '.')
try:
    from backend.infra.observability import init_observability, get_logger, metrics
    print('[OK] Module imports successfully')
except Exception as e:
    print(f'[FAIL] Import error: {e}')
    sys.exit(1)
"@

python -c $importTest
if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] Module imports without errors" -ForegroundColor Green
} else {
    Write-Host "[FAIL] Module import failed" -ForegroundColor Red
    exit 1
}

Write-Host "`n=== VALIDATION COMPLETE ===" -ForegroundColor Cyan
Write-Host @"

NEXT STEPS:
1. Install OpenTelemetry dependencies:
   pip install opentelemetry-api opentelemetry-sdk opentelemetry-instrumentation-fastapi opentelemetry-exporter-otlp python-json-logger

2. Test AI Engine service startup:
   cd microservices/ai_engine
   python main.py

3. Verify structured logs appear (JSON format with service_name field)

4. Check /metrics endpoint (if service starts):
   curl http://localhost:8001/metrics

TODO FOR PHASE 3+:
 [ ] Wire /metrics endpoint in all services
 [ ] Add /health/live and /health/ready endpoints
 [ ] Instrument Execution service
 [ ] Instrument Risk, Portfolio, RL Training services
 [ ] Deploy Jaeger/Tempo for trace collection
 [ ] Configure Prometheus scraping
 [ ] Set up Grafana dashboards

"@
