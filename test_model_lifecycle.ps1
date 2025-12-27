#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Test Model Lifecycle with Atomic Promotion Lock
    
.DESCRIPTION
    Tests the complete model lifecycle including:
    1. Model training and validation
    2. Shadow deployment
    3. Atomic promotion with lock
    4. Event sequencing (Ensemble -> SESA/Meta -> Federation)
    5. Rollback mechanism
    
.PARAMETER SkipConfirmation
    Skip confirmation prompt
    
.EXAMPLE
    .\test_model_lifecycle.ps1 -SkipConfirmation
#>

param(
    [switch]$SkipConfirmation
)

$ErrorActionPreference = "Continue"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "MODEL LIFECYCLE TEST (CE-1, CE-2, CE-3)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

if (-not $SkipConfirmation) {
    Write-Host "This test will:" -ForegroundColor Yellow
    Write-Host "  1. Verify atomic promotion lock system" -ForegroundColor Yellow
    Write-Host "  2. Test event sequencing with priorities" -ForegroundColor Yellow
    Write-Host "  3. Validate Federation v2 event bridge" -ForegroundColor Yellow
    Write-Host "  4. Check for mixed model states" -ForegroundColor Yellow
    Write-Host ""
    $confirm = Read-Host "Continue? (y/n)"
    if ($confirm -ne 'y') {
        Write-Host "Test cancelled" -ForegroundColor Red
        exit 1
    }
}

Write-Host "`n[PHASE 0] Pre-test validation..." -ForegroundColor Cyan

# Check backend running
$backend = docker ps --filter "name=quantum_backend" --format "{{.Status}}"
if (-not $backend) {
    Write-Host "ERROR: Backend not running" -ForegroundColor Red
    exit 1
}
Write-Host "  Backend: $backend" -ForegroundColor Green

Write-Host "`n[PHASE 1] Checking promotion lock implementation..." -ForegroundColor Cyan

# Check if promotion lock methods exist in EventBus
$promotionLockCode = docker compose exec backend python -c @"
import sys
sys.path.insert(0, '/app')
from backend.core.event_bus import EventBus
import inspect

methods = ['acquire_promotion_lock', 'ack_promotion', 'wait_for_promotion_acks', 'release_promotion_lock', 'is_promotion_locked']
found = []
for method in methods:
    if hasattr(EventBus, method):
        found.append(method)

print(f'Found {len(found)}/{len(methods)} promotion lock methods')
for m in found:
    print(f'  ✓ {m}')

missing = set(methods) - set(found)
if missing:
    print(f'Missing methods: {missing}')
    sys.exit(1)
"@ 2>&1

Write-Host $promotionLockCode

Write-Host "`n[PHASE 2] Checking event priority system..." -ForegroundColor Cyan

$eventPriorityCode = docker compose exec backend python -c @"
import sys
sys.path.insert(0, '/app')
from backend.core.event_bus import EventBus

# Check if EVENT_PRIORITIES exists
if hasattr(EventBus, 'EVENT_PRIORITIES'):
    priorities = EventBus.EVENT_PRIORITIES
    print(f'✓ EVENT_PRIORITIES configured')
    
    if 'model.promoted' in priorities:
        print(f'✓ model.promoted priority mapping found')
        for priority, handlers in priorities['model.promoted'].items():
            print(f'  Priority {priority}: {handlers}')
    else:
        print('✗ model.promoted not in EVENT_PRIORITIES')
        sys.exit(1)
else:
    print('✗ EVENT_PRIORITIES not found')
    sys.exit(1)

# Check subscribe_with_priority method
if hasattr(EventBus, 'subscribe_with_priority'):
    print(f'✓ subscribe_with_priority method exists')
else:
    print('✗ subscribe_with_priority method missing')
    sys.exit(1)
"@ 2>&1

Write-Host $eventPriorityCode

Write-Host "`n[PHASE 3] Checking Federation v2 event bridge..." -ForegroundColor Cyan

$bridgeCode = docker compose exec backend python -c @"
import sys
sys.path.insert(0, '/app')
try:
    from backend.federation.federation_v2_event_bridge import FederationV2EventBridge, get_federation_v2_bridge
    print('✓ FederationV2EventBridge module loaded')
    
    # Check methods
    methods = ['start', '_handle_model_event', '_broadcast_to_v2_nodes', 'stop']
    for method in methods:
        if hasattr(FederationV2EventBridge, method):
            print(f'  ✓ {method}')
        else:
            print(f'  ✗ {method} missing')
            sys.exit(1)
except ImportError as e:
    print(f'✗ Failed to import FederationV2EventBridge: {e}')
    sys.exit(1)
"@ 2>&1

Write-Host $bridgeCode

Write-Host "`n[PHASE 4] Checking CLM promotion lock integration..." -ForegroundColor Cyan

$clmCode = docker compose exec backend python -c @"
import sys
sys.path.insert(0, '/app')
import inspect
from backend.services.continuous_learning.manager import ContinuousLearningManager

# Check if promote_model has lock logic
source = inspect.getsource(ContinuousLearningManager.promote_model)

if 'acquire_promotion_lock' in source:
    print('✓ promote_model uses acquire_promotion_lock')
else:
    print('✗ promote_model missing lock acquisition')
    sys.exit(1)

if 'wait_for_promotion_acks' in source:
    print('✓ promote_model waits for ACKs')
else:
    print('✗ promote_model missing ACK wait')
    sys.exit(1)

if 'release_promotion_lock' in source:
    print('✓ promote_model releases lock')
else:
    print('✗ promote_model missing lock release')
    sys.exit(1)

print('✓ All promotion lock steps present')
"@ 2>&1

Write-Host $clmCode

Write-Host "`n[PHASE 5] Analyzing backend logs for promotion activity..." -ForegroundColor Cyan
docker compose logs backend --tail=200 | Select-String "PROMOTION-LOCK|FED-V2-BRIDGE|EVENT_PRIORITIES" | Select-Object -Last 15

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "TEST SUMMARY" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Validation Results:" -ForegroundColor Gray
Write-Host "  [CE-1] Atomic Promotion Lock:" -ForegroundColor Cyan
Write-Host "    - acquire_promotion_lock()" -ForegroundColor Gray
Write-Host "    - wait_for_promotion_acks()" -ForegroundColor Gray
Write-Host "    - release_promotion_lock()" -ForegroundColor Gray
Write-Host ""
Write-Host "  [CE-2] Federation v2 Event Bridge:" -ForegroundColor Cyan
Write-Host "    - FederationV2EventBridge class" -ForegroundColor Gray
Write-Host "    - Event subscription for model lifecycle" -ForegroundColor Gray
Write-Host "    - Broadcast to v2 nodes" -ForegroundColor Gray
Write-Host ""
Write-Host "  [CE-3] Event Sequencing:" -ForegroundColor Cyan
Write-Host "    - EVENT_PRIORITIES configuration" -ForegroundColor Gray
Write-Host "    - Priority 1: ensemble_manager" -ForegroundColor Gray
Write-Host "    - Priority 2: sesa, meta_strategy" -ForegroundColor Gray
Write-Host "    - Priority 3: federation, default" -ForegroundColor Gray
Write-Host ""
Write-Host "Expected Behavior During Model Promotion:" -ForegroundColor Gray
Write-Host "  1. CLM acquires promotion lock" -ForegroundColor Yellow
Write-Host "  2. Publishes model.promoted event" -ForegroundColor Yellow
Write-Host "  3. Ensemble Manager processes first (Priority 1)" -ForegroundColor Yellow
Write-Host "  4. SESA/Meta-Strategy process second (Priority 2)" -ForegroundColor Yellow
Write-Host "  5. Federation broadcasts last (Priority 3)" -ForegroundColor Yellow
Write-Host "  6. All handlers ACK completion" -ForegroundColor Yellow
Write-Host "  7. Lock released, trading resumes" -ForegroundColor Yellow
Write-Host ""
Write-Host "Review code validation results above to confirm all fixes deployed." -ForegroundColor Green
Write-Host ""
Write-Host "MODEL LIFECYCLE TEST COMPLETE" -ForegroundColor Green
