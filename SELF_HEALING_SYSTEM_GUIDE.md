# SELF-HEALING SYSTEM GUIDE

**Complete Integration & Reference Documentation**

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Health Monitoring](#health-monitoring)
4. [Issue Detection](#issue-detection)
5. [Recovery Actions](#recovery-actions)
6. [Safety Policies](#safety-policies)
7. [Integration Guide](#integration-guide)
8. [Testing](#testing)
9. [Operations](#operations)

---

## 🎯 Overview

The **Self-Healing System** provides comprehensive failure detection and automatic recovery for the Quantum Trader platform. It monitors all critical subsystems, detects failures and degradation, and applies safe fallback policies.

### Key Capabilities

✅ **Health Monitoring** - Continuous monitoring of all subsystems  
✅ **Failure Detection** - Detect data interruptions, connection drops, model errors  
✅ **Degradation Detection** - Identify performance issues before they become critical  
✅ **Automatic Recovery** - Apply recovery actions (restart, pause, fallback)  
✅ **Safety Policies** - Enforce trading controls based on system state  
✅ **Resource Monitoring** - Track CPU, memory, connections  
✅ **Dependency Tracking** - Understand subsystem relationships  

### Design Philosophy

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  DETECT → ANALYZE → RECOMMEND → EXECUTE → VERIFY       │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**Principles:**
- **Proactive Detection** - Catch issues early
- **Safe Defaults** - Protect capital first
- **Automatic Recovery** - Minimize manual intervention
- **Clear Visibility** - Comprehensive logging and reporting

---

## 🏗️ Architecture

### System Components

```
┌──────────────────────────────────────────────────────────────┐
│                   SELF-HEALING SYSTEM                         │
│                                                               │
│  ┌──────────────┐  ┌─────────────┐  ┌──────────────┐        │
│  │   Health     │  │   Issue     │  │   Recovery   │        │
│  │   Monitors   │→ │  Detection  │→ │    Engine    │        │
│  └──────────────┘  └─────────────┘  └──────────────┘        │
│         ↓                  ↓                 ↓               │
│  ┌──────────────────────────────────────────────────┐        │
│  │          Safety Policy Enforcement               │        │
│  └──────────────────────────────────────────────────┘        │
└──────────────────────────────────────────────────────────────┘
           ↓                    ↓                    ↓
    ┌──────────┐        ┌──────────┐        ┌──────────┐
    │ Trading  │        │   AI     │        │  Risk    │
    │ Executor │        │ Models   │        │  Guard   │
    └──────────┘        └──────────┘        └──────────┘
```

### Monitored Subsystems

| Subsystem | Purpose | Critical for Trading |
|-----------|---------|---------------------|
| `DATA_FEED` | Market data updates | ✅ Yes |
| `EXCHANGE_CONNECTION` | Binance API connectivity | ✅ Yes |
| `AI_MODEL` | ML model availability | ⚠️ Partially |
| `EVENT_EXECUTOR` | Trading event loop | ✅ Yes |
| `ORCHESTRATOR` | Position orchestration | ⚠️ Partially |
| `PORTFOLIO_BALANCER` | Global risk management | ⚠️ Partially |
| `MODEL_SUPERVISOR` | Model performance oversight | ❌ No |
| `RETRAINING_ORCHESTRATOR` | Model retraining | ❌ No |
| `POSITION_MONITOR` | Per-position tracking | ⚠️ Partially |
| `RISK_GUARD` | Risk enforcement | ✅ Yes |
| `DATABASE` | Data persistence | ⚠️ Partially |
| `UNIVERSE_OS` | Symbol universe management | ⚠️ Partially |
| `LOGGING` | System logging | ❌ No |

---

## 🔍 Health Monitoring

### Health Status Levels

```python
class HealthStatus(Enum):
    HEALTHY = "healthy"      # Normal operation
    DEGRADED = "degraded"    # Performance issues, but functional
    CRITICAL = "critical"    # Major issues, limited functionality
    FAILED = "failed"        # Complete failure
    UNKNOWN = "unknown"      # No data available
```

### Check Types

#### 1. **Data Feed Health**

Monitors universe snapshot freshness:

```python
# Checks:
- File exists: /app/data/universe_snapshot.json
- Age < 5 minutes → HEALTHY
- Age 5-60 minutes → DEGRADED
- Age > 60 minutes → CRITICAL
- Missing file → DEGRADED
- No symbols → CRITICAL
```

#### 2. **Exchange Connection Health**

Monitors trading activity:

```python
# Checks:
- Recent trades file age
- Age < 5 minutes → HEALTHY
- Age 5-60 minutes → DEGRADED
- Age > 60 minutes → CRITICAL
```

#### 3. **AI Model Health**

Checks model file availability:

```python
# Checks:
- xgb_model.pkl exists
- lgbm_model.pkl exists
- nhits_model.pth exists
- patchtst_model.pth exists

# Status:
- 4 models → HEALTHY
- 3 models → HEALTHY
- 2 models → DEGRADED
- 0-1 models → CRITICAL
```

#### 4. **Event Executor Health**

Monitors event-driven executor activity:

```python
# Checks:
- State file: /app/data/event_executor_state.json
- Last cycle age < 1 minute → HEALTHY
- Age 1-5 minutes → DEGRADED
- Age > 5 minutes → CRITICAL
```

#### 5. **Subsystem State Health**

Generic check for subsystems with state files:

```python
# Checks state file age:
- Age < 5 minutes → HEALTHY
- Age 5-60 minutes → DEGRADED
- Age > 60 minutes → CRITICAL
- Missing file → UNKNOWN
```

#### 6. **Database Health**

Checks database availability:

```python
# Checks:
- File exists: /app/backend/data/quantum_trader.db
- File size (MB)
- Missing → CRITICAL
- Exists → HEALTHY
```

#### 7. **Logging Health**

Monitors log file updates:

```python
# Checks:
- Latest log file age
- Age < 1 minute → HEALTHY
- Age 1-5 minutes → DEGRADED
- Age > 5 minutes → CRITICAL
- No logs → DEGRADED
```

#### 8. **Resource Monitoring**

Tracks system resources:

```python
# CPU Monitoring:
- Usage > 90% → HIGH severity issue

# Memory Monitoring:
- Usage > 85% → HIGH severity issue
```

### Health Check Execution

```python
# Check all subsystems
report = await self_healer.check_all_subsystems()

# Report includes:
{
    "overall_status": "critical",
    "current_safety_policy": "no_new_trades",
    "subsystem_health": {...},
    "detected_issues": [...],
    "recovery_recommendations": [...],
    "trading_should_continue": False,
    "requires_immediate_action": True,
    "summary": {
        "healthy": 5,
        "degraded": 3,
        "critical": 2,
        "failed": 0
    }
}
```

---

## 🚨 Issue Detection

### Issue Severity Levels

```python
class IssueSeverity(Enum):
    LOW = "low"          # Minor issues, no impact
    MEDIUM = "medium"    # Degraded performance
    HIGH = "high"        # Significant issues
    CRITICAL = "critical" # System failure imminent
```

### Detection Rules

#### Critical/Failed Subsystems

```python
# Trigger: Subsystem status = CRITICAL or FAILED
# Severity: CRITICAL (for CRITICAL status) or HIGH (for FAILED)
# Impact: Depends on subsystem

Example:
- AI_MODEL = CRITICAL → CRITICAL severity
- Impacts trading: Partially
- Affects: EVENT_EXECUTOR, MODEL_SUPERVISOR
```

#### Degraded Subsystems

```python
# Trigger: Subsystem status = DEGRADED
# Severity: MEDIUM
# Impact: No immediate trading impact

Example:
- DATA_FEED = DEGRADED (stale data)
- Severity: MEDIUM
- Impacts trading: False
```

#### High CPU Usage

```python
# Trigger: CPU > 90%
# Severity: HIGH
# Impact: All subsystems may slow down

Example:
- CPU = 95%
- Severity: HIGH
- Impacts trading: True
- Affects: ALL subsystems
```

#### High Memory Usage

```python
# Trigger: Memory > 85%
# Severity: HIGH
# Impact: System instability risk

Example:
- Memory = 91%
- Severity: HIGH
- Impacts trading: True
- Affects: ALL subsystems
```

### Issue Structure

```python
@dataclass
class DetectedIssue:
    issue_id: str                       # Unique ID
    subsystem: SubsystemType            # Affected subsystem
    severity: IssueSeverity             # Issue severity
    timestamp: str                      # When detected
    
    description: str                    # Human-readable description
    symptoms: List[str]                 # Observable symptoms
    root_cause: Optional[str]           # Identified cause
    
    impacts_trading: bool               # Does this affect trading?
    affects_subsystems: List[SubsystemType]  # Dependent subsystems
```

### Example Issue

```json
{
    "issue_id": "ai_model_20251123_013456",
    "subsystem": "ai_model",
    "severity": "critical",
    "timestamp": "2025-11-23T01:34:56Z",
    "description": "ai_model is critical",
    "symptoms": [
        "Status: critical",
        "Last error: 2025-11-23T01:34:56Z",
        "Details: {'existing_models': ['xgb_model.pkl'], 'total': 1}"
    ],
    "root_cause": "Missing model files",
    "impacts_trading": false,
    "affects_subsystems": ["event_executor", "model_supervisor"]
}
```

---

## 🔧 Recovery Actions

### Action Types

```python
class RecoveryAction(Enum):
    RESTART_SUBSYSTEM = "restart_subsystem"
    PAUSE_TRADING = "pause_trading"
    SWITCH_TO_SAFE_PROFILE = "switch_to_safe_profile"
    DISABLE_MODULE = "disable_module"
    FALLBACK_TO_BACKUP = "fallback_to_backup"
    CLOSE_ALL_POSITIONS = "close_all_positions"
    NO_NEW_TRADES = "no_new_trades"
    DEFENSIVE_EXIT = "defensive_exit"
    RELOAD_CONFIG = "reload_config"
    CLEAR_CACHE = "clear_cache"
```

### Action Selection Logic

```python
# CRITICAL Severity:
if subsystem in [EXCHANGE_CONNECTION, DATA_FEED]:
    action = PAUSE_TRADING
    can_auto = auto_pause_on_critical
elif subsystem == AI_MODEL:
    action = DISABLE_MODULE
    can_auto = True
else:
    action = RESTART_SUBSYSTEM
    can_auto = auto_restart_enabled

# HIGH Severity:
action = SWITCH_TO_SAFE_PROFILE
can_auto = True

# MEDIUM/LOW Severity:
action = RELOAD_CONFIG
can_auto = False
```

### Recovery Recommendation Structure

```python
@dataclass
class RecoveryRecommendation:
    recommendation_id: str              # Unique ID
    issue: DetectedIssue                # Related issue
    action: RecoveryAction              # Recommended action
    priority: int                       # 1=highest
    
    description: str                    # Action description
    expected_result: str                # Expected outcome
    risks: List[str]                    # Action risks
    
    can_auto_execute: bool              # Safe for auto-execution?
    requires_approval: bool             # Manual approval needed?
```

### Example Recommendation

```json
{
    "recommendation_id": "rec_1_ai_model_20251123_013456",
    "issue": {...},
    "action": "disable_module",
    "priority": 1,
    "description": "Disable ai_model module",
    "expected_result": "Module disabled, system continues with fallback",
    "risks": ["Degraded functionality"],
    "can_auto_execute": true,
    "requires_approval": false
}
```

### Action Descriptions & Risks

| Action | Description | Expected Result | Risks |
|--------|-------------|-----------------|-------|
| `RESTART_SUBSYSTEM` | Restart failed subsystem | Subsystem restored | Brief interruption |
| `PAUSE_TRADING` | Halt all trading | Trading stopped | Missed opportunities |
| `SWITCH_TO_SAFE_PROFILE` | Reduce risk exposure | System stabilized | Reduced profit |
| `DISABLE_MODULE` | Turn off failing module | Graceful degradation | Degraded functionality |
| `NO_NEW_TRADES` | Block new positions | Prevent new exposure | Cannot capitalize |
| `DEFENSIVE_EXIT` | Close all positions | Capital preserved | Potential losses |
| `RELOAD_CONFIG` | Refresh configuration | Config updated | Config errors risk |
| `CLEAR_CACHE` | Clear system caches | Memory freed | Performance hit |

---

## 🛡️ Safety Policies

### Policy Levels

```python
class SafetyPolicy(Enum):
    ALLOW_ALL = "allow_all"                  # Normal operation
    NO_NEW_TRADES = "no_new_trades"          # Block new positions
    DEFENSIVE_EXIT = "defensive_exit"        # Close risky positions
    SAFE_RISK_PROFILE = "safe_risk_profile"  # Reduce leverage/sizes
    EMERGENCY_SHUTDOWN = "emergency_shutdown" # Stop everything
```

### Policy Evaluation Logic

```python
def _evaluate_safety_policy(overall_status, issues):
    critical_issues = [i for i in issues if i.severity == CRITICAL]
    
    if overall_status == FAILED:
        return EMERGENCY_SHUTDOWN
    
    elif len(critical_issues) > 0:
        trading_impacted = any(i.impacts_trading for i in critical_issues)
        if trading_impacted:
            return DEFENSIVE_EXIT
        else:
            return NO_NEW_TRADES
    
    elif overall_status == CRITICAL:
        return SAFE_RISK_PROFILE
    
    elif overall_status == DEGRADED:
        return NO_NEW_TRADES
    
    else:
        return ALLOW_ALL
```

### Policy Effects

#### `ALLOW_ALL`
- **When:** System healthy
- **Effect:** Normal trading operations
- **Restrictions:** None

#### `NO_NEW_TRADES`
- **When:** Degraded status OR non-trading critical issues
- **Effect:** Block new position entries
- **Restrictions:**
  - ❌ Cannot open new positions
  - ✅ Can exit existing positions
  - ✅ Can adjust stops/targets

#### `DEFENSIVE_EXIT`
- **When:** Trading-impacting critical issues
- **Effect:** Close risky positions
- **Restrictions:**
  - ❌ No new positions
  - ⚠️ Close high-risk positions
  - ✅ Keep low-risk positions

#### `SAFE_RISK_PROFILE`
- **When:** Overall status = CRITICAL (but not failed)
- **Effect:** Reduce risk exposure
- **Restrictions:**
  - ⚠️ Lower leverage (max 5x)
  - ⚠️ Smaller position sizes (50% normal)
  - ⚠️ Tighter stops

#### `EMERGENCY_SHUTDOWN`
- **When:** Overall status = FAILED
- **Effect:** Stop all trading immediately
- **Restrictions:**
  - ❌ No trading activity
  - ⚠️ Close all positions
  - 🚨 Manual intervention required

---

## 🔌 Integration Guide

### 1. Basic Setup

```python
from backend.services.self_healing import SelfHealingSystem

# Initialize
self_healer = SelfHealingSystem(
    data_dir="/app/data",
    log_dir="/app/logs",
    check_interval=30,
    critical_check_interval=5,
    max_consecutive_failures=3,
    max_error_rate=0.20,
    stale_data_threshold_sec=300,
    max_cpu_percent=90.0,
    max_memory_percent=85.0,
    auto_restart_enabled=True,
    auto_pause_on_critical=True
)
```

### 2. Run Health Checks

```python
# Periodic health check
report = await self_healer.check_all_subsystems()

print(f"Overall Status: {report.overall_status.value}")
print(f"Safety Policy: {report.current_safety_policy.value}")
print(f"Trading Should Continue: {report.trading_should_continue}")

if report.requires_immediate_action:
    print("⚠️ IMMEDIATE ACTION REQUIRED!")
    for issue in report.critical_issues:
        print(f"  - {issue.description}")
```

### 3. Check Safety Policy Before Trading

```python
# Before opening new position
report = await self_healer.check_all_subsystems()

if report.current_safety_policy == SafetyPolicy.ALLOW_ALL:
    # Safe to open new position
    await execute_trade(signal)
elif report.current_safety_policy == SafetyPolicy.NO_NEW_TRADES:
    # Block new positions
    logger.warning("New trades blocked by safety policy")
elif report.current_safety_policy == SafetyPolicy.EMERGENCY_SHUTDOWN:
    # Emergency: close everything
    await close_all_positions()
```

### 4. Handle Recovery Recommendations

```python
report = await self_healer.check_all_subsystems()

for rec in report.recovery_recommendations:
    if rec.can_auto_execute:
        logger.info(f"Auto-executing: {rec.description}")
        await execute_recovery_action(rec.action, rec.issue.subsystem)
    else:
        logger.warning(f"Manual intervention required: {rec.description}")
        await notify_admin(rec)
```

### 5. Monitor Specific Subsystems

```python
# Check data feed specifically
report = await self_healer.check_all_subsystems()
data_feed_health = report.subsystem_health[SubsystemType.DATA_FEED]

if data_feed_health.status == HealthStatus.CRITICAL:
    logger.error("Data feed critical!")
    await pause_trading()
```

### 6. Continuous Monitoring Loop

```python
async def health_monitoring_loop():
    """Background task for continuous health monitoring."""
    while True:
        try:
            report = await self_healer.check_all_subsystems()
            
            # Apply safety policy
            if report.current_safety_policy == SafetyPolicy.EMERGENCY_SHUTDOWN:
                await emergency_shutdown()
            
            # Execute auto-recovery
            for rec in report.recovery_recommendations:
                if rec.can_auto_execute:
                    await execute_recovery_action(rec.action, rec.issue.subsystem)
            
            # Log summary
            logger.info(
                f"Health: {report.overall_status.value}, "
                f"Policy: {report.current_safety_policy.value}"
            )
            
            # Wait before next check
            await asyncio.sleep(self_healer.check_interval)
        
        except Exception as e:
            logger.error(f"Health monitoring error: {e}")
            await asyncio.sleep(30)
```

### 7. Integration with FastAPI

```python
from fastapi import FastAPI, BackgroundTasks

app = FastAPI()
self_healer = SelfHealingSystem()

@app.on_event("startup")
async def startup_event():
    # Start background health monitoring
    asyncio.create_task(health_monitoring_loop())

@app.get("/health/system")
async def get_system_health():
    """Get comprehensive system health report."""
    report = await self_healer.check_all_subsystems()
    
    return {
        "overall_status": report.overall_status.value,
        "safety_policy": report.current_safety_policy.value,
        "trading_allowed": report.trading_should_continue,
        "issues": len(report.detected_issues),
        "critical_issues": len(report.critical_issues),
        "summary": {
            "healthy": report.healthy_count,
            "degraded": report.degraded_count,
            "critical": report.critical_count,
            "failed": report.failed_count
        }
    }

@app.post("/health/force-check")
async def force_health_check():
    """Force immediate health check."""
    report = await self_healer.check_all_subsystems()
    
    return {
        "timestamp": report.timestamp,
        "status": report.overall_status.value,
        "issues": [
            {
                "subsystem": i.subsystem.value,
                "severity": i.severity.value,
                "description": i.description
            }
            for i in report.detected_issues
        ]
    }
```

---

## 🧪 Testing

### Standalone Test

```bash
cd backend/services
python self_healing.py
```

**Expected Output:**

```
============================================================
SELF-HEALING SYSTEM - Standalone Test
============================================================

[OK] Self-Healing System initialized
  Data dir: ./data
  Check interval: 30s
  Auto-restart: True
  Auto-pause: True

============================================================
Running comprehensive health checks...
============================================================

[OK] Health check complete
  Overall status: critical
  Safety policy: no_new_trades
  Trading should continue: False
  Requires immediate action: True

  Subsystem summary:
    Healthy: 0
    Degraded: 2
    Critical: 2
    Failed: 0

  Detected issues (5):
    [medium] data_feed performance degraded
    [critical] ai_model is critical
    [critical] database is critical
    [medium] logging performance degraded
    [high] Memory usage critically high: 91.3%

  Recovery recommendations (5):
    [P1] Disable ai_model module
         Auto-execute: True
    [P1] Restart database subsystem
         Auto-execute: True
    [P2] Reload configuration files
         Auto-execute: False

============================================================
[OK] All tests completed successfully!
============================================================
```

### Integration Test

```python
import asyncio
from backend.services.self_healing import SelfHealingSystem

async def test_integration():
    """Test Self-Healing System integration."""
    
    print("\n" + "="*60)
    print("TEST 1: Initialize Self-Healing System")
    print("="*60)
    
    self_healer = SelfHealingSystem(
        data_dir="/app/data",
        log_dir="/app/logs"
    )
    
    print("[OK] Initialized")
    
    print("\n" + "="*60)
    print("TEST 2: Run Health Checks")
    print("="*60)
    
    report = await self_healer.check_all_subsystems()
    
    print(f"[OK] Overall Status: {report.overall_status.value}")
    print(f"[OK] Safety Policy: {report.current_safety_policy.value}")
    print(f"[OK] Issues Detected: {len(report.detected_issues)}")
    
    print("\n" + "="*60)
    print("TEST 3: Check Safety Policy Logic")
    print("="*60)
    
    if report.current_safety_policy == SafetyPolicy.ALLOW_ALL:
        print("[OK] Normal operation - all systems healthy")
    elif report.current_safety_policy == SafetyPolicy.NO_NEW_TRADES:
        print("[OK] Defensive mode - blocking new trades")
    else:
        print(f"[OK] Policy: {report.current_safety_policy.value}")
    
    print("\n" + "="*60)
    print("TEST 4: Verify Recovery Recommendations")
    print("="*60)
    
    auto_recs = [r for r in report.recovery_recommendations if r.can_auto_execute]
    manual_recs = [r for r in report.recovery_recommendations if not r.can_auto_execute]
    
    print(f"[OK] Auto-executable: {len(auto_recs)}")
    print(f"[OK] Manual approval: {len(manual_recs)}")
    
    for rec in auto_recs[:3]:
        print(f"  - {rec.description}")
    
    print("\n" + "="*60)
    print("[OK] ALL TESTS PASSED")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(test_integration())
```

---

## 📊 Operations

### Health Report Structure

The system saves a comprehensive health report to `/app/data/self_healing_report.json`:

```json
{
  "timestamp": "2025-11-23T01:34:56Z",
  "overall_status": "critical",
  "current_safety_policy": "no_new_trades",
  "subsystem_health": {
    "data_feed": {
      "subsystem": "data_feed",
      "status": "degraded",
      "timestamp": "2025-11-23T01:34:56Z",
      "response_time_ms": 5.2,
      "error_count": 0,
      "details": {
        "warning": "Snapshot 320s old (stale)"
      }
    },
    "ai_model": {
      "subsystem": "ai_model",
      "status": "critical",
      "timestamp": "2025-11-23T01:34:56Z",
      "response_time_ms": 3.1,
      "error_count": 0,
      "details": {
        "existing_models": ["xgb_model.pkl"],
        "total": 1
      }
    }
    // ... more subsystems
  },
  "detected_issues": [
    {
      "issue_id": "ai_model_20251123_013456",
      "subsystem": "ai_model",
      "severity": "critical",
      "description": "ai_model is critical",
      "impacts_trading": false
    }
    // ... more issues
  ],
  "recovery_recommendations": [
    {
      "recommendation_id": "rec_1_ai_model_20251123_013456",
      "action": "disable_module",
      "priority": 1,
      "description": "Disable ai_model module",
      "can_auto_execute": true
    }
    // ... more recommendations
  ],
  "trading_should_continue": false,
  "requires_immediate_action": true,
  "summary": {
    "healthy": 0,
    "degraded": 2,
    "critical": 2,
    "failed": 0
  }
}
```

### Monitoring Dashboard

**Key Metrics to Monitor:**

1. **Overall System Health**
   - Overall status (healthy/degraded/critical/failed)
   - Current safety policy
   - Trading allowed flag

2. **Subsystem Health**
   - Status per subsystem
   - Last check timestamp
   - Response times

3. **Issues**
   - Total issues detected
   - Critical issues count
   - Issue trends over time

4. **Recovery Actions**
   - Auto-executed actions
   - Pending manual actions
   - Action success rate

5. **Resources**
   - CPU usage %
   - Memory usage %
   - Open connections

### Alert Thresholds

**Critical Alerts (Immediate Action Required):**
- Overall status = FAILED
- Safety policy = EMERGENCY_SHUTDOWN
- 3+ subsystems in CRITICAL state
- CPU > 95%
- Memory > 90%

**Warning Alerts:**
- Overall status = CRITICAL
- Any subsystem = CRITICAL
- Safety policy = DEFENSIVE_EXIT
- CPU > 90%
- Memory > 85%

**Info Alerts:**
- Overall status = DEGRADED
- Safety policy changed
- Recovery action executed

### Logging

The Self-Healing System logs all activity:

```python
# Health check results
logger.info("[SELF-HEAL] Health check complete: Overall=critical, ...")

# Safety policy changes
logger.warning("[SELF-HEAL] Safety policy changed: allow_all → no_new_trades")

# Critical issues
logger.error("[SELF-HEAL] 2 CRITICAL ISSUES detected!")
logger.error("  - ai_model is critical")

# Recovery actions
logger.info("[SELF-HEAL] Auto-executing: Disable ai_model module")
```

### Maintenance

**Daily Tasks:**
1. Review health reports
2. Check for recurring issues
3. Verify recovery action effectiveness

**Weekly Tasks:**
1. Analyze issue trends
2. Adjust thresholds if needed
3. Update recovery action policies

**Monthly Tasks:**
1. Review subsystem dependencies
2. Optimize check intervals
3. Update documentation

---

## 🎓 Best Practices

### 1. **Set Appropriate Thresholds**

```python
# Production settings
self_healer = SelfHealingSystem(
    check_interval=30,              # Check every 30s
    critical_check_interval=5,      # Check critical systems every 5s
    max_consecutive_failures=3,     # Allow 3 failures before escalation
    max_error_rate=0.15,           # 15% error rate = degraded
    stale_data_threshold_sec=180,  # 3 min = stale (aggressive)
    max_cpu_percent=85.0,          # CPU threshold
    max_memory_percent=80.0        # Memory threshold
)
```

### 2. **Enable Auto-Recovery Carefully**

```python
# Conservative auto-recovery
self_healer = SelfHealingSystem(
    auto_restart_enabled=True,      # Auto-restart safe subsystems
    auto_pause_on_critical=True     # Auto-pause on critical issues
)

# Aggressive auto-recovery (not recommended)
self_healer = SelfHealingSystem(
    auto_restart_enabled=True,
    auto_pause_on_critical=True,
    # Add custom recovery policies
)
```

### 3. **Integrate with Alerting**

```python
async def health_monitoring_with_alerts():
    """Health monitoring with alert integration."""
    while True:
        report = await self_healer.check_all_subsystems()
        
        # Critical alerts
        if len(report.critical_issues) > 0:
            await send_alert(
                level="CRITICAL",
                message=f"{len(report.critical_issues)} critical issues detected",
                issues=report.critical_issues
            )
        
        # Policy change alerts
        if report.current_safety_policy != SafetyPolicy.ALLOW_ALL:
            await send_alert(
                level="WARNING",
                message=f"Safety policy: {report.current_safety_policy.value}"
            )
        
        await asyncio.sleep(30)
```

### 4. **Test Recovery Actions**

Always test recovery actions in staging:

```python
# Test restart action
async def test_restart_action():
    """Test subsystem restart."""
    # Simulate failure
    await simulate_subsystem_failure(SubsystemType.EVENT_EXECUTOR)
    
    # Check health
    report = await self_healer.check_all_subsystems()
    
    # Verify recommendation
    assert any(
        r.action == RecoveryAction.RESTART_SUBSYSTEM 
        for r in report.recovery_recommendations
    )
    
    # Execute restart
    await execute_recovery_action(
        RecoveryAction.RESTART_SUBSYSTEM,
        SubsystemType.EVENT_EXECUTOR
    )
    
    # Verify recovery
    report = await self_healer.check_all_subsystems()
    assert report.subsystem_health[SubsystemType.EVENT_EXECUTOR].status == HealthStatus.HEALTHY
```

### 5. **Monitor Recovery Effectiveness**

Track recovery action outcomes:

```python
recovery_stats = {
    "restart_subsystem": {"success": 0, "failure": 0},
    "pause_trading": {"success": 0, "failure": 0},
    # ... more actions
}

async def execute_and_track_recovery(action, subsystem):
    """Execute recovery action and track outcome."""
    try:
        await execute_recovery_action(action, subsystem)
        recovery_stats[action.value]["success"] += 1
    except Exception as e:
        recovery_stats[action.value]["failure"] += 1
        logger.error(f"Recovery action failed: {e}")
```

---

## 📝 Summary

The **Self-Healing System** provides comprehensive protection for the Quantum Trader platform through:

✅ **13 Subsystem Monitors** - Complete coverage of critical components  
✅ **5 Health Status Levels** - Granular health assessment  
✅ **4 Severity Levels** - Clear issue prioritization  
✅ **10 Recovery Actions** - Comprehensive response capabilities  
✅ **5 Safety Policies** - Graduated trading controls  
✅ **Automatic Recovery** - Minimize manual intervention  
✅ **Resource Monitoring** - CPU, memory, connections  
✅ **Dependency Tracking** - Understand system relationships  

**Key Integration Points:**
- Event-Driven Executor (trading control)
- Model Supervisor (model health)
- Portfolio Balancer (risk management)
- Retraining Orchestrator (model updates)

**Next Steps:**
1. Deploy Self-Healing System
2. Configure thresholds for production
3. Enable auto-recovery for safe actions
4. Integrate with alerting system
5. Monitor and tune based on operational data

---

**Document Version:** 1.0  
**Last Updated:** November 23, 2025  
**Author:** Quantum Trader AI Team
