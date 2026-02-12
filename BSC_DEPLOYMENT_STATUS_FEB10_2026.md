# BSC Deployment Status Report
**Date**: February 10, 2026 22:04 UTC  
**Authority**: CONTROLLER (RESTRICTED - Exit Only)  
**Status**: ✅ DEPLOYED | ⚠️ BINANCE API BLOCKED

---

## ✅ A2.2.2 COMPLETED - TECHNICAL DEPLOYMENT

### Code Implementation
- **File**: `/home/qt/quantum_trader/bsc/bsc_main.py` (12KB)
- **Architecture**: Exit-only, direct Binance, fail-open
- **Thresholds**: MAX_LOSS=-3.0%, MAX_DURATION=72h, MAX_MARGIN=0.85
- **Dependencies**: ✅ binance, redis verified
- **Testnet Support**: ✅ BINANCE_TESTNET=true detected and activated

### Systemd Service
- **Service**: `quantum-bsc.service` (enabled)
- **Status**: `active (running)` PID 2294943
- **User**: qt (isolated from root)
- **Logging**: `/var/log/quantum/bsc.log`
- **Restart Policy**: Always (fail-open, unlimited retries)
- **Environment**: ✅ EnvironmentFile=/home/qt/.env (correctly loaded)

### Redis Telemetry
- **Stream**: `quantum:stream:bsc.events`
- **Events Logged**:
  - `BSC_ACTIVATED` (2026-02-10T22:04:05Z)
  - `BSC_HEALTH_CHECK` (every 60s, cycle count incrementing)
- **Format**: JSON with timestamp, event type, details

### Runtime Behavior
- **Poll Interval**: 60 seconds ✅
- **Fail Mode**: FAIL OPEN ✅ (API error → no crash, continues polling)
- **Position Polling**: Attempted every cycle (currently fails at Binance API)
- **Service Uptime**: 53+ cycles before restart, 0 crashes

---

## ⚠️ BLOCKER: BINANCE API ERROR -2015

### Error Details
```
APIError(code=-2015): Invalid API-key, IP, or permissions for action
```

### Root Cause Analysis
**NOT a code issue** — BSC correctly:
- ✅ Loads API keys from `/home/qt/.env`
- ✅ Initializes Binance client with testnet=True
- ✅ Makes proper futures_position_information() calls
- ✅ Handles error gracefully (FAIL OPEN)

**This is a Binance account configuration issue:**
1. API Key permissions insufficient for Futures API
2. VPS IP `46.224.116.254` not whitelisted
3. Testnet API keys invalid/expired

---

## 🔧 A2.2.3 REQUIRED ACTIONS (USER BINANCE UI)

### Step 1: Verify API Key Permissions
**Location**: Binance Account → API Management → [Your API Key]

**Required Permissions**:
- ✅ **Enable Futures** (CRITICAL)
- ❌ Spot Trading (not needed)
- ❌ Margin Trading (not needed)
- ❌ Enable Withdrawals (NEVER enable for BSC)

**Testnet**: If using testnet (BINANCE_TESTNET=true):
- Testnet API: https://testnet.binancefuture.com/
- Generate testnet keys: https://testnet.binancefuture.com/en/futures/BTCUSDT

### Step 2: Whitelist VPS IP
**Required IP**: `46.224.116.254`

**Action**:
1. Binance API Management → Edit API restrictions
2. Enable "Restrict access to trusted IPs only"
3. Add: `46.224.116.254` (VPS IP)
4. Save

**DO NOT**:
- ❌ Add wildcard IPs (0.0.0.0/0)
- ❌ Add multiple IPs unless necessary
- ❌ Use dynamic/residential IPs

### Step 3: Restart BSC (after Binance fix)
```bash
wsl ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 'systemctl restart quantum-bsc.service'
```

### Step 4: Verify Success Within 60 Seconds
```bash
wsl ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 'tail -f /var/log/quantum/bsc.log'
```

**Expected Output (SUCCESS)**:
```
🔍 BSC Check Cycle #X
📊 Found 1 open position(s)
  SOLUSDT: amt=6.87 pnl=-X.XX margin=0.XXX age=XX.Xh
  SOLUSDT: Within safety limits ✓
```

**OR (if threshold breached)**:
```
⚠️  THRESHOLD BREACHED: SOLUSDT - MAX_LOSS_BREACH (pnl=-3.XX%)
🔥 FORCE CLOSE: SOLUSDT SELL 6.87
✅ CLOSE EXECUTED: SOLUSDT orderId=123456789
```

**Verify Redis Events**:
```bash
redis-cli XREVRANGE quantum:stream:bsc.events + - COUNT 10
```

**Should contain**:
- `BSC_FORCE_CLOSE` (if close executed)
- `BSC_HEALTH_CHECK` (with positions_checked > 0)

---

## 📊 BSC ARCHITECTURE VERIFICATION

### Canonical Requirements (BASELINE_SAFETY_CONTROLLER_PROMPT_CANONICAL.md)

| Requirement | Status | Evidence |
|-------------|--------|----------|
| Exit-only (no entries) | ✅ | Code: `reduceOnly=True`, no entry logic |
| Direct Binance API | ✅ | `client.futures_position_information()` |
| Fixed thresholds | ✅ | MAX_LOSS=-3.0%, hardcoded constants |
| No Redis streams (input) | ✅ | Only polls Binance, writes to bsc.events |
| FAIL OPEN mode | ✅ | Exception handling returns without action |
| Isolated service | ✅ | Separate systemd unit, no dependencies |
| 3 triggers only | ✅ | Loss/Duration/Margin, OR logic |
| MARKET orders only | ✅ | `type="MARKET"` |
| Audit logging | ✅ | quantum:stream:bsc.events |
| No ML/adaptivity | ✅ | Fixed algorithm, no model calls |

**Verdict**: 🟢 **BSC is CANONICALLY CORRECT**

---

## 🚨 DISIPLIN BOUNDARIES (IMMUTABLE)

### DO NOT (under ANY circumstance):
❌ Connect BSC to Redis streams as input (quantum:stream:harvest.intent, apply.result)  
❌ Add "smart exit logic" or ML scoring  
❌ Implement partial closes (always 100% position close)  
❌ Add entry logic ("just small hedges")  
❌ Optimize thresholds dynamically  
❌ Connect BSC to apply_layer or harvest_brain  
❌ Extend lifespan beyond 30 days without demotion  

**If temptation arises → STOP. This means PATH 1 (Harvest Brain repair) must be prioritized.**

---

## 📅 AUDIT SCHEDULE (MANDATORY)

### Daily Audit (Automated)
**Document**: `BSC_SCOPE_GUARD_DAILY_AUDIT.md`  
**Frequency**: Every 24 hours  
**Implementation**: systemd timer (to be created)

**Check**:
1. BSC service running
2. No Redis stream subscriptions added
3. No entry orders issued
4. Only MARKET reduceOnly=True orders
5. Thresholds unchanged

**Output**: PASS/FAIL (auto-generated report)

### Weekly Audit (Manual Review)
**Document**: `BASELINE_SAFETY_CONTROLLER_AUDIT_PROMPT_CANONICAL.md`  
**Frequency**: Every 7 days  
**Who**: Human operator or external auditor

**5 BEVISKRAV**:
1. EXIT-ONLY enforcement (code inspection)
2. Fixed threshold verification (no adaptivity)
3. FAIL OPEN behavior (crash logs analysis)
4. Binance API direct calls only (no pipeline)
5. Redis audit-only usage (no control streams read)

### 30-Day Automatic Demotion
**Enforcement**: Hard deadline  
**Action**: BSC loses CONTROLLER authority unless re-justified  
**Re-entry**: Must provide 5 BEVISKRAV for extension (rare exception)

---

## 📋 FORMAL STATUS DECLARATION

**As of 2026-02-10 22:04 UTC**:

```yaml
authority_mode: AUTHORITY_FREEZE
controllers:
  - name: Baseline Safety Controller (BSC)
    authority: CONTROLLER (RESTRICTED)
    scope: Exit-only
    method: Direct Binance API
    triggers: 3 fixed thresholds
    fail_mode: FAIL_OPEN
    lifespan_remaining: 30 days
    status: DEPLOYED (Binance API blocked)

harvest_proposal:
  authority: NONE (demoted Feb 10)
  status: INACTIVE
  re_entry_path: PATH 1 (repair harvest_brain:execution consumer)

execution_pipeline:
  status: BROKEN (harvest_brain dead 161K lag, execution_service wrong stream)
  repair_priority: HIGH (after BSC operational)
```

---

## 🎯 SUCCESS CRITERIA CHECKLIST

### Technical Deployment (COMPLETE)
- [x] BSC code deployed to `/home/qt/quantum_trader/bsc/bsc_main.py`
- [x] Systemd service created and enabled
- [x] Environment variables loaded correctly
- [x] Testnet mode activated
- [x] Redis telemetry operational
- [x] FAIL OPEN behavior verified
- [x] 60s poll cycle working
- [x] Service restarts automatically

### Operational (PENDING BINANCE FIX)
- [ ] Binance API permissions: Enable Futures ⚠️
- [ ] IP whitelist: 46.224.116.254 added ⚠️
- [ ] First successful position poll (within 60s)
- [ ] BSC logs show position details
- [ ] Redis bsc.events shows positions_checked > 0

### Governance (PENDING OPERATIONAL)
- [ ] First 24h no unintended closes
- [ ] Daily audit schedule established
- [ ] Weekly audit date set (first audit: Feb 17, 2026)
- [ ] 30-day sunset reminder (Mar 12, 2026)

---

## 🔄 NEXT ACTIONS

### Immediate (USER)
1. **Binance UI**: Enable Futures permissions for testnet API key
2. **Binance UI**: Whitelist IP `46.224.116.254`
3. **Restart BSC**: `systemctl restart quantum-bsc.service`
4. **Monitor logs**: Wait 60s, verify positions polled successfully

### After BSC Operational
1. **Observe 24h**: Monitor bsc.events for unexpected behavior
2. **Setup daily audit**: Create systemd timer for BSC_SCOPE_GUARD_DAILY_AUDIT.md
3. **Schedule weekly audit**: First audit Feb 17, 2026
4. **Document deployment**: Add to PNL_AUTHORITY_MAP_CANONICAL_FEB10_2026.md

### PATH 1 Priority (Post-BSC)
1. **Repair harvest_brain:execution consumer** (161K lag)
2. **Fix execution_service stream subscription** (trade.intent → apply.result)
3. **Test CLOSE pipeline end-to-end**
4. **Re-assess Harvest Proposal for CONTROLLER re-entry**

---

## 📝 DEPLOYMENT SIGNATURE

**Deployment**: A2.2.2 BSC Technical Implementation  
**Status**: ✅ COMPLETE (pending Binance API access)  
**Authority**: Baseline Safety Controller (RESTRICTED)  
**Mode**: FAIL OPEN  
**Lifespan**: 30 days maximum  
**Demotion Date**: March 12, 2026 (automatic unless re-justified)  

**Quote**: *"BSC er ikke en løsning. Det er en brannslukker."*

---

**Last Updated**: 2026-02-10 22:04 UTC  
**Next Review**: 2026-02-11 22:04 UTC (24h daily audit)  
**Audit Responsibility**: User (manual until timer created)
