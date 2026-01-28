# PHASE E1 → VPS Deployment Guide

**Target:** Deploy HarvestBrain to 46.224.116.254 (Hetzner VPS)

## Quick Deploy (5 minutes)

### Step 1: Verify Local Files

```bash
cd c:\quantum_trader

# Verify all E1 files exist
ls -la microservices/harvest_brain/
ls -la etc/quantum/harvest-brain.env.example
ls -la systemd/quantum-harvest-brain.service
ls -la ops/harvest_brain_proof.sh
ls -la ops/harvest_brain_rollback.sh
```

### Step 2: Copy to VPS

```powershell
# From Windows, in WSL or PowerShell:

$sshKey = "$env:USERPROFILE\.ssh\hetzner_fresh"
$vpsHost = "46.224.116.254"
$vpsUser = "root"

# Create target directories
wsl ssh -i $sshKey ${vpsUser}@${vpsHost} 'mkdir -p /opt/quantum/microservices/harvest_brain'

# Copy microservice code
wsl scp -i $sshKey -r microservices/harvest_brain/*.py ${vpsUser}@${vpsHost}:/opt/quantum/microservices/harvest_brain/

# Copy config template
wsl scp -i $sshKey etc/quantum/harvest-brain.env.example ${vpsUser}@${vpsHost}:/etc/quantum/harvest-brain.env

# Copy systemd unit
wsl scp -i $sshKey systemd/quantum-harvest-brain.service ${vpsUser}@${vpsHost}:/etc/systemd/system/

# Copy proof/rollback scripts
wsl scp -i $sshKey ops/harvest_brain_proof.sh ops/harvest_brain_rollback.sh ${vpsUser}@${vpsHost}:/opt/quantum/ops/
```

### Step 3: Verify Permissions on VPS

```bash
# SSH into VPS
ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254

# Verify files
ls -la /opt/quantum/microservices/harvest_brain/
ls -la /etc/quantum/harvest-brain.env
ls -la /etc/systemd/system/quantum-harvest-brain.service

# Fix permissions if needed
chown qt:qt /opt/quantum/microservices/harvest_brain/*
chmod 755 /opt/quantum/microservices/harvest_brain/harvest_brain.py
chmod 644 /etc/quantum/harvest-brain.env
chmod 644 /etc/systemd/system/quantum-harvest-brain.service
chmod 755 /opt/quantum/ops/harvest_brain_*.sh
```

### Step 4: Reload Systemd and Start Service

```bash
# On VPS
sudo systemctl daemon-reload
sudo systemctl enable quantum-harvest-brain
sudo systemctl start quantum-harvest-brain

# Verify running
systemctl status quantum-harvest-brain
```

### Step 5: Run Proof Script

```bash
# On VPS
bash /opt/quantum/ops/harvest_brain_proof.sh
```

**Expected Output:**
```
✅ quantum-harvest-brain is ACTIVE
✅ Config found at /etc/quantum/harvest-brain.env
⚠️  Consumer group may not yet exist (created on first run)
Stream: quantum:stream:harvest.suggestions
  Entries: 0 (will increase as executions flow)
Active dedup keys: 0 (will increase as actions are logged)
✅ Kill-switch is OFF (publishing enabled)
```

### Step 6: Monitor Initial Logs

```bash
# On VPS
journalctl -u quantum-harvest-brain -f
```

**Expected Logs (first run):**
```
quantum-harvest-brain[12345]: Starting HarvestBrainService...
quantum-harvest-brain[12345]: Connecting to Redis 127.0.0.1:6379
quantum-harvest-brain[12345]: Consumer group exists: harvest_brain_group
quantum-harvest-brain[12345]: Listening on quantum:stream:execution.result with group harvest_brain_group
quantum-harvest-brain[12345]: No pending messages, waiting for new events...
```

## Configuration

**File:** `/etc/quantum/harvest-brain.env`

**Critical Settings (Edit Before Starting):**

```bash
# Shadow mode (safe, no real orders)
HARVEST_MODE=shadow

# R-based ladder
HARVEST_MIN_R=0.5                           # Don't harvest unless R >= 0.5
HARVEST_LADDER="0.5:0.25,1.0:0.25,1.5:0.25"  # At R=0.5 close 25%, at R=1.0 close 25%, etc.
HARVEST_SET_BE_AT_R=0.5                     # Move SL to break-even at R >= 0.5

# Idempotency
HARVEST_DEDUP_TTL_SEC=900                   # Dedup key TTL (900 = 15 minutes)

# Safety
HARVEST_REQUIRE_FRESH_SNAPSHOT_SEC=30       # Skip if position > 30s old
HARVEST_KILL_SWITCH_KEY=quantum:kill        # Check this Redis key for emergency stop
```

## Testing in Shadow Mode

### Test 1: Generate Test Execution

```bash
# On VPS, inject a test execution fill
redis-cli <<EOF
XADD quantum:stream:execution.result * symbol ETHUSDT side BUY qty 1.0 price 2500.0 status FILLED timestamp "$(date -u +%s)"
EOF

# Check if harvest suggestion was created
redis-cli XREVRANGE quantum:stream:harvest.suggestions + - COUNT 1
```

### Test 2: Verify Dedup

```bash
# Inject same execution again (should be skipped due to dedup)
redis-cli <<EOF
XADD quantum:stream:execution.result * symbol ETHUSDT side BUY qty 1.0 price 2500.0 status FILLED timestamp "$(date -u +%s)"
EOF

# Should still be only 1 entry in harvest.suggestions (dedup worked)
redis-cli XLEN quantum:stream:harvest.suggestions
```

### Test 3: Test Kill-Switch

```bash
# Activate kill-switch
redis-cli SET quantum:kill 1

# Try to inject execution (should not create harvest suggestion)
redis-cli <<EOF
XADD quantum:stream:execution.result * symbol BTCUSDT side BUY qty 0.5 price 50000.0 status FILLED timestamp "$(date -u +%s)"
EOF

# Deactivate kill-switch
redis-cli SET quantum:kill 0

# Check logs
journalctl -u quantum-harvest-brain -n 10
```

## Live Mode Transition

When validated in shadow mode, transition to live mode:

```bash
# Edit config on VPS
sudo nano /etc/quantum/harvest-brain.env

# Change:
# HARVEST_MODE=shadow
# To:
# HARVEST_MODE=live

# Reload service
sudo systemctl restart quantum-harvest-brain

# Monitor logs
journalctl -u quantum-harvest-brain -f
```

## Troubleshooting

### Service won't start

```bash
# Check logs
journalctl -u quantum-harvest-brain -n 50

# Check syntax
python3 -m py_compile /opt/quantum/microservices/harvest_brain/harvest_brain.py

# Check Redis connection
redis-cli ping
```

### No harvest suggestions appearing

```bash
# 1. Check executions are flowing
redis-cli XLEN quantum:stream:execution.result
redis-cli XREVRANGE quantum:stream:execution.result + - COUNT 3

# 2. Check kill-switch
redis-cli GET quantum:kill

# 3. Check mode
grep HARVEST_MODE /etc/quantum/harvest-brain.env

# 4. Check logs for errors
journalctl -u quantum-harvest-brain -n 30
```

### High CPU usage

- Check log rotation (service may be logging too much)
- Check for stuck positions (update trigger more frequently)
- Monitor with: `ps aux | grep harvest_brain`

## Rollback

If anything goes wrong:

```bash
# Stop service
sudo systemctl stop quantum-harvest-brain

# Disable auto-start
sudo systemctl disable quantum-harvest-brain

# No data loss (Redis streams untouched)
# Can restart anytime with: sudo systemctl start quantum-harvest-brain
```

## Next Phases

✅ **E1:** Scaffold complete (this guide)
⏳ **E2:** Deploy config
⏳ **E5:** Deploy systemd unit
⏳ **E6:** Run proof script
⏳ **E8:** Commit to git main

---

**Ready to deploy? Proceed with Step 1 above.**
