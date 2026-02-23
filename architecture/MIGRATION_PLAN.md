# Live/Research Separation — Migration Plan & Rollback Strategy

**Status:** DRAFT — Not yet executed  
**Date:** 2026-02-21  
**Scope:** Architectural refactor, zero service interruptions  
**Author:** Architectural Refactor Session  

---

## Core Principle

> **Nothing moves until each phase below is explicitly signed off.**  
> Every phase is independently reversible.  
> Services are never restarted except by deliberate action in Phase 4.

---

## Phase Overview

| Phase | Name                          | Risk     | Reversible | Gate Required |
|-------|-------------------------------|----------|------------|---------------|
| 0     | Create directory structure    | None     | Trivially  | None          |
| 1     | Audit current model locations | None     | N/A        | None          |
| 2     | Populate model registry       | Low      | Yes        | Human review  |
| 3     | Symlink live services → registry | Medium | Yes (rm symlink) | Human sign-off |
| 4     | Switch services to new paths  | High     | Yes (revert env) | Full sign-off |
| 5     | Enable systemd units          | High     | Yes (disable) | Staged rollout |
| 6     | Decommission old paths        | Medium   | No (archive) | 30-day wait  |

---

## Phase 0 — Create Directory Structure

**Status:** READY TO EXECUTE  
**Risk:** Zero — creates empty directories only, touches no existing files  
**Command:**
```bash
bash architecture/create_quantum_structure.sh   # from /opt/quantum
```

**Verification:**
```bash
find /opt/quantum/live_core /opt/quantum/research_lab /opt/quantum/model_registry -type d | sort
```

**Rollback:**
```bash
rm -rf /opt/quantum/live_core \
       /opt/quantum/research_lab \
       /opt/quantum/model_registry
```

---

## Phase 1 — Audit Current Model Locations

**Status:** DESIGN (run before Phase 2)  
**Risk:** Zero — read-only audit  

```bash
# Find all .pkl, .pt, .h5 model files currently on the VPS
find /opt/quantum -name "*.pkl" -o -name "*.pt" -o -name "*.h5" | sort > /tmp/model_audit.txt
cat /tmp/model_audit.txt

# Find any model paths currently hardcoded in service env files
grep -r "MODEL_DIR\|MODEL_PATH\|checkpoint" /opt/quantum --include="*.env" --include="*.yaml"
```

**Output required before proceeding:** A table of current model files → intended registry tier (approved or staging).

---

## Phase 2 — Populate Model Registry

**Status:** BLOCKED on Phase 1 completion  
**Risk:** Low — copies, does not move  

For each model identified in Phase 1:

```bash
# Copy (not move) to appropriate registry tier
# Example: current production RL model → approved/
cp /opt/quantum/microservices/rl_training/models/sizing_agent_latest.pt \
   /opt/quantum/model_registry/approved/rl/sizing_agent_v12.0.0.pt

# Example: in-progress research model → staging/
cp /opt/quantum/microservices/rl_calibrator/checkpoints/latest.pt \
   /opt/quantum/model_registry/staging/rl/sizing_agent_v13_candidate.pt
```

Then run first promotion for each current production model:
```bash
sudo -u quantum-admin bash architecture/promote_model.sh \
    --type rl \
    --staging-file rl/sizing_agent_v12.0.0.pt \
    --target-name sizing_agent_v12.0.0.pt \
    --notes "Initial registry promotion — current live model as of 2026-02-21"
```

**Rollback:**
```bash
rm -f /opt/quantum/model_registry/approved/rl/sizing_agent_v12.0.0.pt
# Original file is still at its original path — untouched
```

---

## Phase 3 — Symlink Live Services to Registry

**Status:** BLOCKED on Phase 2 completion  
**Risk:** Medium — services read from symlink; original files unchanged  

Instead of changing service configs immediately, introduce a symlink so services
transparently read from the registry without any config change:

```bash
# Example: Signal engine currently reads from /opt/quantum/models/signal/
# Create a symlink so it can still find models at old path
ln -sfn /opt/quantum/model_registry/approved/signal \
         /opt/quantum/models/signal_registry

# Update service env (single line change, no restart needed until Phase 4):
# OLD: MODEL_DIR=/opt/quantum/models/signal
# NEW: MODEL_DIR=/opt/quantum/model_registry/approved/signal
```

**Verification before any restart:**
```bash
ls -la /opt/quantum/model_registry/approved/signal/
# Confirm all expected model files present
```

**Rollback:**
```bash
rm /opt/quantum/models/signal_registry
# Service continues reading from original path with no interruption
```

---

## Phase 4 — Switch Services to New Paths (REQUIRES MAINTENANCE WINDOW)

**Status:** BLOCKED on Phase 3 + explicit human sign-off  
**Risk:** High — services restart  
**Prerequisites:**
- [ ] All model files verified in registry
- [ ] Symlinks verified pointing to correct files
- [ ] SHA-256 hashes match between registry and previous model paths
- [ ] Maintenance window agreed (recommend low-volatility market period)
- [ ] Rollback engineer on standby

**Execution order (respects service dependency chain):**

```bash
# 1. Stop traders in reverse dependency order
systemctl stop quantum-execution.service
systemctl stop quantum-exit.service
systemctl stop quantum-allocator.service
systemctl stop quantum-risk.service
systemctl stop quantum-signal.service

# 2. Update EnvironmentFiles to point to new paths
# (per-service config in /opt/quantum/live_core/*/config/*.env)

# 3. Copy systemd unit files (review all units before this step)
cp architecture/systemd/quantum-signal.service    /etc/systemd/system/
cp architecture/systemd/quantum-allocator.service /etc/systemd/system/
cp architecture/systemd/quantum-risk.service      /etc/systemd/system/
cp architecture/systemd/quantum-execution.service /etc/systemd/system/
cp architecture/systemd/quantum-exit.service      /etc/systemd/system/
systemctl daemon-reload

# 4. Start in dependency order, verifying health at each step
systemctl start quantum-signal.service
sleep 10; curl -s http://localhost:8010/health | jq .   # must return {"status":"ok"}

systemctl start quantum-allocator.service
sleep 10; curl -s http://localhost:8011/health | jq .

systemctl start quantum-risk.service
sleep 10; curl -s http://localhost:8012/health | jq .

systemctl start quantum-execution.service
sleep 10; curl -s http://localhost:8002/health | jq .

systemctl start quantum-exit.service
sleep 10; curl -s http://localhost:8013/health | jq .
```

**Go/No-Go criteria at each health check:**
- Status must be `"ok"` or `"healthy"`
- No error logs in `journalctl -u quantum-{service} -n 50`
- Redis streams receiving data: `redis-cli XLEN quantum:stream:apply.plan`

**Rollback procedure (Phase 4 failure):**

```bash
# Stop all new units
systemctl stop quantum-execution.service quantum-exit.service \
    quantum-allocator.service quantum-risk.service quantum-signal.service

# Remove new unit files
rm /etc/systemd/system/quantum-{signal,allocator,risk,execution,exit}.service
systemctl daemon-reload

# Restart original Docker-based services
cd /opt/quantum
docker compose up -d signal_engine allocator risk_engine exec_risk_service exit_engine

# Verify recovery
docker ps | grep -E "signal|allocator|risk|exec|exit"
```

---

## Phase 5 — Enable Research Trainer Units (Separate from Live)

**Status:** BLOCKED on Phase 4 stability (≥ 72h live with no incidents)  
**Risk:** Medium — trainers are isolated but consume CPU/memory  

```bash
cp architecture/systemd/quantum-rl-trainer.service  /etc/systemd/system/
cp architecture/systemd/quantum-clm-trainer.service /etc/systemd/system/
systemctl daemon-reload

# Do NOT enable for auto-start — start manually only
systemctl start quantum-rl-trainer.service
systemctl status quantum-rl-trainer.service

# Verify isolation: confirm trainer cannot reach approved/
sudo -u quantum-research ls /opt/quantum/model_registry/approved/
# Expected: "Permission denied"

# Verify trainer writes to staging only
sudo -u quantum-research ls /opt/quantum/model_registry/staging/rl/
# Expected: checkpoint files appearing
```

**Rollback:**
```bash
systemctl stop quantum-rl-trainer.service quantum-clm-trainer.service
rm /etc/systemd/system/quantum-{rl,clm}-trainer.service
systemctl daemon-reload
# Docker-based trainers resumable immediately: docker compose up -d rl_training clm_trainer
```

---

## Phase 6 — Decommission Old Paths (30-day hold)

**Status:** BLOCKED — do not execute for at least 30 days after Phase 4  
**Action:** Archive (do not delete) original model locations  

```bash
# After 30-day verified live operation, archive old paths
mkdir -p /opt/quantum/_pre_migration_archive/$(date +%Y%m%d)
mv /opt/quantum/models /opt/quantum/_pre_migration_archive/$(date +%Y%m%d)/
# Keep archive for 90 days, then delete manually
```

---

## Rollback Matrix (Quick Reference)

| Phase Failed | Time to Rollback | Command Count | Impact             |
|--------------|------------------|---------------|--------------------|
| Phase 0      | < 1 min          | 1             | Zero               |
| Phase 1      | N/A              | N/A           | Read-only          |
| Phase 2      | < 2 min          | 2-5           | Zero (copies only) |
| Phase 3      | < 1 min          | 1 per service | Zero (rm symlink)  |
| Phase 4      | 5-10 min         | 10-15         | Brief trade halt   |
| Phase 5      | < 2 min          | 3             | Trainer stops only |
| Phase 6      | N/A              | Archive, not delete | Review 90d hold |

---

## Permanent Invariants (Never Violate)

1. `approved/` is the only source of truth for live model files
2. `staging/` has zero read access from any live service (ACL enforced)
3. No training loop call can write to `approved/` (filesystem ACL + no code path)
4. `promote_model.sh` is the one and only promotion mechanism
5. Promotion always archives the previous approved model atomically
6. Research services run as `quantum-research` user; live services run as `quantum-live`
7. Service startup failure = no trade, no order. Never silently degrade.

---

*No files moved. No services restarted. This document is planning only.*  
*Next action: Execute Phase 0 after this document is reviewed.*
