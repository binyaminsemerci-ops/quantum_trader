# Ops Tools (P4)

## Universe Service (P0)

**Purpose:** Single source of truth for tradeable symbols. Fetches Binance Futures exchangeInfo and publishes allowed symbol set to Redis for all gates to consume.

**Files:**
- Service: `microservices/universe_service/main.py`
- Config: `/etc/quantum/universe-service.env` (copy from `microservices/universe_service/universe-service.env.example`)
- Systemd: `ops/systemd/quantum-universe-service.service`
- Proof script: `ops/proof_universe.sh`

**Redis Keys:**
- `quantum:cfg:universe:active` - Current active symbols (JSON)
- `quantum:cfg:universe:last_ok` - Last successful fetch (JSON, fail-closed backup)
- `quantum:cfg:universe:meta` - Metadata hash (asof_epoch, last_ok_epoch, count, stale, error)

**Configuration:**
```bash
# /etc/quantum/universe-service.env
UNIVERSE_MODE=testnet           # testnet|mainnet
UNIVERSE_REFRESH_SEC=60         # Fetch interval
UNIVERSE_MAX=800                # Safety cap on symbol count
REDIS_HOST=127.0.0.1
REDIS_PORT=6379
REDIS_DB=0
HTTP_TIMEOUT_SEC=10
```

**Deployment:**
```bash
# 1. Copy example config
sudo cp microservices/universe_service/universe-service.env.example /etc/quantum/universe-service.env
sudo chown qt:qt /etc/quantum/universe-service.env

# 2. Edit config if needed
sudo nano /etc/quantum/universe-service.env

# 3. Install systemd unit
sudo cp ops/systemd/quantum-universe-service.service /etc/systemd/system/
sudo systemctl daemon-reload

# 4. Start service
sudo systemctl enable quantum-universe-service
sudo systemctl start quantum-universe-service

# 5. Verify
sudo systemctl status quantum-universe-service
bash ops/proof_universe.sh
```

**Usage:**
```bash
# Check universe status
bash ops/proof_universe.sh

# View active symbols
redis-cli GET quantum:cfg:universe:active | jq -r '.symbols[]'

# Check metadata
redis-cli HGETALL quantum:cfg:universe:meta

# Service logs
journalctl -u quantum-universe-service -f
```

**Failure Mode:**
- FAIL-CLOSED: On fetch failure, preserves `last_ok` and marks `stale=1`
- Validation: Ensures non-empty symbol list, regex match `^[A-Z0-9]{3,20}USDT$`, count <= UNIVERSE_MAX
- Bootstrap: On boot, copies `last_ok` → `active` if active missing (marked stale until first success)

**Integration:**
Other services read from `quantum:cfg:universe:active` as single source of truth:
```python
import redis
import json

r = redis.Redis(decode_responses=True)
universe_json = r.get('quantum:cfg:universe:active')
universe = json.loads(universe_json)
symbols = universe['symbols']  # List of tradeable symbols
mode = universe['mode']        # testnet or mainnet
```

### P1: P3.3 Universe Integration

**Status:** Integrated (P3.3 Position State Brain)

P3.3 now reads allowlist from Universe Service with fail-closed fallback to `P33_ALLOWLIST` env var.

**Configuration:**
```bash
# /etc/quantum/position-state-brain.env
UNIVERSE_ENABLE=true           # Enable Universe integration (default: true)
UNIVERSE_CACHE_SECONDS=60      # Cache duration (default: 60s)
P33_ALLOWLIST=BTCUSDT          # Fallback if Universe unavailable
```

**Behavior:**
- **Universe fresh** (stale=0): P3.3 uses `quantum:cfg:universe:active` symbols
- **Universe stale** (stale=1): P3.3 falls back to `P33_ALLOWLIST` env var
- **Universe missing**: P3.3 falls back to `P33_ALLOWLIST` env var
- Cache refreshed every 60s (configurable)
- Logs source on startup and refresh: `Allowlist source=universe|fallback`

**Verification:**
```bash
# Check which source P3.3 is using
bash ops/proof_p33_universe_source.sh

# Watch logs for source changes
journalctl -u quantum-position-state-brain -f | grep 'Allowlist source'

# Test integration
python3 ops/test_p33_universe_integration.py
```

**Testing:**
```bash
# Test universe integration
python3 ops/test_p33_universe_integration.py

# This script tests:
# - Universe fresh (stale=0) → P3.3 uses universe symbols
# - Universe stale (stale=1) → P3.3 falls back to P33_ALLOWLIST
# - Universe missing → P3.3 falls back to P33_ALLOWLIST
```

---

## Ops Governor Prompt Generator

Template:
- `ops/ops_prompt_template_v1.txt`

Generator:
- `ops/ops_prompt_fill.py` (stdlib only)

### Example
```bash
python ops/ops_prompt_fill.py \
  --operation "Fix Grafana provisioning wrappers" \
  --objective "Unwrap dashboard JSON and remove UID duplicates under /var/lib/grafana/dashboards/quantum" \
  --risk_class LOW_RISK_CONFIG \
  --blast_radius "Grafana dashboards only" \
  --rollback_strategy "Restore JSON files from git + restart grafana-server" \
  --allowed_paths "/etc/grafana/provisioning" "/var/lib/grafana/dashboards/quantum" \
  --allowed_services "grafana-server"
```

Paste the output into Sonnet/Ops agent to ensure bounded, verifiable, audit-safe execution.

### Usage
```bash
# Show help
python ops/ops_prompt_fill.py --help

# Generate a filled prompt (original mode)
python ops/ops_prompt_fill.py \
  --operation "Your operation name" \
  --objective "What this achieves" \
  --risk_class READ_ONLY \
  --blast_radius "What can be affected" \
  --rollback_strategy "How to revert" \
  --allowed_paths "/path1" "/path2" \
  --allowed_services "service1" "service2"

# Generate a P5 ledger YAML snippet (new in P5)
python ops/ops_prompt_fill.py --ledger \
  --operation "Your operation name" \
  --objective "What this achieves" \
  --risk_class LOW_RISK_CONFIG \
  --blast_radius "What can be affected" \
  --rollback_strategy "git-revert-<commit>" \
  --allowed_paths "/path1" "/path2" \
  --allowed_services "service1" "service2" \
  --changes_summary "1-line summary of what changed" \
  --notes "Optional context"
```

## Documentation
- `docs/OPS_GOVERNOR.md` - Core governance framework (includes P5: Signoff Ledger)
- `docs/OPS_RUNBOOK_TEMPLATE.md` - Template for ops runbooks (with optional ledger section)
- `docs/OPS_CHANGELOG.md` - Operations changelog & signoff ledger (audit trail)

## P5: Change Approval & Signoff Ledger

**Optional** governance layer for operations requiring human approval or audit trails.

**When to use:**
- LOW_RISK_CONFIG: Optional (informal commit reference)
- SERVICE_RESTART / FILESYSTEM_WRITE: Recommended
- High-risk custom ops: Required

**Ledger storage options:**
1. **Markdown ledger:** Append YAML blocks to `docs/OPS_CHANGELOG.md`
2. **Structured files:** Maintain `ops/ledger/YYYY-MM.yaml` (monthly archives)
3. **Git-native:** Use annotated tags: `git tag -a ops-2026-01-27-001 -m "<yaml>"`

**Integration:**
- Fill **Section 6** of runbook template after operation completes
- Commit ledger entry with (or right after) operation changes
- For rollbacks: update entry with `outcome: ROLLBACK` + rollback timestamp

See [docs/OPS_GOVERNOR.md](../docs/OPS_GOVERNOR.md#p5-change-approval--signoff-ledger-optional) for full details.

---

## P5+ Auto-Ledger (Automated Ops Logging)

**Tool:** `ops/ops_ledger_append.py`

Automatically appends YAML ledger entries to `docs/OPS_CHANGELOG.md` after successful deploy/rollback.

### Features
- **Idempotent:** Re-running with same operation_id won't create duplicates
- **Auto-collect:** Git SHA, timestamp, service status, metrics, Redis state, proof file hash
- **Safe:** Deploy still succeeds even if ledger fails (unless `STRICT_LEDGER=true`)
- **Evidence-rich:** Captures proof files, metrics snapshots, Redis keys, service logs

### Usage

#### Manual Invocation
```bash
python3 ops/ops_ledger_append.py \
  --operation "P2.7 Deploy — Portfolio Clusters" \
  --objective "Deploy correlation matrix + capital clustering" \
  --risk_class SERVICE_RESTART \
  --blast_radius "Portfolio gating layer only" \
  --allowed_paths microservices/portfolio_clusters/ \
  --allowed_services quantum-portfolio-clusters quantum-portfolio-gate \
  --changes_summary "Deployed P2.7 with warmup observability" \
  --proof_path "/home/qt/P2_7_PROOF_20260127.txt" \
  --metrics_urls "http://127.0.0.1:8048/metrics" \
  --metrics_grep "p27_corr_ready" \
  --redis_cmds "HGETALL quantum:portfolio:cluster_state" \
  --notes "Manual ledger entry"
```

#### Integrated in Deploy Scripts
Deploy scripts (`p27_deploy_and_proof.sh`) automatically call this after successful proof:

```bash
# Set STRICT_LEDGER=true to fail deployment if ledger fails (default: false)
export STRICT_LEDGER=false

# Run deploy - ledger entry created automatically
bash ops/p27_deploy_and_proof.sh
```

#### Self-Test (No Network)
```bash
# Test auto-generation of operation_id
python3 ops/ops_ledger_append.py \
  --operation "Self-Test" \
  --objective "Test ledger append" \
  --risk_class READ_ONLY \
  --blast_radius "None" \
  --changes_summary "Test entry" \
  --notes "Testing P5+ auto-ledger"

# Check it was appended
tail -30 docs/OPS_CHANGELOG.md
```

#### Validation Examples (P5+.1)
```bash
# Example 1: ROLLBACK without rollback_of (non-strict → warning)
python3 ops/ops_ledger_append.py \
  --operation "Test Rollback Validation" \
  --objective "Test validation behavior" \
  --outcome ROLLBACK \
  --risk_class LOW_RISK_CONFIG \
  --blast_radius "None" \
  --changes_summary "Test" \
  --notes "Expected output: WARNING"
# Expected: "WARNING: ERROR: outcome=ROLLBACK requires --rollback_of parameter (continuing in non-strict mode)"

# Example 2: ROLLBACK with invalid format (non-strict → warning)
python3 ops/ops_ledger_append.py \
  --operation "Test Format Validation" \
  --objective "Test format validation" \
  --outcome ROLLBACK \
  --rollback_of "INVALID-FORMAT" \
  --risk_class LOW_RISK_CONFIG \
  --blast_radius "None" \
  --changes_summary "Test" \
  --notes "Expected output: WARNING"
# Expected: "WARNING: ERROR: rollback_of 'INVALID-FORMAT' does not match expected format OPS-YYYY-MM-DD-NNN (continuing in non-strict mode)"

# Example 3: ROLLBACK with valid rollback_of (success)
python3 ops/ops_ledger_append.py \
  --operation "Test Valid Rollback" \
  --objective "Test valid rollback reference" \
  --outcome ROLLBACK \
  --rollback_of OPS-2026-01-27-010 \
  --risk_class LOW_RISK_CONFIG \
  --blast_radius "None" \
  --changes_summary "Test" \
  --notes "Expected output: SUCCESS"
# Expected: "Auto-generated operation_id: OPS-2026-01-27-XXX" (no warnings)

# Clean up test entries
git checkout docs/OPS_CHANGELOG.md
```

#### Rollback Logging
```bash
# After rolling back a deployment
python3 ops/ops_ledger_append.py \
  --operation "P2.7 Rollback" \
  --objective "Revert P2.7 deployment due to issue" \
  --outcome ROLLBACK \
  --rollback_of OPS-2026-01-27-010 \
  --risk_class SERVICE_RESTART \
  --blast_radius "Portfolio gating layer only" \
  --allowed_services quantum-portfolio-clusters quantum-portfolio-gate \
  --changes_summary "Stopped P2.7, disabled service, reverted to proxy correlation" \
  --notes "Rollback due to [reason]"
```

### CLI Arguments

**Required:**
- `--operation` - Operation name (short description)
- `--objective` - What this operation achieves
- `--risk_class` - READ_ONLY | LOW_RISK_CONFIG | SERVICE_RESTART | FILESYSTEM_WRITE
- `--blast_radius` - What can break
- `--changes_summary` - 1-line summary

**Optional:**
- `--operation_id` - Auto-generated if omitted (OPS-YYYY-MM-DD-NNN)
- `--allowed_paths` - Paths modified (repeatable)
- `--allowed_services` - Services affected (repeatable)
- `--proof_path` - Path to proof file (collects sha256 + size + mtime)
- `--metrics_urls` - Metrics endpoints to curl (repeatable)
- `--metrics_grep` - Regex patterns to filter metrics (repeatable)
- `--redis_cmds` - Redis commands to execute (repeatable, e.g., "HGETALL key")
- `--outcome` - SUCCESS | PARTIAL | ROLLBACK (default: SUCCESS)
- `--rollback_of` - Original operation_id being rolled back (for outcome=ROLLBACK)
- `--notes` - Additional context
- `--strict` - Fail deployment if ledger append fails (default: false)

### Behavior

**Idempotency:**
- If `operation_id` already exists in changelog → exit 0 with "already recorded" message
- Safe to re-run deploy scripts multiple times

**Rollback Logging:**
- Use `--outcome ROLLBACK` for rollback operations
- Use `--rollback_of OPS-YYYY-MM-DD-NNN` to reference the original operation being reverted
- Auto-collects rollback evidence (services stopped, metrics after revert, git SHA)

**Rollback Validation (P5+.1):**
- When `outcome=ROLLBACK`:
  - `--rollback_of` is **required** (must match format `OPS-YYYY-MM-DD-NNN`)
  - Missing or invalid `rollback_of` → **ERROR in strict mode**, **WARNING in non-strict mode**
- Non-strict mode (default): validation failures print warnings but allow ledger entry
- Strict mode (`--strict`): validation failures raise `ValueError` and fail deployment

**Strict Mode:**
- `STRICT_LEDGER=false` (default): Ledger failure prints warning, deployment succeeds
- `STRICT_LEDGER=true`: Ledger failure causes deployment to fail (exit 1)

**Auto-Collection:**
- Git commit SHA (HEAD) + branch
- Hostname
- Timestamp (UTC ISO8601)
- Service status: `systemctl is-active` + last 30 log lines
- Proof file: sha256 hash (first 16 chars), size, mtime
- Metrics: curl endpoints and grep for patterns
- Redis: execute commands and capture output

**Output:**
Appends YAML block to correct month section in `docs/OPS_CHANGELOG.md`:
- Creates month section (e.g., `## 2026-01`) if missing
- Inserts at top of section (most recent first)
- Includes all auto-collected evidence + manual args

### Example Output

**Deploy Operation:**
```yaml
---
operation_id: OPS-2026-01-27-011
operation_name: P2.7 Deploy — Portfolio Clusters (atomic) + P2.6 sync
requested_by: SELF
approved_by: SELF
approval_timestamp: 2026-01-27T06:15:30Z
execution_timestamp: 2026-01-27T06:15:30Z
git_commit: a1b2c3d4
git_branch: main
hostname: quantumtrader-prod-1
risk_class: SERVICE_RESTART
blast_radius: Portfolio gating + clusters only; no execution impact
changes_summary: Deployed P2.7 + synced P2.6 patch; atomic rsync with proof
objective: Deploy P2.7 correlation matrix + capital clustering and verify P2.6 cluster K integration
outcome: SUCCESS
allowed_paths:
  - microservices/portfolio_clusters/
  - microservices/portfolio_gate/main.py
  - deployment/systemd/quantum-portfolio-clusters.service
allowed_services:
  - quantum-portfolio-clusters
  - quantum-portfolio-gate
services_status:
  quantum-portfolio-clusters: active
  quantum-portfolio-gate: active
proof_file:
  sha256: 7f3e9a2b1c4d8e6f
  size_bytes: 15234
  mtime_utc: 2026-01-27T06:15:28Z
metrics_snapshot: |
  # http://127.0.0.1:8048/metrics | grep 'p27_corr_ready'
  p27_corr_ready 1.0
  p27_clusters_count 1.0
redis_snapshot: |
  # redis-cli HGETALL quantum:portfolio:cluster_state
  cluster_stress
  0.788493010732072
  updated_ts
  1769492127
notes: Auto-ledger via P5+ ops_ledger_append.py
```

**Rollback Operation:**
```yaml
---
operation_id: OPS-2026-01-27-012
operation_name: P2.7 Rollback
requested_by: SELF
approved_by: SELF
approval_timestamp: 2026-01-27T08:30:15Z
execution_timestamp: 2026-01-27T08:30:15Z
git_commit: e5f6a7b8
git_branch: main
hostname: quantumtrader-prod-1
risk_class: SERVICE_RESTART
blast_radius: Portfolio gating layer only
changes_summary: Stopped P2.7, disabled service, reverted to proxy correlation
objective: Revert P2.7 deployment due to correlation computation issue
outcome: ROLLBACK
rollback_of: OPS-2026-01-27-010
allowed_services:
  - quantum-portfolio-clusters
  - quantum-portfolio-gate
services_status:
  quantum-portfolio-clusters: inactive
  quantum-portfolio-gate: active
notes: Rollback completed - P2.6 using proxy correlation fallback
```

### Integration Status

**Currently integrated:**
- ✅ `ops/p27_deploy_and_proof.sh` - P2.7 Portfolio Clusters deployment

**To be integrated:**
- ⏳ `ops/p26_deploy_and_proof.sh` - P2.6 Portfolio Gate deployment
- ⏳ `ops/p28_deploy_and_proof.sh` - P2.8 Risk Governor deployment (when created)
- ⏳ Rollback scripts

### Dependencies

**Runtime:** Python 3.10+ (stdlib only, no external packages)

**Optional tools** (for evidence collection):
- `curl` - Metrics collection
- `redis-cli` - Redis snapshots
- `systemctl` - Service status
- `journalctl` - Service logs
- `git` - Commit SHA collection

---

## Infrastructure Audit Scripts

**Tool:** `ops/permit_audit.sh` and `ops/proof_order_id.sh`

Read-only verification scripts for Quantum Trader's permit pipeline and trade execution proof.

### permit_audit.sh - 3-Permit Gate Infrastructure Audit

Audits all three permit sources:
1. **Governor Permits** (`quantum:permit:{plan_id}`)
2. **P2.6 Portfolio Gate** (`quantum:permit:p26:{plan_id}`)
3. **P3.3 Position Gate** (`quantum:permit:p33:{plan_id}`)

#### Usage

**LOCAL execution** (default - runs on current machine):
```bash
# Quick check with 3 samples per type
./ops/permit_audit.sh --sample 3

# JSON output for parsing
./ops/permit_audit.sh --json

# Single sample for quick verification
./ops/permit_audit.sh --sample 1
```

**REMOTE execution** (via SSH to VPS):
```bash
# From Windows/WSL to VPS
./ops/permit_audit.sh --remote --sample 3

# Custom SSH host/key
./ops/permit_audit.sh --remote --host user@host --key ~/.ssh/mykey
```

#### Exit Codes
- **0**: All three permit types present (healthy)
- **2**: One or more permit types missing (infrastructure failure)
- **1**: Error (connection failure, Redis down)

#### Output Example
```
═══════════════════════════════════════════════════════════
Quantum Trader 3-Permit Gate Infrastructure Audit
═══════════════════════════════════════════════════════════
Total Permits: 47

GOVERNOR PERMITS: 15 keys
P2.6 PERMITS:     16 keys
P3.3 PERMITS:     16 keys

STATUS: ✅ ALL THREE PERMIT SOURCES ACTIVE
═══════════════════════════════════════════════════════════
```

### proof_order_id.sh - Trade Execution Proof

Searches `quantum:stream:apply.result` for executed trades with order_id proof.

**Proof criteria:**
- `executed: true`
- `order_id` field exists and non-empty

#### Usage

**LOCAL execution** (default):
```bash
# Check last 50 apply.result entries
./ops/proof_order_id.sh --count 50

# Filter by symbol
./ops/proof_order_id.sh --symbol SOLUSDT --count 100

# Filter by plan_id
./ops/proof_order_id.sh --plan_id harv_sol_001 --count 100
```

**REMOTE execution** (via SSH):
```bash
# From Windows/WSL to VPS
./ops/proof_order_id.sh --remote --count 50

# Verify specific trade remotely
./ops/proof_order_id.sh --remote --symbol BTCUSDT --count 200
```

#### Exit Codes
- **0**: Execution proof found (order_id present)
- **3**: No proof found (zero trades or no order_id)
- **1**: Error (connection failure, Redis down)

#### Output Example
```
═══════════════════════════════════════════════════════════
Quantum Trader Order Execution Proof
═══════════════════════════════════════════════════════════
Searching 50 apply.result entries...

✅ PROOF FOUND: executed=true + order_id present

Entry ID:    1738035441234-0
Symbol:      SOLUSDT
Plan ID:     harv_sol_001
Executed:    true
Order ID:    12345678
Side:        BUY
Quantity:    0.5
Price:       125.45
Timestamp:   2026-01-27 15:30:41
═══════════════════════════════════════════════════════════
```

### Git Workflow Note

**⚠️ IMPORTANT:** VPS is deploy-only. Do NOT commit from VPS.

**Correct workflow:**
1. Modify scripts on Windows/development machine
2. Test locally: `./ops/permit_audit.sh --sample 1`
3. Commit and push from Windows: `git commit && git push`
4. Deploy to VPS: `ssh root@vps 'cd /home/qt/quantum_trader && git pull'`
5. Test on VPS locally: `./ops/permit_audit.sh --sample 1`
6. Test remote access: `./ops/permit_audit.sh --remote --sample 1` (from Windows)

**Scripts default to LOCAL execution** to prevent SSH recursion when run on VPS.
