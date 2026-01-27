# Ops Runbook Template v1.0 (P4)

> Use this runbook for any ops task. Fill placeholders before execution.

## 0) Operation Declaration (required)
**OPERATION_NAME:** <name>  
**OBJECTIVE:** <objective>  
**RISK_CLASS:** <READ_ONLY | LOW_RISK_CONFIG | SERVICE_RESTART | FILESYSTEM_WRITE>  
**BLAST_RADIUS:** <what can break>  
**ROLLBACK_STRATEGY:** <exact rollback plan>  

**ALLOWED_PATHS:**
- <path1>
- <path2>

**ALLOWED_SERVICES:**
- <svc1>
- <svc2>

## 1) Scope Contract (hard boundary)
**ALLOWED**
- Modify ONLY files under ALLOWED_PATHS
- Restart/reload ONLY services under ALLOWED_SERVICES
- Read-only inspection of logs/config/metrics/status

**NOT ALLOWED**
- No trading logic changes
- No writes to trading Redis streams: apply.plan / trade.intent / ai.decision
- No secrets/keys/exchange creds changes
- No Grafana DB modifications
- No firewall/network/user permission changes

## 2) Execution Plan (sequential)
List steps as atomic actions:

1. Step 1: <what>
2. Step 2: <what>
3. Step 3: <what>

## 3) Evidence + Verification Gates
For each step, capture:

[STEP N | UTC timestamp]  
Command:  
Result:  
Status: PASS/FAIL  
Verification:  

Minimum gates:
- systemctl is-active <svc>
- journalctl -u <svc> (filter errors)
- functional curl/promtool check

## 4) Rollback Procedure (must be executable)
Write rollback commands/steps here, not prose.

## 5) Final Report (required)
**EXEC SUMMARY:**  
**CHANGES MADE:**  
**VERIFICATION EVIDENCE:**  
**RISK ASSESSMENT:**  
**NEXT SAFE ACTIONS:**  

## 6) Signoff & Ledger Entry (optional — P5)
> Use for SERVICE_RESTART, FILESYSTEM_WRITE, or multi-person approval workflows.

```yaml
---
operation_id: OPS-YYYY-MM-DD-NNN
operation_name: <from section 0>
requested_by: <username | SELF>
approved_by: <username | SELF>
approval_timestamp: <ISO8601 UTC | use `date -u +"%Y-%m-%dT%H:%M:%SZ"`>
execution_timestamp: <ISO8601 UTC>
risk_class: <from section 0>
blast_radius: <from section 0>
changes_summary: <1-line: what was changed>
rollback_ref: <git commit hash | command | file reference>
outcome: SUCCESS | ROLLBACK | PARTIAL
notes: <optional context>
```

**Ledger Storage:**  
- Append to `docs/OPS_CHANGELOG.md` (create if missing), or  
- Add to `ops/ledger/<YYYY-MM>.yaml`, or  
- Create annotated git tag: `git tag -a ops-<date>-<seq> -m "<yaml block>"`  

**Example:**  
```yaml
---
operation_id: OPS-2026-01-27-001
operation_name: Deploy Prometheus Alert Rules
requested_by: binyamin
approved_by: SELF
approval_timestamp: 2026-01-27T14:32:00Z
execution_timestamp: 2026-01-27T14:35:12Z
risk_class: LOW_RISK_CONFIG
blast_radius: Prometheus alerting only
changes_summary: Added 3 new alert rules for RL agent monitoring
rollback_ref: git-revert-ce8ef922
outcome: SUCCESS
notes: Verified with promtool, reloaded Prometheus, alerts firing correctly.
```  
