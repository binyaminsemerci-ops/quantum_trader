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
