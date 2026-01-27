# Ops Governor v1.0 (P4) — Quantum Trader

## Purpose
Ops Governor v1.0 is the operational governance layer for **audit-safe infrastructure work** in Quantum Trader.
It enforces: **Safety > Proof > Minimalism > Speed**.

This is a *hard* standard for any action that touches production-like environments (VPS, observability stack, configs).

## Governance Layers (context)
1. Trading Governor
2. Risk Governor
3. Analytics Governor
4. **Ops Governor (this document)**

## Core Principles (non-negotiable)
- **Scope Contract first**: Explicit ALLOWED vs NOT ALLOWED boundaries.
- **Rollback-first mindset**: If it can't be rolled back cleanly, it's not ops-safe.
- **Sequential execution**: One step at a time.
- **Verification gates after every step**: Status + logs + functional check.
- **Audit logging**: Timestamped commands and evidence.
- **Fail-safe rule**: When uncertain → **HALT and ASK** (do not guess).

## Mandatory Operation Declaration
Every ops run MUST begin with:

- OPERATION_NAME
- OBJECTIVE
- RISK_CLASS: `READ_ONLY | LOW_RISK_CONFIG | SERVICE_RESTART | FILESYSTEM_WRITE`
- BLAST_RADIUS (what can break)
- ROLLBACK_STRATEGY (how to revert)
- ALLOWED_PATHS (hard allowlist)
- ALLOWED_SERVICES (hard allowlist)

If any of these are missing → operation is invalid.

## Scope Contract Template (copy/paste)
ALLOWED:
- Modify ONLY files under: <ALLOWED_PATHS>
- Restart/reload ONLY services: <ALLOWED_SERVICES>
- Read-only inspection of logs/config/metrics/status

NOT ALLOWED:
- No trading logic modifications
- No writes to trading Redis streams:
  - quantum:stream:apply.plan
  - quantum:stream:trade.intent
  - quantum:stream:ai.decision
- No secrets/keys/exchange creds changes
- No Grafana DB modifications
- No firewall/network/user permission changes

## Execution Contract
For every step:

[STEP N | UTC timestamp]
Command:
<exact command>
Result:
<relevant output>
Status: PASS/FAIL
Verification:
<verification command + output>

Rules:
- If verification FAILS → **STOP** and execute rollback.
- Never do multiple changes without intermediate verification.
- Never "fix forward" blindly.

## Verification Gates (minimum)
Choose what fits the operation, but include at least:
- `systemctl is-active <services>`
- `journalctl -u <service> ... | grep -iE "error|fail|panic|fatal"`
- Functional checks:
  - HTTP: `curl -fsS http://localhost:<port>/<health>`
  - Prometheus: `promtool check config ...`
  - Grafana: provisioning logs, service health, etc.

## Reporting Contract (final output)
Every operation must end with:

- EXEC SUMMARY (1 paragraph)
- CHANGES MADE (files, services)
- VERIFICATION EVIDENCE (key outputs)
- RISK ASSESSMENT (residual risk)
- NEXT SAFE ACTIONS (what to do next, bounded)

## Standard Templates
- See `docs/OPS_RUNBOOK_TEMPLATE.md` for a runnable ops template.
- See `ops/ops_prompt_template_v1.txt` for the canonical prompt body.
- Use `ops/ops_prompt_fill.py` to generate a filled prompt fast.
