# Operations Changelog & Signoff Ledger

> **Purpose:** Audit trail for approved operational changes in Quantum Trader.  
> **Scope:** SERVICE_RESTART, FILESYSTEM_WRITE, or any ops requiring approval/signoff.  
> **Format:** YAML frontmatter blocks (most recent first).

---

## 2026-01

```yaml
---
operation_id: OPS-2026-01-27-001
operation_name: P4 Ops Governor Implementation
requested_by: binyamin
approved_by: SELF
approval_timestamp: 2026-01-27T18:45:00Z
execution_timestamp: 2026-01-27T18:50:33Z
risk_class: LOW_RISK_CONFIG
blast_radius: Repository only (docs/ops tooling)
changes_summary: Added governance docs (OPS_GOVERNOR, RUNBOOK) + CLI prompt generator
rollback_ref: git-revert-ce8ef922
outcome: SUCCESS
notes: Verified with --help and test generation. Pushed to origin/main. P5 ledger added as follow-up.
```

---

## Template (copy for new entries)

```yaml
---
operation_id: OPS-YYYY-MM-DD-NNN
operation_name: <short name>
requested_by: <username | SELF>
approved_by: <username | SELF>
approval_timestamp: <ISO8601 UTC>
execution_timestamp: <ISO8601 UTC>
risk_class: <READ_ONLY | LOW_RISK_CONFIG | SERVICE_RESTART | FILESYSTEM_WRITE>
blast_radius: <what can break>
changes_summary: <1-line summary>
rollback_ref: <git commit | command | procedure>
outcome: SUCCESS | ROLLBACK | PARTIAL
notes: <optional context>
```

---

## Usage Notes

- **Add entries chronologically** (newest at top of month section).
- **Commit with operation changes** or immediately after.
- **Archive monthly** (optional): move old entries to `ops/ledger/YYYY-MM.yaml`.
- **Rollbacks:** Update original entry with `outcome: ROLLBACK` + rollback timestamp.
- **Git-native alternative:** Use annotated tags instead (`git tag -a ops-2026-01-27-001 -m "<yaml>"`).
