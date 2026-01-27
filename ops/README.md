# Ops Tools (P4)

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
