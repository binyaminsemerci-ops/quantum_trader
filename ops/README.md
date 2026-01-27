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

# Generate a filled prompt
python ops/ops_prompt_fill.py \
  --operation "Your operation name" \
  --objective "What this achieves" \
  --risk_class READ_ONLY \
  --blast_radius "What can be affected" \
  --rollback_strategy "How to revert" \
  --allowed_paths "/path1" "/path2" \
  --allowed_services "service1" "service2"
```

## Documentation
- `docs/OPS_GOVERNOR.md` - Core governance framework
- `docs/OPS_RUNBOOK_TEMPLATE.md` - Template for ops runbooks
