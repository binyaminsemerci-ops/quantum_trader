#!/usr/bin/env python3
"""
ops_prompt_fill.py
Generate a filled Ops Governor prompt from a template.
Optionally generate P5 ledger YAML snippets.

Stdlib only. Python 3.10+.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent

DEFAULT_TEMPLATE = Path("ops/ops_prompt_template_v1.txt")


def _block(items: list[str]) -> str:
    if not items:
        return "- <none>"
    return "\n".join(f"- {x}" for x in items)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Generate a filled Ops Governor prompt from ops_prompt_template_v1.txt"
    )
    p.add_argument("--template", default=str(DEFAULT_TEMPLATE), help="Path to template txt")
    p.add_argument("--ledger", action="store_true", help="Generate P5 ledger YAML snippet instead of prompt")
    p.add_argument("--operation", required=True, help="OPERATION_NAME")
    p.add_argument("--objective", required=True, help="OBJECTIVE")
    p.add_argument(
        "--risk_class",
        required=True,
        choices=["READ_ONLY", "LOW_RISK_CONFIG", "SERVICE_RESTART", "FILESYSTEM_WRITE"],
        help="RISK_CLASS",
    )
    p.add_argument("--blast_radius", required=True, help="BLAST_RADIUS")
    p.add_argument("--rollback_strategy", required=True, help="ROLLBACK_STRATEGY")
    p.add_argument(
        "--allowed_paths",
        required=True,
        nargs="+",
        help="One or more allowed paths (space separated)",
    )
    p.add_argument(
        "--allowed_services",
        required=True,
        nargs="+",
        help="One or more allowed services (space separated)",
    )
    
    # Ledger-specific options
    p.add_argument("--operation_id", help="Operation ID (e.g., OPS-2026-01-27-003). Auto-generated if omitted.")
    p.add_argument("--requested_by", default="SELF", help="Username/email of requester (default: SELF)")
    p.add_argument("--approved_by", default="SELF", help="Username/email of approver (default: SELF)")
    p.add_argument("--changes_summary", help="1-line summary of changes (required for ledger mode)")
    p.add_argument("--outcome", choices=["SUCCESS", "ROLLBACK", "PARTIAL"], default="SUCCESS", help="Operation outcome")
    p.add_argument("--notes", help="Optional context/notes for ledger")
    
    args = p.parse_args()

    # Ledger mode
    if args.ledger:
        if not args.changes_summary:
            raise SystemExit("Error: --changes_summary is required when using --ledger")
        
        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        
        # Auto-generate operation_id if not provided
        op_id = args.operation_id
        if not op_id:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            op_id = f"OPS-{today}-NNN"  # User should replace NNN with sequence number
        
        ledger_yaml = f"""```yaml
---
operation_id: {op_id}
operation_name: {args.operation}
requested_by: {args.requested_by}
approved_by: {args.approved_by}
approval_timestamp: {now_utc}
execution_timestamp: {now_utc}
risk_class: {args.risk_class}
blast_radius: {args.blast_radius}
changes_summary: {args.changes_summary}
rollback_ref: {args.rollback_strategy}
outcome: {args.outcome}
notes: {args.notes or 'N/A'}
```"""
        
        print(ledger_yaml.strip())
        print("\n# Add this to docs/OPS_CHANGELOG.md under the current month section")
        print(f"# Replace 'NNN' in operation_id with the next sequence number if auto-generated")
        return 0

    # Prompt mode (original behavior)
    tpl_path = Path(args.template)
    if not tpl_path.exists():
        raise SystemExit(f"Template not found: {tpl_path}")

    tpl = tpl_path.read_text(encoding="utf-8")

    allowed_paths_block = _block(args.allowed_paths)
    allowed_services_block = _block(args.allowed_services)

    filled = tpl
    filled = filled.replace("<OPERATION_NAME>", args.operation)
    filled = filled.replace("<OBJECTIVE>", args.objective)
    filled = filled.replace("<RISK_CLASS>", args.risk_class)
    filled = filled.replace("<BLAST_RADIUS>", args.blast_radius)
    filled = filled.replace("<ROLLBACK_STRATEGY>", args.rollback_strategy)

    # For the inline placeholders we keep the list readable.
    filled = filled.replace("<ALLOWED_PATHS>", ", ".join(args.allowed_paths))
    filled = filled.replace("<ALLOWED_SERVICES>", ", ".join(args.allowed_services))
    filled = filled.replace("<ALLOWED_PATHS_BLOCK>", allowed_paths_block)
    filled = filled.replace("<ALLOWED_SERVICES_BLOCK>", allowed_services_block)

    print(dedent(filled).strip() + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
