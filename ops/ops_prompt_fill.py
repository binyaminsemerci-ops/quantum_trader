#!/usr/bin/env python3
"""
ops_prompt_fill.py
Generate a filled Ops Governor prompt from a template.

Stdlib only. Python 3.10+.
"""

from __future__ import annotations

import argparse
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
    args = p.parse_args()

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
