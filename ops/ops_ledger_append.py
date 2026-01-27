#!/usr/bin/env python3
"""
ops_ledger_append.py - P5+ Automated Ops Ledger Agent

Auto-appends YAML ledger entries to docs/OPS_CHANGELOG.md after deploy/rollback.
Idempotent, evidence-collecting, safe (deploy succeeds even if ledger fails unless strict).

Stdlib only. Python 3.10+.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

CHANGELOG_PATH = Path("docs/OPS_CHANGELOG.md")


def run_cmd(cmd: list[str], check: bool = False, capture: bool = True) -> tuple[int, str]:
    """Run command, return (exit_code, output)"""
    try:
        result = subprocess.run(
            cmd,
            capture_output=capture,
            text=True,
            timeout=10,
            check=check
        )
        return result.returncode, result.stdout.strip()
    except subprocess.TimeoutExpired:
        return 1, "TIMEOUT"
    except Exception as e:
        return 1, f"ERROR: {e}"


def get_git_info() -> tuple[str, str]:
    """Get current git commit SHA and branch"""
    rc, sha = run_cmd(["git", "rev-parse", "HEAD"])
    if rc != 0:
        sha = "UNKNOWN"
    else:
        sha = sha[:8]  # Short SHA
    
    rc, branch = run_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    if rc != 0:
        branch = "UNKNOWN"
    
    return sha, branch


def get_hostname() -> str:
    """Get hostname"""
    rc, hostname = run_cmd(["hostname"])
    return hostname if rc == 0 else "UNKNOWN"


def validate_rollback(outcome: str, rollback_of: str | None, strict: bool) -> None:
    """Validate rollback_of parameter when outcome=ROLLBACK.
    
    Args:
        outcome: Operation outcome (SUCCESS, PARTIAL, ROLLBACK)
        rollback_of: Original operation_id being rolled back
        strict: Whether to fail (True) or warn (False) on validation errors
    
    Raises:
        ValueError: If validation fails and strict=True
    """
    if outcome != "ROLLBACK":
        return  # No validation needed for non-rollback outcomes
    
    # Validation 1: rollback_of must be provided for ROLLBACK outcomes
    if not rollback_of or not rollback_of.strip():
        msg = "ERROR: outcome=ROLLBACK requires --rollback_of parameter"
        if strict:
            raise ValueError(msg)
        else:
            print(f"WARNING: {msg} (continuing in non-strict mode)")
            return
    
    # Validation 2: rollback_of should match operation_id format (OPS-YYYY-MM-DD-NNN)
    pattern = r'^OPS-\d{4}-\d{2}-\d{2}-\d{3}$'
    if not re.match(pattern, rollback_of):
        msg = f"ERROR: rollback_of '{rollback_of}' does not match expected format OPS-YYYY-MM-DD-NNN"
        if strict:
            raise ValueError(msg)
        else:
            print(f"WARNING: {msg} (continuing in non-strict mode)")


def get_service_status(service: str) -> dict:
    """Get systemd service status and recent logs"""
    rc, active = run_cmd(["systemctl", "is-active", service])
    status = active if rc == 0 else "inactive"
    
    # Get last 30 log lines
    rc, logs = run_cmd(["journalctl", "-u", service, "-n", "30", "--no-pager"])
    log_snippet = logs if rc == 0 else "NO_LOGS"
    
    return {"status": status, "logs": log_snippet}


def get_proof_hash(proof_path: str) -> Optional[dict]:
    """Get proof file hash, size, mtime"""
    p = Path(proof_path)
    if not p.exists():
        return None
    
    sha256 = hashlib.sha256(p.read_bytes()).hexdigest()[:16]  # Short hash
    size = p.stat().st_size
    mtime = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).isoformat()
    
    return {"sha256": sha256, "size_bytes": size, "mtime_utc": mtime}


def collect_metrics(urls: list[str], greps: list[str]) -> list[str]:
    """Curl metrics endpoints and grep for patterns"""
    results = []
    for url in urls:
        rc, output = run_cmd(["curl", "-s", url])
        if rc != 0:
            results.append(f"# {url}: FAILED")
            continue
        
        # Apply grep filters
        lines = output.split("\n")
        for grep_pattern in greps:
            filtered = [l for l in lines if re.search(grep_pattern, l, re.IGNORECASE)]
            if filtered:
                results.append(f"# {url} | grep '{grep_pattern}'")
                results.extend(filtered[:10])  # Limit to 10 lines per pattern
    
    return results


def collect_redis(cmds: list[str]) -> list[str]:
    """Execute redis-cli commands"""
    results = []
    for cmd in cmds:
        cmd_parts = ["redis-cli"] + cmd.split()
        rc, output = run_cmd(cmd_parts)
        if rc != 0:
            results.append(f"# redis-cli {cmd}: FAILED")
        else:
            results.append(f"# redis-cli {cmd}")
            results.append(output)
    
    return results


def find_next_operation_id(date_str: str) -> str:
    """Scan changelog for existing IDs on date and generate next sequence number"""
    if not CHANGELOG_PATH.exists():
        return f"OPS-{date_str}-001"
    
    content = CHANGELOG_PATH.read_text(encoding="utf-8")
    pattern = re.compile(rf"operation_id:\s*OPS-{re.escape(date_str)}-(\d+)")
    
    max_seq = 0
    for match in pattern.finditer(content):
        seq = int(match.group(1))
        max_seq = max(max_seq, seq)
    
    next_seq = max_seq + 1
    return f"OPS-{date_str}-{next_seq:03d}"


def operation_id_exists(op_id: str) -> bool:
    """Check if operation_id already exists in changelog"""
    if not CHANGELOG_PATH.exists():
        return False
    
    content = CHANGELOG_PATH.read_text(encoding="utf-8")
    pattern = rf"operation_id:\s*{re.escape(op_id)}"
    return bool(re.search(pattern, content))


def append_ledger_entry(
    operation_id: str,
    operation: str,
    objective: str,
    risk_class: str,
    blast_radius: str,
    allowed_paths: list[str],
    allowed_services: list[str],
    changes_summary: str,
    outcome: str,
    git_commit: str,
    git_branch: str,
    timestamp_utc: str,
    hostname: str,
    services_status: dict,
    proof_info: Optional[dict],
    metrics_lines: list[str],
    redis_lines: list[str],
    notes: str,
    rollback_of: Optional[str] = None,
) -> None:
    """Append YAML entry to docs/OPS_CHANGELOG.md under correct month"""
    
    # Determine month section (YYYY-MM)
    month_section = timestamp_utc[:7]  # "2026-01"
    
    # Read existing content
    if CHANGELOG_PATH.exists():
        content = CHANGELOG_PATH.read_text(encoding="utf-8")
    else:
        # Create initial structure
        content = """# Operations Changelog & Signoff Ledger

> **Purpose:** Audit trail for approved operational changes in Quantum Trader.  
> **Scope:** SERVICE_RESTART, FILESYSTEM_WRITE, or any ops requiring approval/signoff.  
> **Format:** YAML frontmatter blocks (most recent first).

---

"""
    
    # Check if month section exists
    month_pattern = rf"^## {re.escape(month_section)}$"
    if not re.search(month_pattern, content, re.MULTILINE):
        # Insert new month section at top of content (after header)
        header_end = content.find("---\n") + 4
        new_section = f"\n## {month_section}\n\n"
        content = content[:header_end] + new_section + content[header_end:]
    
    # Build YAML entry
    yaml_lines = [
        "```yaml",
        "---",
        f"operation_id: {operation_id}",
        f"operation_name: {operation}",
        "requested_by: SELF",
        "approved_by: SELF",
        f"approval_timestamp: {timestamp_utc}",
        f"execution_timestamp: {timestamp_utc}",
        f"git_commit: {git_commit}",
        f"git_branch: {git_branch}",
        f"hostname: {hostname}",
        f"risk_class: {risk_class}",
        f"blast_radius: {blast_radius}",
        f"changes_summary: {changes_summary}",
        f"objective: {objective}",
        f"outcome: {outcome}",
    ]
    
    # Rollback reference
    if rollback_of:
        yaml_lines.append(f"rollback_of: {rollback_of}")
    
    # Allowed paths and services
    if allowed_paths:
        yaml_lines.append("allowed_paths:")
        for path in allowed_paths:
            yaml_lines.append(f"  - {path}")
    
    if allowed_services:
        yaml_lines.append("allowed_services:")
        for svc in allowed_services:
            yaml_lines.append(f"  - {svc}")
    
    # Services status
    if services_status:
        yaml_lines.append("services_status:")
        for svc, info in services_status.items():
            yaml_lines.append(f"  {svc}: {info['status']}")
    
    # Proof info
    if proof_info:
        yaml_lines.append("proof_file:")
        yaml_lines.append(f"  sha256: {proof_info['sha256']}")
        yaml_lines.append(f"  size_bytes: {proof_info['size_bytes']}")
        yaml_lines.append(f"  mtime_utc: {proof_info['mtime_utc']}")
    
    # Metrics snapshot
    if metrics_lines:
        yaml_lines.append("metrics_snapshot: |")
        for line in metrics_lines[:30]:  # Limit to 30 lines
            yaml_lines.append(f"  {line}")
    
    # Redis snapshot
    if redis_lines:
        yaml_lines.append("redis_snapshot: |")
        for line in redis_lines[:30]:
            yaml_lines.append(f"  {line}")
    
    # Notes
    if notes:
        yaml_lines.append(f"notes: {notes}")
    
    yaml_lines.append("```")
    
    # Combine into block
    yaml_block = "\n".join(yaml_lines) + "\n\n"
    
    # Find insertion point (right after month section header)
    month_header = f"## {month_section}\n"
    insertion_point = content.find(month_header)
    if insertion_point == -1:
        raise RuntimeError(f"Month section {month_section} not found after creation")
    
    insertion_point += len(month_header)
    
    # Insert
    new_content = content[:insertion_point] + "\n" + yaml_block + content[insertion_point:]
    
    # Write back
    CHANGELOG_PATH.write_text(new_content, encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(
        description="P5+ Auto-ledger: Append operation entry to docs/OPS_CHANGELOG.md"
    )
    
    # Core fields
    p.add_argument("--operation_id", help="Operation ID (auto-generated if omitted)")
    p.add_argument("--operation", required=True, help="Operation name")
    p.add_argument("--objective", required=True, help="What this operation achieves")
    p.add_argument("--risk_class", required=True, 
                   choices=["READ_ONLY", "LOW_RISK_CONFIG", "SERVICE_RESTART", "FILESYSTEM_WRITE"])
    p.add_argument("--blast_radius", required=True, help="What can break")
    p.add_argument("--changes_summary", required=True, help="1-line summary of changes")
    
    # Paths and services
    p.add_argument("--allowed_paths", action="append", default=[], help="Allowed paths (repeatable)")
    p.add_argument("--allowed_services", action="append", default=[], help="Allowed services (repeatable)")
    
    # Evidence
    p.add_argument("--proof_path", help="Path to proof file")
    p.add_argument("--metrics_urls", action="append", default=[], help="Metrics URLs to curl (repeatable)")
    p.add_argument("--metrics_grep", action="append", default=[], help="Grep patterns for metrics (repeatable)")
    p.add_argument("--redis_cmds", action="append", default=[], help="Redis commands (repeatable)")
    
    # Optional
    p.add_argument("--outcome", default="SUCCESS", choices=["SUCCESS", "PARTIAL", "ROLLBACK"])
    p.add_argument("--rollback_of", help="Original operation_id being rolled back (for outcome=ROLLBACK)")
    p.add_argument("--notes", default="Auto-generated by P5+ ops_ledger_append.py")
    p.add_argument("--strict", action="store_true", help="Fail deploy if ledger append fails")
    
    args = p.parse_args()
    
    try:
        # Validate rollback parameters
        validate_rollback(args.outcome, args.rollback_of, args.strict)
        
        # Get auto-collected info
        git_commit, git_branch = get_git_info()
        hostname = get_hostname()
        timestamp_utc = datetime.now(timezone.utc).isoformat()
        date_str = timestamp_utc[:10]  # YYYY-MM-DD
        
        # Generate operation_id if needed
        operation_id = args.operation_id
        if not operation_id:
            operation_id = find_next_operation_id(date_str)
            print(f"Auto-generated operation_id: {operation_id}")
        
        # Check idempotency
        if operation_id_exists(operation_id):
            print(f"✓ Operation {operation_id} already recorded - skipping (idempotent)")
            return 0
        
        # Collect service status
        services_status = {}
        for svc in args.allowed_services:
            services_status[svc] = get_service_status(svc)
        
        # Collect proof hash
        proof_info = None
        if args.proof_path:
            proof_info = get_proof_hash(args.proof_path)
            if not proof_info:
                print(f"Warning: Proof file not found: {args.proof_path}")
        
        # Collect metrics
        metrics_lines = []
        if args.metrics_urls:
            metrics_lines = collect_metrics(args.metrics_urls, args.metrics_grep)
        
        # Collect Redis
        redis_lines = []
        if args.redis_cmds:
            redis_lines = collect_redis(args.redis_cmds)
        
        # Append entry
        append_ledger_entry(
            operation_id=operation_id,
            operation=args.operation,
            objective=args.objective,
            risk_class=args.risk_class,
            blast_radius=args.blast_radius,
            allowed_paths=args.allowed_paths,
            allowed_services=args.allowed_services,
            changes_summary=args.changes_summary,
            outcome=args.outcome,
            git_commit=git_commit,
            git_branch=git_branch,
            timestamp_utc=timestamp_utc,
            hostname=hostname,
            services_status=services_status,
            proof_info=proof_info,
            metrics_lines=metrics_lines,
            redis_lines=redis_lines,
            notes=args.notes,
            rollback_of=args.rollback_of,
        )
        
        print(f"✓ Ledger entry appended: {operation_id}")
        print(f"  File: {CHANGELOG_PATH}")
        print(f"  Commit: {git_commit} ({git_branch})")
        print(f"  Services: {', '.join(args.allowed_services)}")
        
        return 0
        
    except Exception as e:
        print(f"ERROR: Ledger append failed: {e}", file=sys.stderr)
        if args.strict:
            print("STRICT_LEDGER=true - failing deployment", file=sys.stderr)
            return 1
        else:
            print("STRICT_LEDGER=false - deployment continues despite ledger failure", file=sys.stderr)
            return 0


if __name__ == "__main__":
    sys.exit(main())
