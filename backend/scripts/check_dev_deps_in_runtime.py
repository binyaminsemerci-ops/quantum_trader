#!/usr/bin/env python3
"""Check for dev-only packages present in the runtime environment.

This script reads `backend/requirements-dev.txt` (ignoring -r and comments),
then compares the package names against `pip list`. If any dev-only package
is installed in the current environment, the script writes a comma-separated
list to `backend/dev_in_runtime.txt` and exits with code 1. Otherwise it
removes that file (if present) and exits 0.

This is intended to be used in CI to warn when developer-only dependencies
are accidentally installed into runtime environments.
"""
from __future__ import annotations
import re
import subprocess
from pathlib import Path
import sys


REQ_DEV = Path("backend/requirements-dev.txt")
OUT = Path("backend/dev_in_runtime.txt")


def parse_req_line(line: str) -> str | None:
    line = line.strip()
    if not line or line.startswith("#") or line.startswith("-r"):
        return None
    # Remove environment markers
    line = line.split(";", 1)[0].strip()
    # Strip extras and version specifiers: take text up to first '[<>=!~'
    m = re.split(r"[\[<>=!~]", line, 1)
    name = m[0].strip()
    if not name:
        return None
    return name.lower()


def installed_packages() -> set[str]:
    # Use pip list --format=freeze for a reliable list in CI
    try:
        res = subprocess.run([sys.executable, "-m", "pip", "list", "--format=freeze"], capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError:
        return set()
    pkgs = set()
    for line in res.stdout.splitlines():
        if not line:
            continue
        # lines look like 'name==1.2.3' or editable installs; split on '=='
        if "==" in line:
            pkgs.add(line.split("==", 1)[0].lower())
        else:
            pkgs.add(line.split("=", 1)[0].lower())
    return pkgs


def main() -> int:
    if not REQ_DEV.exists():
        print("No backend/requirements-dev.txt found; nothing to check.")
        return 0

    dev_names: set[str] = set()
    for ln in REQ_DEV.read_text().splitlines():
        nm = parse_req_line(ln)
        if nm:
            dev_names.add(nm)

    installed = installed_packages()
    found = sorted(n for n in dev_names if n in installed)

    if found:
        OUT.write_text(",".join(found))
        print("Found dev-only packages installed at runtime:", ",".join(found))
        return 1

    if OUT.exists():
        try:
            OUT.unlink()
        except Exception:
            pass
    print("No dev-only packages detected in runtime environment")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
