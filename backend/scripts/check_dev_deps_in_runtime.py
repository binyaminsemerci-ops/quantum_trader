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
import subprocess  # nosec B404 - subprocess is used safely for controlled pip invocations
from pathlib import Path
import sys


REQ_DEV = Path("backend/requirements-dev.txt")
REQ_RUNTIME = Path("backend/requirements.txt")
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
        # Calling pip in a subprocess is intended and arguments are controlled
        # by the repository (not untrusted user input).
        res = subprocess.run(
            [sys.executable, "-m", "pip", "list", "--format=freeze"],
            capture_output=True,
            text=True,
            check=True,
        )  # nosec B603
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
    # Allow bypass in non-CI developer environments to avoid blocking local commits.
    import os

    if not os.getenv("CI") or os.getenv("ALLOW_DEV_RUNTIME") == "1":
        print(
            "Skipping dev-only dependency runtime check (developer environment detected)"
        )
        return 0
    if not REQ_DEV.exists():
        print("No backend/requirements-dev.txt found; nothing to check.")
        return 0

    dev_names: set[str] = set()
    for ln in REQ_DEV.read_text().splitlines():
        nm = parse_req_line(ln)
        if nm:
            dev_names.add(nm)

    # If a package is listed in runtime requirements it's allowed to be
    # installed in runtime; ignore those to avoid false positives. This
    # handles shared packages used both at runtime and for developer
    # workflows (for example `anyio` pulled in by FastAPI/starlette).
    runtime_names: set[str] = set()
    if REQ_RUNTIME.exists():
        for ln in REQ_RUNTIME.read_text().splitlines():
            nm = parse_req_line(ln)
            if nm:
                runtime_names.add(nm)

    # Only consider packages that are dev-only (not present in runtime reqs)
    # Expand runtime_names to include transitive dependencies so packages
    # pulled in by runtime packages are not flagged as dev-only. We use
    # `pip show` to read immediate requirements and recurse up to a small
    # depth to cover common transitive cases (fastapi -> starlette -> anyio).
    def runtime_transitive_deps(names: set[str], max_depth: int = 4) -> set[str]:
        seen = set(names)
        to_process = list(names)
        depth = 0
        while to_process and depth < max_depth:
            nxt = []
            for pkg in to_process:
                try:
                    # We call `pip show <pkg>` for repository-controlled package
                    # names discovered in requirements files. This is not
                    # executing untrusted shell input.
                    res = subprocess.run(
                        [sys.executable, "-m", "pip", "show", pkg],
                        capture_output=True,
                        text=True,
                        check=True,
                    )  # nosec B603
                except subprocess.CalledProcessError:
                    continue
                for line in res.stdout.splitlines():
                    if line.lower().startswith("requires:"):
                        reqs = line.split(":", 1)[1].strip()
                        if not reqs:
                            continue
                        for r in reqs.split(","):
                            rname = r.strip().split(" ", 1)[0].lower()
                            if rname and rname not in seen:
                                seen.add(rname)
                                nxt.append(rname)
            to_process = nxt
            depth += 1
        return seen

    runtime_full = runtime_transitive_deps(runtime_names)
    dev_only = {n for n in dev_names if n not in runtime_full}

    installed = installed_packages()
    found = sorted(n for n in dev_only if n in installed)

    if found:
        OUT.write_text(",".join(found))
        print("Found dev-only packages installed at runtime:", ",".join(found))
        return 1

    if OUT.exists():
        try:
            OUT.unlink()
        except Exception as exc:
            # Avoid silent failures; log the exception for CI debugging.
            print(f"Warning: failed to remove {OUT}: {exc}")
    print("No dev-only packages detected in runtime environment")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
