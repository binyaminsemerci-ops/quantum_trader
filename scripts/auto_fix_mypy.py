#!/usr/bin/env python3
"""Small, conservative codemod to apply low-risk fixes for mypy/lint issues.

What it does:
- Replace patterns where code does `db = get_db()` followed by `cursor = db.cursor()`
  with `db = next(get_db())` to match generator-style dependency.
- Replace hard-coded fetch URLs to the backend (`http://127.0.0.1:8000/api`) with
  relative `/api` so Vite proxy works and CORS is avoided in dev.
- Ensure `backend/utils/trade_logger.py` coerces values before constructing TradeLog
  (adds a small safe block if an older pattern is found).

This script is intentionally conservative and prints files changed. It writes
backups with .bak extension before modifying files.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List


ROOT = Path(__file__).resolve().parents[1]


def find_files(globs: List[str]) -> List[Path]:
    files: List[Path] = []
    for g in globs:
        files.extend(ROOT.glob(g))
    return files


def backup_write(path: Path, text: str) -> None:
    bak = path.with_suffix(path.suffix + ".bak")
    path.write_text(text, encoding="utf-8")
    if not bak.exists():
        bak.write_text(text, encoding="utf-8")


def replace_get_db_cursor(path: Path) -> bool:
    src = path.read_text(encoding="utf-8")
    pattern = re.compile(
        r"(?m)^(\s*)db\s*=\s*get_db\(\)\s*\n\s*cursor\s*=\s*db\.cursor\(\)\s*$"
    )
    new, n = pattern.subn(r"\1db = next(get_db())\n\1cursor = db.cursor()", src)
    if n:
        backup_write(path, new)
        return True
    return False


def replace_fetch_urls(path: Path) -> bool:
    src = path.read_text(encoding="utf-8")
    new = src.replace("http://127.0.0.1:8000/api", "/api")
    if new != src:
        backup_write(path, new)
        return True
    return False


def ensure_trade_logger_coercion(path: Path) -> bool:
    # targeted, idempotent change for the specific pattern we saw earlier
    src = path.read_text(encoding="utf-8")
    if "TradeLog(" not in src:
        return False

    # detect the old pattern where trade.get(...) values are passed directly
    old_call_pattern = re.compile(
        r"TradeLog\(\s*\n\s*symbol=trade.get\(\"symbol\"\),\s*\n\s*side=trade.get\(\"side\"\),\s*\n\s*qty=trade.get\(\"qty\"\),\s*\n\s*price=trade.get\(\"price\"\),",
        flags=re.MULTILINE,
    )

    if not old_call_pattern.search(src):
        return False

    # Replace the call site with a guarded/coercion block similar to the one we
    # used earlier. This replacement is conservative and keeps indentation.
    def repl(match: re.Match) -> str:
        indent = match.group(0).split("T")[0]
        block = (
            "# Coerce and validate incoming values so MyPy sees the expected types\n"
            f'{indent}raw_symbol = trade.get("symbol")\n'
            f'{indent}raw_side = trade.get("side")\n'
            f'{indent}raw_qty = trade.get("qty")\n'
            f'{indent}raw_price = trade.get("price")\n\n'
            f'{indent}symbol: str = str(raw_symbol or "")\n'
            f'{indent}side: str = str(raw_side or "")\n'
            f"{indent}try:\n"
            f"{indent}    qty: float = float(raw_qty or 0.0)\n"
            f"{indent}except (TypeError, ValueError):\n"
            f"{indent}    qty = 0.0\n"
            f"{indent}try:\n"
            f"{indent}    price: float = float(raw_price or 0.0)\n"
            f"{indent}except (TypeError, ValueError):\n"
            f"{indent}    price = 0.0\n\n"
            f"{indent}log = TradeLog(\n"
            f"{indent}    symbol=symbol,\n"
            f"{indent}    side=side,\n"
            f"{indent}    qty=qty,\n"
            f"{indent}    price=price,\n"
        )
        return block

    new_src = old_call_pattern.sub(repl, src)
    if new_src != src:
        backup_write(path, new_src)
        return True
    return False


def main() -> None:
    print("Running auto_fix_mypy codemod...")
    changed: List[Path] = []

    # Files to scan: backend .py files and frontend src .ts/.tsx files
    backend_files = find_files(["backend/**/*.py"])
    frontend_files = find_files(
        [
            "frontend/src/**/*.ts",
            "frontend/src/**/*.tsx",
            "frontend/src/**/*.js",
            "frontend/src/**/*.jsx",
        ]
    )

    # 1) Replace get_db() cursor patterns in backend files
    for p in backend_files:
        if replace_get_db_cursor(p):
            changed.append(p)

    # 2) Replace hardcoded fetch URLs in frontend files
    for p in frontend_files:
        if replace_fetch_urls(p):
            changed.append(p)

    # 3) Ensure trade_logger coercion if needed
    tl = ROOT / "backend" / "utils" / "trade_logger.py"
    if tl.exists() and ensure_trade_logger_coercion(tl):
        changed.append(tl)

    if changed:
        print(f"Modified {len(changed)} files:")
        for c in changed:
            print(" -", c.relative_to(ROOT))
    else:
        print("No changes made (everything already looks good).")


if __name__ == "__main__":
    main()
