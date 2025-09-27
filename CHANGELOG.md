# Changelog

## 2025-09-23 — Frontend TypeScript migration cleanup

- Archived legacy frontend backups into `backups/migrated-archive/20250923` (non-destructive).
- Replaced archived `index.jsx` with a conservative re-export stub (backup kept as `index.jsx.bak`).
- Added TypeScript generator `scripts/generate-reexports.ts` and usage README.
- Moved tracked `.bak` files into `backups/migrated-archive/20250923/baks/` and removed them from their original tracked locations (PR #19).
- Updated `.gitignore` to ignore Vite artifacts and untracked build outputs; untracked build artifacts (PR #18).
- Verified locally: `tsc --noEmit`, `vite build`, and `vitest --run` all passed.

See PRs: #17 (migration prep), #18 (cleanup build artifacts), #19 (archive .bak files)

## 2025-09-27 — Backwards-compatible exchange API alias

- Added `get_adapter(name, ...)` alias in `backend.utils.exchanges` which wraps the canonical `get_exchange_client(...)` factory and emits a `DeprecationWarning` to signal deprecation.
- Added `backend/tests/test_exchanges_adapter_alias.py` to lock the alias behavior into CI.
- Migration note: update any external callers to use `get_exchange_client(...)`. Plan to remove `get_adapter(...)` in a future major release after callers have migrated.
