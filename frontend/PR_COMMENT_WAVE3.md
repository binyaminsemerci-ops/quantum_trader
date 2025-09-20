Wave 3 (ts-migration) ready for review

Summary
- Migrated small components conservatively: `TradeForm`, `TradeHistory`, `OHLCVChart`, `EquityChart` and tidied `TradeList`.
- Added mounted guards, used typed `api` helpers, and avoided runtime changes.

Validation
- `npx tsc --noEmit` — no new type errors locally
- `npx vitest --run` — 2 tests passing locally

Artifacts
- `frontend/PR_WAVE3_UPDATE.md` (in-branch summary)
- Headless screenshot: `artifacts/dev_screenshot_5173.png`

Request
- Please run CI & review; smallest scope change to keep review quick. Happy to iterate on any feedback.
