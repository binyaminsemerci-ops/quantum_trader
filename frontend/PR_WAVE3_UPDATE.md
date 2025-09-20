Wave 3 (ts-migration): TradeForm + charts + trade history

- Files changed in this wave:
  - `frontend/src/components/TradeForm.tsx` — switched to typed `api.post`, added mounted guard and tighter form typing.
  - `frontend/src/components/dashboard/TradeHistory.tsx` — switched to `api.getTrades()`, hoisted fetch for Refresh button, added mounted guards and defensive parsing.
  - `frontend/src/components/OHLCVChart.tsx` — switched to `api.getChart()`, parsed `OHLCV[]` with safe timestamp coercion.
  - `frontend/src/components/EquityChart.tsx` — tightened hook return typing and guarded `chartData`.
  - `frontend/src/components/dashboard/PriceChart.tsx` — migrated to typed `api.getChart()` usage, added prop/state types and mounted guards.
  - `frontend/src/components/dashboard/PriceChart.tsx` — migrated to typed `api.getChart()` helper, added mounted guards and prop/state types.

Validation performed:
- `npx tsc --noEmit` (no new type errors locally)
- `npx vitest --run` (2 tests passing locally)

Notes for reviewers:
- Changes are conservative and additive — small runtime-safe guard additions.
- If CI shows issues with any tests, revert the specific file and I will iterate.

Artifacts:
- Headless dev screenshot: `artifacts/dev_screenshot_5173.png`
  - Fresh headless dev screenshot: `artifacts/dev_screenshot_5174.png`
  - New headless dev screenshot (IPv4 fix): `artifacts/dev_screenshot_5175.png`
- Fresh headless dev screenshot (new): `artifacts/dev_screenshot_5174.png`
