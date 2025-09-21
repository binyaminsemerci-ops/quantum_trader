Migration checklist for `ts-migration` branch

Overview
- This branch incrementally migrates frontend components to TypeScript with conservative changes.
- Purpose: minimize runtime risk while improving type coverage and enabling stricter checks.

What changed (high level)
- Converted/typed many small components: PriceChart, TradeList, SignalsList, ApiTest, StatsCard, AnalyticsCards, RiskCards, Watchlist, TradeLog/TradeLogs, Header, Sidebar, Toast, ErrorBanner.
- Centralized test setup for Vitest: `frontend/vitest.setup.ts` and `frontend/vitest.config.ts`.
- Tightened API helper typings in `frontend/src/utils/api.ts` and created a generic `ApiResponse<T>`.
- Centralized `StatSummary` into `frontend/src/types/index.ts` and used across dashboard hooks/components.

Review checklist
- Run typecheck locally
  - cd frontend
  - npx tsc --noEmit
- Run tests locally
  - cd frontend
  - npx vitest --run
- Files of interest (high priority review)
  - frontend/src/utils/api.ts (typed request wrapper)
  - frontend/src/types/index.ts (shared types)
  - frontend/src/components/dashboard/PriceChart.tsx (chart data guards)
  - frontend/src/components/analysis/SignalsList.tsx (API usage and runtime guards)
  - frontend/src/components/TradeList.tsx (safe id handling)
- Verify no visual regressions on the main dashboard (smoke test)
  - Start dev server per repo README and open the Dashboard page.

Notes & rationale
- Changes are intentionally conservative: prefer runtime guards, explicit casts when necessary, and keeping behavior unchanged.
- Some components remain re-exports (compatibility) to preserve existing import paths.
- Vitest setup centralization unblocks consistent use of `@testing-library/jest-dom` matchers across tests.

If you find an area that needs stricter typing, open an issue and I can follow up with a dedicated migration for that subsystem.
