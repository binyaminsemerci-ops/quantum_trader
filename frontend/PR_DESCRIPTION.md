Title: Incremental TypeScript migration (conservative waves) — ts-migration

Summary

This branch (ts-migration) contains a conservative, incremental TypeScript migration of selected frontend components and cross-cutting tooling improvements. The goal is to tighten types and reduce runtime casts while keeping all changes low-risk and well-tested.

What changed (high level)

- Conservative component typings and runtime guards across multiple small waves
  - BalanceCard, TradeTable, LoaderOverlay (typed and hardened)
  - TradeList, Header, Sidebar, AnalyticsCards, RiskCards, Watchlist, TradeLog (conservative migrations)
  - PriceChart and SignalsList safety hardening (URL encoding, NaN guards, axios retained where necessary)
- Centralized types
  - `frontend/src/types/index.ts` contains shared types (OHLCV, Signal, Trade, TradeSummary, StatSummary, ApiResponse)
- API helper typing
  - `frontend/src/utils/api.ts` now exposes conservative domain-typed helpers: `getStats(): ApiResponse<StatSummary>`, `getTrades(): ApiResponse<Trade[]>`, `getChart(): ApiResponse<OHLCV[]>` while keeping balance endpoints loosely typed.
- Testing/tooling
  - `frontend/vitest.setup.ts` and `frontend/vitest.config.ts` added to register jest-dom matchers and ensure jsdom environment for Vitest
  - Husky install preparation scripts and README fixes were added earlier in the migration (see PR checklist file)

Recent commits (most relevant)

- 80b7463 frontend(api): consume ApiResponse<T> in BalanceCard and ApiTest (safe consumers)
- 96f0122 frontend(api): add conservative domain types for api helpers (getTrades/getStats/getChart)
- 0d1ebef frontend: conservative typings - BalanceCard, TradeTable, LoaderOverlay
- 4cc9e1b chore(frontend): centralize StatSummary into types, tighten cross-cutting types, add PR migration checklist
- 0b4cf8b frontend: dashboard analytics - conservative typing for RiskCards/AnalyticsCards and ensure Watchlist/TradeLog compatibility

Validation performed

- Type-check: `npx tsc --noEmit` (frontend) — no type diagnostics after edits
- Unit tests: `npx vitest --run` — existing frontend tests passing (2 tests)
- Small smoke checks: local builds and manual spot-checking of components in dev server (recommended reviewer step)

How to run locally

From the repository root:

```pwsh
# frontend
cd frontend
npm install
npx tsc --noEmit
npx vitest --run
npm run dev # optionally to run dev server
```

Reviewer checklist

- [ ] Type-safety: spot-check moved components (BalanceCard, TradeTable, PriceChart, SignalsList, TradeList, AnalyticsCards) and ensure there are no unsafe any-casts altering behavior.
- [ ] Runtime behavior: run the dev server and confirm key pages render (dashboard, trades, signals) and the app doesn't log unexpected errors.
- [ ] Tests: run the local test suite and confirm green (see validation steps above).
- [ ] API contracts: confirm backend endpoints still return compatible payloads for typed helpers (getStats/getTrades/getChart). If the backend shape differs, recommend small adapter changes in `frontend/src/utils/api.ts` rather than UI changes.
- [ ] CI: ensure pipeline (if any) that runs type-check and tests passes for this branch.

Notes & follow-ups

- Migration strategy: prefer small waves of 2–4 files to keep reviews bite-sized. Components were migrated conservatively (guards, optional props, one-line re-exports kept where needed).
- Next actions: continue migrating another wave of components (recommend: `TradeHistory`, `OHLCVChart`, `EquityChart`, `TradeForm`) or tighten api helper signatures further and update direct callers.

If you'd like, I can post this as the PR body to PR #1 and add the reviewer checklist as a comment. Let me know and I'll post it and then start the next migration wave.
