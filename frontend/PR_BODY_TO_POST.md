Ready-to-post PR body for PR #1

Title: Incremental TypeScript migration (conservative waves) — ts-migration

Short summary

This PR performs a conservative, incremental TypeScript migration of frontend components and cross-cutting types and tooling. The focus is on small, safe waves that add runtime guards and stronger typings without changing runtime behavior.

Files/areas touched (representative)

- frontend/src/components/
  - BalanceCard (typed, defensive), TradeTable (typed), LoaderOverlay (props typed), TradeList, Header, Sidebar, AnalyticsCards, RiskCards, Watchlist, TradeLog, PriceChart (safety guards), SignalsList (conservative edits)
- frontend/src/utils/api.ts (typed request wrapper and domain helpers)
- frontend/src/types/index.ts (shared domain types)
- frontend/vitest.setup.ts, frontend/vitest.config.ts (test setup)
- frontend/PR_DESCRIPTION.md (this branch's summary)

Validation

- Type-check: npx tsc --noEmit (frontend) — no type diagnostics
- Tests: npx vitest --run — all existing frontend tests pass (2 tests)

Reviewer checklist

- Run the validation steps locally (see below).
- Inspect migrated components for unexpected any usage or changed behavior.
- Run dev server and spot-check the dashboard and trades pages.
- Verify that backend endpoints used by typed helpers (getStats/getTrades/getChart) return compatible payload shapes.

Commands to run locally

```pwsh
cd frontend
npm install
npx tsc --noEmit
npx vitest --run
npm run dev
```

Notes

- Migration approach: keep each change small and reversible. If a change causes test or runtime issues, revert and follow a smaller change path.
- Next steps: continue with small waves (2–4 files). Recommend next candidates: TradeHistory, OHLCVChart, EquityChart, TradeForm.

If you want, I'll post this as the PR body to PR #1 and add the reviewer checklist as a top-level comment; let me know and I'll proceed (requires GitHub API/CLI access).