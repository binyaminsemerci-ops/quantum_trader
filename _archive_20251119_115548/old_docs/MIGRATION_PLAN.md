Migration plan for converting frontend JS/JSX to TS/TSX

Goals
- Incrementally migrate components to TypeScript (.ts/.tsx) with minimal disruption.
- Keep existing imports working by replacing legacy `.jsx` files with re-export stubs pointing to canonical `.tsx` implementations.
- Automate repetitive tasks (stubbing) with a helper script and create backups for safety.

Phases
1. Prep
   - Ensure `frontend/tsconfig.json` includes only `src` and excludes backups/legacy.
   - Add `npm run typecheck` (done).
2. Wave migrations
   - Group components by areas (dashboard, trading, analysis, widgets).
   - For each component: create `Component.tsx` (typed conservatively), replace `Component.jsx` with `export { default } from './Component.tsx'`.
   - Run `npx tsc --noEmit -p frontend/tsconfig.json` and `npm run build --prefix frontend` after each wave.
3. Types consolidation
   - Create `frontend/src/types` for shared types (Trade, Signal, OHLCV, API responses).
   - Replace `any` gradually with tighter types.
4. CI & tests
   - Add `npm run typecheck` to CI and set up unit tests for migrated components.

Script: `scripts/generate-reexports.js`
- Scans `frontend/src` for `.jsx` files which have a matching `.tsx` or `.ts` file with the same basename.
- Backs up the original `.jsx` (copy to `frontend/backups/`), then replaces the `.jsx` with a safe re-export stub.

Rollback
- All replaced files are backed up in `frontend/backups/` with timestamped copies. To rollback, restore from backup.

Notes
- Work in small PRs (1–10 files) to make reviews straightforward.
- Keep `allowJs` disabled in tsconfig until the repo is fully migrated to avoid mixing JS into the typegraph.

Completed waves
----------------
- Wave 1 (safe utilities):
   - `frontend/src/utils/api.ts` (replaced `frontend/src/utils/api.js`)
   - `frontend/src/services/api.ts` (replaced `frontend/src/services/api.js`)
   - `frontend/src/services/twitterService.ts` (replaced `frontend/src/services/twitterService.js`)

These were migrated conservatively and validated locally with `tsc --noEmit` and Vitest.

Next focus areas
-----------------
- Surface the new gross exposure and position snapshots in the dashboard widgets once the telemetry TS types land.
