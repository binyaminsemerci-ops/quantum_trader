Reviewer checklist for PR #1 (ts-migration)

Please use this checklist when reviewing the TypeScript migration changes in `ts-migration`.

- [ ] Run type-check locally: `cd frontend && npx tsc --noEmit`
- [ ] Run tests: `cd frontend && npx vitest --run`
- [ ] Start dev server and smoke the main dashboard/trades pages: `cd frontend && npm run dev`
- [ ] Inspect migrated components for:
  - no unsafe casts to `any` or `// @ts-ignore` left behind
  - conservative runtime guards (optional props are checked before use)
  - no change in runtime behavior (compare with `main` if unsure)
- [ ] API contract check: ensure backend endpoints still return shapes matching the typed helpers (getStats/getTrades/getChart). If not, prefer changing the small adapter in `frontend/src/utils/api.ts`.
- [ ] Security/edge cases: ensure no sensitive data is leaked or logged during migration (no credentials in logs).

If you'd like me to address review feedback I can apply small follow-up commits (2–4 files per wave) and keep the PR small and reviewable.
