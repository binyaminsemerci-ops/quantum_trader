## Summary

- Short description of the change

## Files migrated / changed

- List migrated files (for TypeScript migration PRs)

## Type notes

- Type coverage notes (any `any` left intentionally)

## Testing & QA

- Local commands run:
  - `npm --prefix frontend run typecheck`
  - `npm --prefix frontend run test:frontend`
  - `pytest -v backend/tests`
- Smoke tests performed (routes/pages verified)

## CI

- Link to CI run(s)

## Risk & rollback

- Risk summary
- Rollback plan: restore from `frontend/backups/` if needed

## Reviewers

- Add at least one reviewer familiar with trading logic if this touches orders/models
