Title: feat(frontend): convert remaining .jsx -> .tsx (batch)

Description

This PR converts the remaining `.jsx` source files under `frontend/src` to TypeScript (`.tsx`) and archives original `.jsx` files where a `.tsx` variant already existed.

What changed

- Renamed tracked `.jsx` -> `.tsx` where no `.tsx` counterpart existed.
- Archived original `.jsx` files into `frontend/backups/unconverted/<original-path>` when a `.tsx` already existed to avoid overwriting.
- Fixed a circular re-export and a handful of small typing issues discovered during conversion.

Why

- Consolidate the frontend onto TypeScript/TSX for stronger typing, safer refactors, and better editor support.
- Preserve originals for review or rollback while keeping the working tree clean.

Verification

- TypeScript: `npm run typecheck` (tsc --noEmit) — passes on branch.
- Unit tests: `npm run test:frontend` — passes locally on branch.

Notes

- The `frontend/backups/unconverted/` directory contains archived `.jsx` files. Inspect before deleting. If you prefer to preserve directory hierarchy differently, I can update the branch.
- If CI requires any additional adjustments (ESLint, build-time transforms, etc.), we can address them in follow-ups.

Checklist

- [ ] Review files in `frontend/backups/unconverted/`
- [ ] Run local smoke (dev server) and spot-check UI
- [ ] Merge after CI passes

Signed-off-by: migration bot
