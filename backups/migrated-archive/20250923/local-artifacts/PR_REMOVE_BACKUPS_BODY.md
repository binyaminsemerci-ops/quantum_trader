Title: chore(frontend): move tracked .bak and App.restored.jsx into frontend/backups/legacy/

Description

This PR moves tracked backup files ("*.bak") and `App.restored.jsx` out of their original locations into a consolidated archive at `frontend/backups/legacy/`.

Why

- These files are leftover from the incremental JSX→TSX migration and were committed while we validated changes.
- Keeping them in the code paths increases noise and maintenance burden; archiving keeps history while cleaning the tree.

What changed

- Moved tracked backup files and `App.restored.jsx` to `frontend/backups/legacy/` (flattened filenames to avoid recreation of the original directory tree in the archive).
- No functional code changed; this is a bookkeeping-only commit.

Verification

- TypeScript typecheck: `tsc --noEmit` (ran locally on branch) — no errors.
- Unit tests (Vitest): `npm run test:frontend` — all tests passed locally on the branch.

Notes / Options

- Preservation of directory structure: currently the archive uses flattened filenames. If preferred, I can instead preserve the original directory layout under `frontend/backups/legacy/<original-path>` — tell me and I'll update the branch.
- Local untracked backups: there are untracked local copies (for example `tmp_backups/`). I did not delete them. If you want them removed permanently, say so and I'll remove them as an additional step.

Checklist

- [ ] Review moved files in `frontend/backups/legacy/`
- [ ] Confirm whether to preserve original directory hierarchy in the archive
- [ ] Merge PR when ready

Signed-off-by: migration bot
