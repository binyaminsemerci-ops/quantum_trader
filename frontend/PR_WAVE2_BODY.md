Wave 2 TypeScript migration - conservative components and stubs

What changed
- Added three small TypeScript components: BalanceCard.tsx, StatsCard.tsx, ApiTest.tsx
- Replaced corresponding .jsx files with re-export stubs and created .bak backups
- Added a small re-export generator script at scripts/generate-reexports.js (dry-run by default)
- Added @types/react and @types/react-dom to frontend devDependencies
- Added a trivial Vitest smoke test so CI will run tests during PR

Validation performed
- Local TypeScript typecheck (tsc) ran successfully after fixes
- Vitest smoke test ran successfully with `vitest run --environment=node`

Notes
- The generator is conservative and requires `--yes` to actually write files.
- I avoided adding react-bootstrap types; SignalsList was converted to a plain HTML table to reduce dependency/type churn.
