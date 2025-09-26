# Frontend (quantum_trader)

Quick notes for developing the frontend locally.

Commands

Install dependencies:

```
npm install
```

Start dev server:

```
npm run dev
```

Typecheck (TypeScript):

```
npm run typecheck
```

Run frontend tests (Vitest):

```
npm run test:frontend
```

Notes

- The project uses Vite + React + TypeScript.
- Tests use Vitest; tests import vitest helpers explicitly where globals are not enabled.
- The `frontend/src/utils/position.ts` util contains a small position-size calculator used by `ChartView` and covered by tests.
